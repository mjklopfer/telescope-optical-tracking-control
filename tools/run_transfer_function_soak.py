#!/usr/bin/env python3
"""Run a reproducible camera + encoder + motor transfer-function soak.

The dashboard must already be in Transfer-Function Test mode with a valid
learned mapping and a live motor link. The runner validates learned matrix
signs against the configured artificial plant, turns tracking ON, samples the
dashboard for the requested duration, and always requests OFF in ``finally``.
It writes samples.jsonl, progress.json, and summary.json beneath a timestamped
directory so transient camera or Bridge failures remain inspectable.
"""

import argparse
import asyncio
import json
import math
from pathlib import Path
import time
import urllib.request


def read_status(url):
    with urllib.request.urlopen(url, timeout=4) as response:
        return json.load(response)


def post_command(url, payload):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode('utf-8'), method='POST',
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=6) as response:
        return json.load(response)


async def command(url, payload):
    return await asyncio.to_thread(post_command, url, payload)


async def wait_for_output(status_url, enabled, timeout=6):
    deadline = time.monotonic() + timeout
    latest = None
    while time.monotonic() < deadline:
        latest = await asyncio.to_thread(read_status, status_url)
        hardware = latest.get('hardware_loop') or {}
        output_matches = bool(hardware.get('enabled')) is bool(enabled)
        motion_settled = (enabled or hardware.get('motion') in
                          ('idle', 'paused', 'stale_target_stopped'))
        if output_matches and motion_settled:
            return latest
        await asyncio.sleep(.1)
    raise RuntimeError(
        f'Motor output did not confirm {"ON" if enabled else "OFF"} within {timeout:g}s')


async def wait_for_ready(status_url, timeout=20):
    """Wait through dashboard/mode startup instead of failing a valid test race."""
    deadline = time.monotonic() + timeout
    latest = None
    while time.monotonic() < deadline:
        latest = await asyncio.to_thread(read_status, status_url)
        hardware = latest.get('hardware_loop') or {}
        camera_ready = (int(latest.get('frame', -1)) >= 0
                        and latest.get('state') not in
                        ('STARTING', 'CAMERA RECOVERY', 'CAMERA OFFLINE'))
        if (camera_ready and latest.get('ui_mode') == 'transfer'
                and hardware.get('connection') == 'live'
                and hardware.get('feedback_live', True)
                and not hardware.get('enabled')
                and (latest.get('orientation_calibration') or {}).get('valid')):
            return latest
        await asyncio.sleep(.2)
    raise RuntimeError('Dashboard did not reach ready Transfer mode within '
                       f'{timeout:g}s; latest state={latest!r}')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')
    temporary.replace(path)


def signed_matrix_check(status):
    transfer = status.get('transfer_function_test') or {}
    coefficients = transfer.get('coefficients') or {}
    learned = (status.get('orientation_calibration') or {}).get('matrix_px_per_degree')
    if not learned or len(learned) != 2:
        raise RuntimeError('No learned 2x2 transfer mapping is loaded')
    expected = [
        [-float(coefficients['azimuth_x_px_per_degree']),
         -float(coefficients['elevation_x_px_per_degree'])],
        [-float(coefficients['azimuth_y_px_per_degree']),
         -float(coefficients['elevation_y_px_per_degree'])],
    ]
    comparisons = []
    for row in range(2):
        for column in range(2):
            actual = float(learned[row][column])
            target = expected[row][column]
            sign_ok = (target == 0 and abs(actual) < 1e-6) or actual * target > 0
            relative_error = (abs(actual - target) / abs(target)
                              if target else abs(actual))
            comparisons.append({
                'row': row, 'column': column, 'learned': actual,
                'expected': target, 'sign_ok': sign_ok,
                'relative_error': relative_error})
    if not all(item['sign_ok'] for item in comparisons):
        raise RuntimeError('Learned transfer matrix has an incorrect axis sign')
    return comparisons


async def run(args):
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    run_directory = Path(args.output_directory).expanduser() / stamp
    run_directory.mkdir(parents=True, exist_ok=False)
    sample_path = run_directory / 'samples.jsonl'
    progress_path = run_directory / 'progress.json'
    summary_path = run_directory / 'summary.json'
    started = time.time()
    initial = await wait_for_ready(args.status_url)
    hardware = initial.get('hardware_loop') or {}
    matrix_check = signed_matrix_check(initial)
    baseline_recoveries = int((initial.get('camera_recovery') or {}).get('attempts') or 0)
    baseline_reconnects = int(((hardware.get('bridge') or {}).get(
        'dashboard_reconnections')) or 0)
    initial_positions = hardware.get('encoder_positions') or {}
    initial_azimuth = float(initial_positions['Azimuth']['raw_degrees'])
    initial_elevation = float(initial_positions['Elevation']['raw_degrees'])
    metrics = {
        'samples': 0, 'status_fetch_failures': 0, 'non_live_link_samples': 0,
        'unexpected_output_off_samples': 0, 'error_samples': 0,
        'camera_metadata_missing_samples': 0, 'nonadvancing_frame_intervals': 0,
        'camera_recovery_increase': 0, 'bridge_reconnect_increase': 0,
        'frames_advanced': 0, 'max_azimuth_travel_degrees': 0.0,
        'max_elevation_travel_degrees': 0.0,
    }
    first_frame = last_frame = None
    last_progress = 0.0
    await command(args.command_url, {'action': 'set_hil_enabled', 'enabled': True})
    await wait_for_output(args.status_url, True)
    try:
        with sample_path.open('w', encoding='utf-8') as stream:
            deadline = time.monotonic() + args.duration
            while time.monotonic() < deadline:
                loop_started = time.monotonic()
                try:
                    status = await asyncio.to_thread(read_status, args.status_url)
                    hardware = status.get('hardware_loop') or {}
                    positions = hardware.get('encoder_positions') or {}
                    frame = int(status.get('frame', -1))
                    if first_frame is None:
                        first_frame = frame
                    if last_frame is not None and frame <= last_frame:
                        metrics['nonadvancing_frame_intervals'] += 1
                    last_frame = frame
                    metrics['frames_advanced'] = frame - first_frame
                    metrics['samples'] += 1
                    metrics['non_live_link_samples'] += (
                        hardware.get('connection') != 'live'
                        or not hardware.get('feedback_live', True))
                    metrics['unexpected_output_off_samples'] += not bool(hardware.get('enabled'))
                    error = status.get('error') or hardware.get('error')
                    metrics['error_samples'] += bool(error)
                    camera = status.get('camera_auto') or {}
                    metrics['camera_metadata_missing_samples'] += (
                        camera.get('exposure_time_us') is None)
                    recoveries = int((status.get('camera_recovery') or {}).get('attempts') or 0)
                    reconnects = int(((hardware.get('bridge') or {}).get(
                        'dashboard_reconnections')) or 0)
                    metrics['camera_recovery_increase'] = recoveries - baseline_recoveries
                    metrics['bridge_reconnect_increase'] = reconnects - baseline_reconnects
                    azimuth = float(positions['Azimuth']['raw_degrees'])
                    elevation = float(positions['Elevation']['raw_degrees'])
                    azimuth_delta = (azimuth - initial_azimuth + 180) % 360 - 180
                    elevation_delta = (elevation - initial_elevation + 180) % 360 - 180
                    metrics['max_azimuth_travel_degrees'] = max(
                        metrics['max_azimuth_travel_degrees'], abs(azimuth_delta))
                    metrics['max_elevation_travel_degrees'] = max(
                        metrics['max_elevation_travel_degrees'], abs(elevation_delta))
                    record = {
                        'utc_epoch_s': time.time(), 'frame': frame,
                        'state': status.get('state'),
                        'camera_exposure_time_us': camera.get('exposure_time_us'),
                        'camera_analogue_gain': camera.get('analogue_gain'),
                        'camera_recovery_attempts': recoveries,
                        'motor_link': hardware.get('connection'),
                        'encoder_feedback_live': hardware.get('feedback_live'),
                        'motor_output': hardware.get('enabled'),
                        'motion': hardware.get('motion'),
                        'bridge_reconnections': reconnects,
                        'azimuth_raw_degrees': azimuth,
                        'elevation_raw_degrees': elevation,
                        'error_x_px': status.get('error_x_px'),
                        'error_y_px': status.get('error_y_px'),
                        'error': error,
                    }
                except Exception as error:  # retain the soak through brief API loss
                    metrics['status_fetch_failures'] += 1
                    record = {'utc_epoch_s': time.time(), 'status_fetch_error': str(error)}
                stream.write(json.dumps(record, separators=(',', ':')) + '\n')
                stream.flush()
                elapsed = time.time() - started
                if elapsed - last_progress >= args.progress_interval:
                    progress = dict(
                        status='running', elapsed_seconds=elapsed,
                        requested_duration_seconds=args.duration,
                        run_directory=str(run_directory), metrics=dict(metrics))
                    write_json(progress_path, progress)
                    print(json.dumps(progress), flush=True)
                    last_progress = elapsed
                await asyncio.sleep(max(0.0, args.interval -
                                        (time.monotonic() - loop_started)))
    finally:
        try:
            await command(args.command_url, {'action': 'set_hil_enabled', 'enabled': False})
            await wait_for_output(args.status_url, False)
        except Exception as error:
            metrics['shutdown_error'] = str(error)
    ended = time.time()
    final = await asyncio.to_thread(read_status, args.status_url)
    summary = {
        'status': 'complete', 'started_utc_epoch_s': started,
        'ended_utc_epoch_s': ended, 'elapsed_seconds': ended - started,
        'requested_duration_seconds': args.duration,
        'matrix_sign_check': matrix_check, 'metrics': metrics,
        'final': {
            'camera_state': final.get('state'),
            'motor_link': (final.get('hardware_loop') or {}).get('connection'),
            'motor_output': (final.get('hardware_loop') or {}).get('enabled'),
            'motion': (final.get('hardware_loop') or {}).get('motion'),
            'camera_recovery': final.get('camera_recovery'),
        },
        'run_directory': str(run_directory),
    }
    write_json(summary_path, summary)
    write_json(progress_path, summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=1800)
    parser.add_argument('--interval', type=float, default=.5)
    parser.add_argument('--progress-interval', type=float, default=60)
    parser.add_argument('--status-url', default='http://127.0.0.1:8000/api/status')
    parser.add_argument('--command-url', default='http://127.0.0.1:8000/api/command')
    parser.add_argument('--output-directory', default='logs/transfer-function-soak')
    args = parser.parse_args()
    if args.duration <= 0 or args.interval <= 0 or args.progress_interval <= 0:
        parser.error('duration, interval and progress interval must be positive')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
