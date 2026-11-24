#!/usr/bin/env python3
"""Browser dashboard for live optical tracking and guarded synthetic evaluation.

The dashboard deliberately has no telescope or motor-control imports. It captures
one camera/video source, detects multiple bright or dark objects, assigns short-lived track
IDs, serves an MJPEG preview and sends timestamped telemetry over a WebSocket.

Examples:
  python tools/vision_dashboard.py --source synthetic --polarity dark
  python tools/vision_dashboard.py --source v4l2 --device /dev/video1
  ssh -L 8000:127.0.0.1:8000 mklopfer@raspberrypi.local

Then open http://127.0.0.1:8000. The server binds to loopback by default.
"""

import argparse
import asyncio
from collections import deque
from contextlib import asynccontextmanager
import csv
import glob
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time

import cv2
import numpy as np


def enumerate_cameras():
    """Return Pi/libcamera and Linux V4L2 capture devices."""
    cameras = []
    try:
        from picamera2 import Picamera2
        for index, info in enumerate(Picamera2.global_camera_info()):
            cameras.append({
                'device': f'picamera2:{index}',
                'name': f"{info.get('Model', 'Raspberry Pi CSI camera')} (Picamera2)",
                'settings': [
                    {'width': 640, 'height': 480, 'fps': 30.0},
                    {'width': 640, 'height': 480, 'fps': 20.0},
                    {'width': 1280, 'height': 720, 'fps': 15.0},
                ],
            })
    except (ImportError, RuntimeError):
        pass
    for device in sorted(glob.glob('/dev/video*'),
                         key=lambda value: int(re.search(r'(\d+)$', value).group(1))):
        number = re.search(r'(\d+)$', device).group(1)
        name_path = Path('/sys/class/video4linux') / f'video{number}' / 'name'
        try:
            name = name_path.read_text(encoding='utf-8').strip()
        except OSError:
            name = Path(device).name
        settings = []
        try:
            result = subprocess.run(
                ['v4l2-ctl', '-d', device, '--list-formats-ext'],
                capture_output=True, text=True, timeout=2, check=False)
            current_size = None
            for line in result.stdout.splitlines():
                size_match = re.search(r'Size: Discrete (\d+)x(\d+)', line)
                if size_match:
                    current_size = (int(size_match.group(1)), int(size_match.group(2)))
                    continue
                rate_match = re.search(r'Interval: Discrete [^(]+\(([0-9.]+) fps\)', line)
                if current_size and rate_match:
                    setting = {'width': current_size[0], 'height': current_size[1],
                               'fps': float(rate_match.group(1))}
                    if setting not in settings:
                        settings.append(setting)
        except (OSError, subprocess.SubprocessError):
            pass
        if settings:
            cameras.append({'device': device, 'name': name, 'settings': settings})
    return cameras


def create_feature_tracker(name, lock_size=False):
    """Create an OpenCV single-object tracker across main/legacy namespaces."""
    name = str(name).upper()
    if name not in ('CSRT', 'KCF', 'MOSSE'):
        raise ValueError('Feature tracker must be CSRT, KCF, or MOSSE')
    factory_name = f'Tracker{name}_create'
    factory = getattr(cv2, factory_name, None)
    if name == 'CSRT' and lock_size and factory and hasattr(cv2, 'TrackerCSRT_Params'):
        params = cv2.TrackerCSRT_Params()
        if hasattr(params, 'number_of_scales'):
            params.number_of_scales = 1
        if hasattr(params, 'scale_lr'):
            params.scale_lr = 0.0
        return factory(params)
    if factory is None and hasattr(cv2, 'legacy'):
        factory = getattr(cv2.legacy, factory_name, None)
    if factory is None:
        raise RuntimeError(f'OpenCV {name} tracker is not available in this environment')
    return factory()


def _json_camera_control_value(value):
    """Convert libcamera scalar, tuple and rectangle values to JSON-safe data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return [_json_camera_control_value(item) for item in value]
    if all(hasattr(value, field) for field in ('x', 'y', 'width', 'height')):
        return [int(value.x), int(value.y), int(value.width), int(value.height)]
    enum_value = getattr(value, 'value', None)
    if isinstance(enum_value, (bool, int, float, str)):
        return enum_value
    return str(value)


def normalize_camera_controls(controls):
    """Return driver-advertised minimum, maximum and default camera controls."""
    normalized = {}
    for name, information in (controls or {}).items():
        try:
            minimum, maximum, default = tuple(information)
        except (TypeError, ValueError):
            continue
        normalized[str(name)] = {
            'minimum': _json_camera_control_value(minimum),
            'maximum': _json_camera_control_value(maximum),
            'default': _json_camera_control_value(default),
        }
    return normalized


def hil_target_from_pixel(x, y, width, height, envelope_degrees):
    """Map a synthetic screen point into a bounded local azimuth/elevation target."""
    scale = float(envelope_degrees) * .9
    azimuth = (180 + (float(x) - width / 2) / (width / 2) * scale) % 360
    elevation = -(float(y) - height / 2) / (height / 2) * scale
    return azimuth, elevation


def hil_marker_from_status(status, width, height, envelope_degrees):
    """Map encoder positions in the local test reference back into screen pixels."""
    positions = status.get('positions', {})
    azimuth = positions.get('Azimuth', {}).get('reference_degrees')
    elevation = positions.get('Elevation', {}).get('reference_degrees')
    if azimuth is None or elevation is None:
        return None
    azimuth_delta = (float(azimuth) - 180 + 180) % 360 - 180
    scale = float(envelope_degrees) * .9
    return {'x': width / 2 + azimuth_delta / scale * width / 2,
            'y': height / 2 - float(elevation) / scale * height / 2,
            'azimuth_degrees': float(azimuth),
            'elevation_degrees': float(elevation)}


def learn_pixel_mount_transform(azimuth_vector, elevation_vector, step_degrees):
    """Build pixel/degree and degree/pixel matrices from two axis probes."""
    if isinstance(step_degrees, (tuple, list)):
        azimuth_step, elevation_step = map(float, step_degrees)
    else:
        azimuth_step = elevation_step = float(step_degrees)
    if not all(.003 <= abs(value) <= 1.2
               for value in (azimuth_step, elevation_step)):
        raise ValueError('Measured mapping movement must be 0.003–1.2 degrees')
    az_dx, az_dy = map(float, azimuth_vector)
    el_dx, el_dy = map(float, elevation_vector)
    if min(math.hypot(az_dx, az_dy), math.hypot(el_dx, el_dy)) < 4:
        raise ValueError('Probe motion was under 4 pixels; select a larger safe probe')
    a, c = az_dx / azimuth_step, az_dy / azimuth_step
    b, d = el_dx / elevation_step, el_dy / elevation_step
    determinant = a * d - b * c
    if abs(determinant) < .15 * math.hypot(a, c) * math.hypot(b, d):
        raise ValueError('Measured axis vectors are nearly parallel; lock a stable feature and retry')
    inverse = [[d / determinant, -b / determinant],
               [-c / determinant, a / determinant]]
    return {'matrix_px_per_degree': [[a, b], [c, d]],
            'inverse_degrees_per_pixel': inverse,
            'azimuth_vector_px': [az_dx, az_dy],
            'elevation_vector_px': [el_dx, el_dy],
            'requested_step_degrees': max(abs(azimuth_step), abs(elevation_step)),
            'measured_step_degrees': [azimuth_step, elevation_step]}


def wrapped_encoder_delta(current, baseline):
    """Return the shortest signed delta between two 24-bit encoder readings."""
    modulus = 1 << 24
    return (int(current) - int(baseline) + modulus // 2) % modulus - modulus // 2


def learned_mount_delta(error_x, error_y, calibration):
    """Return mount movement needed to drive the measured pixel error to zero."""
    inverse = calibration['inverse_degrees_per_pixel']
    return (-(inverse[0][0] * error_x + inverse[0][1] * error_y),
            -(inverse[1][0] * error_x + inverse[1][1] * error_y))


class BrightObjectTracker:
    """Detect bright/dark contours and associate them with temporary track IDs."""

    def __init__(self, threshold=160, min_area=2, max_objects=20,
                 match_distance=80, max_missed=8, polarity='bright'):
        self.threshold = int(threshold)
        self.min_area = float(min_area)
        self.max_objects = int(max_objects)
        self.match_distance = float(match_distance)
        self.max_missed = int(max_missed)
        self.tracks = {}
        self.next_id = 1
        self.selected_id = None
        self.last_detections = []
        self.set_polarity(polarity)

    def set_threshold(self, value):
        value = int(value)
        if not 1 <= value <= 254:
            raise ValueError('Threshold must be between 1 and 254')
        self.threshold = value

    def set_polarity(self, value):
        value = str(value).lower()
        if value not in ('bright', 'dark'):
            raise ValueError('Polarity must be bright or dark')
        self.polarity = value
        self.tracks.clear()
        self.selected_id = None
        self.last_detections = []

    def _candidates(self, gray):
        threshold_mode = (cv2.THRESH_BINARY if self.polarity == 'bright'
                          else cv2.THRESH_BINARY_INV)
        _level, mask = cv2.threshold(gray, self.threshold, 255, threshold_mode)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[-2]
        found = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue
            moments = cv2.moments(contour)
            if moments['m00'] <= 0:
                continue
            x = float(moments['m10'] / moments['m00'])
            y = float(moments['m01'] / moments['m00'])
            bx, by, width, height = cv2.boundingRect(contour)
            brightness = float(cv2.mean(gray, mask=cv2.drawContours(
                np.zeros_like(gray), [contour], -1, 255, -1))[0])
            contrast = ((brightness - self.threshold) / max(1, 255 - self.threshold)
                        if self.polarity == 'bright'
                        else (self.threshold - brightness) / max(1, self.threshold))
            confidence = min(1.0, max(0.0, contrast) *
                             min(1.0, area / 20.0))
            found.append(dict(x=x, y=y, bbox=[int(bx), int(by), int(width), int(height)],
                              area=area, brightness=brightness, confidence=confidence))
        found.sort(key=lambda item: item['area'], reverse=True)
        return found[:self.max_objects]

    def detect(self, gray):
        candidates = self._candidates(gray)
        available = set(self.tracks)
        assignments = []
        remaining = list(candidates)

        def assign(candidate, track_id):
            prior = self.tracks.get(track_id)
            if prior:
                observed_dx = candidate['x'] - prior['x']
                observed_dy = candidate['y'] - prior['y']
                candidate['vx'] = .75 * prior.get('vx', 0) + .25 * observed_dx
                candidate['vy'] = .75 * prior.get('vy', 0) + .25 * observed_dy
            else:
                candidate['vx'] = candidate['vy'] = 0.0
            candidate['track_id'] = track_id
            candidate['missed'] = 0
            assignments.append(candidate)

        # Reserve the selected ID first. This prevents a larger distractor from
        # stealing it when a blinking target reappears near its predicted path.
        if self.selected_id in available and remaining:
            track = self.tracks[self.selected_id]
            gate = min(self.match_distance, 30) * (1 + min(track.get('missed', 0), 8) * .05)
            candidate = min(remaining, key=lambda item: math.hypot(
                item['x'] - track['x'], item['y'] - track['y']))
            distance = math.hypot(candidate['x'] - track['x'], candidate['y'] - track['y'])
            if distance <= gate:
                assign(candidate, self.selected_id)
                remaining.remove(candidate)
                available.remove(self.selected_id)

        for candidate in remaining:
            nearest = None
            nearest_distance = self.match_distance
            for track_id in available:
                track = self.tracks[track_id]
                distance = math.hypot(candidate['x'] - track['x'],
                                      candidate['y'] - track['y'])
                if distance <= nearest_distance:
                    nearest, nearest_distance = track_id, distance
            if nearest is None:
                nearest = self.next_id
                self.next_id += 1
            else:
                available.remove(nearest)
            assign(candidate, nearest)

        for track_id, track in list(self.tracks.items()):
            if not any(item['track_id'] == track_id for item in assignments):
                track = dict(track)
                track['missed'] = track.get('missed', 0) + 1
                if track['missed'] <= self.max_missed:
                    track['x'] += track.get('vx', 0)
                    track['y'] += track.get('vy', 0)
                    track['bbox'][0] = round(track['bbox'][0] + track.get('vx', 0))
                    track['bbox'][1] = round(track['bbox'][1] + track.get('vy', 0))
                    self.tracks[track_id] = track
                else:
                    del self.tracks[track_id]
        for item in assignments:
            self.tracks[item['track_id']] = dict(item)

        self.last_detections = assignments
        return [dict(item, selected=item['track_id'] == self.selected_id)
                for item in assignments]

    def select(self, track_id):
        track_id = int(track_id)
        if not any(item['track_id'] == track_id for item in self.last_detections):
            raise ValueError('That object is no longer visible')
        self.selected_id = track_id
        return track_id

    def select_at(self, x, y):
        x, y = float(x), float(y)
        if not self.last_detections:
            raise ValueError('No visible objects to select')
        inside = []
        for item in self.last_detections:
            bx, by, width, height = item['bbox']
            if bx <= x <= bx + width and by <= y <= by + height:
                inside.append(item)
        choices = inside or self.last_detections
        selected = min(choices, key=lambda item: math.hypot(item['x'] - x,
                                                             item['y'] - y))
        self.selected_id = selected['track_id']
        return self.selected_id

    def clear_selection(self):
        self.selected_id = None


class VisionDashboardEngine:
    def __init__(self, source='synthetic', device='/dev/video0', width=640,
                 height=480, fps=20, threshold=160, jpeg_quality=65,
                 log_directory='vision-dashboard-logs', polarity='bright',
                 hil_host=None, hil_port=2000, hil_envelope_degrees=4,
                 hil_dispatch_rate_hz=4, hil_display_span_degrees=5,
                 ui_mode='demo'):
        self.source, self.device = source, device
        self.desired_source = source
        self.ui_mode = ui_mode
        self.aim_mode = 'center'
        self.aim_point = (.5, .5)
        self.live_lock_box_fraction = .12
        self.image_horizontal_reversed = False
        self.image_vertical_reversed = False
        self.last_selected = None
        self.feature_samples = deque(maxlen=240)
        self.orientation_history = deque(maxlen=120)
        self.orientation_sanity = {'status': 'idle', 'message': None}
        self.orientation_calibration = {
            'status': 'not learned', 'valid': False,
            'message': 'Lock a stable feature, then turn tracking ON to run soft start.'}
        self.orientation_calibration_thread = None
        self.orientation_cancel = threading.Event()
        self.soft_start_step_degrees = .5
        self.soft_start_slew_rate = 1
        self.soft_start_ramp_started = None
        data_root = os.getenv('TELESCOPE_CONTROL_DATA_DIR', os.getenv(
            'COSMOS90GT_DATA_DIR', Path.home() / '.telescope-control'))
        self.orientation_profile_path = (Path(data_root) /
            'vision-orientation-profiles.json')
        self.bridge_status = {'management_reachable': False, 'verified_proxy': False,
                              'message': 'Bridge status not checked yet.'}
        self.bridge_status_next_poll = 0.0
        self.bridge_monitor_thread = None
        self.bridge_manual_reconnects = 0
        self.bridge_service_restart_thread = None
        self.bridge_service_restart_in_progress = False
        self.bridge_service_restart_count = 0
        self.bridge_service_restart_result = None
        self.cameras = enumerate_cameras()
        if str(self.device).lower() == 'auto' and self.cameras:
            preferred = next((camera for camera in self.cameras
                              if 'usb' in camera['name'].lower()
                              or 'uvc' in camera['name'].lower()), self.cameras[0])
            self.device = preferred['device']
        if str(self.device).startswith('picamera2:'):
            self.source = self.desired_source = 'picamera2'
        self.camera_reopen_requested = False
        self.camera_recovery_requested = False
        self.camera_recovery_count = 0
        self.camera_metadata = {}
        self.camera_control_ranges = {}
        self.camera_auto_exposure = True
        self.camera_exposure_value = 0.0
        self.camera_manual_exposure_us = 10000
        self.camera_manual_gain = 1.0
        self.curve_black = 0
        self.curve_white = 255
        self.curve_gamma = 1.0
        self.width, self.height, self.requested_fps = int(width), int(height), float(fps)
        self.jpeg_quality = int(jpeg_quality)
        self.tracker = BrightObjectTracker(threshold=threshold, polarity=polarity,
                                           max_missed=max(8, round(float(fps) * 2.5)))
        self.tracking_mode = 'feature-roi' if ui_mode == 'live' else 'threshold'
        self.feature_algorithm = 'CSRT'
        self.feature_lock_size = True
        self.feature_initial_size = None
        self.feature_reference_patch = None
        self.feature_similarity_threshold = .22
        self.feature_similarity_baseline = 1.0
        self.feature_invalid_frames = 0
        self.feature_reacquiring = False
        self.feature_reacquire_candidate = None
        self.feature_reacquire_confirmations = 0
        self.feature_last_confirmed = None
        self.feature_tracker = None
        self.feature_roi_pending = None
        self.feature_selected_id = None
        self.log_directory = Path(log_directory).expanduser()
        self.lock = threading.RLock()
        self.frame_ready = threading.Condition(self.lock)
        self.stop_event = threading.Event()
        self.thread = None
        self.capture = None
        self.jpeg = None
        self.telemetry = {'frame': -1, 'state': 'STARTING', 'motor_output': 'DISABLED'}
        self.record_requested = False
        self.record_path = None
        self.record_stream = None
        self.record_writer = None
        self.errors = deque(maxlen=1200)
        self.hil_host = hil_host
        self.hil_port = int(hil_port)
        self.hil_envelope_degrees = float(hil_envelope_degrees)
        self.hil_dispatch_rate_hz = float(hil_dispatch_rate_hz)
        self.hil_display_span_degrees = float(hil_display_span_degrees)
        if not .1 <= self.hil_display_span_degrees <= 10:
            raise ValueError('Display span must be 0.1–10 degrees')
        self.hil_controller = None
        self.hil_enabled = False
        self.hil_link_action = None
        self.hil_link_action_thread = None
        self.hil_link_action_cancel = threading.Event()
        self.hil_sequence = 0
        self.hil_last_submit = 0.0
        self.hil_previous_target = None
        self.hil_target_rate = None
        self.hil_last_visible_target = None
        self.hil_current_target = None
        self.hil_loss_mode = 'stop'
        self.hil_coast_seconds = .5
        self.hil_last_locked_marker = None
        self.hil_return_started = None
        self.vibration_filter_enabled = False
        self.vibration_threshold_px = 3.0
        self.vibration_filtered_point = None
        self.vibration_candidate = None
        self.vibration_candidate_frames = 0
        self.hil_control_preset = 'off'
        self.hil_prediction_horizon = 0.0
        self.hil_proportional_lead = 0.0
        self.hil_correction_deadband = 0.0
        self.hil_auto_horizon = .1
        self.hil_commanded_target = None
        self.hil_error = None
        self.synthetic_blink = {'enabled': False, 'fade_out_seconds': .5,
                                'hidden_seconds': .5, 'fade_in_seconds': .5,
                                'period_seconds': 4.0, 'phase': 'steady',
                                'intensity': None}
        self._load_orientation_profile()

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        # Motor/link state is independent of camera health.  Establish it
        # first so a failed CSI open still leaves a usable dashboard shell and
        # an active Recover Camera action.
        if self.hil_host and not self.hil_controller:
            self.hil_controller = self._new_position_controller()
        if self.hil_host and not (self.bridge_monitor_thread and
                                  self.bridge_monitor_thread.is_alive()):
            self.bridge_monitor_thread = threading.Thread(
                target=self._bridge_monitor, name='bridge-status', daemon=True)
            self.bridge_monitor_thread.start()
        self.camera_recovery_requested = False
        self.thread = threading.Thread(target=self._run, name='vision-capture', daemon=True)
        self.thread.start()

    def _new_position_controller(self):
        from position_control import PositionController
        controller = PositionController(
            self.hil_host, self.hil_port,
            envelope_degrees=self.hil_envelope_degrees,
            dispatch_rate_hz=self.hil_dispatch_rate_hz)
        controller.start()
        return controller

    def stop(self):
        self.orientation_cancel.set()
        self.stop_event.set()
        with self.frame_ready:
            self.frame_ready.notify_all()
        if self.thread:
            self.thread.join(3)
        if self.hil_controller:
            self.hil_controller.stop()
            if self.hil_controller.thread:
                self.hil_controller.thread.join(4)
        if self.bridge_monitor_thread:
            self.bridge_monitor_thread.join(2)
        self._close_capture()
        self._close_recording()

    def _orientation_profile_key(self):
        return f'{self.hil_host or "display-only"}|{self.device}|{self.width}x{self.height}'

    def _load_orientation_profile(self):
        try:
            payload = json.loads(self.orientation_profile_path.read_text(encoding='utf-8'))
            profile = payload.get(self._orientation_profile_key())
            if profile:
                profile = dict(profile)
                profile.update(status='warm profile available', valid=False,
                               warm_profile=True,
                               message=('Saved camera-to-mount mapping found. Tracking ON will '
                                        'validate it with bounded probe movements before use.'))
                self.orientation_calibration = profile
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return

    def _save_orientation_profile(self, calibration):
        self.orientation_profile_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            payload = json.loads(self.orientation_profile_path.read_text(encoding='utf-8'))
        except (OSError, ValueError, json.JSONDecodeError):
            payload = {}
        profile = {key: value for key, value in calibration.items()
                   if key not in ('status', 'valid', 'message', 'phase', 'warm_profile')}
        profile.update(camera_device=self.device, width=self.width, height=self.height,
                       telescope_host=self.hil_host, saved_utc_epoch_s=time.time())
        payload[self._orientation_profile_key()] = profile
        temporary = self.orientation_profile_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
        temporary.replace(self.orientation_profile_path)

    def clear_orientation_profile(self):
        if self.hil_enabled or (self.orientation_calibration_thread
                                and self.orientation_calibration_thread.is_alive()):
            raise ValueError('Turn tracking output OFF and wait for soft start to finish')
        try:
            payload = json.loads(self.orientation_profile_path.read_text(encoding='utf-8'))
        except (OSError, ValueError, json.JSONDecodeError):
            payload = {}
        payload.pop(self._orientation_profile_key(), None)
        self.orientation_profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.orientation_profile_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
        self._invalidate_orientation('Saved mapping cleared; run soft start before tracking.')

    def _discard_saved_orientation(self):
        try:
            payload = json.loads(self.orientation_profile_path.read_text(encoding='utf-8'))
            payload.pop(self._orientation_profile_key(), None)
            self.orientation_profile_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    def _return_to_last_lock_after_fault(self):
        marker = self.hil_last_locked_marker
        if not marker or not self.hil_controller:
            return
        try:
            self.orientation_cancel.clear()
            self._probe_target(marker['azimuth_degrees'], marker['elevation_degrees'], timeout=4)
            self.hil_controller.pause()
        except (ConnectionError, RuntimeError, ValueError):
            self.hil_controller.pause()

    def _poll_bridge_management(self, force=False):
        now = time.monotonic()
        if not self.hil_host or (not force and now < self.bridge_status_next_poll):
            return self.bridge_status
        self.bridge_status_next_poll = now + 2.0
        sock = None
        try:
            sock = socket.create_connection((self.hil_host, 3000), .35)
            sock.settimeout(.35)

            def read(command):
                sock.sendall(command.encode('ascii') + b'\n')
                data = bytearray()
                deadline = time.monotonic() + .7
                while len(data) < 16384 and time.monotonic() < deadline:
                    chunk = sock.recv(min(4096, 16384 - len(data)))
                    if not chunk:
                        break
                    data.extend(chunk)
                    if b'\n> ' in data:
                        break
                return data.decode('utf-8', 'replace').strip().rstrip('>').strip()

            version = read('version')
            if not version.startswith('COSMOSBridge-AMW007-'):
                self.bridge_status = {
                    'management_reachable': True, 'verified_proxy': False,
                    'version': version,
                    'message': 'Endpoint answered on management port but is not the verified Control Bridge.'}
            else:
                identity = json.loads(read('get cosmos.proxy'))
                telemetry = json.loads(read('get cosmos.telemetry'))
                self.bridge_status = {
                    'management_reachable': True, 'verified_proxy': True,
                    'version': version, 'ready': bool(identity.get('ready')),
                    'data_port': identity.get('data_port'),
                    'management_port': identity.get('management_port'),
                    'cached_positions': telemetry.get('positions', {}),
                    'message': ('Bridge management API verified.' if identity.get('ready')
                                else 'Bridge process replied but is not ready.')}
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.bridge_status = {
                'management_reachable': False, 'verified_proxy': False,
                'message': f'Bridge management API unavailable: {error}'}
        finally:
            if sock:
                sock.close()
        return self.bridge_status

    def _bridge_monitor(self):
        while not self.stop_event.is_set():
            self._poll_bridge_management(force=True)
            self.stop_event.wait(2.0)

    def reconnect_bridge_client(self):
        """Restart this dashboard's motor link without restarting Control Bridge."""
        if not self.hil_controller:
            raise ValueError('No bridge endpoint was configured at launch')
        if self.hil_enabled:
            self.set_hil_enabled(False)
        old_controller = self.hil_controller
        old_controller.pause()
        old_controller.stop()
        if old_controller.thread:
            old_controller.thread.join(5)
        if old_controller.thread and old_controller.thread.is_alive():
            raise RuntimeError('Previous dashboard connection did not close; retry after it stops')
        self.hil_controller = self._new_position_controller()
        self.bridge_manual_reconnects += 1
        self.hil_previous_target = None
        self.hil_last_visible_target = None
        self.hil_current_target = None
        self.hil_commanded_target = None
        self.hil_error = 'Dashboard bridge connection restarted; waiting for both fresh encoders.'
        self.bridge_status_next_poll = 0.0
        return self.hil_controller.status()

    def restart_bridge_service(self):
        """Restart the Pi user service, then rebuild and verify the motor link.

        The command and unit name are fixed; the browser cannot provide shell
        input.  This works without sudo because both services run under the
        same systemd user manager as the dashboard.
        """
        if self.bridge_service_restart_thread and self.bridge_service_restart_thread.is_alive():
            return
        if not self.hil_controller:
            raise ValueError('No bridge endpoint was configured at launch')
        if self.hil_enabled:
            self.set_hil_enabled(False)
        self.bridge_service_restart_in_progress = True
        self.bridge_service_restart_result = 'Restart requested; stopping the old motor link.'
        self.hil_error = self.bridge_service_restart_result

        def worker():
            try:
                self.hil_controller.pause()
                result = subprocess.run(
                    ['/usr/bin/systemctl', '--user', 'restart', 'celestron-bridge.service'],
                    capture_output=True, text=True, timeout=20, check=False)
                if result.returncode:
                    detail = (result.stderr or result.stdout).strip()
                    raise RuntimeError(
                        'Control Bridge service restart failed' +
                        (f': {detail}' if detail else '.'))
                self.bridge_service_restart_count += 1
                self.bridge_service_restart_result = (
                    'Control Bridge restarted; waiting for its management endpoint.')
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    status = self._poll_bridge_management(force=True)
                    if status.get('verified_proxy'):
                        break
                    time.sleep(.25)
                else:
                    raise ConnectionError(
                        'Control Bridge restarted, but its management endpoint did not return.')

                self.reconnect_bridge_client()
                self.bridge_service_restart_result = (
                    'Control Bridge is running; waiting for fresh motor feedback.')
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    if self.hil_controller.status().get('connection') == 'live':
                        self.bridge_service_restart_result = (
                            'Control Bridge and motor link recovered successfully.')
                        self.hil_error = None
                        return
                    time.sleep(.1)
                raise ConnectionError(
                    'Control Bridge restarted, but fresh replies from both motors did not arrive.')
            except (OSError, subprocess.SubprocessError, ConnectionError,
                    RuntimeError, ValueError) as error:
                self.bridge_service_restart_result = str(error)
                self.hil_error = str(error)
            finally:
                self.bridge_service_restart_in_progress = False

        self.bridge_service_restart_thread = threading.Thread(
            target=worker, name='bridge-service-restart', daemon=True)
        self.bridge_service_restart_thread.start()

    def _open_capture(self):
        if self.source == 'synthetic':
            return None
        if self.source == 'picamera2':
            from picamera2 import Picamera2
            index = int(str(self.device).split(':', 1)[1])
            capture = Picamera2(index)
            self.camera_control_ranges = normalize_camera_controls(
                capture.camera_controls)
            configuration = capture.create_video_configuration(
                main={'size': (self.width, self.height), 'format': 'RGB888'},
                controls={'FrameRate': self.requested_fps})
            capture.configure(configuration)
            controls = {'AeEnable': self.camera_auto_exposure,
                        'AwbEnable': True,
                        'ExposureValue': self.camera_exposure_value}
            if not self.camera_auto_exposure:
                controls.update(ExposureTime=self.camera_manual_exposure_us,
                                AnalogueGain=self.camera_manual_gain)
            capture.set_controls(controls)
            capture.start()
            return capture
        self.camera_control_ranges = {}
        device = int(self.device) if str(self.device).isdigit() else self.device
        capture = cv2.VideoCapture(device)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.requested_fps)
        if not capture.isOpened():
            raise RuntimeError('Camera/video source could not be opened: ' + str(self.device))
        return capture

    def _close_capture(self):
        capture, self.capture = self.capture, None
        if not capture:
            return
        if self.source == 'picamera2':
            try:
                capture.stop()
            finally:
                capture.close()
        else:
            capture.release()

    def _synthetic(self, index):
        if self.tracker.polarity == 'dark':
            frame = np.random.default_rng(index).normal(
                247, 2, (self.height, self.width)).clip(0, 255).astype(np.uint8)
            values = (10, 35, 65)
        else:
            frame = np.random.default_rng(index).normal(
                8, 2, (self.height, self.width)).clip(0, 255).astype(np.uint8)
            values = (245, 215, 190)
        t = index / max(1, self.requested_fps)
        objects = (
            (self.width * (.30 + .18 * math.sin(t * 1.1)),
             self.height * (.45 + .22 * math.sin(t * .8)), 6, values[0]),
            (self.width * (.70 + .12 * math.sin(t * .7 + 1.2)),
             self.height * (.55 + .18 * math.sin(t * 1.3 + .4)), 9, values[1]),
            (self.width * (.50 + .30 * math.sin(t * .35 + 2.1)),
             self.height * (.25 + .08 * math.sin(t * 1.7)), 4, values[2]),
        )
        blink_value = values[1]
        blink_phase = 'steady'
        if self.synthetic_blink['enabled']:
            phase = t % self.synthetic_blink['period_seconds']
            fade_out = self.synthetic_blink['fade_out_seconds']
            hidden = self.synthetic_blink['hidden_seconds']
            fade_in = self.synthetic_blink['fade_in_seconds']
            background = 247 if self.tracker.polarity == 'dark' else 8
            if phase < fade_out:
                fraction = phase / fade_out
                blink_value = round(values[1] + (background - values[1]) * fraction)
                blink_phase = 'fading_out'
            elif phase < fade_out + hidden:
                blink_value = background
                blink_phase = 'hidden'
            elif phase < fade_out + hidden + fade_in:
                fraction = (phase - fade_out - hidden) / fade_in
                blink_value = round(background + (values[1] - background) * fraction)
                blink_phase = 'fading_in'
            else:
                blink_phase = 'visible'
        self.synthetic_blink['phase'] = blink_phase
        self.synthetic_blink['intensity'] = blink_value
        for object_index, (x, y, radius, value) in enumerate(objects):
            if object_index == 1:
                value = blink_value
            cv2.circle(frame, (int(x), int(y)), radius, value, -1)
        return frame

    def _open_recording(self):
        self.log_directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        self.record_path = self.log_directory / f'vision-track-{stamp}.csv'
        self.record_stream = self.record_path.open('w', newline='', encoding='utf-8')
        self.record_writer = csv.writer(self.record_stream)
        self.record_writer.writerow((
            'utc_epoch_s', 'monotonic_ns', 'frame', 'state', 'selected_track_id',
            'x_px', 'y_px', 'error_x_px', 'error_y_px', 'bbox_x', 'bbox_y',
            'bbox_width', 'bbox_height', 'confidence', 'brightness',
            'object_count', 'capture_fps', 'processing_ms'))

    def _close_recording(self):
        if self.record_stream:
            self.record_stream.flush()
            self.record_stream.close()
        self.record_stream = self.record_writer = None

    def set_recording(self, enabled):
        with self.lock:
            self.record_requested = bool(enabled)

    def set_hil_enabled(self, enabled):
        if not self.hil_controller:
            raise ValueError('Hardware-loop mode was not enabled at launch')
        if enabled:
            status = self.hil_controller.status()
            if status.get('connection') != 'live':
                raise ValueError('Motor link is not live; wait for fresh encoder feedback')
            if self.ui_mode == 'live' and not self.orientation_calibration.get('valid'):
                # Camera rotation and inversion must be measured.  Do not guess
                # from manual Normal/Reversed switches: start the two-axis
                # mapping routine, which arms tracking only after it succeeds.
                self.start_orientation_learning(self.soft_start_step_degrees)
                return
            if not status.get('local_reference_set'):
                self.hil_controller.reference(180, 0)
            self.hil_controller.arm()
            self.hil_enabled = True
            self.hil_error = None
            self.soft_start_ramp_started = time.monotonic()
            self.orientation_history.clear()
            self.orientation_sanity = {'status': 'monitoring', 'message': None}
        else:
            self.hil_link_action_cancel.set()
            self.hil_link_action = None
            self.orientation_cancel.set()
            self.hil_enabled = False
            self.hil_controller.pause()
            self.soft_start_ramp_started = None
            self.orientation_history.clear()
            self.orientation_sanity = {'status': 'idle', 'message': None}

    def _run_when_motor_link_live(self, action, step=None, rate=None):
        """Reconnect stale motor feedback, then perform the requested action."""
        if not self.hil_controller:
            raise ValueError('Motor control was not configured at launch')
        if self.hil_link_action_thread and self.hil_link_action_thread.is_alive():
            return
        if action == 'learn' and not self.last_selected:
            raise ValueError('Lock a stable visible feature before learning the mapping')
        if self.hil_controller.status().get('connection') == 'live':
            if action == 'enable':
                self.set_hil_enabled(True)
            else:
                self.start_orientation_learning(step, rate)
            return
        self.hil_link_action_cancel.clear()
        self.hil_link_action = action
        self.hil_error = ('Reconnecting motor control; waiting for fresh azimuth and '
                          'elevation encoder replies.')

        def worker():
            try:
                self.reconnect_bridge_client()
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline and not self.hil_link_action_cancel.is_set():
                    if self.hil_controller.status().get('connection') == 'live':
                        if action == 'enable':
                            self.set_hil_enabled(True)
                        else:
                            self.start_orientation_learning(step, rate)
                        return
                    time.sleep(.1)
                if not self.hil_link_action_cancel.is_set():
                    raise ConnectionError(
                        'Motor reconnect timed out; no fresh reply from both axes. '
                        'Press RESUME / ON to retry.')
            except (ConnectionError, RuntimeError, ValueError) as error:
                self.hil_enabled = False
                self.hil_error = str(error)
            finally:
                self.hil_link_action = None

        self.hil_link_action_thread = threading.Thread(
            target=worker, name=f'motor-link-{action}', daemon=True)
        self.hil_link_action_thread.start()

    def request_hil_enable(self):
        self._run_when_motor_link_live('enable')

    def request_orientation_learning(self, step=None, rate=None):
        self._run_when_motor_link_live('learn', step, rate)

    def set_hil_dispatch_rate(self, value):
        if not self.hil_controller:
            raise ValueError('Hardware-loop mode was not enabled at launch')
        if self.hil_enabled:
            raise ValueError('Turn motor output OFF before changing dispatch rate')
        self.hil_controller.set_dispatch_rate(value)
        self.hil_dispatch_rate_hz = self.hil_controller.dispatch_rate_hz

    def set_hil_display_span(self, value):
        value = float(value)
        if self.hil_enabled:
            raise ValueError('Turn motor output OFF before changing display span')
        if value not in (.1, .5, 1, 5, 10):
            raise ValueError('Select a supported display span')
        self.hil_display_span_degrees = value

    def set_hil_travel_envelope(self, enabled, degrees=5):
        if not self.hil_controller:
            raise ValueError('Hardware-loop mode was not enabled at launch')
        if self.hil_enabled:
            raise ValueError('Turn motor output OFF before changing the travel envelope')
        self.hil_controller.set_travel_envelope(bool(enabled), degrees)
        self.hil_envelope_degrees = self.hil_controller.envelope
        self._invalidate_orientation(
            'Travel-envelope setting changed; lock a feature and run soft start again.')

    def set_hil_loss_behavior(self, mode, coast_seconds=.5):
        if self.hil_enabled:
            raise ValueError('Turn motor output OFF before changing target-loss behavior')
        if mode not in ('stop', 'predictive-coast', 'return-last-lock'):
            raise ValueError('Select Stop, Predictive coast, or Return to last lock')
        coast_seconds = float(coast_seconds)
        if coast_seconds not in (.25, .5, 1, 2, 3, 5):
            raise ValueError('Predictive coast must be 0.25, 0.5, 1, 2, 3, or 5 seconds')
        self.hil_loss_mode = mode
        self.hil_coast_seconds = coast_seconds

    def set_vibration_filter(self, enabled, threshold_px=3):
        if self.hil_enabled:
            raise ValueError('Turn tracking output OFF before changing vibration filtering')
        threshold_px = float(threshold_px)
        if not 0.5 <= threshold_px <= 25:
            raise ValueError('Vibration threshold must be 0.5–25 pixels')
        self.vibration_filter_enabled = bool(enabled)
        self.vibration_threshold_px = threshold_px
        self.vibration_filtered_point = None
        self.vibration_candidate = None
        self.vibration_candidate_frames = 0

    def _filter_selected(self, selected):
        if selected and self.tracking_mode == 'feature-roi':
            # CSRT can continue returning a mathematically valid rectangle
            # after most of the selected feature has left the image.  That is
            # not a credible optical lock.  Reject severely clipped boxes so
            # the mount freezes in-place instead of following background or
            # an effectively empty ROI.  A mildly clipped edge target remains
            # usable by the adaptive calibration routine below.
            frame_width = max(1, self.telemetry.get('width', self.width))
            frame_height = max(1, self.telemetry.get('height', self.height))
            box_x, box_y, box_width, box_height = selected['bbox']
            inside_width = max(0, min(frame_width, box_x + box_width) - max(0, box_x))
            inside_height = max(0, min(frame_height, box_y + box_height) - max(0, box_y))
            visible_fraction = ((inside_width * inside_height) /
                                max(1, box_width * box_height))
            center_inside = (0 <= selected['x'] < frame_width and
                             0 <= selected['y'] < frame_height)
            if not center_inside or visible_fraction < .72:
                return None
        if not selected or not self.vibration_filter_enabled:
            if selected:
                self.vibration_filtered_point = (float(selected['x']), float(selected['y']))
            return selected
        raw = (float(selected['x']), float(selected['y']))
        if self.vibration_filtered_point is None:
            self.vibration_filtered_point = raw
        distance = math.hypot(raw[0] - self.vibration_filtered_point[0],
                              raw[1] - self.vibration_filtered_point[1])
        if distance <= self.vibration_threshold_px:
            alpha = .2
            self.vibration_filtered_point = (
                self.vibration_filtered_point[0] * (1 - alpha) + raw[0] * alpha,
                self.vibration_filtered_point[1] * (1 - alpha) + raw[1] * alpha)
            self.vibration_candidate = None
            self.vibration_candidate_frames = 0
        else:
            if (self.vibration_candidate is not None and
                    math.hypot(raw[0] - self.vibration_candidate[0],
                               raw[1] - self.vibration_candidate[1]) <=
                    self.vibration_threshold_px):
                self.vibration_candidate_frames += 1
            else:
                self.vibration_candidate = raw
                self.vibration_candidate_frames = 1
            if self.vibration_candidate_frames >= 3:
                self.vibration_filtered_point = raw
                self.vibration_candidate = None
                self.vibration_candidate_frames = 0
        filtered = dict(selected)
        filtered['raw_x'], filtered['raw_y'] = raw
        filtered['x'], filtered['y'] = self.vibration_filtered_point
        filtered['vibration_filtered'] = True
        return filtered

    def set_synthetic_blink(self, enabled, fade_out, hidden, fade_in, period):
        if self.source != 'synthetic':
            raise ValueError('Blink testing is available only for the synthetic source')
        if self.hil_enabled:
            raise ValueError('Turn motor output OFF before changing blink timing')
        fade_out, hidden, fade_in, period = map(float,
                                                (fade_out, hidden, fade_in, period))
        if (not .1 <= fade_out <= 10 or not 0 <= hidden <= 10
                or not .1 <= fade_in <= 10 or not .5 <= period <= 30
                or fade_out + hidden + fade_in >= period):
            raise ValueError('Fade 0.1–10s, hidden 0–10s, period 0.5–30s; cycle must leave visible time')
        self.synthetic_blink.update(enabled=bool(enabled),
                                    fade_out_seconds=fade_out,
                                    hidden_seconds=hidden,
                                    fade_in_seconds=fade_in,
                                    period_seconds=period)

    def _reset_feature_guard(self, clear_reference=False):
        self.feature_similarity_baseline = 1.0
        self.feature_invalid_frames = 0
        self.feature_reacquiring = False
        self.feature_reacquire_candidate = None
        self.feature_reacquire_confirmations = 0
        self.feature_last_confirmed = None
        if clear_reference:
            self.feature_reference_patch = None

    def set_tracking_mode(self, mode, algorithm='CSRT', lock_size=True):
        if self.hil_enabled:
            raise ValueError('Turn motor output OFF before changing tracking mode')
        if mode not in ('threshold', 'feature-roi'):
            raise ValueError('Tracking mode must be Thresholded objects or Feature ROI')
        if mode == 'feature-roi':
            create_feature_tracker(algorithm, lock_size)
        self.tracking_mode = mode
        self.feature_algorithm = str(algorithm).upper()
        self.feature_lock_size = bool(lock_size)
        self.feature_tracker = None
        self.feature_roi_pending = None
        self.feature_selected_id = None
        self.feature_initial_size = None
        self._reset_feature_guard(clear_reference=True)
        self.tracker.clear_selection()

    def set_operating_mode(self, mode):
        if self.hil_enabled:
            raise ValueError('Turn tracking output OFF before changing Live/Demo mode')
        if mode not in ('live', 'demo'):
            raise ValueError('Select Live Mode or Demo Mode')
        self.ui_mode = mode
        self.desired_source = (('picamera2' if str(self.device).startswith('picamera2:')
                                else 'v4l2') if mode == 'live' else 'synthetic')
        self.aim_mode = 'center'
        self.aim_point = (.5, .5)
        self.feature_tracker = None
        self.feature_roi_pending = None
        self.feature_selected_id = None
        self.feature_initial_size = None
        self._reset_feature_guard(clear_reference=True)
        self.tracker.clear_selection()
        self.tracking_mode = 'feature-roi' if mode == 'live' else 'threshold'
        self.hil_previous_target = None
        self.hil_last_visible_target = None
        self.hil_current_target = None
        self.hil_commanded_target = None
        self.hil_error = None
        self._invalidate_orientation('Operating mode changed; soft start must be relearned.')
        self._load_orientation_profile()

    def set_aim_mode(self, mode):
        if self.hil_enabled:
            raise ValueError('Turn tracking output OFF before changing the aim point')
        if mode not in ('center', 'hold'):
            raise ValueError('Select Center target or Hold current frame position')
        self.aim_mode = mode
        if mode == 'center':
            self.aim_point = (.5, .5)
        elif self.last_selected:
            width = max(1, self.telemetry.get('width', self.width))
            height = max(1, self.telemetry.get('height', self.height))
            self.aim_point = (self.last_selected['x'] / width,
                              self.last_selected['y'] / height)

    def set_image_orientation(self, horizontal, vertical):
        if self.hil_enabled:
            raise ValueError('Turn tracking output OFF before changing image orientation')
        if horizontal not in ('normal', 'reversed') or vertical not in ('normal', 'reversed'):
            raise ValueError('Image orientation must be Normal or Reversed for each axis')
        self.image_horizontal_reversed = horizontal == 'reversed'
        self.image_vertical_reversed = vertical == 'reversed'
        self._invalidate_orientation('Manual image direction changed; soft start must be relearned.')
        self.orientation_history.clear()
        self.orientation_sanity = {'status': 'idle', 'message': None}

    def _invalidate_orientation(self, message):
        self.orientation_calibration = {'status': 'not learned', 'valid': False,
                                        'message': message}
        self.soft_start_ramp_started = None

    def _stable_feature_sample(self, after, timeout=3):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self.stop_event.is_set():
            if self.orientation_cancel.is_set():
                raise RuntimeError('Soft start canceled; motor output remains OFF')
            with self.lock:
                samples = [item for item in self.feature_samples if item[0] >= after]
                selected = self.last_selected
            if selected and len(samples) >= 4:
                recent = samples[-4:]
                return (sum(item[1] for item in recent) / len(recent),
                        sum(item[2] for item in recent) / len(recent))
            time.sleep(.04)
        raise RuntimeError('Feature lock was not stable during soft start')

    def _reseed_feature_at(self, point):
        """Recreate the feature tracker at a known pixel after a probe return."""
        with self.lock:
            frame_width = max(1, self.telemetry.get('width', self.width))
            frame_height = max(1, self.telemetry.get('height', self.height))
            box_width, box_height = self.feature_initial_size or (
                round(frame_width * self.live_lock_box_fraction),
                round(frame_height * self.live_lock_box_fraction))
            box_width = max(8, min(frame_width, int(box_width)))
            box_height = max(8, min(frame_height, int(box_height)))
            left = round(max(0, min(frame_width - box_width,
                                    float(point[0]) - box_width / 2)))
            top = round(max(0, min(frame_height - box_height,
                                   float(point[1]) - box_height / 2)))
            self.feature_tracker = None
            self.feature_roi_pending = (left, top, box_width, box_height)
            self.feature_selected_id = None
            self.last_selected = None
            self.feature_samples.clear()
            self._reset_feature_guard(clear_reference=False)

    def _return_to_probe_start(self, baseline, baseline_px):
        """Return to exact starting encoders and reacquire only the same ROI."""
        self._probe_target(*baseline)
        self._reseed_feature_at(baseline_px)
        return self._stable_feature_sample(time.monotonic(), timeout=4)

    def _rate_axis_displacement(self, axis, direction, distance, rate,
                                require_feature=True):
        """Move one motor with native slew level 1-9 until encoders reach distance."""
        controller = self.hil_controller
        motor = 'Azimuth' if axis == 0 else 'Elevation'
        address = 0x10 if axis == 0 else 0x11
        status = controller.status()
        start_count = status.get('positions', {}).get(motor, {}).get('encoder')
        if start_count is None:
            raise RuntimeError(f'Fresh {motor.lower()} encoder feedback is required')
        controller.arm()
        controller.slew_rate(address, direction, int(rate))
        timeout = max(8, min(75, 8 + float(distance) * 80 / max(1, rate)))
        deadline = time.monotonic() + timeout
        lost_since = None
        try:
            while time.monotonic() < deadline:
                status = controller.status()
                if status.get('connection') != 'live':
                    raise RuntimeError('Motor link stopped responding during rate-limited probe')
                count = status.get('positions', {}).get(motor, {}).get('encoder')
                if count is not None:
                    moved = abs(wrapped_encoder_delta(count, start_count) * 360 / (1 << 24))
                    if moved >= distance:
                        return status
                with self.lock:
                    visible = self.last_selected is not None
                if require_feature and not visible:
                    lost_since = lost_since or time.monotonic()
                    if time.monotonic() - lost_since >= .35:
                        raise RuntimeError('Selected feature was lost during rate-limited probe')
                else:
                    lost_since = None
                time.sleep(.04)
            raise RuntimeError(
                f'Rate-{rate} probe did not reach {distance:.3f}\u00b0 before timeout')
        finally:
            controller.freeze()
            time.sleep(.20)

    def _return_to_probe_start_rate(self, baseline, baseline_px, rate):
        """Return both encoder references slowly, then re-seed the original ROI."""
        for axis, motor in enumerate(('Azimuth', 'Elevation')):
            status = self.hil_controller.status()
            current = status.get('positions', {}).get(motor, {}).get('reference_degrees')
            if current is None:
                raise RuntimeError(f'Cannot return: {motor.lower()} encoder is unavailable')
            difference = ((baseline[axis] - current + 180) % 360 - 180
                          if axis == 0 else baseline[axis] - current)
            if abs(difference) >= .01:
                self._rate_axis_displacement(
                    axis, 1 if difference > 0 else -1, abs(difference),
                    min(int(rate), 2), require_feature=False)
        self._reseed_feature_at(baseline_px)
        return self._stable_feature_sample(time.monotonic(), timeout=4)

    def _adaptive_axis_probe(self, axis, baseline, baseline_counts, baseline_px,
                             requested_step, slew_rate):
        """Measure one axis while preferring movement toward frame center.

        An edge feature can disappear if the first guessed motor direction is
        wrong.  Every rejected attempt returns to the recorded encoder start,
        recreates the original ROI, and tries a smaller/reversed movement.  A
        box is never accepted merely because CSRT returned coordinates.
        """
        frame_width = max(1, self.telemetry.get('width', self.width))
        frame_height = max(1, self.telemetry.get('height', self.height))
        center = (frame_width / 2, frame_height / 2)
        baseline_error = (baseline_px[0] - center[0], baseline_px[1] - center[1])
        baseline_distance = math.hypot(*baseline_error)
        near_edge = (baseline_px[0] < frame_width * .22 or
                     baseline_px[0] > frame_width * .78 or
                     baseline_px[1] < frame_height * .22 or
                     baseline_px[1] > frame_height * .78)
        minimum_step = .005
        attempts = []
        maximum_step = float(requested_step)
        # Treat the selected value as a maximum.  Begin with a much smaller
        # motion so a narrow-field feature is not thrown out of frame before
        # the pixel direction is known, then escalate only when the measured
        # pixel response is too small.
        step = min(.05, maximum_step)
        direction = 1
        for attempt_number in range(8):
            try:
                status = self._rate_axis_displacement(
                    axis, direction, step, slew_rate, require_feature=True)
                moved_px = self._stable_feature_sample(time.monotonic(), timeout=2.5)
                vector = (moved_px[0] - baseline_px[0],
                          moved_px[1] - baseline_px[1])
                pixel_motion = math.hypot(*vector)
                moved_error = (moved_px[0] - center[0], moved_px[1] - center[1])
                moved_distance = math.hypot(*moved_error)
                crossed_center = (baseline_error[0] * moved_error[0] < 0 or
                                  baseline_error[1] * moved_error[1] < 0)
                overshot = (crossed_center and moved_distance >
                             max(8, baseline_distance * .35))
                outward = (near_edge and moved_distance >
                           baseline_distance + max(3, pixel_motion * .10))
                attempts.append({'direction': direction, 'step_degrees': step,
                                 'pixel_motion': pixel_motion,
                                 'distance_before_px': baseline_distance,
                                 'distance_after_px': moved_distance,
                                 'result': ('overshoot' if overshot else
                                            'outward' if outward else 'accepted')})
                if pixel_motion >= 3 and not outward and not overshot:
                    motor = 'Azimuth' if axis == 0 else 'Elevation'
                    encoder = status['positions'][motor]['encoder']
                    counts = wrapped_encoder_delta(encoder, baseline_counts[axis])
                    measured_step = counts * 360 / (1 << 24)
                    if abs(measured_step) < .003:
                        raise RuntimeError('Probe motor displacement was too small to measure')
                    return status, vector, counts, measured_step, attempts
                self._return_to_probe_start_rate(baseline, baseline_px, slew_rate)
                if overshot:
                    step = max(minimum_step, step / 2)
                elif outward:
                    direction *= -1
                elif pixel_motion < 3 and step < maximum_step:
                    step = min(maximum_step, step * 2)
                else:
                    step = max(minimum_step, step / 2)
                    direction *= -1
            except (ConnectionError, RuntimeError, ValueError) as error:
                attempts.append({'direction': direction, 'step_degrees': step,
                                 'result': 'lost-or-failed', 'message': str(error)})
                self._return_to_probe_start_rate(baseline, baseline_px, slew_rate)
                step = max(minimum_step, step / 2) if not near_edge else step
                direction *= -1
            self.orientation_calibration.update(
                message=(f'{"Azimuth" if axis == 0 else "Elevation"} probe '
                         f'retrying toward frame center at {step:.3f}\u00b0…'),
                retry_count=attempt_number + 1, probe_attempts=list(attempts))
        raise RuntimeError(
            f'{"Azimuth" if axis == 0 else "Elevation"} calibration could not '
            'produce a credible inward feature movement; telescope returned to start')

    def _probe_target(self, azimuth, elevation, timeout=None):
        controller = self.hil_controller
        starting = controller.status().get('positions', {})
        starting_az = starting.get('Azimuth', {}).get('reference_degrees')
        starting_el = starting.get('Elevation', {}).get('reference_degrees')
        requested_distance = max(
            abs((starting_az - azimuth + 180) % 360 - 180)
            if starting_az is not None else 0,
            abs(starting_el - elevation) if starting_el is not None else 0)
        # This geared Alt-Az mount is not a sub-arcsecond positioning stage.
        # A fixed 0.004-degree acceptance rejected motion that is adequate for
        # learning how camera pixels map to the two motor axes.
        # Repeated live tests on the COSMOS 90GT showed finite GoTo settling
        # roughly 0.05-0.08 degrees from the requested encoder position.  A
        # 0.1-degree floor accepts that real gear-train behavior while still
        # providing ample pixel displacement for direction learning.
        tolerance = max(.10, min(.20, requested_distance * .20))
        timeout = (max(5, min(15, 5 + requested_distance * 5))
                   if timeout is None else float(timeout))
        controller.arm()
        with self.lock:
            self.hil_sequence += 1
            sequence = self.hil_sequence
        controller.submit(azimuth % 360, elevation, sequence, time.time(), ttl=2)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.orientation_cancel.is_set():
                raise RuntimeError('Soft start canceled; motor output remains OFF')
            status = controller.status()
            if status.get('connection') != 'live':
                raise RuntimeError('Motor link stopped responding during soft start')
            positions = status.get('positions', {})
            az = positions.get('Azimuth', {}).get('reference_degrees')
            el = positions.get('Elevation', {}).get('reference_degrees')
            if az is not None and el is not None:
                az_error = abs((az - azimuth + 180) % 360 - 180)
                if az_error <= tolerance and abs(el - elevation) <= tolerance:
                    controller.pause()
                    time.sleep(.25)
                    return status
            time.sleep(.04)
        controller.pause()
        raise RuntimeError(
            f'Soft-start probe did not reach its target within {tolerance:.3f}\u00b0')

    def _learn_orientation_worker(self, step, slew_rate):
        baseline = None
        try:
            controller = self.hil_controller
            status = controller.status()
            if status.get('connection') != 'live':
                raise RuntimeError('Fresh motor feedback is required for soft start')
            if not status.get('local_reference_set'):
                controller.reference(180, 0)
                status = controller.status()
            positions = status.get('positions', {})
            baseline = (positions['Azimuth']['reference_degrees'],
                        positions['Elevation']['reference_degrees'])
            baseline_counts = (positions['Azimuth']['encoder'],
                               positions['Elevation']['encoder'])
            if None in baseline:
                raise RuntimeError('Both encoder positions must be fresh for soft start')

            self.orientation_calibration.update(
                status='running', message='Measuring azimuth pixel direction…',
                phase='azimuth probe')
            before_time = time.monotonic()
            baseline_px = self._stable_feature_sample(before_time)
            az_status, az_vector, az_counts, az_step, az_attempts = (
                self._adaptive_axis_probe(0, baseline, baseline_counts,
                                          baseline_px, step, slew_rate))
            self._return_to_probe_start_rate(baseline, baseline_px, slew_rate)

            self.orientation_calibration.update(
                message='Measuring elevation pixel direction…', phase='elevation probe')
            baseline_px = self._stable_feature_sample(time.monotonic())
            el_status, el_vector, el_counts, el_step, el_attempts = (
                self._adaptive_axis_probe(1, baseline, baseline_counts,
                                          baseline_px, step, slew_rate))
            self._return_to_probe_start_rate(baseline, baseline_px, slew_rate)

            learned = learn_pixel_mount_transform(az_vector, el_vector,
                                                  (az_step, el_step))
            learned.update(encoder_delta_counts=[az_counts, el_counts],
                           adaptive_probe_attempts={
                               'azimuth': az_attempts, 'elevation': el_attempts},
                           pixels_per_encoder_count=[
                               math.hypot(*az_vector) / abs(az_counts),
                               math.hypot(*el_vector) / abs(el_counts)])
            learned.update(status='learned', valid=True,
                           message='Camera orientation learned; tracking authority is ramping up slowly.')
            with self.lock:
                self.orientation_calibration = learned
                self._save_orientation_profile(learned)
                self.soft_start_ramp_started = time.monotonic()
                self.hil_error = None
                if not self.last_selected:
                    raise RuntimeError('Feature lock was lost before tracking could start')
                controller.arm()
                self.hil_enabled = True
        except (ConnectionError, KeyError, RuntimeError, ValueError) as error:
            if self.hil_controller:
                self.hil_controller.pause()
            with self.lock:
                self.hil_enabled = False
                self.hil_error = str(error)
                self.orientation_calibration = {
                    'status': 'failed', 'valid': False, 'message': str(error)}
        finally:
            if baseline and self.hil_controller and not self.orientation_calibration.get('valid'):
                try:
                    self.orientation_cancel.clear()
                    self._return_to_probe_start_rate(
                        baseline, baseline_px if 'baseline_px' in locals() else
                        (self.width / 2, self.height / 2), slew_rate)
                except (ConnectionError, RuntimeError, ValueError):
                    self.hil_controller.pause()
            self.orientation_cancel.clear()

    def start_orientation_learning(self, step=None, rate=None):
        if not self.hil_controller or self.ui_mode != 'live':
            raise ValueError('Soft start requires Live Mode and a configured motor link')
        if self.hil_enabled:
            raise ValueError('Turn tracking output OFF before relearning camera orientation')
        if not self.last_selected:
            raise ValueError('Lock a stable visible feature before starting soft start')
        if self.hil_controller.status().get('connection') != 'live':
            raise ValueError('Motor link is not live; use Learn again to reconnect')
        if self.orientation_calibration_thread and self.orientation_calibration_thread.is_alive():
            raise ValueError('Soft start is already running')
        step = self.soft_start_step_degrees if step is None else float(step)
        if step not in (.005, .01, .02, .05, .1, .5, 1.0):
            raise ValueError(
                'Select a mapping probe of 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, or 1 degree')
        self.soft_start_step_degrees = step
        rate = self.soft_start_slew_rate if rate is None else int(rate)
        if rate not in range(1, 10):
            raise ValueError('Select calibration slew level 1-9')
        self.soft_start_slew_rate = rate
        self.hil_error = None
        self.orientation_cancel.clear()
        self.orientation_calibration = {
            'status': 'running', 'valid': False,
            'message': 'Soft start queued; telescope will make two bounded probe movements.',
            'phase': 'starting', 'step_degrees': step, 'slew_rate': rate}
        self.orientation_calibration_thread = threading.Thread(
            target=self._learn_orientation_worker, args=(step, rate),
            name='camera-orientation-learning', daemon=True)
        self.orientation_calibration_thread.start()

    def set_feature_point(self, x, y, box_fraction=.12):
        if self.hil_enabled:
            raise ValueError('Turn tracking output OFF before selecting a new feature')
        x, y, box_fraction = map(float, (x, y, box_fraction))
        if (not 0 <= x <= 1 or not 0 <= y <= 1
                or not .03 <= box_fraction <= .30):
            raise ValueError('Select a point inside the video; lock box must be 3–30%')
        self.tracking_mode = 'feature-roi'
        self.feature_algorithm = 'CSRT'
        self.feature_lock_size = True
        half = box_fraction / 2
        left = max(0, min(1 - box_fraction, x - half))
        top = max(0, min(1 - box_fraction, y - half))
        self.set_feature_roi(left, top, box_fraction, box_fraction)
        self.aim_point = ((.5, .5) if self.aim_mode == 'center' else (x, y))

    def clear_target(self):
        self.tracker.clear_selection()
        self.feature_tracker = None
        self.feature_roi_pending = None
        self.feature_selected_id = None
        self.feature_initial_size = None
        self._reset_feature_guard(clear_reference=True)
        self.last_selected = None
        self.hil_previous_target = None
        self.hil_last_visible_target = None
        self.hil_current_target = None
        self.hil_commanded_target = None

    def reset_live_view(self):
        if self.hil_enabled:
            raise ValueError('Turn tracking output OFF before resetting the live view')
        self.clear_target()
        self.aim_mode = 'center'
        self.aim_point = (.5, .5)
        self.image_horizontal_reversed = False
        self.image_vertical_reversed = False
        self.orientation_history.clear()
        self.orientation_sanity = {'status': 'idle', 'message': None}
        self.errors.clear()
        self.hil_error = None
        self.camera_reopen_requested = self.ui_mode == 'live'

    def refresh_cameras(self):
        if self.hil_enabled:
            raise ValueError('Turn tracking output OFF before refreshing cameras')
        self.cameras = enumerate_cameras()
        return self.cameras

    def recover_camera_service(self):
        """Retry only camera capture; never replace the dashboard process."""
        if self.camera_recovery_requested:
            return
        self.camera_recovery_requested = True
        self.camera_recovery_count += 1
        if self.hil_controller:
            try:
                self.hil_controller.freeze()
            except Exception as error:
                self.errors.append(f'Camera recovery motor pause: {error}')
        with self.frame_ready:
            self.telemetry = dict(
                self.telemetry,
                state='CAMERA RECOVERY',
                error=('Camera recovery requested. Motion is frozen; the motor '
                       'link and bridge remain active.'))
            self.frame_ready.notify_all()
        if self.thread and self.thread.is_alive():
            # The capture loop will reopen on its next iteration.  A driver
            # timeout normally unwinds the thread first; if it has already
            # done so, start a fresh capture worker below.
            self.camera_reopen_requested = True
        else:
            self.start()

    def set_camera(self, device, width, height, fps):
        if self.hil_enabled:
            raise ValueError('Turn tracking output OFF before changing cameras')
        device = str(device)
        width, height, fps = int(width), int(height), float(fps)
        camera = next((item for item in self.cameras if item['device'] == device), None)
        if not camera:
            raise ValueError('Refresh cameras and select an available capture device')
        if not any(item['width'] == width and item['height'] == height
                   and abs(item['fps'] - fps) < .01 for item in camera['settings']):
            raise ValueError('Select a resolution and frame rate advertised by that camera')
        self.device, self.width, self.height, self.requested_fps = device, width, height, fps
        self.desired_source = ('picamera2' if device.startswith('picamera2:') else 'v4l2')
        self.camera_reopen_requested = True
        self.feature_tracker = None
        self.feature_roi_pending = None
        self.feature_selected_id = None
        self._reset_feature_guard(clear_reference=True)
        self._invalidate_orientation('Camera or resolution changed; soft start must be relearned.')
        self._load_orientation_profile()

    def set_camera_exposure(self, automatic, exposure_value=0,
                            exposure_time_us=10000, analogue_gain=1.0):
        if self.hil_enabled:
            raise ValueError('Turn tracking output OFF before changing camera exposure')
        automatic = bool(automatic)
        exposure_value = float(exposure_value)
        exposure_time_us = int(exposure_time_us)
        analogue_gain = float(analogue_gain)
        def limits(name, fallback):
            detected = self.camera_control_ranges.get(name, {})
            return (float(detected.get('minimum', fallback[0])),
                    float(detected.get('maximum', fallback[1])))

        ev_limits = limits('ExposureValue', (-4, 4))
        exposure_limits = limits('ExposureTime', (100, 1000000))
        gain_limits = limits('AnalogueGain', (1, 32))
        if (not ev_limits[0] <= exposure_value <= ev_limits[1]
                or not exposure_limits[0] <= exposure_time_us <= exposure_limits[1]
                or not gain_limits[0] <= analogue_gain <= gain_limits[1]):
            raise ValueError(
                f'Exposure compensation {ev_limits[0]:g} to {ev_limits[1]:g}; '
                f'manual exposure {exposure_limits[0]:g}–{exposure_limits[1]:g} us; '
                f'gain {gain_limits[0]:g}–{gain_limits[1]:g}')
        self.camera_auto_exposure = automatic
        self.camera_exposure_value = exposure_value
        self.camera_manual_exposure_us = exposure_time_us
        self.camera_manual_gain = analogue_gain
        if self.source != 'picamera2' or not self.capture:
            if str(self.device).startswith('picamera2:'):
                self.camera_reopen_requested = True
            return
        controls = ({'AeEnable': True, 'ExposureValue': exposure_value}
                    if automatic else
                    {'AeEnable': False, 'ExposureTime': exposure_time_us,
                     'AnalogueGain': analogue_gain})
        self.capture.set_controls(controls)

    def set_image_curve(self, black, white, gamma):
        black, white, gamma = int(black), int(white), float(gamma)
        if not 0 <= black < white <= 255 or not .2 <= gamma <= 3:
            raise ValueError('Image curve requires black 0–254, white above black through 255, gamma 0.2–3.0')
        self.curve_black, self.curve_white, self.curve_gamma = black, white, gamma

    def _apply_image_curve(self, gray):
        if self.curve_black == 0 and self.curve_white == 255 and self.curve_gamma == 1:
            return gray
        values = np.arange(256, dtype=np.float32)
        values = np.clip((values - self.curve_black) /
                         (self.curve_white - self.curve_black), 0, 1)
        values = np.power(values, 1 / self.curve_gamma) * 255
        return cv2.LUT(gray, values.astype(np.uint8))

    def set_hil_control_strategy(self, preset, horizon=.25, proportional=.2,
                                 deadband=.03):
        if self.hil_enabled:
            raise ValueError('Turn motor output OFF before changing control strategy')
        presets = {'off': (0.0, 0.0, 0.0), 'conservative': (.1, .1, .05),
                   'balanced': (.25, .2, .03), 'aggressive': (.5, .35, .02)}
        if preset in presets:
            horizon, proportional, deadband = presets[preset]
        elif preset == 'auto':
            horizon, proportional, deadband = self.hil_auto_horizon, .15, .03
        elif preset != 'custom':
            raise ValueError('Select Off, Conservative, Balanced, Aggressive, Auto, or Custom')
        horizon, proportional, deadband = map(float, (horizon, proportional, deadband))
        if (not 0 <= horizon <= 1.5 or not 0 <= proportional <= 1
                or not 0 <= deadband <= .5):
            raise ValueError('Prediction horizon 0–1.5s, proportional lead 0–1, deadband 0–0.5°')
        self.hil_control_preset = preset
        self.hil_prediction_horizon = horizon
        self.hil_proportional_lead = proportional
        self.hil_correction_deadband = deadband

    def _apply_outer_loop(self, desired, marker):
        azimuth, elevation = desired
        horizon = (self.hil_auto_horizon if self.hil_control_preset == 'auto'
                   else self.hil_prediction_horizon)
        az_error = el_error = 0.0
        if marker:
            az_error = (azimuth - marker['azimuth_degrees'] + 180) % 360 - 180
            el_error = elevation - marker['elevation_degrees']
        if self.hil_control_preset == 'auto' and self.hil_target_rate and marker:
            estimates = []
            for error, rate in ((az_error, self.hil_target_rate['azimuth_deg_s']),
                                (el_error, self.hil_target_rate['elevation_deg_s'])):
                if abs(rate) >= .1 and error * rate > 0:
                    estimates.append(min(1.0, abs(error / rate)))
            if estimates:
                estimate = sum(estimates) / len(estimates)
                self.hil_auto_horizon = .9 * self.hil_auto_horizon + .1 * estimate
                horizon = self.hil_auto_horizon
        rate = self.hil_target_rate or {'azimuth_deg_s': 0.0, 'elevation_deg_s': 0.0}
        commanded_azimuth = (azimuth + rate['azimuth_deg_s'] * horizon
                             + self.hil_proportional_lead * az_error)
        commanded_elevation = (elevation + rate['elevation_deg_s'] * horizon
                               + self.hil_proportional_lead * el_error)
        if marker and abs(az_error) <= self.hil_correction_deadband:
            commanded_azimuth = marker['azimuth_degrees']
        if marker and abs(el_error) <= self.hil_correction_deadband:
            commanded_elevation = marker['elevation_degrees']
        if self.hil_envelope_degrees > 0:
            limit = self.hil_envelope_degrees * .95
            az_delta = max(-limit, min(
                limit, (commanded_azimuth - 180 + 180) % 360 - 180))
            commanded_elevation = max(-limit, min(limit, commanded_elevation))
            commanded = ((180 + az_delta) % 360, commanded_elevation)
        else:
            commanded = (commanded_azimuth % 360, commanded_elevation)
        self.hil_commanded_target = {'azimuth_degrees': commanded[0],
                                     'elevation_degrees': commanded[1],
                                     'prediction_horizon_seconds': horizon,
                                     'proportional_lead': self.hil_proportional_lead,
                                     'correction_deadband_degrees': self.hil_correction_deadband}
        return commanded

    def set_feature_roi(self, x, y, width, height):
        if self.tracking_mode != 'feature-roi':
            raise ValueError('Select Feature ROI tracking mode first')
        if self.hil_enabled:
            raise ValueError('Turn motor output OFF before selecting a new feature')
        frame_width = self.telemetry.get('width', self.width)
        frame_height = self.telemetry.get('height', self.height)
        values = [float(value) for value in (x, y, width, height)]
        if (not all(math.isfinite(value) for value in values)
                or not all(0 <= value <= 1 for value in values)
                or values[2] < .01 or values[3] < .01
                or values[0] + values[2] > 1 or values[1] + values[3] > 1):
            raise ValueError('Drag a valid feature box inside the video')
        self.feature_roi_pending = tuple(round(value) for value in
                                         (values[0] * frame_width,
                                          values[1] * frame_height,
                                          values[2] * frame_width,
                                          values[3] * frame_height))
        self.feature_initial_size = self.feature_roi_pending[2:]
        self.feature_tracker = None
        self.feature_selected_id = None
        self._reset_feature_guard(clear_reference=True)

    @staticmethod
    def _feature_detection(bbox, confidence=1.0):
        x, y, width, height = (float(value) for value in bbox)
        return {'track_id': 1, 'x': x + width / 2, 'y': y + height / 2,
                'bbox': [round(x), round(y), round(width), round(height)],
                'area': width * height, 'brightness': 0.0,
                'confidence': float(confidence),
                'missed': 0, 'selected': True}

    @staticmethod
    def _extract_feature_patch(gray, bbox):
        frame_height, frame_width = gray.shape[:2]
        x, y, width, height = (round(float(value)) for value in bbox)
        if (width < 4 or height < 4 or x < 0 or y < 0 or
                x + width > frame_width or y + height > frame_height):
            return None
        return gray[y:y + height, x:x + width].copy()

    def _feature_similarity(self, gray, bbox):
        candidate = self._extract_feature_patch(gray, bbox)
        reference = self.feature_reference_patch
        if candidate is None or reference is None:
            return None
        if candidate.shape != reference.shape:
            candidate = cv2.resize(candidate, (reference.shape[1], reference.shape[0]))
        # Normalized correlation resists uniform exposure changes.  A nearly
        # textureless ROI is never promoted to a credible reacquisition.
        if float(np.std(reference)) < 4 or float(np.std(candidate)) < 4:
            return 0.0
        return float(cv2.matchTemplate(
            candidate, reference, cv2.TM_CCOEFF_NORMED)[0, 0])

    def _feature_candidate_is_credible(self, bbox, similarity, now=None):
        """Reject tracker drift without making the normal lock less sensitive."""
        if similarity is None or not math.isfinite(similarity):
            return False
        threshold = max(self.feature_similarity_threshold,
                        min(.55, self.feature_similarity_baseline * .45))
        if similarity < threshold:
            return False
        x, y, width, height = (float(value) for value in bbox)
        center = (x + width / 2, y + height / 2)
        now = time.monotonic() if now is None else float(now)
        if self.feature_last_confirmed:
            previous_x, previous_y, previous_time = self.feature_last_confirmed
            jump = math.hypot(center[0] - previous_x, center[1] - previous_y)
            diagonal = max(8.0, math.hypot(width, height))
            predicted = 0.0
            if len(self.feature_samples) >= 2:
                first, last = self.feature_samples[-2], self.feature_samples[-1]
                dt = max(.001, last[0] - first[0])
                velocity = math.hypot(last[1] - first[1], last[2] - first[2]) / dt
                predicted = velocity * max(.001, now - previous_time)
            if jump > max(diagonal * 3.0, diagonal + predicted * 3.0):
                return False
        self.feature_similarity_baseline = (
            .95 * self.feature_similarity_baseline + .05 * similarity)
        self.feature_last_confirmed = (center[0], center[1], now)
        return True

    def _feature_reacquisition_match(self, gray):
        """Find the original ROI in the full frame only when the match is unique."""
        reference = self.feature_reference_patch
        if reference is None or float(np.std(reference)) < 4:
            return None
        frame_height, frame_width = gray.shape[:2]
        ref_height, ref_width = reference.shape[:2]
        if ref_width > frame_width or ref_height > frame_height:
            return None
        scores = cv2.matchTemplate(gray, reference, cv2.TM_CCOEFF_NORMED)
        _, best, _, location = cv2.minMaxLoc(scores)
        threshold = max(.60, min(.82, self.feature_similarity_baseline * .70))
        if not math.isfinite(best) or best < threshold:
            return None
        remaining = scores.copy()
        radius_x, radius_y = max(2, ref_width // 2), max(2, ref_height // 2)
        left, top = max(0, location[0] - radius_x), max(0, location[1] - radius_y)
        right = min(remaining.shape[1], location[0] + radius_x + 1)
        bottom = min(remaining.shape[0], location[1] + radius_y + 1)
        remaining[top:bottom, left:right] = -1
        second = float(np.max(remaining)) if remaining.size else -1.0
        if best < .85 and best - second < .05:
            return None
        return ((float(location[0]), float(location[1]),
                 float(ref_width), float(ref_height)), float(best))

    def _attempt_feature_reacquisition(self, gray, feature_frame):
        """Require two consistent full-frame matches before resuming motion."""
        match = self._feature_reacquisition_match(gray)
        if not match:
            self.feature_reacquire_candidate = None
            self.feature_reacquire_confirmations = 0
            return None
        bbox, score = match
        x, y, width, height = bbox
        center = (x + width / 2, y + height / 2)
        previous = self.feature_reacquire_candidate
        consistent = bool(previous and math.hypot(
            center[0] - previous[0], center[1] - previous[1])
            <= max(8.0, math.hypot(width, height)))
        self.feature_reacquire_candidate = center
        self.feature_reacquire_confirmations = (
            self.feature_reacquire_confirmations + 1 if consistent else 1)
        if self.feature_reacquire_confirmations < 2:
            return None
        tracker = create_feature_tracker(self.feature_algorithm,
                                         self.feature_lock_size)
        initialized = tracker.init(feature_frame, bbox)
        if initialized is False:
            self.feature_reacquire_confirmations = 0
            return None
        self.feature_tracker = tracker
        self.feature_initial_size = (round(width), round(height))
        self.feature_invalid_frames = 0
        self.feature_reacquiring = False
        self.feature_reacquire_candidate = None
        self.feature_reacquire_confirmations = 0
        self.feature_similarity_baseline = max(.60, score)
        self.feature_last_confirmed = (center[0], center[1], time.monotonic())
        return self._feature_detection(bbox, score)

    def _hil_update(self, selected, width, height):
        if not self.hil_controller:
            return {'available': False, 'enabled': False,
                    'description': 'Display-only mode; no motor endpoint was supplied.'}
        status = self.hil_controller.status()
        if (status.get('connection') == 'live'
                and self.hil_error == 'Dashboard bridge connection restarted; waiting for both fresh encoders.'):
            self.hil_error = None
        marker = hil_marker_from_status(status, width, height,
                                        self.hil_display_span_degrees / 2)
        if self.hil_enabled:
            if status.get('connection') != 'live':
                self.hil_enabled = False
                self.hil_controller.pause()
                self.hil_error = 'Motor output stopped because encoder feedback is not live.'
            else:
                try:
                    target_now = time.monotonic()
                    target = None
                    if selected:
                        if marker:
                            self.hil_last_locked_marker = {
                                'azimuth_degrees': marker['azimuth_degrees'],
                                'elevation_degrees': marker['elevation_degrees'],
                                'utc_epoch_s': time.time()}
                        self.hil_return_started = None
                        if self.ui_mode == 'live':
                            if not marker:
                                raise RuntimeError('Fresh encoder feedback is required for optical corrections')
                            if not self.orientation_calibration.get('valid'):
                                raise RuntimeError('Camera-to-mount mapping is not validated; run soft start')
                            anchor_x = self.aim_point[0] * width
                            anchor_y = self.aim_point[1] * height
                            delta_azimuth, delta_elevation = learned_mount_delta(
                                selected['x'] - anchor_x, selected['y'] - anchor_y,
                                self.orientation_calibration)
                            ramp_age = max(0, target_now - (self.soft_start_ramp_started
                                                           or target_now))
                            authority = min(1.0, .1 + .9 * ramp_age / 10)
                            correction_limit = .01 + .14 * authority
                            magnitude = math.hypot(delta_azimuth, delta_elevation)
                            if magnitude > correction_limit:
                                scale = correction_limit / magnitude
                                delta_azimuth *= scale
                                delta_elevation *= scale
                            azimuth = (marker['azimuth_degrees'] + delta_azimuth) % 360
                            elevation = marker['elevation_degrees'] + delta_elevation
                        else:
                            azimuth, elevation = hil_target_from_pixel(
                                selected['x'], selected['y'], width, height,
                                self.hil_display_span_degrees / 2)
                        if self.hil_previous_target:
                            previous_time, previous_azimuth, previous_elevation = self.hil_previous_target
                            delta_time = target_now - previous_time
                            if delta_time > 0:
                                self.hil_target_rate = {
                                    'azimuth_deg_s': ((azimuth - previous_azimuth + 180) % 360 - 180) / delta_time,
                                    'elevation_deg_s': (elevation - previous_elevation) / delta_time,
                                }
                        self.hil_previous_target = (target_now, azimuth, elevation)
                        self.hil_last_visible_target = (target_now, azimuth, elevation)
                        target = (azimuth, elevation)
                        self.hil_error = None
                    elif (self.hil_loss_mode == 'predictive-coast'
                          and self.hil_last_visible_target and self.hil_target_rate
                          and target_now - self.hil_last_visible_target[0] <= self.hil_coast_seconds):
                        lost_for = target_now - self.hil_last_visible_target[0]
                        _seen, last_azimuth, last_elevation = self.hil_last_visible_target
                        target = ((last_azimuth + self.hil_target_rate['azimuth_deg_s'] * lost_for) % 360,
                                  last_elevation + self.hil_target_rate['elevation_deg_s'] * lost_for)
                        self.hil_error = f'Predictive coast active for {lost_for:.2f}s; awaiting selected track.'
                    elif self.hil_loss_mode == 'return-last-lock' and self.hil_last_locked_marker:
                        if self.hil_return_started is None:
                            self.hil_return_started = target_now
                        if target_now - self.hil_return_started <= self.hil_coast_seconds:
                            target = (self.hil_last_locked_marker['azimuth_degrees'],
                                      self.hil_last_locked_marker['elevation_degrees'])
                            self.hil_error = ('Lock lost; returning to the last encoder position '
                                              'where the feature was confirmed, then searching there.')
                        else:
                            if status.get('motion') != 'paused':
                                self.hil_controller.freeze()
                            self.hil_error = ('Returned toward the last confirmed lock position; '
                                              'motion is frozen while the armed link waits for a credible target.')
                    else:
                        if status.get('motion') != 'paused':
                            self.hil_controller.freeze()
                        self.hil_error = ('Target lost; motion is frozen, but the motor link remains armed. '
                                          'A credible reacquisition resumes tracking automatically.')
                    if (target is not None and target_now - self.hil_last_submit
                            >= 1 / self.hil_dispatch_rate_hz):
                        if not status.get('armed'):
                            self.hil_controller.arm()
                        self.hil_sequence += 1
                        ramping = (self.ui_mode == 'live' and self.soft_start_ramp_started
                                   and target_now - self.soft_start_ramp_started < 10)
                        commanded_target = (target if ramping else
                                            self._apply_outer_loop(target, marker))
                        self.hil_controller.submit(
                            commanded_target[0], commanded_target[1],
                            self.hil_sequence, time.time(), ttl=.75)
                        self.hil_current_target = {'azimuth_degrees': target[0],
                                                   'elevation_degrees': target[1]}
                        self.hil_last_submit = target_now
                except (ConnectionError, RuntimeError, ValueError) as error:
                    self.hil_enabled = False
                    self.hil_controller.pause()
                    self.hil_error = str(error)
        encoder_error = None
        if marker and self.hil_current_target:
            encoder_error = {
                'azimuth_degrees': ((self.hil_current_target['azimuth_degrees']
                                     - marker['azimuth_degrees'] + 180) % 360 - 180),
                'elevation_degrees': (self.hil_current_target['elevation_degrees']
                                      - marker['elevation_degrees']),
            }
        return {'available': True, 'enabled': self.hil_enabled,
                'host': self.hil_host, 'port': self.hil_port,
                'envelope_degrees': self.hil_envelope_degrees,
                'travel_envelope_enabled': self.hil_envelope_degrees > 0,
                'display_span_degrees': self.hil_display_span_degrees,
                'dispatch_rate_hz': self.hil_dispatch_rate_hz,
                'connection': status.get('connection'), 'motion': status.get('motion'),
                'armed': status.get('armed'), 'marker': marker,
                'encoder_positions': status.get('positions', {}),
                'cycle_count': status.get('cycle_count'),
                'cycle_overruns': status.get('cycle_overruns'),
                'last_achieved_cycle_rate_hz': status.get('last_achieved_cycle_rate_hz'),
                'target_rate': self.hil_target_rate,
                'current_target': self.hil_current_target,
                'commanded_target': self.hil_commanded_target,
                'control_preset': self.hil_control_preset,
                'prediction_horizon_seconds': (self.hil_auto_horizon
                                                if self.hil_control_preset == 'auto'
                                                else self.hil_prediction_horizon),
                'proportional_lead': self.hil_proportional_lead,
                'correction_deadband_degrees': self.hil_correction_deadband,
                'encoder_error': encoder_error,
                'loss_mode': self.hil_loss_mode,
                'coast_seconds': self.hil_coast_seconds,
                'last_locked_encoder_position': self.hil_last_locked_marker,
                'vibration_filter': {'enabled': self.vibration_filter_enabled,
                                     'threshold_px': self.vibration_threshold_px},
                'rate_warning': status.get('rate_warning'),
                'orientation_calibration': dict(self.orientation_calibration),
                'soft_start_ramp_seconds_remaining': (max(
                    0, 10 - (time.monotonic() - self.soft_start_ramp_started))
                    if self.soft_start_ramp_started else None),
                'link_action_pending': self.hil_link_action,
                'bridge': dict(self.bridge_status,
                               dashboard_connection=status.get('connection'),
                               dashboard_reconnections=status.get('reconnections'),
                               dashboard_manual_reconnects=self.bridge_manual_reconnects,
                               service_restart_in_progress=self.bridge_service_restart_in_progress,
                               service_restart_count=self.bridge_service_restart_count,
                               service_restart_result=self.bridge_service_restart_result,
                               dashboard_last_error=status.get('last_error')),
                'error': self.hil_error or status.get('last_error'),
                'description': (('Live camera feature plus encoder feedback; optical output remains experimental.'
                                 if self.ui_mode == 'live' else
                                 'Synthetic target plus real encoder feedback; this is not optical closed-loop tracking.'))}

    def _check_orientation_sanity(self, error_x, error_y, hil):
        marker = hil.get('marker')
        if (self.ui_mode != 'live' or not self.hil_enabled or not marker
                or error_x is None or error_y is None):
            self.orientation_history.clear()
            return
        now = time.monotonic()
        sample = (now, float(error_x), float(error_y),
                  marker['azimuth_degrees'], marker['elevation_degrees'])
        self.orientation_history.append(sample)
        oldest = next((item for item in self.orientation_history
                       if now - item[0] >= 1.2), None)
        if not oldest:
            return
        _then, old_x, old_y, old_az, old_el = oldest
        _now, new_x, new_y, new_az, new_el = sample
        az_motion = (new_az - old_az + 180) % 360 - 180
        el_motion = new_el - old_el
        old_error = math.hypot(old_x, old_y)
        new_error = math.hypot(new_x, new_y)
        mount_motion = math.hypot(az_motion, el_motion)
        if (old_error >= 8 and new_error >= old_error * 1.25
                and mount_motion >= .02):
            message = ('Optical error increased by more than 25% after measurable mount motion. '
                       'Tracking output was stopped; relearn the camera-to-mount mapping.')
            self.hil_enabled = False
            self.hil_controller.pause()
            self.hil_error = message
            self._discard_saved_orientation()
            self.orientation_calibration = {
                'status': 'invalidated', 'valid': False,
                'message': ('Previous mapping diverged and was discarded. The mount is returning '
                            'to the last confirmed lock position; run soft start again.')}
            self.orientation_sanity = {'status': 'fault', 'message': message}
            hil['enabled'] = False
            hil['armed'] = False
            hil['error'] = message
            self.orientation_history.clear()
            threading.Thread(target=self._return_to_last_lock_after_fault,
                             name='orientation-fault-return', daemon=True).start()

    def _record(self, telemetry, selected):
        if self.record_requested and not self.record_writer:
            self._open_recording()
        if not self.record_requested and self.record_writer:
            self._close_recording()
        if not self.record_writer:
            return
        bbox = selected['bbox'] if selected else ['', '', '', '']
        self.record_writer.writerow((
            telemetry['utc_epoch_s'], telemetry['monotonic_ns'], telemetry['frame'],
            telemetry['state'], telemetry['selected_track_id'],
            selected['x'] if selected else '', selected['y'] if selected else '',
            telemetry['error_x_px'], telemetry['error_y_px'], *bbox,
            selected['confidence'] if selected else '',
            selected['brightness'] if selected else '', len(telemetry['objects']),
            telemetry['capture_fps'], telemetry['processing_ms']))

    def _run(self):
        started = time.monotonic()
        frame_index = 0
        consecutive_capture_timeouts = 0
        try:
            self.capture = self._open_capture()
            while not self.stop_event.is_set():
                cycle = time.monotonic()
                acquired_ns = time.monotonic_ns()
                with self.lock:
                    desired_source = self.desired_source
                    reopen_camera = self.camera_reopen_requested
                    self.camera_reopen_requested = False
                if desired_source != self.source or reopen_camera:
                    self._close_capture()
                    self.source = desired_source
                    self.capture = self._open_capture()
                if self.source == 'synthetic':
                    gray = self._synthetic(frame_index)
                elif self.source == 'picamera2':
                    try:
                        # Picamera2 otherwise waits forever when the CSI sensor
                        # is detected but its frontend stops delivering frames.
                        # A bounded wait keeps Recover Camera and the service
                        # controls usable without rebooting the Pi.
                        request = self.capture.capture_request(wait=2.0)
                    except TimeoutError as error:
                        consecutive_capture_timeouts += 1
                        if consecutive_capture_timeouts > 3:
                            raise RuntimeError(
                                'CSI camera timed out after three automatic reopen attempts') from error
                        with self.frame_ready:
                            self.telemetry = dict(
                                self.telemetry, state='CAMERA RECOVERY',
                                error=(f'CSI frame timeout; automatic reopen attempt '
                                       f'{consecutive_capture_timeouts} of 3.'))
                            self.frame_ready.notify_all()
                        self._close_capture()
                        self.stop_event.wait(1)
                        if self.stop_event.is_set():
                            break
                        self.capture = self._open_capture()
                        continue
                    consecutive_capture_timeouts = 0
                    try:
                        frame = request.make_array('main')
                        self.camera_metadata = request.get_metadata()
                    finally:
                        request.release()
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                else:
                    ok, frame = self.capture.read()
                    if not ok:
                        raise RuntimeError('Camera/video source stopped returning frames')
                    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = self._apply_image_curve(gray)
                height, width = gray.shape[:2]
                with self.lock:
                    if self.tracking_mode == 'threshold':
                        objects = self.tracker.detect(gray)
                        selected_id = self.tracker.selected_id
                    else:
                        feature_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                        if self.feature_roi_pending:
                            self.feature_tracker = create_feature_tracker(
                                self.feature_algorithm, self.feature_lock_size)
                            roi = self.feature_roi_pending
                            self.feature_roi_pending = None
                            initialized = self.feature_tracker.init(feature_frame, roi)
                            if initialized is False:
                                raise RuntimeError('OpenCV could not initialize that feature ROI')
                            self.feature_selected_id = 1
                            self.feature_reference_patch = self._extract_feature_patch(gray, roi)
                            self._reset_feature_guard(clear_reference=False)
                            self.feature_last_confirmed = (
                                roi[0] + roi[2] / 2, roi[1] + roi[3] / 2,
                                time.monotonic())
                            objects = [self._feature_detection(roi)]
                        elif self.feature_tracker:
                            found, bbox = self.feature_tracker.update(feature_frame)
                            if found and self.feature_lock_size and self.feature_initial_size:
                                # Keep the feature-box dimensions separate from the
                                # camera-frame dimensions. Reusing width/height here
                                # caused telemetry to report the ROI as the full frame;
                                # every subsequent click then shrank the coordinate
                                # system again and visually magnified the overlay.
                                box_x, box_y, box_width, box_height = bbox
                                locked_width, locked_height = self.feature_initial_size
                                bbox = (box_x + box_width / 2 - locked_width / 2,
                                        box_y + box_height / 2 - locked_height / 2,
                                        locked_width, locked_height)
                            similarity = self._feature_similarity(gray, bbox) if found else None
                            if (found and self._feature_candidate_is_credible(
                                    bbox, similarity)):
                                self.feature_invalid_frames = 0
                                self.feature_reacquiring = False
                                objects = [self._feature_detection(bbox, similarity)]
                            else:
                                objects = []
                                self.feature_invalid_frames += 1
                                if self.feature_invalid_frames >= 3:
                                    self.feature_tracker = None
                                    self.feature_reacquiring = True
                                    self.feature_reacquire_candidate = None
                                    self.feature_reacquire_confirmations = 0
                        elif (self.feature_reference_patch is not None
                              and self.feature_selected_id is not None):
                            self.feature_reacquiring = True
                            reacquired = self._attempt_feature_reacquisition(
                                gray, feature_frame)
                            objects = [reacquired] if reacquired else []
                        else:
                            objects = []
                        selected_id = self.feature_selected_id
                    selected = next((item for item in objects
                                     if item['track_id'] == selected_id), None)
                    selected = self._filter_selected(selected)
                    self.last_selected = selected
                    if selected:
                        self.feature_samples.append((time.monotonic(),
                                                     float(selected['x']),
                                                     float(selected['y'])))
                    if selected_id is None:
                        state = 'SEARCHING'
                    elif selected:
                        state = 'LOCKED'
                    elif (self.tracking_mode == 'threshold'
                          and self.tracker.selected_id in self.tracker.tracks):
                        state = 'COASTING'
                    elif self.feature_reacquiring:
                        state = 'REACQUIRING'
                    else:
                        state = 'LOST'
                    anchor_x = self.aim_point[0] * width
                    anchor_y = self.aim_point[1] * height
                    error_x = selected['x'] - anchor_x if selected else None
                    error_y = selected['y'] - anchor_y if selected else None
                    if selected:
                        self.errors.append((time.monotonic(), error_x, error_y))
                    hil = self._hil_update(selected, width, height)
                    self._check_orientation_sanity(error_x, error_y, hil)
                encoded, data = cv2.imencode(
                    '.jpg', gray, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                if not encoded:
                    raise RuntimeError('JPEG encoding failed')
                processing_ms = (time.monotonic_ns() - acquired_ns) / 1e6
                elapsed = max(.001, time.monotonic() - started)
                telemetry = {
                    'utc_epoch_s': time.time(), 'monotonic_ns': acquired_ns,
                    'frame': frame_index, 'width': width, 'height': height,
                    'state': state,
                    'motor_output': ('ENABLED' if hil.get('enabled') else 'DISABLED'),
                    'hardware_loop': hil,
                    'objects': objects, 'selected_track_id': selected_id,
                    'tracking_mode': self.tracking_mode,
                    'feature_reacquiring': self.feature_reacquiring,
                    'feature_invalid_frames': self.feature_invalid_frames,
                    'feature_similarity_baseline': self.feature_similarity_baseline,
                    'ui_mode': self.ui_mode,
                    'aim_mode': self.aim_mode,
                    'aim_point': {'x': self.aim_point[0], 'y': self.aim_point[1]},
                    'image_orientation': {
                        'horizontal': ('reversed' if self.image_horizontal_reversed else 'normal'),
                        'vertical': ('reversed' if self.image_vertical_reversed else 'normal')},
                    'orientation_sanity': dict(self.orientation_sanity),
                    'orientation_calibration': dict(self.orientation_calibration),
                    'camera_device': self.device,
                    'camera_settings': {'width': self.width, 'height': self.height,
                                        'fps': self.requested_fps},
                    'camera_auto': {
                        'available': self.source == 'picamera2',
                        'enabled': self.camera_auto_exposure,
                        'exposure_value': self.camera_exposure_value,
                        'manual_exposure_time_us': self.camera_manual_exposure_us,
                        'manual_analogue_gain': self.camera_manual_gain,
                        'exposure_time_us': self.camera_metadata.get('ExposureTime'),
                        'analogue_gain': self.camera_metadata.get('AnalogueGain'),
                        'digital_gain': self.camera_metadata.get('DigitalGain'),
                        'limits': {
                            name: dict(self.camera_control_ranges[name])
                            for name in ('ExposureValue', 'ExposureTime', 'AnalogueGain')
                            if name in self.camera_control_ranges
                        },
                    },
                    'image_curve': {'black': self.curve_black,
                                    'white': self.curve_white,
                                    'gamma': self.curve_gamma},
                    'focus_score': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                    'cameras': self.cameras,
                    'feature_algorithm': self.feature_algorithm,
                    'feature_lock_size': self.feature_lock_size,
                    'error_x_px': error_x, 'error_y_px': error_y,
                    'capture_fps': (frame_index + 1) / elapsed,
                    'processing_ms': processing_ms, 'threshold': self.tracker.threshold,
                    'polarity': self.tracker.polarity,
                    'recording': self.record_requested,
                    'record_path': str(self.record_path) if self.record_path else None,
                    'synthetic_blink': dict(self.synthetic_blink),
                }
                with self.frame_ready:
                    self.jpeg = data.tobytes()
                    self.telemetry = telemetry
                    self._record(telemetry, selected)
                    self.frame_ready.notify_all()
                frame_index += 1
                if self.source == 'synthetic':
                    self.stop_event.wait(max(0, 1 / self.requested_fps -
                                             (time.monotonic() - cycle)))
        except Exception as error:
            if self.hil_controller and self.hil_enabled:
                try:
                    self.hil_controller.freeze()
                except Exception:
                    pass
            with self.frame_ready:
                self.telemetry = dict(
                    self.telemetry, state='CAMERA OFFLINE',
                    error=(f'Camera unavailable: {error}. Motor link remains available; '
                           'use Recover Camera to retry capture.'))
                self.frame_ready.notify_all()
        finally:
            self._close_capture()
            self._close_recording()

    def snapshot(self):
        with self.lock:
            snapshot = dict(self.telemetry)
            # A camera exception can occur before the first frame.  Build the
            # non-camera status dynamically so Live Mode, bridge diagnostics,
            # motor feedback and recovery controls remain usable.
            if self.hil_controller and 'hardware_loop' not in snapshot:
                snapshot.update(
                    width=self.width, height=self.height, ui_mode=self.ui_mode,
                    tracking_mode=self.tracking_mode, selected_track_id=None,
                    camera_device=self.device, cameras=self.cameras,
                    hardware_loop=self._hil_update(None, self.width, self.height))
            return snapshot

    def mjpeg(self):
        last_frame = -1
        while not self.stop_event.is_set():
            with self.frame_ready:
                self.frame_ready.wait_for(
                    lambda last=last_frame: self.telemetry.get('frame', -1) != last or
                    self.stop_event.is_set(), timeout=1)
                if self.stop_event.is_set():
                    break
                last_frame = self.telemetry.get('frame', -1)
                jpeg = self.jpeg
            if jpeg:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       jpeg + b'\r\n')

    def command(self, message):
        action = message.get('action')
        with self.lock:
            if action == 'select_track':
                self.tracker.select(message['track_id'])
            elif action == 'select_at':
                x = float(message['x']) * self.telemetry.get('width', self.width)
                y = float(message['y']) * self.telemetry.get('height', self.height)
                self.tracker.select_at(x, y)
            elif action == 'clear_selection':
                if self.hil_enabled:
                    raise ValueError('Turn tracking output OFF before clearing the target')
                self.clear_target()
            elif action == 'set_threshold':
                self.tracker.set_threshold(message['value'])
            elif action == 'set_polarity':
                self.tracker.set_polarity(message['value'])
            elif action == 'set_recording':
                self.record_requested = bool(message['enabled'])
            elif action == 'set_hil_enabled':
                if bool(message['enabled']):
                    self.request_hil_enable()
                else:
                    self.set_hil_enabled(False)
            elif action == 'set_hil_dispatch_rate':
                self.set_hil_dispatch_rate(message['value'])
            elif action == 'set_hil_display_span':
                self.set_hil_display_span(message['value'])
            elif action == 'set_hil_travel_envelope':
                self.set_hil_travel_envelope(message.get('enabled', False),
                                             message.get('degrees', 5))
            elif action == 'set_hil_loss_behavior':
                self.set_hil_loss_behavior(message['mode'], message.get('coast_seconds', .5))
            elif action == 'set_vibration_filter':
                self.set_vibration_filter(message.get('enabled', False),
                                          message.get('threshold_px', 3))
            elif action == 'set_synthetic_blink':
                self.set_synthetic_blink(message.get('enabled', False),
                                         message['fade_out_seconds'], message['hidden_seconds'],
                                         message['fade_in_seconds'], message['period_seconds'])
            elif action == 'set_tracking_mode':
                self.set_tracking_mode(message['mode'], message.get('algorithm', 'CSRT'),
                                       message.get('lock_size', True))
            elif action == 'set_feature_roi':
                self.set_feature_roi(message['x'], message['y'],
                                     message['width'], message['height'])
            elif action == 'set_feature_point':
                self.set_feature_point(message['x'], message['y'],
                                       message.get('box_fraction', .12))
            elif action == 'set_operating_mode':
                self.set_operating_mode(message['mode'])
            elif action == 'set_aim_mode':
                self.set_aim_mode(message['mode'])
            elif action == 'set_image_orientation':
                self.set_image_orientation(message['horizontal'], message['vertical'])
            elif action == 'start_orientation_learning':
                self.request_orientation_learning(message.get('step'), message.get('rate'))
            elif action == 'clear_orientation_profile':
                self.clear_orientation_profile()
            elif action == 'refresh_cameras':
                self.refresh_cameras()
            elif action == 'recover_camera':
                self.recover_camera_service()
            elif action == 'set_camera':
                self.set_camera(message['device'], message['width'],
                                message['height'], message['fps'])
            elif action == 'set_camera_exposure':
                self.set_camera_exposure(
                    message.get('automatic', True), message.get('exposure_value', 0),
                    message.get('exposure_time_us', 10000),
                    message.get('analogue_gain', 1))
            elif action == 'set_image_curve':
                self.set_image_curve(message['black'], message['white'],
                                     message['gamma'])
            elif action == 'refresh_bridge_status':
                self.bridge_status_next_poll = 0.0
            elif action == 'reconnect_bridge_client':
                self.reconnect_bridge_client()
            elif action == 'restart_bridge_service':
                self.restart_bridge_service()
            elif action == 'reset_live_view':
                self.reset_live_view()
            elif action == 'set_hil_control_strategy':
                self.set_hil_control_strategy(message['preset'],
                                              message.get('horizon', .25),
                                              message.get('proportional', .2),
                                              message.get('deadband', .03))
            else:
                raise ValueError('Unknown dashboard action')
        return self.snapshot()


def create_app(engine, static_directory=None):
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import FileResponse, StreamingResponse
    except ImportError as error:
        raise RuntimeError('Install the vision-dashboard optional dependencies') from error

    static_directory = Path(static_directory or
                            Path(__file__).with_name('vision_dashboard_static'))

    @asynccontextmanager
    async def lifespan(_app):
        engine.start()
        yield
        engine.stop()

    app = FastAPI(title='Telescope Visual Tracking Dashboard', lifespan=lifespan)

    @app.get('/')
    async def index():
        return FileResponse(static_directory / 'index.html')

    @app.get('/video_feed')
    async def video_feed():
        return StreamingResponse(engine.mjpeg(),
                                 media_type='multipart/x-mixed-replace; boundary=frame')

    @app.get('/api/status')
    async def status():
        return engine.snapshot()

    @app.get('/api/camera-controls')
    async def camera_controls():
        return {'camera_device': engine.device,
                'controls': dict(engine.camera_control_ranges)}

    @app.websocket('/ws')
    async def websocket(websocket: WebSocket):
        await websocket.accept()
        last_frame = -2
        try:
            while True:
                snapshot = engine.snapshot()
                if snapshot.get('frame') != last_frame:
                    await websocket.send_json(snapshot)
                    last_frame = snapshot.get('frame')
                try:
                    message = await asyncio.wait_for(websocket.receive_json(), timeout=.05)
                    try:
                        engine.command(message)
                    except (KeyError, RuntimeError, TypeError, ValueError) as error:
                        await websocket.send_json({'event': 'command_error',
                                                   'message': str(error)})
                except asyncio.TimeoutError:
                    pass
        except WebSocketDisconnect:
            pass

    return app


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('live-run', 'display-only', 'synthetic-hardware-loop'),
                        default='live-run')
    parser.add_argument('--source', choices=('synthetic', 'v4l2', 'picamera2', 'file'),
                        default='synthetic')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=float, default=20)
    parser.add_argument('--threshold', type=int, default=160)
    parser.add_argument('--polarity', choices=('bright', 'dark'), default='bright')
    parser.add_argument('--jpeg-quality', type=int, default=65)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--allow-lan', action='store_true')
    parser.add_argument('--log-directory', default='vision-dashboard-logs')
    parser.add_argument('--hil-host', help='Enable guarded synthetic hardware-loop mode through this bridge host')
    parser.add_argument('--hil-port', type=int, default=2000)
    parser.add_argument('--hil-envelope-degrees', type=float, default=0)
    parser.add_argument('--hil-dispatch-rate-hz', type=float, default=4)
    parser.add_argument('--hil-display-span-degrees', type=float, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == 'synthetic-hardware-loop':
        if not args.hil_host:
            raise SystemExit('synthetic-hardware-loop mode requires --hil-host')
        args.source = 'synthetic'
        args.polarity = 'dark'
    elif args.mode == 'live-run':
        if not args.hil_host:
            raise SystemExit('live-run mode requires --hil-host')
        args.source = ('picamera2' if str(args.device).startswith('picamera2:')
                       else 'v4l2')
    if args.host not in ('127.0.0.1', 'localhost', '::1') and not args.allow_lan:
        raise SystemExit('Non-loopback binding requires --allow-lan; SSH tunneling is recommended')
    if not 1 <= args.jpeg_quality <= 95 or not 1 <= args.threshold <= 254:
        raise SystemExit('JPEG quality must be 1–95 and threshold 1–254')
    if not 0 <= args.hil_envelope_degrees <= 90:
        raise SystemExit('Hardware-loop envelope must be 0 (disabled)–90 degrees')
    if not 1 <= args.hil_dispatch_rate_hz <= 20:
        raise SystemExit('Hardware-loop dispatch rate must be 1–20 Hz')
    engine = VisionDashboardEngine(
        source=args.source, device=args.device, width=args.width, height=args.height,
        fps=args.fps, threshold=args.threshold, jpeg_quality=args.jpeg_quality,
        log_directory=args.log_directory, polarity=args.polarity,
        hil_host=args.hil_host, hil_port=args.hil_port,
        hil_envelope_degrees=args.hil_envelope_degrees,
        hil_dispatch_rate_hz=args.hil_dispatch_rate_hz,
        hil_display_span_degrees=args.hil_display_span_degrees,
        ui_mode=('live' if args.mode == 'live-run'
                 or args.source in ('v4l2', 'picamera2', 'file') else 'demo'))
    app = create_app(engine)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, access_log=False,
                timeout_graceful_shutdown=3)


if __name__ == '__main__':
    main()
