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

from network_info import route_source_address


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


def estimate_backlash_breakpoint(trials, minimum_trials=3):
    """Estimate repeatable encoder dead travel before coherent image motion.

    Each trial contains ``(encoder_degrees, pixel_distance)`` samples acquired
    after reversing an already-loaded axis.  Candidate breakpoints are scored
    as a flat pre-motion region followed by a positive linear response.  The
    median of accepted trial breakpoints rejects vibration and tracker jitter;
    spread is returned so callers can refuse an inconsistent measurement.
    """
    estimates = []
    diagnostics = []
    for samples in trials:
        clean = sorted((float(x), float(y)) for x, y in samples
                       if math.isfinite(float(x)) and math.isfinite(float(y)))
        if len(clean) < 6:
            diagnostics.append({'accepted': False, 'reason': 'fewer than 6 samples'})
            continue
        x = np.asarray([item[0] for item in clean], dtype=float)
        y = np.asarray([item[1] for item in clean], dtype=float)
        baseline_values = y[:min(3, len(y))]
        baseline = float(np.median(baseline_values))
        noise = float(np.median(np.abs(baseline_values - baseline)))
        activation = baseline + max(1.5, noise * 4)
        best = None
        for index in range(2, len(clean) - 3):
            tail_x, tail_y = x[index:], y[index:]
            slope, intercept = np.polyfit(tail_x, tail_y, 1)
            if slope <= 0:
                continue
            breakpoint = (activation - intercept) / slope
            if not 0 <= breakpoint <= x[-1]:
                continue
            pre_residual = float(np.mean((y[:index] - baseline) ** 2))
            predicted = slope * tail_x + intercept
            post_residual = float(np.mean((tail_y - predicted) ** 2))
            total_variance = float(np.sum((tail_y - np.mean(tail_y)) ** 2))
            r_squared = (1.0 - float(np.sum((tail_y - predicted) ** 2)) /
                         total_variance) if total_variance > 1e-9 else 0.0
            score = pre_residual + post_residual
            candidate = (score, float(breakpoint), float(slope), index,
                         float(r_squared))
            if best is None or candidate[0] < best[0]:
                best = candidate
        if (best is None or best[4] < .75 or
                float(np.max(y) - baseline) < 3):
            diagnostics.append({'accepted': False,
                                'reason': ('poor linear response or probable '
                                           'tracker jump')})
            continue
        estimates.append(best[1])
        diagnostics.append({'accepted': True, 'breakpoint_degrees': best[1],
                            'slope_pixels_per_degree': best[2],
                            'fit_score': best[0], 'fit_start_index': best[3],
                            'r_squared': best[4]})
    minimum_trials = int(minimum_trials)
    if len(estimates) < minimum_trials:
        raise ValueError(
            f'Backlash characterization needs {minimum_trials} consistent '
            'reversals; '
            'reacquire a distinctive feature and retry')
    median = float(np.median(estimates))
    spread = float(np.median(np.abs(np.asarray(estimates) - median)))
    if spread > max(.025, median * .5):
        raise ValueError(
            f'Backlash reversals were inconsistent (median {median:.3f}\u00b0, '
            f'spread {spread:.3f}\u00b0)')
    return {'degrees': median, 'spread_degrees': spread,
            'accepted_trials': len(estimates), 'trial_fits': diagnostics}


def learned_mount_delta(error_x, error_y, calibration):
    """Return mount movement needed to drive the measured pixel error to zero."""
    inverse = calibration['inverse_degrees_per_pixel']
    return (-(inverse[0][0] * error_x + inverse[0][1] * error_y),
            -(inverse[1][0] * error_x + inverse[1][1] * error_y))


def transfer_mount_delta(error_x, error_y, coefficients):
    """Invert the known artificial encoder-to-pixel plant.

    Transfer mode renders ``pixel = path - M * encoder_delta``.  The encoder
    correction which cancels the current pixel error is therefore
    ``M^-1 * error``.  This intentionally does not use a saved camera mapping:
    a stale real-camera profile would make the synthetic plant test a test of
    the wrong matrix and can reduce valid commands below the mount's backlash.
    """
    a = float(coefficients['azimuth_x_px_per_degree'])
    b = float(coefficients['elevation_x_px_per_degree'])
    c = float(coefficients['azimuth_y_px_per_degree'])
    d = float(coefficients['elevation_y_px_per_degree'])
    determinant = a * d - b * c
    if abs(determinant) < 1e-9:
        raise ValueError('Artificial transfer matrix is singular')
    return ((d * error_x - b * error_y) / determinant,
            (-c * error_x + a * error_y) / determinant)


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
                 hil_dispatch_rate_hz=4, hil_feedback_timeout_seconds=10,
                 hil_display_span_degrees=5,
                 ui_mode='demo', hil_route_target=None,
                 bridge_recovery_quiet_seconds=8):
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
        self.orientation_auto_reversed = False
        self.orientation_calibration = {
            'status': 'not learned', 'valid': False,
            'message': 'Lock a stable feature, then turn tracking ON to run soft start.'}
        self.orientation_calibration_thread = None
        self.orientation_cancel = threading.Event()
        self.soft_start_step_degrees = .05
        self.soft_start_slew_rate = 5
        self.soft_start_ramp_started = None
        data_root = os.getenv('TELESCOPE_OPTICAL_TRACKING_CONTROL_DATA_DIR', os.getenv(
            'TELESCOPE_CONTROL_DATA_DIR', os.getenv(
                'COSMOS90GT_DATA_DIR', Path.home() / '.telescope-optical-tracking-control')))
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
        # Camera enumeration can take tens of seconds on a Pi with many V4L2
        # nodes or a slow CSI frontend. Never hold HTTP startup behind it.
        self.cameras = []
        self.camera_inventory_thread = None
        self.camera_inventory_status = 'Camera discovery has not started.'
        self.camera_device_auto = str(self.device).lower() == 'auto'
        if str(self.device).startswith('picamera2:'):
            self.source = self.desired_source = 'picamera2'
        self.camera_reopen_requested = False
        self.camera_recovery_requested = False
        self.camera_recovery_count = 0
        self.camera_recovery_thread = None
        self.camera_recovery_in_progress = False
        self.camera_recovery_status = 'Camera recovery has not been requested.'
        self.camera_recovery_result = None
        self.camera_metadata = {}
        self.camera_control_ranges = {}
        self.camera_auto_exposure = True
        self.camera_gain_locked = False
        self.camera_software_ae_last = 0.0
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
        # KCF is substantially lighter than CSRT on the Pi and keeps CSI
        # capture draining while a live feature is locked.
        self.feature_algorithm = 'KCF'
        self.feature_lock_size = True
        self.feature_initial_size = None
        self.feature_reference_patch = None
        self.feature_similarity_threshold = .22
        self.feature_similarity_baseline = 1.0
        self.feature_invalid_frames = 0
        self.feature_reacquiring = False
        self.feature_reacquire_candidate = None
        self.feature_reacquire_confirmations = 0
        self.manual_lock_hold = None
        self.manual_lock_hold_generation = 0
        self.hil_reference_rebasing = False
        self.feature_last_confirmed = None
        self.feature_tracker = None
        self.feature_roi_pending = None
        self.feature_selected_id = None
        self.feature_error = None
        self.log_directory = Path(log_directory).expanduser()
        self.lock = threading.RLock()
        self.frame_ready = threading.Condition(self.lock)
        self.stop_event = threading.Event()
        self.thread = None
        self.capture = None
        self.jpeg = None
        # Keep the first WebSocket snapshot schema complete even when camera
        # discovery or CSI recovery has not produced a frame yet.  The browser
        # must remain usable for Bridge status and recovery controls.
        self.telemetry = {
            'frame': -1,
            'state': 'STARTING',
            'motor_output': 'DISABLED',
            'objects': [],
            'selected_track_id': None,
            'error_x_px': None,
            'error_y_px': None,
            'capture_fps': None,
            'processing_ms': None,
            'recording': False,
            'record_path': None,
        }
        self.record_requested = False
        self.record_path = None
        self.record_stream = None
        self.record_writer = None
        self.errors = deque(maxlen=1200)
        self.hil_host = hil_host
        self.hil_route_target = hil_route_target
        self.hil_port = int(hil_port)
        self.bridge_recovery_quiet_seconds = max(
            0.0, float(bridge_recovery_quiet_seconds))
        self.hil_envelope_degrees = float(hil_envelope_degrees)
        self.hil_dispatch_rate_hz = float(hil_dispatch_rate_hz)
        # Measured effective one-second pulse average for native slew level 5
        # on the COSMOS 90GT profile. The UI exposes all measured levels 1-9.
        self.hil_max_tracking_slew_deg_s = .0753
        self.hil_feedback_timeout_seconds = float(hil_feedback_timeout_seconds)
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
        # Development default: keep following the last measured trajectory
        # through a temporary visual dropout instead of freezing the mount.
        self.hil_loss_mode = 'predictive-coast'
        self.hil_coast_seconds = .25
        self.hil_last_locked_marker = None
        self.hil_return_started = None
        self.vibration_filter_enabled = False
        self.vibration_threshold_px = 3.0
        self.vibration_filtered_point = None
        self.vibration_candidate = None
        self.vibration_candidate_frames = 0
        self.hil_control_preset = 'balanced'
        self.hil_prediction_horizon = .05
        self.hil_proportional_lead = .55
        self.hil_derivative_damping = .10
        self.hil_correction_deadband = .02
        self.hil_control_error_state = None
        self.hil_filtered_error_rate = (0.0, 0.0)
        self.hil_control_terms = None
        self.hil_initial_trajectory_enabled = False
        self.hil_initial_trajectory_window = 2.0
        self.hil_initial_trajectory = {
            'valid': False, 'message': 'Initial trajectory learning is OFF.'}
        self.hil_auto_horizon = .1
        self.hil_commanded_target = None
        self.hil_error = None
        self.synthetic_blink = {'enabled': False, 'fade_out_seconds': .5,
                                'hidden_seconds': .5, 'fade_in_seconds': .5,
                                'period_seconds': 4.0, 'phase': 'steady',
                                'intensity': None}
        # Transfer-Function Test simulates the camera response from the
        # real mount encoders. The cross-axis terms force the learner to
        # identify a full 2x2 transform instead of assuming orthogonal axes.
        self.transfer_function = {
            'azimuth_x_px_per_degree': 180.0,
            'azimuth_y_px_per_degree': 25.0,
            'elevation_x_px_per_degree': -18.0,
            'elevation_y_px_per_degree': -150.0,
        }
        self.transfer_encoder_origin = None
        self.transfer_path_started = time.monotonic()
        self.transfer_pan_duration_seconds = 30 * 60
        self.transfer_frozen_path = None
        self.transfer_state = {
            'encoder_live': False, 'encoder_delta_degrees': None,
            'path_offset_pixels': None, 'plant_offset_pixels': None}
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
        self._start_camera_inventory()
        self.camera_recovery_requested = False
        self.thread = threading.Thread(target=self._run, name='vision-capture', daemon=True)
        self.thread.start()

    def _start_camera_inventory(self):
        if self.camera_inventory_thread and self.camera_inventory_thread.is_alive():
            return

        def worker():
            self.camera_inventory_status = 'Searching for CSI and USB cameras.'
            try:
                cameras = enumerate_cameras()
                with self.lock:
                    self.cameras = cameras
                    if self.camera_device_auto and cameras:
                        preferred = next((camera for camera in cameras
                                          if 'usb' in camera['name'].lower()
                                          or 'uvc' in camera['name'].lower()), cameras[0])
                        self.device = preferred['device']
                        self.desired_source = ('picamera2'
                                               if self.device.startswith('picamera2:')
                                               else 'v4l2')
                        self.camera_reopen_requested = True
                self.camera_inventory_status = (
                    f'Camera discovery complete: {len(cameras)} capture device(s) found.')
            except Exception as error:
                self.camera_inventory_status = f'Camera discovery failed: {error}'

        self.camera_inventory_thread = threading.Thread(
            target=worker, name='camera-inventory', daemon=True)
        self.camera_inventory_thread.start()

    def _new_position_controller(self):
        from position_control import PositionController
        controller = PositionController(
            self.hil_host, self.hil_port,
            envelope_degrees=self.hil_envelope_degrees,
            feedback_timeout=self.hil_feedback_timeout_seconds,
            dispatch_rate_hz=self.hil_dispatch_rate_hz,
            stop_on_divergence=False)
        controller.start()
        return controller

    def _current_bridge_host(self):
        """Return the current local route address used to reach the telescope.

        The bridge listener runs on the Pi's LAN address. DHCP can briefly
        assign one address during boot and then replace it. When a route target
        is configured, re-resolve that local address instead of retaining the
        stale launch-time value.
        """
        if not self.hil_route_target:
            return self.hil_host
        return route_source_address(self.hil_route_target, self.hil_port)

    def _refresh_bridge_host(self):
        current = self._current_bridge_host()
        if not current:
            raise ConnectionError('No routed local IPv4 address is available for the telescope.')
        changed = current != self.hil_host
        self.hil_host = current
        if changed:
            self.bridge_status_next_poll = 0.0
        return changed

    def _stop_position_controller(self):
        controller = self.hil_controller
        if not controller:
            return
        controller.pause()
        controller.stop()
        if controller.thread:
            controller.thread.join(5)
        if controller.thread and controller.thread.is_alive():
            raise RuntimeError('Previous dashboard connection did not close; retry after it stops')
        self.hil_controller = None

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
        telescope = self.hil_route_target or self.hil_host or 'display-only'
        source = 'transfer-function-test' if self.ui_mode == 'transfer' else 'camera'
        return (f'{source}|{telescope}|{self.device}|{self.width}x{self.height}|'
                f'hrev{int(self.image_horizontal_reversed)}|'
                f'vrev{int(self.image_vertical_reversed)}')

    def _load_orientation_profile(self):
        try:
            profile_path = self.orientation_profile_path
            if not profile_path.exists():
                legacy_path = Path.home() / '.cosmos90gt' / 'vision-orientation-profiles.json'
                if legacy_path.exists():
                    profile_path = legacy_path
            payload = json.loads(profile_path.read_text(encoding='utf-8'))
            profile = payload.get(self._orientation_profile_key())
            if not profile:
                # The bridge computer's LAN address can change.  The mapping is
                # a relationship between this camera/resolution and the mount,
                # so do not strand a measured mapping under an old bridge IP.
                profile = next((candidate for candidate in payload.values()
                                if candidate.get('camera_device') == self.device
                                and candidate.get('width') == self.width
                                and candidate.get('height') == self.height
                                and bool(candidate.get(
                                    'image_horizontal_reversed', False)) ==
                                    self.image_horizontal_reversed
                                and bool(candidate.get(
                                    'image_vertical_reversed', False)) ==
                                    self.image_vertical_reversed
                                and candidate.get('profile_source', 'camera') ==
                                    ('transfer-function-test'
                                     if self.ui_mode == 'transfer' else 'camera')
                                and candidate.get('inverse_degrees_per_pixel')), None)
            if profile:
                profile = dict(profile)
                inverse = profile.get('inverse_degrees_per_pixel')
                usable = (isinstance(inverse, list) and len(inverse) == 2
                          and all(isinstance(row, list) and len(row) == 2
                                  for row in inverse))
                profile.update(status=('loaded' if usable else 'invalid'), valid=usable,
                               warm_profile=True,
                               message=('Saved measured camera-to-mount mapping loaded and ready.'
                                        if usable else
                                        'Saved camera-to-mount mapping is incomplete; run Learn once.'))
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
                       telescope_host=self.hil_route_target or self.hil_host,
                       image_horizontal_reversed=self.image_horizontal_reversed,
                       image_vertical_reversed=self.image_vertical_reversed,
                       profile_source=('transfer-function-test'
                                       if self.ui_mode == 'transfer' else 'camera'),
                       saved_utc_epoch_s=time.time())
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
            service_state = None
            try:
                service = subprocess.run(
                    ['/usr/bin/systemctl', '--user', 'is-active',
                     'celestron-bridge.service'],
                    capture_output=True, text=True, timeout=.75, check=False)
                service_state = service.stdout.strip() or None
            except (OSError, subprocess.SubprocessError):
                pass
            message = f'Bridge management API unavailable: {error}'
            if service_state in ('active', 'activating'):
                message = (f'Control Bridge service is {service_state}, but its management '
                           'port is not ready. It is retrying the telescope connection; '
                           'verify telescope power, Wi-Fi association and endpoint address.')
            self.bridge_status = {
                'management_reachable': False, 'verified_proxy': False,
                'service_state': service_state, 'message': message}
        finally:
            if sock:
                sock.close()
        return self.bridge_status

    def _bridge_monitor(self):
        while not self.stop_event.is_set():
            if self.hil_route_target and not self.bridge_service_restart_in_progress:
                try:
                    if self._current_bridge_host() != self.hil_host:
                        self.bridge_service_restart_result = (
                            'The Pi LAN address changed; automatic Bridge recovery started.')
                        self.restart_bridge_service()
                except OSError:
                    pass
            self._poll_bridge_management(force=True)
            self.stop_event.wait(2.0)

    def reconnect_bridge_client(self):
        """Restart this dashboard's motor link without restarting Control Bridge."""
        if not self.hil_host and not self.hil_route_target:
            raise ValueError('No bridge endpoint was configured at launch')
        if self.hil_enabled:
            self.set_hil_enabled(False)
        self._stop_position_controller()
        self._refresh_bridge_host()
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
        if not self.hil_host and not self.hil_route_target:
            raise ValueError('No bridge endpoint was configured at launch')
        if self.hil_enabled:
            self.set_hil_enabled(False)
        self.bridge_service_restart_in_progress = True
        self.bridge_service_restart_result = 'Restart requested; stopping the old motor link.'
        self.hil_error = self.bridge_service_restart_result

        def worker():
            try:
                self.bridge_service_restart_result = (
                    'Stage 1/4: releasing the dashboard motor connection.')
                self._stop_position_controller()
                result = subprocess.run(
                    ['/usr/bin/systemctl', '--user', 'stop', 'celestron-bridge.service'],
                    capture_output=True, text=True, timeout=20, check=False)
                if result.returncode:
                    detail = (result.stderr or result.stdout).strip()
                    raise RuntimeError(
                        'Control Bridge service stop failed' +
                        (f': {detail}' if detail else '.'))
                self.bridge_service_restart_result = (
                    'Stage 2/4: Bridge stopped; allowing the legacy Wi-Fi socket to close.')
                if self.stop_event.wait(self.bridge_recovery_quiet_seconds):
                    raise RuntimeError('Bridge recovery was cancelled during shutdown.')
                changed = self._refresh_bridge_host()
                self.bridge_service_restart_result = (
                    'Stage 3/4: starting the Bridge on ' + self.hil_host +
                    (' after refreshing the Pi LAN address.' if changed else '.'))
                result = subprocess.run(
                    ['/usr/bin/systemctl', '--user', 'start', 'celestron-bridge.service'],
                    capture_output=True, text=True, timeout=20, check=False)
                if result.returncode:
                    detail = (result.stderr or result.stdout).strip()
                    raise RuntimeError(
                        'Control Bridge service start failed' +
                        (f': {detail}' if detail else '.'))
                self.bridge_service_restart_count += 1
                self.bridge_service_restart_result = (
                    'Stage 3/4: waiting for the Bridge management endpoint.')
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    status = self._poll_bridge_management(force=True)
                    if status.get('verified_proxy'):
                        break
                    time.sleep(.25)
                else:
                    raise ConnectionError(
                        'Control Bridge restarted, but its management endpoint did not return.')

                self.hil_controller = self._new_position_controller()
                self.bridge_manual_reconnects += 1
                self.bridge_service_restart_result = (
                    'Stage 4/4: Bridge is running; waiting for both motor encoders.')
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

    def _hybrid_target(self, frame, index):
        """Overlay one deterministic target while retaining the real camera frame."""
        height, width = frame.shape[:2]
        t = index / max(1, self.requested_fps)
        x = width * (.5 + .22 * math.sin(t * .45))
        y = height * (.5 + .18 * math.sin(t * .31 + .7))
        radius = max(6, round(min(width, height) * .015))
        # A dark rim plus bright core remains visible on bright or dark camera
        # scenes. Control uses the known synthetic coordinates directly; scene
        # thresholding cannot steal the target identity.
        cv2.circle(frame, (round(x), round(y)), radius + 3, 0, -1)
        cv2.circle(frame, (round(x), round(y)), radius, 255, -1)
        return {'track_id': 1, 'x': float(x), 'y': float(y),
                'bbox': [round(x - radius), round(y - radius),
                         radius * 2, radius * 2],
                'area': float(math.pi * radius * radius),
                'brightness': 255.0, 'confidence': 1.0,
                'missed': 0, 'selected': True, 'synthetic': True}

    @staticmethod
    def _wrapped_degrees_delta(current, origin):
        return (float(current) - float(origin) + 180.0) % 360.0 - 180.0

    def _transfer_path(self):
        """Return a one-way, 30-minute artificial-sky pan in camera pixels."""
        elapsed = time.monotonic() - self.transfer_path_started
        progress = max(0.0, min(1.0, elapsed / self.transfer_pan_duration_seconds))
        # Traverse 32% of the frame horizontally while a shallow half-sine
        # adds cross-axis motion.  At 640x480 this is only ~0.114 px/s in X,
        # slow enough for a 30-minute camera/motor/link stability soak.
        return (self.width * .32 * (progress - .5),
                self.height * .08 * math.sin(math.pi * progress))

    def _transfer_target(self, frame):
        """Render the real-encoder-driven artificial optical plant.

        Pixel position = moving object path - M * encoder displacement. The
        scene therefore moves opposite telescope motion. During Learn/Validate
        the path freezes so the learner measures only the encoder response.
        """
        height, width = frame.shape[:2]
        learning = self.orientation_calibration.get('status') == 'running'
        if learning:
            if self.transfer_frozen_path is None:
                self.transfer_frozen_path = self._transfer_path()
            path_x, path_y = self.transfer_frozen_path
        else:
            self.transfer_frozen_path = None
            path_x, path_y = self._transfer_path()

        positions = {}
        connection = 'disconnected'
        if self.hil_controller:
            status = self.hil_controller.status()
            connection = status.get('connection', 'unknown')
            positions = status.get('positions', {})
        azimuth = positions.get('Azimuth', {}).get('raw_degrees')
        elevation = positions.get('Elevation', {}).get('raw_degrees')
        encoder_live = connection == 'live' and azimuth is not None and elevation is not None
        if encoder_live and self.transfer_encoder_origin is None:
            self.transfer_encoder_origin = (float(azimuth), float(elevation))

        delta_azimuth = delta_elevation = 0.0
        if encoder_live and self.transfer_encoder_origin:
            delta_azimuth = self._wrapped_degrees_delta(
                azimuth, self.transfer_encoder_origin[0])
            delta_elevation = self._wrapped_degrees_delta(
                elevation, self.transfer_encoder_origin[1])
        coefficients = self.transfer_function
        plant_x = (coefficients['azimuth_x_px_per_degree'] * delta_azimuth
                   + coefficients['elevation_x_px_per_degree'] * delta_elevation)
        plant_y = (coefficients['azimuth_y_px_per_degree'] * delta_azimuth
                   + coefficients['elevation_y_px_per_degree'] * delta_elevation)
        x = width / 2 + path_x - plant_x
        y = height / 2 + path_y - plant_y
        radius = max(6, round(min(width, height) * .015))
        cv2.circle(frame, (round(x), round(y)), radius + 3, 0, -1)
        cv2.circle(frame, (round(x), round(y)), radius, 255, -1)
        self.transfer_state = {
            'encoder_live': encoder_live,
            'encoder_origin_degrees': self.transfer_encoder_origin,
            'encoder_delta_degrees': [delta_azimuth, delta_elevation],
            'path_offset_pixels': [path_x, path_y],
            'plant_offset_pixels': [plant_x, plant_y],
            'path_frozen_for_learning': learning,
            'path_mode': '30-minute-slow-pan',
            'path_elapsed_seconds': min(
                self.transfer_pan_duration_seconds,
                time.monotonic() - self.transfer_path_started),
            'path_duration_seconds': self.transfer_pan_duration_seconds,
            'coefficients': dict(coefficients),
        }
        return {'track_id': 1, 'x': float(x), 'y': float(y),
                'bbox': [round(x - radius), round(y - radius), radius * 2, radius * 2],
                'area': float(math.pi * radius * radius),
                'brightness': 255.0, 'confidence': 1.0,
                'missed': 0, 'selected': True, 'synthetic': True,
                'transfer_function_test': True}

    def set_transfer_function(self, azimuth_x, azimuth_y, elevation_x, elevation_y):
        if self.ui_mode != 'transfer':
            raise ValueError('Transfer terms can be changed only in Transfer-Function Test')
        if self.hil_enabled:
            raise ValueError('Turn tracking output OFF before changing the transfer function')
        values = [float(value) for value in
                  (azimuth_x, azimuth_y, elevation_x, elevation_y)]
        if not all(math.isfinite(value) and abs(value) <= 1000 for value in values):
            raise ValueError('Each transfer term must be finite and within -1000 to 1000 px/degree')
        determinant = values[0] * values[3] - values[2] * values[1]
        scale = max(1.0, math.hypot(values[0], values[1]) *
                    math.hypot(values[2], values[3]))
        if abs(determinant) / scale < .05:
            raise ValueError('Transfer matrix is singular or nearly singular; use two distinct axis vectors')
        self.transfer_function = {
            'azimuth_x_px_per_degree': values[0],
            'azimuth_y_px_per_degree': values[1],
            'elevation_x_px_per_degree': values[2],
            'elevation_y_px_per_degree': values[3],
        }
        self.transfer_encoder_origin = None
        self.transfer_path_started = time.monotonic()
        self.transfer_frozen_path = None
        self._invalidate_orientation('Transfer function changed; run Learn / validate to identify it.')

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
            if self.ui_mode in ('live', 'transfer') and not self.orientation_calibration.get('valid'):
                # Learning improves the camera/mount transform, but it must not
                # gate basic tracking.  Start with the display-span scale and
                # conventional camera axes; the live sanity monitor reverses
                # the correction if observed error grows after mount motion.
                # This provisional transform is deliberately not persisted.
                self.orientation_calibration = {
                    'status': 'provisional',
                    'valid': True,
                    'provisional': True,
                    'inverse_degrees_per_pixel': [
                        [self.hil_display_span_degrees / max(1, self.width), 0.0],
                        [0.0, self.hil_display_span_degrees / max(1, self.height)]],
                    'message': ('Tracking started with a provisional camera-to-mount '
                                'mapping; Learn can refine it later.')}
            if not status.get('local_reference_set'):
                self.hil_controller.reference(180, 0)
            self.hil_controller.arm()
            self.hil_enabled = True
            self.hil_error = None
            self.soft_start_ramp_started = None
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

    def manual_jog(self, axis, direction, rate):
        """Start/renew immediate operator jogging without optical calibration."""
        if not self.hil_controller:
            raise ValueError('Motor link is not connected; use Recover Bridge and Motor Link')
        axis_id = {'azimuth': 0x10, 'elevation': 0x11}.get(str(axis).lower())
        if axis_id is None:
            raise ValueError('Manual jog axis must be azimuth or elevation')
        if self.hil_enabled:
            self.set_hil_enabled(False)
        self.hil_controller.manual_jog(axis_id, int(direction), int(rate), 2.0)
        self.hil_error = None

    def manual_stop(self):
        """Priority manual Stop; safe to call when the link is unavailable."""
        self.hil_enabled = False
        if self.hil_controller:
            self.hil_controller.manual_stop()

    def zero_encoder_marker(self):
        """Put the current encoder position at the display center reference.

        This changes only the controller's relative session reference. It does
        not write raw motor encoders or establish celestial alignment. Active
        optical correction is briefly frozen, while the Bridge link and the
        operator's armed tracking request are retained.
        """
        if not self.hil_controller:
            raise ValueError('Motor link is not connected; use Recover Bridge and Motor Link')
        status = self.hil_controller.status()
        if status.get('connection') != 'live':
            raise ValueError('Fresh motor feedback is required to zero the encoder marker')
        self.hil_reference_rebasing = True
        try:
            self.hil_controller.freeze()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                status = self.hil_controller.status()
                if status.get('motion') in ('idle', 'paused', 'stale_target_stopped'):
                    break
                time.sleep(.02)
            else:
                raise RuntimeError('Motor motion did not pause; encoder marker was not changed')
            self.hil_controller.reference(180, 0)
            self.hil_previous_target = None
            self.hil_current_target = None
            self.hil_last_visible_target = None
            self.hil_target_rate = None
            self.hil_last_locked_marker = {
                'azimuth_degrees': 180.0, 'elevation_degrees': 0.0,
                'utc_epoch_s': time.time()}
            self.hil_error = ('Encoder marker zeroed at the center cross. This is a '
                              'relative session reference, not telescope alignment.')
        finally:
            self.hil_reference_rebasing = False

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

    def abort_orientation_learning(self):
        thread = self.orientation_calibration_thread
        if not (thread and thread.is_alive()):
            raise ValueError('No learning sequence is running')
        self.orientation_cancel.set()
        self.hil_enabled = False
        if self.hil_controller:
            self.hil_controller.pause()
        self.hil_error = 'Learning abort requested; Stop sent and Bridge link retained.'

    def set_hil_dispatch_rate(self, value):
        if not self.hil_controller:
            raise ValueError('Hardware-loop mode was not enabled at launch')
        if self.hil_enabled:
            raise ValueError('Turn motor output OFF before changing dispatch rate')
        self.hil_controller.set_dispatch_rate(value)
        self.hil_dispatch_rate_hz = self.hil_controller.dispatch_rate_hz

    def set_hil_feedback_timeout(self, value):
        if not self.hil_controller:
            raise ValueError('Hardware-loop mode was not enabled at launch')
        self.hil_controller.set_feedback_timeout(value)
        self.hil_feedback_timeout_seconds = self.hil_controller.feedback_timeout

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
            # A mode button must remain actionable. Stop active tracking before
            # changing its source instead of trapping the operator in a mode.
            self.set_hil_enabled(False)
        if mode not in ('live', 'hybrid', 'transfer', 'demo'):
            raise ValueError('Select Live, Hybrid Camera, Transfer-Function, or Demo Mode')
        self.ui_mode = mode
        # Hybrid and Transfer-Function Test deliberately keep the physical
        # camera capture draining. Their synthetic targets replace detector
        # output for control, but the live background exposes CSI/USB capture
        # failures that occur only while the mount is moving.
        self.desired_source = (('picamera2' if str(self.device).startswith('picamera2:')
                                else 'v4l2')
                               if mode in ('live', 'hybrid', 'transfer') else 'synthetic')
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
        self.transfer_encoder_origin = None
        self.transfer_path_started = time.monotonic()
        self.transfer_frozen_path = None
        self._invalidate_orientation('Operating mode changed; soft start must be relearned.')
        self._load_orientation_profile()
        if not (self.thread and self.thread.is_alive()):
            self.start()

    def set_aim_mode(self, mode):
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
        if horizontal not in ('normal', 'reversed') or vertical not in ('normal', 'reversed'):
            raise ValueError('Image orientation must be Normal or Reversed for each axis')
        requested = (horizontal == 'reversed', vertical == 'reversed')
        if requested == (self.image_horizontal_reversed,
                         self.image_vertical_reversed):
            return
        if self.hil_enabled:
            self.set_hil_enabled(False)
        self.image_horizontal_reversed = horizontal == 'reversed'
        self.image_vertical_reversed = vertical == 'reversed'
        self.clear_target()
        self._invalidate_orientation(
            'Preview orientation changed; select the feature again and relearn motor mapping.')
        self._load_orientation_profile()
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
            if selected and len(samples) >= 6:
                recent = samples[-6:]
                center_x = float(np.median([item[1] for item in recent]))
                center_y = float(np.median([item[2] for item in recent]))
                spread = max(math.hypot(item[1] - center_x,
                                        item[2] - center_y)
                             for item in recent)
                box = selected.get('bbox') or [0, 0, 20, 20]
                allowed_spread = max(1.5, min(5.0,
                    math.hypot(float(box[2]), float(box[3])) * .08))
                if spread <= allowed_spread:
                    return center_x, center_y
            time.sleep(.04)
        raise RuntimeError(
            'Feature lock centroid was unstable or jumped during learning; '
            'reacquire a distinctive feature and retry')

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
        """Return both encoder references at the selected bounded calibration rate.

        Earlier builds forced every return to rate 2.  A sequence of rejected
        probes can accumulate more than a degree of travel, and the COSMOS 90GT
        may not cover that distance at rate 2 before the bounded probe timeout.
        That left learning failed during cleanup even though the outward probe
        worked.  Use the operator-selected calibration rate for the return too;
        the encoder target still bounds travel and ``finally`` always freezes
        both motors.
        """
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
                    int(rate), require_feature=False)
        self._reseed_feature_at(baseline_px)
        return self._stable_feature_sample(time.monotonic(), timeout=4)

    def _characterize_axis_backlash(self, axis, baseline, baseline_px,
                                    slew_rate, trial_count=3):
        """Measure encoder dead travel after reversals using image response.

        The axis is first loaded opposite the trial direction.  It is then
        reversed through 0.02-degree encoder increments while settled feature
        displacement is sampled.  A flat-plus-linear breakpoint is fitted per
        trial and the median of three alternating-direction trials becomes the
        take-up distance used by the mapping probe.
        """
        motor = 'Azimuth' if axis == 0 else 'Elevation'
        load_distance = .08
        increment = .02
        maximum_sweep = .20
        trial_samples = []
        trial_records = []
        accepted_directions = []
        maximum_attempts = max(int(trial_count), 5)
        for attempt_number in range(maximum_attempts):
            if len(trial_samples) >= int(trial_count):
                break
            direction = 1 if attempt_number % 2 == 0 else -1
            samples = []
            pixel_points = []
            try:
                self.orientation_calibration.update(
                    message=(f'{motor} backlash attempt {attempt_number + 1}/'
                             f'{maximum_attempts} ({len(trial_samples)}/'
                             f'{trial_count} accepted): loading gears…'),
                    phase=f'{motor.lower()} backlash characterization')
                self._rate_axis_displacement(
                    axis, -direction, load_distance, slew_rate,
                    require_feature=True)
                reversal_px = self._stable_feature_sample(
                    time.monotonic(), timeout=2.5)
                reversal_status = self.hil_controller.status()
                reversal_encoder = reversal_status['positions'][motor]['encoder']
                samples.append((0.0, 0.0))
                pixel_points.append((0.0, 0.0))
                box_width, box_height = self.feature_initial_size or (40, 40)
                jump_limit = max(12.0, math.hypot(
                    float(box_width), float(box_height)) * .55)
                for sample_number in range(round(maximum_sweep / increment)):
                    if self.orientation_cancel.is_set():
                        raise RuntimeError('Soft start canceled; motor output remains OFF')
                    self._rate_axis_displacement(
                        axis, direction, increment, slew_rate,
                        require_feature=True)
                    point = self._stable_feature_sample(
                        time.monotonic(), timeout=2.5)
                    status = self.hil_controller.status()
                    encoder = status['positions'][motor]['encoder']
                    encoder_degrees = abs(wrapped_encoder_delta(
                        encoder, reversal_encoder) * 360 / (1 << 24))
                    pixel_distance = math.hypot(
                        point[0] - reversal_px[0], point[1] - reversal_px[1])
                    pixel_vector = (point[0] - reversal_px[0],
                                    point[1] - reversal_px[1])
                    previous_point = pixel_points[-1]
                    incremental_jump = math.hypot(
                        pixel_vector[0] - previous_point[0],
                        pixel_vector[1] - previous_point[1])
                    if incremental_jump > jump_limit:
                        raise RuntimeError(
                            f'Feature lock jumped {incremental_jump:.1f} px '
                            f'(limit {jump_limit:.1f} px)')
                    if (samples and pixel_distance + 3 < samples[-1][1] and
                            samples[-1][1] >= 4):
                        raise RuntimeError(
                            'Feature path reversed during a one-direction '
                            'backlash sweep; probable lock hunt')
                    samples.append((encoder_degrees, pixel_distance))
                    pixel_points.append(pixel_vector)
                    self.orientation_calibration.update(
                        message=(f'{motor} backlash attempt {attempt_number + 1}/'
                                 f'{maximum_attempts}: {encoder_degrees:.3f}\u00b0 encoder, '
                                 f'{pixel_distance:.1f} px response…'))
                final_vector = pixel_points[-1]
                final_magnitude = math.hypot(*final_vector)
                if final_magnitude < 3:
                    raise RuntimeError('No coherent optical movement followed encoder sweep')
                normalized_direction = (
                    direction * final_vector[0] / final_magnitude,
                    direction * final_vector[1] / final_magnitude)
                if accepted_directions:
                    agreement = min(
                        normalized_direction[0] * item[0] +
                        normalized_direction[1] * item[1]
                        for item in accepted_directions)
                    if agreement < .70:
                        raise RuntimeError(
                            f'Feature direction disagreed with earlier trials '
                            f'(cosine {agreement:.2f}); probable lock jump')
                # Reject a noisy/nonlinear response now so the bounded attempt
                # budget can automatically collect a replacement trial.
                single_fit = estimate_backlash_breakpoint(
                    [samples], minimum_trials=1)
                trial_samples.append(samples)
                accepted_directions.append(normalized_direction)
                trial_records.append({
                    'accepted': True,
                    'attempt': attempt_number + 1,
                    'direction': direction,
                    'fit': single_fit['trial_fits'][0],
                    'samples': [{'encoder_degrees': item[0],
                                 'pixel_distance': item[1]} for item in samples]})
            except (ConnectionError, KeyError, RuntimeError, ValueError) as error:
                trial_records.append({'accepted': False,
                    'attempt': attempt_number + 1, 'direction': direction, 'samples': [
                    {'encoder_degrees': item[0], 'pixel_distance': item[1]}
                    for item in samples], 'error': str(error)})
            finally:
                self._return_to_probe_start_rate(
                    baseline, baseline_px, slew_rate)
        if len(trial_samples) < int(trial_count):
            raise RuntimeError(
                f'{motor} learning accepted only {len(trial_samples)} of '
                f'{maximum_attempts} backlash attempts. The feature lock '
                'jumped or was too noisy; reacquire a distinctive feature and retry.')
        estimate = estimate_backlash_breakpoint(trial_samples)
        estimate['trials'] = trial_records
        estimate['axis'] = motor
        return estimate

    def _adaptive_axis_probe(self, axis, baseline, baseline_counts, baseline_px,
                             requested_step, slew_rate, backlash_degrees):
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
                # The repeated reversal routine estimates the encoder-only
                # region. Preload just beyond that breakpoint, settle, then
                # establish the actual pixel/encoder measurement baseline.
                takeup_distance = max(.02, min(.25,
                                               float(backlash_degrees) + .01))
                self.orientation_calibration.update(
                    message=(f'{"Azimuth" if axis == 0 else "Elevation"} '
                             f'backlash take-up {takeup_distance:.3f}\u00b0; '
                             'measured probe follows after settling…'),
                    phase=('azimuth backlash take-up' if axis == 0 else
                           'elevation backlash take-up'))
                self._rate_axis_displacement(
                    axis, direction, takeup_distance, slew_rate,
                    require_feature=True)
                takeup_px = self._stable_feature_sample(
                    time.monotonic(), timeout=2.5)
                takeup_status = self.hil_controller.status()
                if takeup_status.get('connection') != 'live':
                    raise RuntimeError(
                        'Encoder feedback was not live after backlash take-up')
                motor = 'Azimuth' if axis == 0 else 'Elevation'
                takeup_encoder = takeup_status['positions'][motor]['encoder']
                takeup_counts = wrapped_encoder_delta(
                    takeup_encoder, baseline_counts[axis])
                takeup_pixel_motion = math.hypot(
                    takeup_px[0] - baseline_px[0],
                    takeup_px[1] - baseline_px[1])
                status = self._rate_axis_displacement(
                    axis, direction, step, slew_rate, require_feature=True)
                moved_px = self._stable_feature_sample(time.monotonic(), timeout=2.5)
                # Pair the settled image measurement with an encoder reading
                # taken after the same stop/settling interval.  The status
                # returned by _rate_axis_displacement is the threshold-crossing
                # sample from just before its finally block freezes the motors;
                # using it inflated learned px/degree values when the gear
                # train coasted slightly during the bounded settle delay.
                status = self.hil_controller.status()
                if status.get('connection') != 'live':
                    raise RuntimeError('Encoder feedback was not live after the probe settled')
                vector = (moved_px[0] - takeup_px[0],
                          moved_px[1] - takeup_px[1])
                pixel_motion = math.hypot(*vector)
                measurement_error = (takeup_px[0] - center[0],
                                     takeup_px[1] - center[1])
                measurement_distance = math.hypot(*measurement_error)
                moved_error = (moved_px[0] - center[0], moved_px[1] - center[1])
                moved_distance = math.hypot(*moved_error)
                crossed_center = (measurement_error[0] * moved_error[0] < 0 or
                                  measurement_error[1] * moved_error[1] < 0)
                overshot = (crossed_center and moved_distance >
                             max(8, measurement_distance * .35))
                outward = (near_edge and moved_distance >
                           measurement_distance + max(3, pixel_motion * .10))
                attempts.append({'direction': direction, 'step_degrees': step,
                                 'backlash_takeup_degrees': takeup_distance,
                                 'backlash_takeup_encoder_counts': takeup_counts,
                                 'backlash_takeup_pixel_motion': takeup_pixel_motion,
                                 'pixel_motion': pixel_motion,
                                 'distance_before_px': measurement_distance,
                                 'distance_after_px': moved_distance,
                                 'result': ('overshoot' if overshot else
                                            'outward' if outward else 'accepted')})
                if pixel_motion >= 3 and not outward and not overshot:
                    encoder = status['positions'][motor]['encoder']
                    counts = wrapped_encoder_delta(encoder, takeup_encoder)
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
            self.orientation_calibration.update(
                message='Characterizing azimuth backlash from repeated reversals…',
                phase='azimuth backlash characterization')
            az_backlash = self._characterize_axis_backlash(
                0, baseline, baseline_px, slew_rate)
            az_status, az_vector, az_counts, az_step, az_attempts = (
                self._adaptive_axis_probe(0, baseline, baseline_counts,
                                          baseline_px, step, slew_rate,
                                          az_backlash['degrees']))
            self._return_to_probe_start_rate(baseline, baseline_px, slew_rate)

            self.orientation_calibration.update(
                message='Characterizing elevation backlash from repeated reversals…',
                phase='elevation backlash characterization')
            baseline_px = self._stable_feature_sample(time.monotonic())
            el_backlash = self._characterize_axis_backlash(
                1, baseline, baseline_px, slew_rate)
            self.orientation_calibration.update(
                message='Measuring elevation pixel direction…', phase='elevation probe')
            el_status, el_vector, el_counts, el_step, el_attempts = (
                self._adaptive_axis_probe(1, baseline, baseline_counts,
                                          baseline_px, step, slew_rate,
                                          el_backlash['degrees']))
            self._return_to_probe_start_rate(baseline, baseline_px, slew_rate)

            learned = learn_pixel_mount_transform(az_vector, el_vector,
                                                  (az_step, el_step))
            learned.update(encoder_delta_counts=[az_counts, el_counts],
                           adaptive_probe_attempts={
                               'azimuth': az_attempts, 'elevation': el_attempts},
                           backlash_characterization={
                               'azimuth': az_backlash,
                               'elevation': el_backlash},
                           pixels_per_encoder_count=[
                               math.hypot(*az_vector) / abs(az_counts),
                               math.hypot(*el_vector) / abs(el_counts)])
            learned.update(
                status='learned', valid=True,
                message=(('Artificial transfer function learned and saved; '
                          if self.ui_mode == 'transfer' else
                          'Camera orientation learned and saved; ') +
                         'telescope returned to its starting position. '
                         'Press Motor Output ON to track.'))
            with self.lock:
                self.orientation_calibration = learned
                self._save_orientation_profile(learned)
                self.soft_start_ramp_started = None
                self.hil_error = None
                self.hil_enabled = False
                controller.pause()
        except (ConnectionError, KeyError, RuntimeError, ValueError) as error:
            canceled = self.orientation_cancel.is_set()
            if self.hil_controller:
                self.hil_controller.pause()
            with self.lock:
                self.hil_enabled = False
                self.hil_error = str(error)
                if canceled:
                    self._load_orientation_profile()
                    if self.orientation_calibration.get('valid'):
                        self.orientation_calibration.update(
                            status='loaded',
                            message=('Learning aborted; previous saved mapping restored. '
                                     'Bridge link remains available.'))
                    else:
                        self.orientation_calibration = {
                            'status': 'aborted', 'valid': False,
                            'message': 'Learning aborted; no saved mapping is available.'}
                else:
                    self.orientation_calibration = {
                        'status': 'failed', 'valid': False, 'message': str(error)}
        finally:
            # A learning failure must never leave the last rate command active.
            # Freeze before attempting the optional return and freeze again
            # afterward.  This retains the warm Bridge client; it does not tear
            # down or restart the shared telescope connection.
            if self.hil_controller:
                self.hil_controller.freeze()
            if (baseline and self.hil_controller
                    and not self.orientation_calibration.get('valid')
                    and not self.orientation_cancel.is_set()):
                try:
                    self.orientation_cancel.clear()
                    self._return_to_probe_start_rate(
                        baseline, baseline_px if 'baseline_px' in locals() else
                        (self.width / 2, self.height / 2), slew_rate)
                except (ConnectionError, RuntimeError, ValueError):
                    self.hil_controller.pause()
            if self.hil_controller:
                self.hil_controller.freeze()
            self.orientation_cancel.clear()

    def start_orientation_learning(self, step=None, rate=None):
        if not self.hil_controller or self.ui_mode not in ('live', 'transfer'):
            raise ValueError('Learning requires Live or Transfer-Function Test mode and a motor link')
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
        x, y, box_fraction = map(float, (x, y, box_fraction))
        if (not 0 <= x <= 1 or not 0 <= y <= 1
                or not .03 <= box_fraction <= .30):
            raise ValueError('Select a point inside the video; lock box must be 3–30%')
        self.tracking_mode = 'feature-roi'
        self.feature_lock_size = True
        half = box_fraction / 2
        left = max(0, min(1 - box_fraction, x - half))
        top = max(0, min(1 - box_fraction, y - half))
        self.set_feature_roi(left, top, box_fraction, box_fraction)
        self.aim_point = ((.5, .5) if self.aim_mode == 'center' else (x, y))

    def set_manual_lock_hold(self, active, x=None, y=None, box_fraction=.12):
        """Pin the visual lock under the pointer without dropping motor authority."""
        if active:
            x, y, box_fraction = map(float, (x, y, box_fraction))
            if (not 0 <= x <= 1 or not 0 <= y <= 1 or
                    not .03 <= box_fraction <= .30):
                raise ValueError('Manual lock hold must remain inside the video')
            if (not self.manual_lock_hold or
                    not self.manual_lock_hold.get('active')):
                self.manual_lock_hold_generation += 1
            self.manual_lock_hold = {
                'active': True, 'x': x, 'y': y,
                'box_fraction': box_fraction,
                'started_monotonic': (self.manual_lock_hold or {}).get(
                    'started_monotonic', time.monotonic())}
            self.hil_previous_target = None
            if self.hil_controller:
                self.hil_controller.freeze()
            self.hil_error = ('PURPLE MANUAL LOCK HOLD — motor correction '
                              'frozen; green automatic lock remains visible.')
            return
        held = self.manual_lock_hold
        if not held:
            return
        x = held['x'] if x is None else float(x)
        y = held['y'] if y is None else float(y)
        box_fraction = held['box_fraction'] if box_fraction is None else float(box_fraction)
        generation = self.manual_lock_hold_generation
        self.manual_lock_hold = dict(
            held, active=False, x=x, y=y, box_fraction=box_fraction,
            release_pending=True, released_monotonic=time.monotonic(),
            apply_delay_seconds=1.0)
        self.hil_error = ('Purple manual point released; one-second confirmation '
                          'window before it replaces the green automatic lock.')

        def apply_after_confirmation():
            with self.lock:
                held_now = self.manual_lock_hold
                if (generation != self.manual_lock_hold_generation or
                        not held_now or held_now.get('active')):
                    return
                final_x, final_y = held_now['x'], held_now['y']
                final_size = held_now['box_fraction']
                self.manual_lock_hold = None
                self.aim_mode = 'center'
                self.set_feature_point(final_x, final_y, final_size)
                self.hil_error = ('Purple point committed; initializing the '
                                  'replacement green feature lock.')

        timer = threading.Timer(1.0, apply_after_confirmation)
        timer.daemon = True
        timer.start()

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
        self._start_camera_inventory()
        return self.cameras

    def recover_camera_service(self):
        """Request capture reopen without taking the web dashboard offline."""
        if self.camera_recovery_thread and self.camera_recovery_thread.is_alive():
            return
        self.camera_recovery_requested = True
        self.camera_recovery_count += 1
        self.camera_recovery_in_progress = True
        self.camera_recovery_status = 'Request accepted; stopping and reopening camera capture.'
        self.camera_recovery_result = None
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
        self.camera_reopen_requested = True
        starting_frame = self.telemetry.get('frame', -1)

        def worker():
            deadline = time.monotonic() + 12
            last_frame = starting_frame
            advancing_frames = 0
            try:
                while time.monotonic() < deadline:
                    if not (self.thread and self.thread.is_alive()):
                        self.camera_recovery_status = (
                            'Capture worker stopped; starting a fresh capture worker.')
                        self.start()
                    current_frame = self.telemetry.get('frame', -1)
                    if current_frame != last_frame and current_frame >= 0:
                        advancing_frames += 1
                        last_frame = current_frame
                    if advancing_frames >= 3:
                        self.camera_recovery_status = (
                            f'Camera recovered; {advancing_frames} advancing frames '
                            f'confirmed through frame {current_frame}.')
                        self.camera_recovery_result = 'recovered'
                        return
                    self.camera_recovery_status = (
                        f'Confirming camera recovery: {advancing_frames} of 3 '
                        'advancing frames received.')
                    time.sleep(.2)
                self.camera_reopen_requested = True
                self.camera_recovery_status = (
                    'No fresh frame yet. The dashboard remains online and the '
                    'background camera worker will keep retrying with bounded backoff.')
                self.camera_recovery_result = 'background retry active'
            except (OSError, subprocess.SubprocessError, RuntimeError) as error:
                self.camera_recovery_status = str(error)
                self.camera_recovery_result = 'failed'
            finally:
                self.camera_recovery_requested = False
                self.camera_recovery_in_progress = False

        self.camera_recovery_thread = threading.Thread(
            target=worker, name='camera-recovery', daemon=True)
        self.camera_recovery_thread.start()

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

    def set_camera_agc_lock(self, locked):
        """Freeze the current auto-exposure result or return to automatic control."""
        locked = bool(locked)
        if locked:
            self.camera_gain_locked = False
            exposure = int(self.camera_metadata.get(
                'ExposureTime', self.camera_manual_exposure_us))
            gain = float(self.camera_metadata.get(
                'AnalogueGain', self.camera_manual_gain))
            self.set_camera_exposure(
                False, self.camera_exposure_value, exposure, gain)
        else:
            self.camera_gain_locked = False
            self.set_camera_exposure(
                True, self.camera_exposure_value,
                self.camera_manual_exposure_us, self.camera_manual_gain)

    def set_camera_gain_lock(self, locked):
        """Hold analogue gain while software continues adjusting exposure time."""
        locked = bool(locked)
        if locked:
            exposure = int(self.camera_metadata.get(
                'ExposureTime', self.camera_manual_exposure_us))
            gain = float(self.camera_metadata.get(
                'AnalogueGain', self.camera_manual_gain))
            self.camera_gain_locked = True
            self.set_camera_exposure(
                False, self.camera_exposure_value, exposure, gain)
            self.camera_gain_locked = True
            self.camera_software_ae_last = 0.0
        else:
            self.camera_gain_locked = False
            self.set_camera_exposure(
                True, self.camera_exposure_value,
                self.camera_manual_exposure_us, self.camera_manual_gain)

    def _adjust_locked_gain_exposure(self, gray, now=None):
        if not self.camera_gain_locked or not self.capture:
            return
        now = time.monotonic() if now is None else float(now)
        if now - self.camera_software_ae_last < .5:
            return
        self.camera_software_ae_last = now
        percentile = max(1.0, float(np.percentile(gray, 90)))
        target = float(np.clip(128 * (2 ** (self.camera_exposure_value / 2)),
                               24, 230))
        ratio = target / percentile
        if .92 <= ratio <= 1.08:
            return
        current = int(self.camera_metadata.get(
            'ExposureTime', self.camera_manual_exposure_us))
        limits = self.camera_control_ranges.get('ExposureTime', {})
        minimum = int(limits.get('minimum', 1))
        maximum = int(limits.get('maximum', 1000000))
        step = float(np.clip(ratio ** .35, .75, 1.33))
        exposure = int(np.clip(round(current * step), minimum, maximum))
        if exposure == current:
            return
        self.camera_manual_exposure_us = exposure
        self.capture.set_controls({
            'AeEnable': False, 'ExposureTime': exposure,
            'AnalogueGain': self.camera_manual_gain})

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

    def _apply_image_orientation(self, gray):
        """Transform pixels before preview, click selection and tracking."""
        if self.image_horizontal_reversed and self.image_vertical_reversed:
            return cv2.flip(gray, -1)
        if self.image_horizontal_reversed:
            return cv2.flip(gray, 1)
        if self.image_vertical_reversed:
            return cv2.flip(gray, 0)
        return gray

    def set_hil_control_strategy(self, preset, horizon=.05, proportional=.55,
                                 derivative=.10, deadband=.02):
        if self.hil_enabled:
            raise ValueError('Turn motor output OFF before changing control strategy')
        presets = {
            # Off is the legacy full-error controller retained for comparison.
            'off': (0.0, 1.0, 0.0, 0.0),
            'conservative': (0.0, .35, .14, .03),
            'balanced': (.05, .55, .10, .02),
            'aggressive': (.10, .80, .05, .01),
        }
        if preset in presets:
            horizon, proportional, derivative, deadband = presets[preset]
        elif preset == 'auto':
            horizon, proportional, derivative, deadband = (
                self.hil_auto_horizon, .55, .10, .02)
        elif preset != 'custom':
            raise ValueError('Select Off, Conservative, Balanced, Aggressive, Auto, or Custom')
        horizon, proportional, derivative, deadband = map(
            float, (horizon, proportional, derivative, deadband))
        if (not 0 <= horizon <= 1.5 or not 0 <= proportional <= 1
                or not 0 <= derivative <= .5 or not 0 <= deadband <= .5):
            raise ValueError(
                'Prediction horizon 0–1.5s, position gain 0–1, '
                'derivative damping 0–0.5s, deadband 0–0.5°')
        self.hil_control_preset = preset
        self.hil_prediction_horizon = horizon
        self.hil_proportional_lead = proportional
        self.hil_derivative_damping = derivative
        self.hil_correction_deadband = deadband
        self.hil_control_error_state = None
        self.hil_filtered_error_rate = (0.0, 0.0)

    def _damped_mount_correction(self, azimuth_error, elevation_error, now):
        """Apply proportional approach, derivative braking and center deadband."""
        azimuth_error, elevation_error = map(
            float, (azimuth_error, elevation_error))
        derivative_azimuth = derivative_elevation = 0.0
        if self.hil_control_error_state:
            previous_time, previous_azimuth, previous_elevation = (
                self.hil_control_error_state)
            delta_time = max(.02, min(1.0, float(now) - previous_time))
            raw_azimuth = (azimuth_error - previous_azimuth) / delta_time
            raw_elevation = (elevation_error - previous_elevation) / delta_time
            derivative_azimuth = (.70 * self.hil_filtered_error_rate[0] +
                                  .30 * raw_azimuth)
            derivative_elevation = (.70 * self.hil_filtered_error_rate[1] +
                                    .30 * raw_elevation)
        self.hil_filtered_error_rate = (derivative_azimuth,
                                        derivative_elevation)
        self.hil_control_error_state = (
            float(now), azimuth_error, elevation_error)

        def axis(error, rate):
            if abs(error) <= self.hil_correction_deadband:
                return 0.0
            correction = (self.hil_proportional_lead * error +
                          self.hil_derivative_damping * rate)
            # Derivative noise must not reverse an otherwise valid correction;
            # braking may reduce it to zero, never drive away from the target.
            if correction * error < 0:
                correction = 0.0
            return correction

        correction = (axis(azimuth_error, derivative_azimuth),
                      axis(elevation_error, derivative_elevation))
        self.hil_control_terms = {
            'raw_error_degrees': [azimuth_error, elevation_error],
            'error_rate_degrees_per_second': [derivative_azimuth,
                                              derivative_elevation],
            'correction_degrees': list(correction),
            'position_gain': self.hil_proportional_lead,
            'derivative_damping_seconds': self.hil_derivative_damping,
            'deadband_degrees': self.hil_correction_deadband,
        }
        return correction

    def set_hil_max_tracking_slew(self, degrees_per_second):
        degrees_per_second = float(degrees_per_second)
        if (not math.isfinite(degrees_per_second)
                or not .001 <= degrees_per_second <= 5):
            raise ValueError('Maximum tracking correction rate must be 0.001–5 degrees per second')
        self.hil_max_tracking_slew_deg_s = degrees_per_second

    def set_initial_trajectory_learning(self, enabled, window_seconds=2.0):
        if self.hil_enabled:
            raise ValueError('Turn motor output OFF before changing initial trajectory learning')
        window_seconds = float(window_seconds)
        if window_seconds not in (1.0, 2.0, 3.0, 5.0, 10.0):
            raise ValueError('Initial trajectory window must be 1, 2, 3, 5, or 10 seconds')
        self.hil_initial_trajectory_enabled = bool(enabled)
        self.hil_initial_trajectory_window = window_seconds
        self.hil_initial_trajectory = {
            'valid': False,
            'message': ('Waiting for a stable locked feature while motor output is OFF.'
                        if enabled else 'Initial trajectory learning is OFF.')}

    def _update_initial_trajectory(self, selected, now):
        if (not self.hil_initial_trajectory_enabled or self.hil_enabled or
                not selected or self.ui_mode != 'live' or
                not self.orientation_calibration.get('valid')):
            return
        window = self.hil_initial_trajectory_window
        samples = [item for item in self.feature_samples
                   if now - item[0] <= window]
        if len(samples) < 8 or samples[-1][0] - samples[0][0] < window * .75:
            self.hil_initial_trajectory = {
                'valid': False, 'sample_count': len(samples),
                'window_seconds': window,
                'message': 'Collecting stable pre-motion feature trajectory…'}
            return
        times = np.asarray([item[0] - samples[0][0] for item in samples])
        xs = np.asarray([item[1] for item in samples])
        ys = np.asarray([item[2] for item in samples])
        vx, x0 = np.polyfit(times, xs, 1)
        vy, y0 = np.polyfit(times, ys, 1)
        residual = np.hypot(xs - (vx * times + x0),
                            ys - (vy * times + y0))
        rms = float(np.sqrt(np.mean(residual ** 2)))
        span = float(times[-1] - times[0])
        path = float(math.hypot(vx, vy) * span)
        confidence = max(0.0, min(1.0, 1.0 - rms / max(2.0, path)))
        azimuth_rate, elevation_rate = learned_mount_delta(
            float(vx), float(vy), self.orientation_calibration)
        valid = confidence >= .60 and rms <= 5.0
        self.hil_initial_trajectory = {
            'valid': valid, 'sample_count': len(samples),
            'window_seconds': window, 'observed_seconds': span,
            'pixel_rate': {'x_px_s': float(vx), 'y_px_s': float(vy)},
            'mount_rate': {'azimuth_deg_s': float(azimuth_rate),
                           'elevation_deg_s': float(elevation_rate)},
            'residual_rms_px': rms, 'confidence': confidence,
            'message': (('Initial slow-motion trajectory ready.' if valid else
                         'Initial trajectory was noisy; keep the lock still '
                         'or reacquire and collect another window.'))}

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
        commanded_azimuth = azimuth + rate['azimuth_deg_s'] * horizon
        commanded_elevation = elevation + rate['elevation_deg_s'] * horizon
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
        # Clicking a new live feature is the recovery path after the previous
        # feature leaves the frame. Keep the Bridge/motor link armed; the next
        # frame initializes this replacement tracker while the control loop has
        # no selected target and therefore cannot use the discarded feature.
        self.hil_previous_target = None
        self.hil_control_error_state = None
        self.hil_filtered_error_rate = (0.0, 0.0)
        self.hil_error = 'New optical target selected; initializing feature lock.'

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
        # OpenCV's Python bindings require native integer coordinates here.
        # Template matching returns floats (and can return NumPy scalar types),
        # which KCF rejects as an unparseable boundingBox.
        bbox = tuple(int(round(float(value))) for value in bbox)
        try:
            initialized = tracker.init(feature_frame, bbox)
        except (cv2.error, TypeError, ValueError) as error:
            self.feature_error = f'Feature tracker could not accept reacquisition box: {error}'
            self.feature_reacquire_confirmations = 0
            return None
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
        self.feature_error = None
        self.feature_last_confirmed = (center[0], center[1], time.monotonic())
        return self._feature_detection(bbox, score)

    def _hil_update(self, selected, width, height):
        if not self.hil_controller:
            if not (self.hil_host or self.hil_route_target):
                return {'available': False, 'enabled': False,
                        'description': 'Display-only mode; no motor endpoint was supplied.'}
            return {
                'available': True, 'enabled': False,
                'host': self.hil_host, 'port': self.hil_port,
                'route_target': self.hil_route_target,
                'connection': ('recovering' if self.bridge_service_restart_in_progress
                               else 'disconnected'),
                'motion': 'idle', 'armed': False, 'marker': None,
                'encoder_positions': {}, 'link_action_pending': self.hil_link_action,
                'bridge': dict(
                    self.bridge_status,
                    dashboard_connection=('recovering' if self.bridge_service_restart_in_progress
                                          else 'disconnected'),
                    dashboard_reconnections=0,
                    dashboard_manual_reconnects=self.bridge_manual_reconnects,
                    service_restart_in_progress=self.bridge_service_restart_in_progress,
                    service_restart_count=self.bridge_service_restart_count,
                    service_restart_result=self.bridge_service_restart_result,
                    dashboard_last_error=self.hil_error),
                'error': self.hil_error,
                'description': ('Motor endpoint is configured but its client is stopped. '
                                'Use Recover Bridge and Motor Link.')}
        status = self.hil_controller.status()
        if (status.get('connection') == 'live'
                and self.hil_error == 'Dashboard bridge connection restarted; waiting for both fresh encoders.'):
            self.hil_error = None
        marker = hil_marker_from_status(status, width, height,
                                        self.hil_display_span_degrees / 2)
        feedback_live = status.get('feedback_live')
        if feedback_live is None:
            # Compatibility with older controller/status providers that did
            # not expose encoder freshness independently of socket state.
            feedback_live = status.get('connection') == 'live'
        self._update_initial_trajectory(selected, time.monotonic())
        if self.hil_enabled:
            if self.hil_reference_rebasing:
                self.hil_controller.freeze()
                self.hil_error = ('Rebasing the relative encoder marker at the center cross; '
                                  'the motor link remains armed.')
            elif self.manual_lock_hold:
                self.hil_controller.freeze()
                self.hil_error = ('PURPLE MANUAL LOCK HOLD — motor correction '
                                  'frozen while the green automatic lock remains visible.')
            elif status.get('connection') != 'live':
                # A transient dashboard-to-Bridge interruption must not clear
                # the operator's ON request.  The controller reconnects in its
                # own worker; freeze target traffic while it does so, retain
                # the armed UI state, and resume from fresh encoder feedback.
                self.hil_controller.freeze()
                self.hil_error = ('Motor link interrupted; corrections are frozen while the '
                                  'Bridge connection recovers. Tracking remains ON.')
            else:
                try:
                    if not feedback_live:
                        self.hil_error = (
                            'Encoder watchdog: one or both replies are stale. The socket is '
                            'still live, scheduled polling continues, and encoder confirmation '
                            'does not gate target dispatch.')
                    if not status.get('local_reference_set'):
                        # PositionController intentionally clears local offsets
                        # after reconnect because it cannot know whether the
                        # telescope power-cycled.  In this optical session the
                        # dashboard has a stronger continuity signal: the last
                        # confirmed encoder-referenced optical lock. Restore
                        # that local frame instead of disabling output.
                        reference = self.hil_last_locked_marker
                        if not reference and self.hil_current_target:
                            reference = self.hil_current_target
                        if reference:
                            self.hil_controller.reference(
                                reference['azimuth_degrees'],
                                reference['elevation_degrees'])
                            status = self.hil_controller.status()
                            marker = hil_marker_from_status(
                                status, width, height,
                                self.hil_display_span_degrees / 2)
                            self.hil_error = ('Motor feedback resumed; the last optical-session '
                                              'reference was restored automatically.')
                    target_now = time.monotonic()
                    target = None
                    if selected:
                        if marker:
                            self.hil_last_locked_marker = {
                                'azimuth_degrees': marker['azimuth_degrees'],
                                'elevation_degrees': marker['elevation_degrees'],
                                'utc_epoch_s': time.time()}
                        self.hil_return_started = None
                        if self.ui_mode in ('live', 'transfer'):
                            if not marker:
                                raise RuntimeError('Fresh encoder feedback is required for optical corrections')
                            if not self.orientation_calibration.get('valid'):
                                raise RuntimeError('Camera-to-mount mapping is unavailable')
                            anchor_x = self.aim_point[0] * width
                            anchor_y = self.aim_point[1] * height
                            error_x = selected['x'] - anchor_x
                            error_y = selected['y'] - anchor_y
                            if self.ui_mode == 'transfer':
                                delta_azimuth, delta_elevation = transfer_mount_delta(
                                    error_x, error_y, self.transfer_function)
                            else:
                                delta_azimuth, delta_elevation = learned_mount_delta(
                                    error_x, error_y, self.orientation_calibration)
                            if self.ui_mode == 'live':
                                delta_azimuth, delta_elevation = (
                                    self._damped_mount_correction(
                                        delta_azimuth, delta_elevation,
                                        target_now))
                                initial = self.hil_initial_trajectory
                                if (self.hil_initial_trajectory_enabled and
                                        initial.get('valid')):
                                    lead_time = max(
                                        1 / max(1.0, self.hil_dispatch_rate_hz),
                                        self.hil_prediction_horizon)
                                    rate = initial['mount_rate']
                                    delta_azimuth += rate['azimuth_deg_s'] * lead_time
                                    delta_elevation += rate['elevation_deg_s'] * lead_time
                            # Convert the current pixel error with the measured
                            # mapping, then use the damped live correction. The
                            # step cap prevents calibration noise from turning
                            # one frame into a multi-degree jump.
                            # Limit the finite correction target sent on each
                            # dispatch. This is software tracking authority in
                            # degrees/second; it does not rewrite the mount's
                            # native manual slew-rate table.
                            correction_limit = (self.hil_max_tracking_slew_deg_s /
                                                max(1.0, self.hil_dispatch_rate_hz))
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
                        if feedback_live:
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
                        commanded_target = self._apply_outer_loop(target, marker)
                        self.hil_controller.submit(
                            commanded_target[0], commanded_target[1],
                            self.hil_sequence, time.time(), ttl=.75)
                        self.hil_current_target = {'azimuth_degrees': target[0],
                                                   'elevation_degrees': target[1]}
                        self.hil_last_submit = target_now
                except (ConnectionError, RuntimeError, ValueError) as error:
                    # Keep the requested tracking state through recoverable
                    # feedback/mapping interruptions. Freeze motion until the
                    # next fresh cycle rather than forcing another user click.
                    self.hil_controller.freeze()
                    self.hil_error = (f'{error}; corrections are frozen and will retry while '
                                      'tracking remains ON.')
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
                'route_target': self.hil_route_target,
                'envelope_degrees': self.hil_envelope_degrees,
                'travel_envelope_enabled': self.hil_envelope_degrees > 0,
                'display_span_degrees': self.hil_display_span_degrees,
                'dispatch_rate_hz': self.hil_dispatch_rate_hz,
                'max_tracking_slew_deg_s': self.hil_max_tracking_slew_deg_s,
                'feedback_poll_rate_hz': status.get(
                    'configured_feedback_poll_rate_hz', 4),
                'feedback_timeout_seconds': status.get(
                    'feedback_timeout_seconds', self.hil_feedback_timeout_seconds),
                'connection': status.get('connection'), 'motion': status.get('motion'),
                'feedback_live': feedback_live,
                'armed': status.get('armed'), 'marker': marker,
                'encoder_positions': status.get('positions', {}),
                'cycle_count': status.get('cycle_count'),
                'cycle_overruns': status.get('cycle_overruns'),
                'encoder_poll_count': status.get('encoder_poll_count'),
                'runtime_rx_count': status.get('runtime_rx_count'),
                'target_confirmation': status.get('target_confirmation'),
                'last_achieved_cycle_rate_hz': status.get('last_achieved_cycle_rate_hz'),
                'target_rate': self.hil_target_rate,
                'current_target': self.hil_current_target,
                'commanded_target': self.hil_commanded_target,
                'control_preset': self.hil_control_preset,
                'prediction_horizon_seconds': (self.hil_auto_horizon
                                                if self.hil_control_preset == 'auto'
                                                else self.hil_prediction_horizon),
                'proportional_lead': self.hil_proportional_lead,
                'derivative_damping_seconds': self.hil_derivative_damping,
                'correction_deadband_degrees': self.hil_correction_deadband,
                'control_terms': self.hil_control_terms,
                'initial_trajectory': dict(self.hil_initial_trajectory,
                    enabled=self.hil_initial_trajectory_enabled,
                    configured_window_seconds=self.hil_initial_trajectory_window),
                'manual_lock_hold': (dict(self.manual_lock_hold)
                                     if self.manual_lock_hold else None),
                'reference_rebasing': self.hil_reference_rebasing,
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
            if not self.orientation_auto_reversed:
                inverse = self.orientation_calibration.get('inverse_degrees_per_pixel')
                if inverse:
                    self.orientation_calibration['inverse_degrees_per_pixel'] = [
                        [-float(value) for value in row] for row in inverse]
                    self.orientation_auto_reversed = True
                    message = ('Error increased after mount motion; correction direction was '
                               'reversed automatically and tracking remains ON.')
                    self.orientation_calibration.update(
                        status='auto-corrected', valid=True, message=message)
                    self.hil_error = message
                    self.orientation_sanity = {'status': 'auto-corrected',
                                               'message': message}
            else:
                message = ('Error is still increasing after automatic direction reversal; '
                           'tracking remains ON. Use OFF if the feature leaves the frame.')
                self.hil_error = message
                self.orientation_sanity = {'status': 'warning', 'message': message}
            self.orientation_history.clear()

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
        """Supervise capture without coupling camera health to HTTP availability."""
        retry_delay = 1.0
        while not self.stop_event.is_set():
            starting_frame = int(self.telemetry.get('frame', -1) or -1)
            self._capture_once()
            if self.stop_event.is_set():
                break
            ending_frame = int(self.telemetry.get('frame', -1) or -1)
            if ending_frame != starting_frame and ending_frame >= 0:
                retry_delay = 1.0
            else:
                retry_delay = min(15.0, retry_delay * 1.7)
            self.camera_recovery_status = (
                f'Camera offline; background retry in {retry_delay:.1f} seconds. '
                'The web dashboard and motor-link controls remain available.')
            self.camera_recovery_result = 'background retry active'
            if self.stop_event.wait(retry_delay):
                break

    def _capture_once(self):
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
                self._adjust_locked_gain_exposure(gray, cycle)
                gray = self._apply_image_curve(gray)
                gray = self._apply_image_orientation(gray)
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
                            roi = tuple(int(round(float(value)))
                                        for value in self.feature_roi_pending)
                            self.feature_roi_pending = None
                            try:
                                initialized = self.feature_tracker.init(feature_frame, roi)
                            except (cv2.error, TypeError, ValueError) as error:
                                initialized = False
                                self.feature_error = (
                                    f'Feature tracker rejected the selected box: {error}')
                            if initialized is False:
                                self.feature_tracker = None
                                self.feature_reacquiring = True
                                objects = []
                            else:
                                self.feature_error = None
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
                                self.feature_error = None
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
                    if self.ui_mode == 'hybrid':
                        hybrid = self._hybrid_target(gray, frame_index)
                        objects = [hybrid]
                        selected_id = 1
                    elif self.ui_mode == 'transfer':
                        transfer = self._transfer_target(gray)
                        objects = [transfer]
                        selected_id = 1
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
                    'tracking_error': self.feature_error,
                    'ui_mode': self.ui_mode,
                    'aim_mode': self.aim_mode,
                    'aim_point': {'x': self.aim_point[0], 'y': self.aim_point[1]},
                    'image_orientation': {
                        'horizontal': ('reversed' if self.image_horizontal_reversed else 'normal'),
                        'vertical': ('reversed' if self.image_vertical_reversed else 'normal')},
                    'orientation_sanity': dict(self.orientation_sanity),
                    'orientation_calibration': dict(self.orientation_calibration),
                    'transfer_function_test': (dict(self.transfer_state)
                                               if self.ui_mode == 'transfer' else None),
                    'camera_device': self.device,
                    'camera_settings': {'width': self.width, 'height': self.height,
                                        'fps': self.requested_fps},
                    'camera_auto': {
                        'available': self.source == 'picamera2',
                        'enabled': self.camera_auto_exposure,
                        'gain_locked': self.camera_gain_locked,
                        'agc_locked': (not self.camera_auto_exposure and
                                       not self.camera_gain_locked),
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
            with self.frame_ready:
                self.telemetry = dict(
                    self.telemetry, state='CAMERA OFFLINE',
                    objects=[], error_x_px=None, error_y_px=None,
                    error=(f'Camera unavailable: {error}. Motor link remains armed; '
                           'use Recover Camera to retry capture.'))
                self.frame_ready.notify_all()
        finally:
            self._close_capture()
            self._close_recording()

    def snapshot(self):
        with self.lock:
            snapshot = dict(self.telemetry)
            # These are operator-controlled state, not frame-derived values.
            # Keep them current even when the last camera frame is stale.
            snapshot.update(
                ui_mode=self.ui_mode,
                tracking_mode=self.tracking_mode,
                camera_device=self.device,
                cameras=self.cameras,
                camera_inventory_status=self.camera_inventory_status,
                image_curve={'black': self.curve_black,
                             'white': self.curve_white,
                             'gamma': self.curve_gamma},
                image_orientation={
                    'horizontal': ('reversed' if self.image_horizontal_reversed
                                   else 'normal'),
                    'vertical': ('reversed' if self.image_vertical_reversed
                                 else 'normal')},
                camera_auto={
                    'available': self.source == 'picamera2' or
                                 str(self.device).startswith('picamera2:'),
                    'enabled': self.camera_auto_exposure,
                    'gain_locked': self.camera_gain_locked,
                    'agc_locked': (not self.camera_auto_exposure and
                                   not self.camera_gain_locked),
                    'exposure_value': self.camera_exposure_value,
                    'manual_exposure_time_us': self.camera_manual_exposure_us,
                    'manual_analogue_gain': self.camera_manual_gain,
                    'exposure_time_us': self.camera_metadata.get('ExposureTime'),
                    'analogue_gain': self.camera_metadata.get('AnalogueGain'),
                    'digital_gain': self.camera_metadata.get('DigitalGain'),
                    'limits': {
                        name: dict(self.camera_control_ranges[name])
                        for name in ('ExposureValue', 'ExposureTime', 'AnalogueGain')
                        if name in self.camera_control_ranges},
                })
            snapshot['camera_recovery'] = {
                'in_progress': self.camera_recovery_in_progress,
                'attempts': self.camera_recovery_count,
                'status': self.camera_recovery_status,
                'result': self.camera_recovery_result,
            }
            # A camera exception can occur before the first frame.  Build the
            # non-camera status dynamically so Live Mode, bridge diagnostics,
            # motor feedback and recovery controls remain usable.
            if (self.hil_controller or self.hil_host or self.hil_route_target) and 'hardware_loop' not in snapshot:
                snapshot.update(
                    width=self.width, height=self.height, ui_mode=self.ui_mode,
                    tracking_mode=self.tracking_mode, selected_track_id=None,
                    camera_device=self.device, cameras=self.cameras,
                    hardware_loop=self._hil_update(None, self.width, self.height))
            # Camera telemetry is frame-based and can legitimately stop
            # advancing during CSI recovery. Never let that frozen frame make
            # the live motor controller appear OFF, connected, or fresh when
            # its current state says otherwise.
            if self.hil_controller:
                live = self.hil_controller.status()
                hardware = dict(snapshot.get('hardware_loop') or {})
                hardware.update(
                    enabled=self.hil_enabled,
                    connection=live.get('connection'),
                    feedback_live=live.get('feedback_live'),
                    motion=live.get('motion'),
                    armed=live.get('armed'),
                    encoder_positions=live.get('positions', {}),
                    cycle_count=live.get('cycle_count'),
                    cycle_overruns=live.get('cycle_overruns'),
                    feedback_poll_rate_hz=live.get(
                        'configured_feedback_poll_rate_hz'),
                    encoder_poll_count=live.get('encoder_poll_count'),
                    runtime_rx_count=live.get('runtime_rx_count'),
                    target_confirmation=live.get('target_confirmation'),
                    last_achieved_cycle_rate_hz=live.get(
                        'last_achieved_cycle_rate_hz'),
                    error=self.hil_error or live.get('last_error'))
                snapshot['hardware_loop'] = hardware
                snapshot['motor_output'] = ('ENABLED' if self.hil_enabled
                                            else 'DISABLED')
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
            elif action == 'manual_jog':
                self.manual_jog(message['axis'], message['direction'], message['rate'])
            elif action == 'manual_stop':
                self.manual_stop()
            elif action == 'set_hil_dispatch_rate':
                self.set_hil_dispatch_rate(message['value'])
            elif action == 'set_hil_feedback_timeout':
                self.set_hil_feedback_timeout(message['value'])
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
            elif action == 'set_transfer_function':
                self.set_transfer_function(
                    message['azimuth_x'], message['azimuth_y'],
                    message['elevation_x'], message['elevation_y'])
            elif action == 'set_aim_mode':
                self.set_aim_mode(message['mode'])
            elif action == 'center_selected':
                if not self.last_selected:
                    raise ValueError('The selected feature is not locked yet; retry after the green box appears')
                self.set_aim_mode('center')
                self.request_hil_enable()
            elif action == 'manual_lock_hold':
                self.set_manual_lock_hold(
                    message.get('active', False), message.get('x'),
                    message.get('y'), message.get('box_fraction', .12))
            elif action == 'zero_encoder_marker':
                self.zero_encoder_marker()
            elif action == 'set_image_orientation':
                self.set_image_orientation(message['horizontal'], message['vertical'])
            elif action == 'start_orientation_learning':
                self.request_orientation_learning(message.get('step'), message.get('rate'))
            elif action == 'abort_orientation_learning':
                self.abort_orientation_learning()
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
            elif action == 'set_camera_agc_lock':
                self.set_camera_agc_lock(message.get('locked', True))
            elif action == 'set_camera_gain_lock':
                self.set_camera_gain_lock(message.get('locked', True))
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
                                              message.get('horizon', .05),
                                              message.get('proportional', .55),
                                              message.get('derivative', .10),
                                              message.get('deadband', .02))
            elif action == 'set_hil_max_tracking_slew':
                self.set_hil_max_tracking_slew(message['value'])
            elif action == 'set_initial_trajectory_learning':
                self.set_initial_trajectory_learning(
                    message.get('enabled', False),
                    message.get('window_seconds', 2.0))
            else:
                raise ValueError('Unknown dashboard action')
        return self.snapshot()


def create_app(engine, static_directory=None):
    try:
        from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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

    @app.post('/api/command')
    async def command_api(message: dict):
        """Reliable request/response command path for local automation.

        The browser keeps using the bidirectional WebSocket. Test runners and
        companion processes use this endpoint so a command cannot be lost when
        a short-lived WebSocket closes before its receive task consumes JSON.
        """
        try:
            engine.command(message)
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {'accepted_action': message.get('action'),
                'status': engine.snapshot()}

    @app.websocket('/ws')
    async def websocket(websocket: WebSocket):
        await websocket.accept()
        last_frame = -2
        last_send = 0.0
        try:
            while True:
                snapshot = engine.snapshot()
                now = time.monotonic()
                if snapshot.get('frame') != last_frame or now - last_send >= .5:
                    await websocket.send_json(snapshot)
                    last_frame = snapshot.get('frame')
                    last_send = now
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
    parser.add_argument('--hil-route-target',
                        help=('Telescope address used to re-resolve this computer\'s current '
                              'Bridge address after DHCP/network changes'))
    parser.add_argument('--hil-port', type=int, default=2000)
    parser.add_argument('--hil-envelope-degrees', type=float, default=0)
    parser.add_argument('--hil-dispatch-rate-hz', type=float, default=4)
    parser.add_argument('--hil-feedback-timeout-seconds', type=float, default=10)
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
    if not 3 <= args.hil_feedback_timeout_seconds <= 60:
        raise SystemExit('Hardware-loop feedback timeout must be 3–60 seconds')
    engine = VisionDashboardEngine(
        source=args.source, device=args.device, width=args.width, height=args.height,
        fps=args.fps, threshold=args.threshold, jpeg_quality=args.jpeg_quality,
        log_directory=args.log_directory, polarity=args.polarity,
        hil_host=args.hil_host, hil_port=args.hil_port,
        hil_route_target=args.hil_route_target,
        hil_envelope_degrees=args.hil_envelope_degrees,
        hil_dispatch_rate_hz=args.hil_dispatch_rate_hz,
        hil_feedback_timeout_seconds=args.hil_feedback_timeout_seconds,
        hil_display_span_degrees=args.hil_display_span_degrees,
        ui_mode=('live' if args.mode == 'live-run'
                 or args.source in ('v4l2', 'picamera2', 'file') else 'demo'))
    app = create_app(engine)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, access_log=False,
                timeout_graceful_shutdown=3)


if __name__ == '__main__':
    main()
