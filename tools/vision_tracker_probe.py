#!/usr/bin/env python3
"""Low-latency vision-tracker probe for Raspberry Pi development.

This deliberately does not command telescope motors.  It validates a camera or
recorded-video source, detects the brightest compact target, timestamps each
frame, and writes processing latency and centroid measurements as CSV.

Examples:
  python3 tools/vision_tracker_probe.py --source synthetic --seconds 10
  python3 tools/vision_tracker_probe.py --source v4l2 --device /dev/video0
  python3 tools/vision_tracker_probe.py --source file --device sample.mp4

The future ASI120 adapter should produce the same grayscale NumPy frame plus a
monotonic acquisition timestamp.  Tracking and mount-control code must not
depend on Pi-camera, V4L2, INDI, or ZWO-specific APIs.
"""

import argparse
import csv
import math
import time
from pathlib import Path

import cv2
import numpy as np


def synthetic_frame(index, width, height, fps):
    """Return a repeatable moving bright target with mild sensor-like noise."""
    t = index / fps
    x = int(width * (0.5 + 0.35 * math.sin(t * 1.7)))
    y = int(height * (0.5 + 0.30 * math.sin(t * 1.1 + 0.7)))
    rng = np.random.RandomState(index & 0xFFFF)
    frame = rng.normal(8, 2, (height, width)).clip(0, 255).astype(np.uint8)
    cv2.circle(frame, (x, y), 4, 245, -1)
    return frame, x, y


def brightest_centroid(gray, threshold):
    """Find the largest bright compact region and return centroid/confidence."""
    _level, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours_result = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_result[-2]
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if moments['m00'] <= 0:
        return None
    x = moments['m10'] / moments['m00']
    y = moments['m01'] / moments['m00']
    area = cv2.contourArea(contour)
    confidence = min(1.0, area / 40.0)
    return x, y, area, confidence


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('synthetic', 'v4l2', 'file'),
                        default='synthetic')
    parser.add_argument('--device', default='/dev/video0')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--seconds', type=float, default=10.0)
    parser.add_argument('--threshold', type=int, default=160)
    parser.add_argument('--csv', default='vision-tracker-probe.csv')
    return parser.parse_args()


def main():
    args = parse_args()
    capture = None
    if args.source != 'synthetic':
        device = int(args.device) if args.device.isdigit() else args.device
        capture = cv2.VideoCapture(device)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_FPS, args.fps)
        if not capture.isOpened():
            raise SystemExit('Camera/video source could not be opened: ' + args.device)

    csv_path = Path(args.csv).expanduser()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    index = 0
    detected = 0
    latency_total_ms = 0.0
    max_error_px = 0.0

    with csv_path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('utc_epoch_s', 'monotonic_ns', 'frame', 'source',
                         'x_px', 'y_px', 'error_x_px', 'error_y_px', 'area_px',
                         'confidence', 'processing_ms', 'synthetic_true_x_px',
                         'synthetic_true_y_px'))
        while time.monotonic() - started < args.seconds:
            acquired_ns = time.monotonic_ns()
            true_x = true_y = ''
            if args.source == 'synthetic':
                gray, true_x, true_y = synthetic_frame(
                    index, args.width, args.height, args.fps)
            else:
                ok, frame = capture.read()
                if not ok:
                    break
                gray = (frame if len(frame.shape) == 2 else
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

            result = brightest_centroid(gray, args.threshold)
            processing_ms = (time.monotonic_ns() - acquired_ns) / 1e6
            latency_total_ms += processing_ms
            if result:
                x, y, area, confidence = result
                error_x = x - gray.shape[1] / 2.0
                error_y = y - gray.shape[0] / 2.0
                detected += 1
                if args.source == 'synthetic':
                    max_error_px = max(max_error_px,
                                       math.hypot(x - true_x, y - true_y))
                row = (time.time(), acquired_ns, index, args.source, x, y,
                       error_x, error_y, area, confidence, processing_ms,
                       true_x, true_y)
            else:
                row = (time.time(), acquired_ns, index, args.source, '', '',
                       '', '', '', 0.0, processing_ms, true_x, true_y)
            writer.writerow(row)
            index += 1

            if args.source == 'synthetic':
                deadline = started + index / args.fps
                delay = deadline - time.monotonic()
                if delay > 0:
                    time.sleep(delay)

    if capture:
        capture.release()
    elapsed = time.monotonic() - started
    effective_fps = index / elapsed if elapsed else 0.0
    mean_processing_ms = latency_total_ms / index if index else 0.0
    print('frames={0} detections={1} elapsed_s={2:.3f} fps={3:.2f} '
          'mean_processing_ms={4:.3f} max_centroid_error_px={5:.3f} csv={6}'.format(
              index, detected, elapsed, effective_fps, mean_processing_ms,
              max_error_px, csv_path))
    if not index or not detected:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
