# Telescope Optical Tracking Control

Raspberry Pi visual tracking, camera control, target reacquisition and bounded
Celestron AUX motor output for telescope-mounted cameras.

This repository is the new development home for the **Telescope Visual Tracking
Dashboard**. It contains the live-camera and synthetic-test modes extracted from
[`mjklopfer/celestron-telescope-network-control`](https://github.com/mjklopfer/celestron-telescope-network-control)
at commit `54fbc7fb08b6b494a84f21ef3a0cf963797ed8c3`.

The original repository remains the home of the Celestron Wi-Fi discovery,
SkyQLink/WiFly provisioning and Control Bridge desktop application. Nothing was
removed from it during this extraction.

## Current capabilities

- Browser-based live video with click-to-lock feature tracking.
- Threshold-object Demo Mode and synthetic hardware-loop testing.
- CSRT feature tracking with correlation and jump credibility checks.
- Lost-target latch that freezes correction rather than hunting blindly.
- Automatic reacquisition only after two consistent, unique matches to the
  original selected image patch.
- Adaptive camera-to-mount mapping that begins at no more than `0.05°` and treats
  the selected movement as a maximum.
- Encoder feedback, tracking-error graphs and adjustable 1–20 Hz command loop;
  4 Hz remains the conservative default.
- Runtime Picamera2 hardware-control enumeration and camera-specific exposure,
  exposure-compensation and analogue-gain limits.
- Camera-recovery and motor-link/bridge-recovery controls.

Physical optical closed-loop performance is still under supervised development.
Passing software tests does not establish unattended tracking safety, mechanical
clearance or reliable high-rate AUX throughput.

## Relationship to the Celestron bridge

The dashboard can connect directly to a compatible Celestron/AUX TCP endpoint.
For the legacy COSMOS 90GT SkyQLink/WiFly adapter, run the companion Control
Bridge from `celestron-telescope-network-control` and point this dashboard at the
bridge computer on port `2000`. The read-only bridge management endpoint on port
`3000` enables status and restart reporting.

Only one process or control application can own the telescope endpoint at a time.

## Raspberry Pi environment

The current tested host is a Raspberry Pi 3B+ running Raspberry Pi OS Trixie
32-bit (`armv7l`) and Python 3.13.5. The working stack uses the operating system's
OpenCV/NumPy packages plus a virtual environment created with system site packages:

```console
sudo apt update
sudo apt install python3-venv python3-opencv python3-numpy python3-picamera2
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e .
```

Start a live dashboard manually:

```console
PYTHONPATH=. .venv/bin/python tools/vision_dashboard.py \
  --mode live-run --source picamera2 --device picamera2:0 \
  --width 640 --height 480 --fps 20 \
  --hil-host 127.0.0.1 --hil-port 2000 \
  --hil-envelope-degrees 0 --hil-display-span-degrees 5 \
  --hil-dispatch-rate-hz 4 --host 0.0.0.0 --allow-lan --port 8000
```

Open `http://<pi-address>:8000/`. Motor output starts OFF.

For the managed user service, inspect and run:

```console
tools/install_pi_dashboard_service.sh
```

The service expects this checkout at `~/telescope-optical-tracking-control`. Override paths and
addresses with the environment variables documented in
`tools/run_pi_vision_dashboard.sh`.

## Testing

The included tests use simulated adapters and do not move physical hardware:

```console
python -m unittest -v test_motor_control.py test_position_control.py test_vision_dashboard.py
```

See [the complete Pi vision-stack engineering notes](docs/PI-VISION-STACK.md) and
[the captured OV5647 control manifest](hardware/camera-profiles/ov5647-libcamera-controls.json).

## Repository boundary

This initial extraction intentionally carries the tested AUX framing, endpoint
locking and WiFly route-validation modules used by the position controller. A
later cleanup can package those internals under a dedicated namespace after the
live interfaces stabilize; behavioral changes should not be mixed into the
repository split.

