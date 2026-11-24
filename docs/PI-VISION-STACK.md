# Raspberry Pi vision-tracking evaluation stack

This is an evaluation environment, not a validated autonomous telescope tracker.
It can start with the user service manager when the supplied service is installed.
Display-only mode never owns the telescope endpoint.
Live Mode and the explicitly selected synthetic Demo Mode can own one bridge
client connection and command bounded physical movement only after output is armed.

## Test host and installed base

Validated on 2026-11-22 on a Raspberry Pi 3B+, Raspberry Pi OS Trixie 32-bit
(`armv7l`), Python 3.13.5. The isolated environment is:

```text
/home/mklopfer/telescope-optical-tracking-control/.venv
```

It uses the operating system's OpenCV 4.10.0, NumPy 2.2.4, SciPy 1.15.3 and
PySerial 3.5, plus isolated installs of:

- Skyfield 1.55, SGP4 2.27 and jplephem 2.24;
- FilterPy 2.0.0 and simple-pid 2.0.1;
- imutils 0.5.4, tifffile 2026.11.16, PyYAML and Requests;
- geocoder 1.38.1 for evaluating legacy candidate code only.

OpenCV exposes CSRT, KCF, legacy MOSSE and DNN support. Hamlib `rotctld` 4.6.2
and Gpredict are installed separately. The system reported NTP synchronized with
the clock in UTC internally and the display timezone set to America/Los_Angeles.
No GPS/PPS receiver is installed or verified.

The driver-advertised OV5647 camera controls captured from
`Picamera2(0).camera_controls` are preserved in
[`hardware/camera-profiles/ov5647-libcamera-controls.json`](../hardware/camera-profiles/ov5647-libcamera-controls.json).
That manifest records all returned ranges and defaults, the capture environment,
an observed automatic-exposure state and the generic dashboard limits that did
not match the hardware. Enumerate controls again for a different sensor, driver,
camera mode or operating-system image; do not assume this profile is universal.
The dashboard now enumerates the selected Picamera2 device when it opens, applies
the detected exposure-compensation, exposure-time and analogue-gain limits to the
controls and server-side validation, and restores the selected manual values after
a camera reopen. The complete detected manifest is available read-only at
`/api/camera-controls`. USB/V4L2 devices without Picamera2 control enumeration use
the documented conservative fallback limits.

The camera-only probe is at `tools/vision_tracker_probe.py` in this repository and
was copied to `/home/mklopfer/telescope-optical-tracking-control/tools`. It never commands motors.
A 640x480 USB-camera bright-centroid run processed 96 frames in 5.012 seconds
(19.15 FPS, mean processing time 52.0 ms). This does not include a detector,
registration, Kalman/PID, web display or motor I/O.

## Candidate source inventory

Candidate repositories are shallow source checkouts below
`/home/mklopfer/telescope-optical-tracking-control/candidates`. They were not installed into the
system Python and have no telescope authority.

| Candidate and exact checkout | Useful material | Current result / boundary |
| --- | --- | --- |
| AstronomyLiveYt/SatTraker `0d97f492313e` (2023-10-05) | TLE and video-guidance application structure | Source parses on Python 3.13, but directly imports Windows `win32com`, pins Python-era NumPy/OpenCV packages and needs a new backend. Not runnable on this Pi as checked out. GPL-3.0. |
| AstronomyLiveYt/RocketTraker `0b751a4ab41c` (2026-02-08) | FlightClub ingestion, trajectory and optical-guidance application structure | Source parses, but directly imports `win32com` and pins Windows/old scientific packages. The reusable pieces need extraction and tests. Apache-2.0. |
| esa/pypogs `a0340bf0480b` (2022-03-11) | Closed-loop system, alignment and spot-tracker design | Source parses. Its Celestron backend opens a 9600-baud NexStar hand-controller serial port; no INDI backend was found. Requirements pin old Astropy/NumPy/SciPy versions incompatible with the current base. The checkout does not include an obvious root license file; review reuse terms before copying code. |
| EPFLRocketTeam/MantisTracker `187b4b9e712d` (2021-01-21) | MOSSE/KCF/SiamRPN/YOLO comparison assets | Large 2.1 GB source/data checkout. Base source parses, but startup selected an incompatible bundled `libdarknet_MAC_CPU.so` and failed with an invalid ELF header on armv7. Full old ML stack is not installed. GPL-2.0. |
| UCI-Rocket-Project/rocket-tracker `26863aad5a7e` (2025-10-15) | Kalman/PID/simulation/ZWO design reference | Source parses, but its own README says current dependency versions no longer run; requirements include DeepSparse/JAX/TensorFlow/ONNX stacks unsuitable for this Pi. Submodules were not initialized and no root license file was present. Reference only until selectively ported. |

`compileall` succeeding means Python can parse the files. It is not an import,
runtime, camera, trajectory, motor or safety test.

## Recommended evaluation order

1. Keep all output disconnected from motors. Benchmark V4L2, CSI and future
   ASI120 capture through one grayscale-frame plus monotonic-timestamp adapter.
2. Compare threshold/centroid, MOSSE, KCF and CSRT on recorded clips. Record loss,
   false reacquisition, latency and CPU/thermal throttling.
3. Export a compact custom detector off-device to an armv7-compatible format and
   benchmark OpenCV DNN. Do not install an unsupported PyTorch/Ultralytics stack
   on the Pi merely to satisfy a package name.
4. Add registration/background subtraction, then bounded Kalman prediction and
   PID in simulation. Treat target age/uncertainty as a Stop/reacquire condition.
5. Adapt selected trajectory/vision components to this project's authenticated
   latest-target contract. Do not let candidate applications open the telescope.
6. Characterize discrete AUX `0x24`/`0x25` direction/speed behavior separately.
   Implement a rate backend only after measured degrees/second, update latency,
   reversal/backlash, braking and watchdog tests.
7. Perform supervised movement only after clearance, explicit authorization and
   a bounded simulation/replay test. Preserve source commit, model, video, camera
   settings, clock state, rates and logs for every result.

## Transport choices

The existing MCP target worker can use SkyQLink or another compatible TCP endpoint,
but its native target path remains experimental. A future ESP32 AUX bridge or a
USB-to-AUX adapter can reduce transport jitter. Neither changes the need for a
compliant AUX electrical interface, single-owner arbitration, one serialized
request/reply at a time and Stop priority. A generic USB-UART cable is not a safe
AUX adapter.

## Browser dashboard

`tools/vision_dashboard.py` and `tools/vision_dashboard_static/index.html` provide
an evaluation dashboard. Install the optional web dependencies in an
environment that already has OpenCV:

```console
python -m pip install -e ".[vision-dashboard]"
python tools/vision_dashboard.py --mode display-only --source synthetic
```

For the currently identified USB camera on the test Pi:

```console
python tools/vision_dashboard.py --mode display-only --source v4l2 --device /dev/video1
```

From another computer, create an SSH tunnel and open the local URL:

```console
ssh -L 8000:127.0.0.1:8000 mklopfer@raspberrypi.local
# Open http://127.0.0.1:8000
```

The page shows client-rendered bounding boxes, bright/dark object polarity,
object IDs, selected-object error vector, capture/processing status, a selectable
10/30/60-second X/Y centroid drift chart and a 100-sample bullseye. Select an
object by clicking its box or using the visible-object list; either method can
change the active object. **Thresholded objects** is the Demo Mode default.

**Feature ROI** is an explicit alternative for a crater or other textured feature.
Drag a rectangle around it and select CSRT, KCF or MOSSE. CSRT is the robust default
but is substantially slower on the Pi. **Lock Feature ROI size** defaults ON to
prevent scale drift and growing boxes; disable it only when real target scale is
expected to change. CSRT tolerates moderate scale/appearance/field rotation, not
arbitrary rotation. KCF and MOSSE trade robustness for speed. CSV recording is
explicitly enabled from the page. Display-only mode does not load telescope
control and labels motor output unavailable.

## Live optical run mode

Live Mode is the normal startup page. It enumerates Linux V4L2 capture cameras,
lists the resolution/frame-rate combinations advertised by each camera, and can
refresh that inventory after a camera is added. Apply camera changes only while
tracking output is OFF. On the validated Pi, `auto` selects the USB UVC camera in
preference to a generic video node.

Start Control Bridge first, with no other data client, then run:

```console
PYTHONPATH=/home/mklopfer/telescope-optical-tracking-control \
  /home/mklopfer/telescope-optical-tracking-control/.venv/bin/python \
  /home/mklopfer/telescope-optical-tracking-control/tools/vision_dashboard.py \
  --mode live-run --device auto --hil-host 192.168.1.149 --hil-port 2000 \
  --hil-envelope-degrees 5 --hil-display-span-degrees 5 \
  --hil-dispatch-rate-hz 4 --host 0.0.0.0 --allow-lan --port 8000 \
  --log-directory /home/mklopfer/telescope-optical-tracking-control/logs/live-optical
```

### Automatic Pi startup

The standalone repository installs the dashboard as a user-level systemd service.
The legacy Celestron Control Bridge remains a separate companion service from
`celestron-telescope-network-control`; start it first when the selected telescope
requires that proxy. Neither project requires a root-owned service file:

```console
cd /home/mklopfer/telescope-optical-tracking-control
./tools/install_pi_dashboard_service.sh
```

This enables and starts `telescope-vision-dashboard.service`, which uses
`Restart=always`. When installed separately, the companion Bridge keeps retrying
when the telescope Wi-Fi adapter is temporarily unavailable. On the validated Raspberry Pi OS
desktop, graphical automatic login starts the user service manager after boot.
For a headless Pi with no user login, an administrator must additionally enable
lingering once with `sudo loginctl enable-linger "$USER"`.

Useful checks are:

```console
systemctl --user status celestron-bridge.service
systemctl --user status telescope-vision-dashboard.service
systemctl --user restart telescope-vision-dashboard.service
```

The launcher discovers the Pi's routed IPv4 address and adds this standalone
checkout to `PYTHONPATH`; no address or import-path edit is required after a
normal DHCP address change.

If the page loads but remains at frame `-1` / **STARTING**, restart only
`telescope-vision-dashboard.service`. A libcamera `Camera frontend has timed out`
message means the CSI sensor was detected but delivered no image data; if one
service restart does not recover it, power down and reseat or replace the ribbon
cable/sensor. If the Bridge service shows repeated restarts while the dashboard
remains available, inspect its status: the normal cause is the telescope adapter
refusing or resetting the upstream connection. The service keeps retrying, but it
cannot repair a lost SSID configuration or power-cycle the physical adapter.

Click a distinct high-contrast point while output is OFF. The page creates a
fixed-size CSRT region around the click. Choose **Bring selected feature to frame
center** or **Track at its present field position** before selecting the point.
The second choice records the clicked location as the desired image location.
Camera rotation, mirroring and axis direction are measured by the bounded
**Learn / validate motor mapping** routine rather than guessed with manual
Normal/Reversed selectors.

The mapping movement selector is a **maximum**, not the first movement. Each
axis now starts at no more than 0.05 degrees and increases only when the measured
feature displacement is under 3 pixels, never beyond the selected maximum. An
edge feature is probed toward frame center. A detected overshoot returns to the
starting encoder position and retries with a smaller movement. Normal optical
corrections also ramp from roughly 0.024 degrees per command during initial
acquisition and remain capped at 0.15 degrees per command. These software limits
address the large initial jump seen in early tests, but physical repeatability
still requires another supervised telescope session.

Feature ROI tracking retains the original clicked image patch. Three consecutive
non-credible CSRT results release the drifting tracker and change the state to
**REACQUIRING**. Motor corrections freeze; the program does not scan the mount or
promote arbitrary new points. It searches the camera frame for a unique match to
the original patch and recreates CSRT only after two spatially consistent frames.
This preserves a sensitive normal lock while preventing aimless hunting after the
object leaves the field. A returned original feature can therefore reestablish
tracking automatically. Similar-looking repeated texture can remain ambiguous,
so use a distinctive ROI and watch the reported state.

Live Mode draws a compact fixed-size green lock box but deliberately omits the
long green target-to-aim error vector used by Demo Mode; that vector became an
oversized stripe on a responsive browser display. **Reset live view** requires
output OFF, clears the target and plotted history, restores center aim, and
reopens the selected camera without changing
its chosen device or advertised resolution/frame-rate setting.

Live corrections are incremental: each correction uses the latest encoder
position plus the calibrated pixel error, rather than treating image coordinates
as absolute celestial coordinates. The display-span value is the assumed camera
field width used to convert pixels to degrees; an incorrect value produces an
incorrect correction magnitude. Tracking output defaults OFF. A direction sanity
monitor compares sustained pixel error with actual encoder motion. If an axis
repeatedly moves opposite the required correction for at least about 1.2 seconds,
with meaningful encoder displacement and no improvement, output is frozen and
the page reports that the measured mapping must be relearned. It never silently
guesses or reverses an axis.

The common **Control Bridge and motor link** panel reads the bridge's dedicated read-only
management endpoint on port 3000 in a background thread, outside the video and
motor loop. It reports verified proxy identity, bridge readiness, dashboard data
connection, manual/automatic reconnect counts, cached motor-feedback age and the
current connection error. **Reconnect Motor Link** pauses optical output if needed,
closes and recreates this dashboard's port-2000 client, then requires both encoders
to become fresh again. **Restart Bridge & Reconnect Motors** restarts the Pi user
service, waits for its verified management endpoint, recreates the client and
reports whether fresh replies returned from both axes. Neither action restarts the
telescope Wi-Fi adapter or another control application. If the adapter itself is
offline or refuses connections, the service continues retrying and the page states
that telescope-side condition explicitly.

Verified on 2026-11-22 on the Pi: the UVC camera produced live 640x480 frames,
the bridge management identity reported Ready, cached replies for both axes were
fresh, and a supervised dashboard-client reconnect returned to a live connection
with zero motor output. This does not yet verify physical optical closed-loop
tracking; click-lock, lost-feature latching, two-frame reacquisition and direction
safety are software/OpenCV tested only.

The bright-contour IDs are temporary nearest-neighbor associations, not semantic
identities. Crossing objects, occlusion or large inter-frame movement can exchange
or replace IDs. This first dashboard is for measuring and selecting candidate
tracks; it is not evidence of safe closed-loop telescope control.

## Saved synthetic hardware-loop mode

This supervised mode maps a selected moving black object on a white synthetic
frame into a local encoder target. The synthetic target is not optical feedback.
The cyan circle labeled **Encoder Reported Location** comes only from live mount
encoder replies, so its lag, stepping and error remain visible independently of
the requested target.

Start the Control Bridge first, with no other client connected. Then run:

```console
PYTHONPATH=/home/mklopfer/telescope-optical-tracking-control \
  /home/mklopfer/telescope-optical-tracking-control/.venv/bin/python \
  /home/mklopfer/telescope-optical-tracking-control/tools/vision_dashboard.py \
  --mode synthetic-hardware-loop --hil-host 192.168.1.149 \
  --hil-port 2000 --hil-envelope-degrees 5 --hil-display-span-degrees 5 \
  --hil-dispatch-rate-hz 4 \
  --host 0.0.0.0 --allow-lan --port 8000 \
  --log-directory /home/mklopfer/telescope-optical-tracking-control/logs/synthetic-hardware-loop
```

Replace the bridge address with the Pi or computer actually running the bridge.
Open `http://<pi-address>:8000`, select an object, and deliberately enable
**Hardware-in-the-loop motor output**. OFF sends Stop and leaves encoder polling
live. Link loss, stale feedback, stale targets, travel-envelope violation or
measured divergence disarms the controller and attempts Stop. The local reference
maps the initial encoders to screen center; it does not write telescope alignment.

The launch above uses the hard software maximum of 5 degrees from the initial
position on each axis. The synthetic display span can be changed while motor
output is OFF among 0.1, 0.5, 1, 5 and 10 degrees edge-to-edge per axis. A 10-degree
span maps to approximately ±4.5 degrees because a 10% margin is retained. The
page also reports the selected target's current azimuth/elevation demand in
degrees/second. The dispatch default is 4 Hz.
The page permits 1, 2, 4, 6, 8, 10, 15 and 20 Hz while output is OFF. Values above
4 Hz are unverified; 15–20 Hz are stress-test requests, not throughput promises.
The page reports full start-to-start achieved cycle rate and serialized-cycle
overruns. The controller still performs two encoder reads and up to two target
request/reply exchanges per cycle.

Verified on 2026-11-22 through the Pi bridge and physical COSMOS 90GT: both axes
responded, live encoder feedback drove the cyan marker, a ±4-degree test envelope
was enforced, OFF disarmed and paused the worker, and stationary encoder counts
were unchanged across a post-Stop check. This verifies the synthetic command and
encoder-reporting path; it does not validate optical acquisition, real-camera
closed-loop tracking, sustained 15–20 Hz operation or unattended use.

The motor control is an explicit OFF/ON rocker with a pilot light. It defaults
OFF after every launch and ON is unavailable until the bridge and fresh encoders
are confirmed. Faults and controller disarm return it to OFF.

Target loss defaults to **Stop and reacquire**. Optional predictive coast advances
the last measured velocity for 0.25, 0.5, 1, 2, 3 or 5 seconds, then Stops; five
seconds is aggressive. Selected threshold tracks use predicted-position gating
through short dark intervals and do not deliberately substitute another flash.
Crossing targets inside one gate can still be ambiguous. The synthetic blink test
provides adjustable fade-out, hidden, fade-in and repeat timing for testing this.

Outer-loop prediction is Off by default. Conservative, Balanced, Aggressive,
Auto-estimate and Custom modes add bounded velocity feed-forward and proportional
lead. There is no external integral term: the mount already has an internal
position servo, and integral windup through backlash could fight it. All corrected
targets are clamped inside 95% of the hard envelope.

The outer loop also provides a **correction deadband** from 0 to 0.5 degrees.
When an axis error is inside that threshold, the dashboard holds that axis at its
reported encoder position instead of issuing another tiny reversal. This is a
software chatter threshold only: it does not measure, write or replace the
telescope controller's stored backlash setting. Start with the preset value and
increase it only enough to stop visible reversal chatter, because an excessive
deadband intentionally permits more centering error.

Two encoder panels appear only with a requested target and live replies. **Encoder
following error** plots requested target minus Encoder Reported Location. A second
panel plots signed minimum/maximum AZ/EL error in five-second windows to expose lag
and overshoot. The cyan Encoder Reported Location circle always comes from replies.

Automatic backlash writing is not implemented. Future estimation should use
multiple controlled low-speed reversals, separate transport/image latency from
mechanical deadband, report confidence, retain prior values, and require explicit
apply, readback, retest and rollback approval.
