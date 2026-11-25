# Development resume checkpoint — 2026-11-25

## 2026-11-25 closeout update

- Used the companion application's existing discovery implementation, not a new
  scanner. It found the physical SkyQLink/WiFly RN-171 at `192.168.1.231:2000`,
  WiFly `2.40-CEL` (03-05-2012), MAC `00:06:66:54:a2:bb`, and motor firmware
  `5.24` on both axes. Discovery made no settings or motion changes.
- After Bridge restart, its management endpoint was verified Ready; both encoder
  streams were fresh and polling measured about 3.998 Hz with zero observed
  overruns.
- Fixed a dashboard startup crash caused by startup telemetry legitimately
  omitting `objects`. Backend defaults now publish the complete schema and the
  browser treats a missing/non-array object list as empty.
- Deferred opening the endless MJPEG request until after the document load event.
  The dashboard and Bridge recovery controls now load when a camera provides no
  bytes instead of leaving the browser document perpetually pending.
- RESUME / ON now reports why it is unavailable when there is no advancing
  camera frame or selected live target instead of appearing inert.
- The OV5647 initially blocked testing: both the dashboard and an independent
  `rpicam-still` probe repeatedly reported `Camera frontend has timed out`. After
  the final dashboard service restart it recovered and advanced at about 20.6
  FPS. The intermittent CSI fault remains unexplained; do not call it fixed.
- The final Bridge management report was Ready and verified, but the dashboard's
  motor client remained `connecting` without cached encoder positions. Motion
  readiness was not reestablished or tested.
- Pre-login auto-start remains incomplete because user lingering is still off.
  Normal graphical-login startup is the verified path.
- No real-camera closed-loop tracking or fresh physical movement was performed
  during this closeout.

## Frozen state

Development was deliberately paused after a live lunar test with the Moon very
near zenith. Do not use that geometry to judge normal azimuth tracking. The
alt-az mount was near its zenith keyhole, where azimuth motion has little
translational authority and can appear primarily as field rotation.

The short-lived change that made adaptive feed-forward the default was undone.
The default outer-loop preset is again **Off — latest position only**. Feed-forward
presets remain available for later controlled evaluation.

## Live system addresses and services

- Raspberry Pi: `192.168.1.149`
- Telescope Wi-Fi interface: `192.168.1.231:2000`
- Tracking dashboard: `http://192.168.1.149:8000/`
- Pi user services: `celestron-bridge.service` and
  `telescope-vision-dashboard.service`
- Pi deployment directories: `/home/mklopfer/vision-tracking` and
  `/home/mklopfer/celestron-telescope-network-control`

## Verified during this session

- Added **Hybrid Camera Test** mode. It keeps the real CSI camera, real Bridge,
  real encoder polling and real telescope active, overlays a deterministic
  synthetic dot on the camera image, and supplies that dot's known coordinates
  directly to motion control. Real scene features cannot steal the synthetic
  target. The Hybrid button presents the camera controls, not the demo-only
  controls.
- A stationary two-minute baseline passed with the same Bridge PID, zero
  dashboard reconnects, fresh encoder ages and approximately 4.00 Hz polling.
- Four bounded manual jogs at slew 5 (azimuth positive/negative and elevation
  positive/negative, 0.75 seconds each) all moved the expected encoder and
  stopped in paused state with zero reconnects or Bridge errors.
- A 60-second moving Hybrid Camera Test passed: 60/60 samples stayed ON, zero
  dashboard reconnects, zero non-live intervals, zero reported errors, both
  axes followed the generated path, and achieved command rate remained about
  3.98–4.00 Hz. Stop/OFF was sent at completion.
- These results isolate the observed dropout to the real optical
  target/mapping/tracker path or its interaction with that control path; basic
  Bridge polling, manual motion and repeatable synthetic target motion were
  stable in the tested intervals.
- CSI camera capture and `/video_feed` were live. The backend continuously
  advanced frames; a browser MJPEG reconnection handler was added because the
  displayed stream could remain blank after a stream/service interruption.
- Bridge management reported ready and fresh azimuth/elevation encoder replies.
- Default mapping probe is **0.05 degrees** and default manual/calibration slew
  level is **5**. Both remain adjustable.
- The running motor target dispatch was corrected from an accidental 1 Hz to
  the intended conservative default of **4 Hz**.
- A mapping pass completed and returned to its starting encoder position.
- Measured mapping during the near-zenith lunar test:
  - Azimuth: 4.138 pixels from 0.056198 degrees / 2619 encoder counts.
  - Elevation: 32.141 pixels from 0.061176 degrees / 2851 encoder counts.
  - Azimuth optical response was barely above the acceptance threshold and is
    not a reliable normal-sky calibration result.
- A KCF reacquisition bug that passed float/NumPy bounding boxes into OpenCV was
  corrected. True loss now clears the stale optical cross instead of leaving a
  frozen cross while the encoder marker continues moving.
- Learning has an Abort action. A failed/aborted learner sends Stop and retains
  the Bridge rather than intentionally tearing it down.
- Basic tracking no longer requires a completed Learn mapping; a provisional
  mapping can start motion. This remains experimental and needs a non-zenith
  validation.
- Focused position-control and reconnect tests passed locally.

## Important unresolved behavior

- **Required architecture correction:** The Control Bridge must be the single
  persistent owner and authority for the telescope TCP/Wi-Fi session. The vision
  dashboard must not duplicate telescope connection ownership, reconnect logic,
  encoder freshness policy, keep-alive behavior or local-reference lifecycle in
  its own `PositionController` client. That duplication caused the repeated
  design/debug loop observed in this session.
- Add a local Bridge control interface (loopback HTTP/WebSocket or Unix socket)
  for `set axis rate/target`, `Stop`, encoder/status subscription, connection
  health, sequence acknowledgement and explicit session/reference events. The
  dashboard sends desired motion to that interface and consumes authoritative
  Bridge telemetry. It does not connect to a Celestron-compatible data port as
  if it were another external astronomy application.
- Encoder polling, telescope keep-alive, TCP reconnect, retry timing, command
  sequencing and bounded encoder-loss coasting belong inside the Bridge. A
  dashboard/browser disconnect must not disturb the Bridge-to-telescope link.
  When the telescope link itself reconnects, the Bridge reports whether the
  prior reference is continuous or invalidated; clients do not independently
  guess and rebuild it.
- **Temporary development continuity mode:** For motion/feed-forward validation,
  encoder feedback is telemetry, not motion authority. Once the operator turns
  tracking ON, a missing or late encoder sample changes the reported state to
  `FEEDBACK_INTERRUPTED` but must not disconnect the Bridge client, clear ON,
  disarm, send Stop, or discard the last motion request. The Bridge continues
  the last commanded axis rate/trajectory while the camera and telemetry remain
  visible so the operator/AI can judge whether optical tracking is working.
- When encoders resume in development continuity mode, report the time without
  feedback, the predicted versus reported position delta and the reconciliation
  action. Do not hide the experiment by knocking out motion control. Explicit
  operator Stop/OFF remains immediate. A separately configured terminal-failure
  policy may end motion, but an ordinary encoder-reporting blip may not.
- Development continuity mode is specifically for validating motion and
  encoder-loss feed-forward. It is not the intended permanent production
  default; production policy will be chosen only after measured testing.
- Before more optical tuning, build an automated Bridge soak test: sustained
  encoder polling plus representative rate/Stop commands for at least one hour,
  with zero unintended client release, measured latency/packet loss, reconnect
  reason codes, and deterministic recovery after an intentionally broken
  telescope socket.
- **Feed-forward clarification:** In this project, the requested feed-forward
  behavior primarily means surviving a temporary **encoder-feedback loss**. Do
  not clear tracking ON or drop the Bridge client. Continue the last credible
  axis rate/trajectory open-loop for a bounded, configurable coast interval;
  visibly report that feedback is unavailable; then reconcile smoothly to new
  encoder readings when they resume. This is distinct from the existing
  outer-loop target-position prediction/lead presets, which must remain optional.
- Implement encoder-loss coasting with explicit states such as
  `TRACKING_FEEDBACK`, `COASTING_NO_ENCODER`, `RECONCILING`, and
  `FEEDBACK_TIMEOUT`. The Bridge remains the connection authority. A transient
  missing encoder reply must not by itself flip the user's motor-output request
  to OFF. Define a configurable coast timeout and a separate terminal-failure
  policy before resuming live tests.
- During live tracking the dashboard motor client reconnected twice. A 4 Hz
  requested loop achieved about 2.07 Hz with 198 overruns in the observed run.
- The old behavior that cleared motor output when feedback stopped being live
  was changed locally: transient interruption now freezes corrections, retains
  the ON request, and restores the last optical-session local reference after
  fresh encoder replies return. This passed focused tests but still needs a
  deliberate live reconnect test away from zenith.
- Near zenith, the learned camera-to-mount matrix is poorly conditioned for a
  single tracked feature. Future work should estimate a dynamic 2x2 Jacobian,
  use multiple lunar features to measure field rotation, and explicitly report
  zenith-keyhole geometry.
- The apparent azimuth failure may also have coincided with the selected lunar
  feature reaching the camera-frame boundary. Treat this as an unverified
  hypothesis. The next test must log the feature bounding box, frame-edge state,
  azimuth command, encoder response and optical displacement together so a
  frame-exit loss is not mistaken for a failed azimuth motor command.
- Automatic direction reversal changed the saved mapping after error increased.
  Do not assume that near-zenith auto-corrected mapping is valid at lower
  elevation.
- Actual frame processing with KCF was roughly 135 ms per frame (about 7 FPS in
  the heavier live state), despite the camera itself being capable of a higher
  capture rate. Decouple capture/display from tracking computation before
  expecting consistently smooth video.
- Minimum effective slew-rate and backlash characterization were discussed but
  are not implemented. They should recommend measured values before any
  telescope backlash write.
- The complete suites must pass in the closeout commit; any failing test is a
  release blocker rather than an accepted known failure.

## First test next session

1. Refactor and soak-test the Bridge as the only telescope-session owner before
   resuming optical tracking tests.
2. Use a terrestrial feature or the Moon below roughly 70 degrees elevation,
   with ample mechanical clearance.
3. Confirm Bridge ready, fresh encoders, and live camera frames.
4. Lock one distinct feature and run the 0.05 degree / slew-5 mapping once.
5. Verify both axis vectors are comfortably above the pixel-motion threshold
   and not nearly parallel.
6. Turn tracking ON and record horizontal/vertical pixel error, encoder motion,
   reconnect count, achieved dispatch rate and overruns for at least 60 seconds.
7. Intentionally suppress or delay encoder replies in a non-moving/synthetic
   test and verify the new encoder-loss feed-forward state retains ON, coasts
   for the configured interval, and reconciles without a command jump.
8. Only after latest-position control converges should Conservative/Auto
   feed-forward be evaluated as an optional improvement.

## Repository state

This checkpoint is included with the 2026-11-25 closeout changes. Generated
captures and runtime logs remain local and are intentionally excluded from Git.

