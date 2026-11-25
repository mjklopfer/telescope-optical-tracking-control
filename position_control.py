"""Experimental latest-target AUX controller for supervised small tracking tests.

MC_GOTO_FAST (0x02) takes a 24-bit absolute encoder target. Unlike an unbounded
rate command, loss of the connection leaves only the last finite target in the
mount. This does NOT prove physical clearance or guarantee delivery of Stop.
The operator configures a symmetric encoder travel envelope of at most 5 degrees
from the initial verified position. Do not use this experimental backend for an
unattended launch/ISS pass. The native target command uses the mount's own speed;
the calibrated timed-pulse rates are not a configurable GOTO velocity setting.

Network I/O runs in one worker. MCP/stream calls only replace the latest target;
there is no stale-command queue. Both axis targets are transmitted consecutively,
not simultaneously on the serial wire. The conservative default dispatch and
feedback-poll rates are 4 Hz. Runtime encoder polls are scheduled writes, not
synchronous request/reply exchanges: incoming AUX frames are parsed whenever they
arrive and never gate the next motor command. Rates above 4 Hz are experimental.
Position targets are transmitted to both axes without waiting for their optional
acknowledgements; encoder age and observed motion provide asynchronous health data.
Missing feedback freezes new motion while the warm socket continues polling. A
real socket failure triggers Stop attempts, TCP close and bounded-rate reconnection.
Reconnection requires both fresh encoders and a NEW target received after the link
is live. Last known position remains labeled with age; it is never treated as live
feedback. Encoder references are local transforms, NOT a shared
celestial model. A power cycle can invalidate them: do not assume reconnect proves
the encoder origin survived. Reset/re-align after any suspected power cycle.
"""
from collections import deque
import math
import socket
import threading
import time
from celestron_probe import FrameParser, Recorder
from endpoint_lock import EndpointLease
from motor_control import AXES, angle, relative_angle, motor_packet
from setup_backend import check_direct_route, validate_endpoint


def target_packet(axis, encoder):
    if axis not in AXES or type(encoder) is not int or not 0 <= encoder < 1 << 24:
        raise ValueError('Invalid motor target')
    body = bytes((6, 0x20, axis, 0x02)) + encoder.to_bytes(3, 'big')
    return b'\x3b' + body + bytes((-sum(body) & 255,))


class PositionController:
    def __init__(self, host, port=2000, envelope_degrees=5, feedback_timeout=10.0,
                 dispatch_rate_hz=4, target_change_threshold_degrees=.001,
                 feedback_poll_rate_hz=4, motion_observation_timeout=3.0,
                 target_confirmation_tolerance_degrees=.02,
                 stop_on_divergence=True):
        self.host, self.port = validate_endpoint(host, port)
        if (not 0 <= envelope_degrees <= 90 or not 3 <= feedback_timeout <= 60
                or not math.isfinite(dispatch_rate_hz) or not 1 <= dispatch_rate_hz <= 20
                or not math.isfinite(target_change_threshold_degrees)
                or not math.isfinite(feedback_poll_rate_hz)
                or not 1 <= feedback_poll_rate_hz <= 20
                or not math.isfinite(motion_observation_timeout)
                or not .5 <= motion_observation_timeout <= 30
                or not math.isfinite(target_confirmation_tolerance_degrees)
                or not .001 <= target_confirmation_tolerance_degrees <= 1
                or not 0 <= target_change_threshold_degrees <= .1):
            raise ValueError('Optional envelope 0 (disabled)–90 degrees; feedback timeout 3–60 seconds; dispatch rate 1–20 Hz; target threshold 0–0.1 degrees')
        self.envelope = envelope_degrees
        self.feedback_timeout = feedback_timeout
        self.dispatch_rate_hz = float(dispatch_rate_hz)
        self.feedback_poll_rate_hz = float(feedback_poll_rate_hz)
        self.target_change_threshold_degrees = float(target_change_threshold_degrees)
        self.cycle_interval = 1 / self.dispatch_rate_hz
        self.feedback_poll_interval = 1 / self.feedback_poll_rate_hz
        self.motion_observation_timeout = float(motion_observation_timeout)
        self.target_confirmation_tolerance_degrees = float(
            target_confirmation_tolerance_degrees)
        self.stop_on_divergence = bool(stop_on_divergence)
        self.rate_warning = (None if self.dispatch_rate_hz <= 4 else
                             ('STRESS TEST: requested rates above 10 Hz are unverified on the mount and Bridge; '
                              'watch achieved dispatch timing, encoder age and overruns.' if self.dispatch_rate_hz > 10 else
                              'EXPERIMENTAL: rates above the conservative 4 Hz baseline '
                              '(intervals below 250 ms) are unverified and may cause delayed or missing AUX replies.'))
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.thread = None
        self.sock = None
        self.recorder = None
        self.lease = EndpointLease(self.host, self.port)
        self.positions, self.position_times, self.origin = {}, {}, {}
        self.offsets = None
        self.armed = False
        self.target = None
        self.rate_request = None
        self.rate_sequence = 0
        self.manual_request = None
        self.manual_sequence = 0
        self.sequence = -1
        self.live_since = 0
        self.connection = 'disconnected'
        self.motion = 'idle'
        self.last_error = None
        self.reconnections = 0
        self.cycle_count = 0
        self.cycle_overruns = 0
        self.last_cycle_duration_ms = None
        self.last_cycle_period_ms = None
        self.max_cycle_duration_ms = 0.0
        self.encoder_poll_count = 0
        self.runtime_rx_count = 0
        self.target_confirmation = {
            'state': 'idle',
            'message': 'No finite position target has been sent.'}
        self.history = deque(maxlen=80)

    def set_feedback_timeout(self, seconds):
        """Set when encoder feedback is reported stale without closing the socket.

        Missing encoder replies do not prove that the TCP link is dead.  The
        controller keeps the socket warm and continues polling; only an actual
        socket close or network error starts reconnection.
        """
        seconds = float(seconds)
        if not math.isfinite(seconds) or not 3 <= seconds <= 60:
            raise ValueError('Motor feedback stale timeout must be 3–60 seconds')
        with self.lock:
            self.feedback_timeout = seconds
        self.event('feedback_timeout_changed', seconds=seconds)

    def set_dispatch_rate(self, dispatch_rate_hz):
        """Change requested cycle rate while disarmed; 4 Hz remains the default."""
        dispatch_rate_hz = float(dispatch_rate_hz)
        if not math.isfinite(dispatch_rate_hz) or not 1 <= dispatch_rate_hz <= 20:
            raise ValueError('Dispatch rate must be 1–20 Hz')
        with self.lock:
            if self.armed:
                raise RuntimeError('Turn motor output OFF before changing dispatch rate')
            self.dispatch_rate_hz = dispatch_rate_hz
            self.cycle_interval = 1 / dispatch_rate_hz
            self.rate_warning = (None if dispatch_rate_hz <= 4 else
                                 ('STRESS TEST: requested rates above 10 Hz are unverified on the mount and Bridge; '
                                  'watch achieved dispatch timing, encoder age and overruns.' if dispatch_rate_hz > 10 else
                                  'EXPERIMENTAL: rates above the conservative 4 Hz baseline '
                                  '(intervals below 250 ms) are unverified and may cause delayed or missing AUX replies.'))
        self.event('dispatch_rate_changed', dispatch_rate_hz=dispatch_rate_hz,
                   warning=self.rate_warning)

    def set_feedback_poll_rate(self, feedback_poll_rate_hz):
        """Change independent encoder-poll cadence without gating motor writes."""
        feedback_poll_rate_hz = float(feedback_poll_rate_hz)
        if not math.isfinite(feedback_poll_rate_hz) or not 1 <= feedback_poll_rate_hz <= 20:
            raise ValueError('Feedback poll rate must be 1–20 Hz')
        with self.lock:
            self.feedback_poll_rate_hz = feedback_poll_rate_hz
            self.feedback_poll_interval = 1 / feedback_poll_rate_hz
        self.event('feedback_poll_rate_changed',
                   feedback_poll_rate_hz=feedback_poll_rate_hz)

    def set_travel_envelope(self, enabled, degrees=5):
        degrees = float(degrees)
        if not 0.1 <= degrees <= 90:
            raise ValueError('Enabled travel envelope must be 0.1–90 degrees')
        with self.lock:
            if self.armed:
                raise RuntimeError('Turn motor output OFF before changing travel envelope')
            self.envelope = degrees if enabled else 0.0
            self.origin = dict(self.positions) if self.positions else {}
            self.offsets = None
            self.target = None
        self.event('travel_envelope_changed', enabled=bool(enabled), degrees=self.envelope)

    def event(self, kind, **fields):
        item = dict(event=kind, utc=time.time(), **fields)
        with self.lock:
            self.history.append(item)
        if self.recorder:
            self.recorder.record(kind, **fields)

    def start(self):
        if self.thread:
            raise RuntimeError('Already started')
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def reference(self, azimuth, elevation):
        if not (math.isfinite(azimuth) and math.isfinite(elevation)
                and 0 <= azimuth < 360 and -90 <= elevation <= 90):
            raise ValueError('Reference requires azimuth [0,360), elevation [-90,90]')
        with self.lock:
            if self.connection != 'live' or self.motion not in ('idle', 'paused', 'stale_target_stopped'):
                raise RuntimeError('Connect and pause before setting the local pointing reference')
            # At the supported 1 Hz stability setting, one axis sample can be
            # older than 0.5 seconds while the pair is still valid.  Use the
            # controller's configured feedback timeout instead of a conflicting
            # hard-coded threshold.
            if any(time.monotonic() - self.position_times.get(a, 0) >
                   self.feedback_timeout for a in AXES):
                raise RuntimeError('Fresh feedback required')
            self.offsets = {0x10: azimuth - angle(self.positions[0x10]),
                            0x11: elevation - angle(self.positions[0x11])}
            self.target = None
        self.event('local_reference', azimuth=azimuth, elevation=elevation)

    def submit(self, azimuth, elevation, sequence, utc, ttl=.75):
        now = time.time()
        if (type(sequence) is not int or not all(math.isfinite(v) for v in (azimuth, elevation, utc, ttl))
                or not 0 <= azimuth < 360 or not -90 <= elevation <= 90
                or not .1 <= ttl <= 2 or not -.2 <= now - utc <= ttl):
            raise ValueError('Invalid or stale target')
        with self.lock:
            if self.offsets is None:
                raise RuntimeError('Set a local pointing reference first')
            if sequence <= self.sequence:
                raise ValueError('Sequence must increase')
            targets = {}
            for axis, degrees in ((0x10, azimuth), (0x11, elevation)):
                raw = (degrees - self.offsets[axis]) % 360
                encoder = round(raw * (1 << 24) / 360) % (1 << 24)
                if (self.envelope > 0 and
                        abs(relative_angle(encoder, self.origin[axis])) > self.envelope):
                    raise ValueError('Target exceeds confirmed experimental travel envelope')
                targets[axis] = encoder
            self.sequence = sequence
            self.rate_request = None
            self.target = dict(encoders=targets, utc=utc, ttl=ttl, received=time.monotonic(), sequence=sequence)
        return {'accepted_sequence': sequence, 'queued_commands': 1, 'latest_target_replaces_previous': True}

    def arm(self):
        with self.lock:
            if self.connection != 'live' or self.offsets is None:
                raise RuntimeError('Live feedback and local reference required')
            self.armed = True

    def pause(self):
        with self.lock:
            self.target = None
            self.rate_request = None
            self.manual_request = None
            self.armed = False
        self.pause_event.set()

    def freeze(self):
        """Stop motion without dropping control authority or encoder polling.

        Optical target loss is not a reason to release the motor session.  The
        controller remains armed so a credible reacquisition (or a local
        nudge) can submit a new target immediately.  ``pause()`` remains the
        explicit user-requested OFF/disarm operation.
        """
        with self.lock:
            stop_required = bool(
                self.target or self.rate_request or self.manual_request
                or self.motion not in ('idle', 'paused', 'stale_target_stopped'))
            self.target = None
            self.rate_request = None
            self.manual_request = None
            keep_armed = self.armed
            already_pending = self.pause_event.is_set()
        if stop_required and not already_pending:
            self.pause_event.set()
            self.event('motion_frozen', armed=keep_armed)
            return True
        return False

    def slew_rate(self, axis, direction, rate):
        """Start one axis at native Celestron slew level 1-9.

        The network worker remains the sole AUX socket owner.  Calibration can
        therefore request a slow continuous movement while the same worker
        continues collecting encoder feedback and can stop it via ``freeze``.
        """
        if axis not in AXES or direction not in (-1, 1) or rate not in range(1, 10):
            raise ValueError('Rate slew requires motor axis 0x10/0x11, direction -1/+1 and rate 1-9')
        with self.lock:
            if not self.armed or self.connection != 'live' or self.offsets is None:
                raise RuntimeError('Live, referenced and armed motor control is required')
            self.target = None
            self.rate_sequence += 1
            self.rate_request = {
                'axis': axis, 'direction': direction, 'rate': rate,
                'sequence': self.rate_sequence}
        self.event('rate_slew_requested', axis=axis, direction=direction, rate=rate,
                   sequence=self.rate_sequence)

    def manual_jog(self, axis, direction, rate, maximum_seconds=2.0):
        """Immediately jog one axis while fresh encoder feedback is live.

        Manual operator motion is deliberately independent of optical target
        selection, camera mapping, local sky reference and tracking arm state.
        The browser renews this short request while a button is held; expiry or
        release sends Stop. Network I/O remains owned by the worker thread.
        """
        maximum_seconds = float(maximum_seconds)
        if axis not in AXES or direction not in (-1, 1) or rate not in range(1, 10):
            raise ValueError('Manual jog requires axis 0x10/0x11, direction -1/+1 and speed 1-9')
        if not math.isfinite(maximum_seconds) or not .25 <= maximum_seconds <= 3:
            raise ValueError('Manual jog renewal window must be 0.25–3 seconds')
        with self.lock:
            now = time.monotonic()
            if (self.connection != 'live' or any(
                    now - self.position_times.get(axis_id, 0) > self.feedback_timeout
                    for axis_id in AXES)):
                raise RuntimeError('Fresh azimuth and elevation encoder feedback is required')
            self.armed = False
            self.target = None
            self.rate_request = None
            self.manual_sequence += 1
            self.manual_request = {
                'axis': axis, 'direction': direction, 'rate': rate,
                'sequence': self.manual_sequence,
                'deadline': now + maximum_seconds}
        self.event('manual_jog_requested', axis=axis, direction=direction,
                   rate=rate, sequence=self.manual_sequence)

    def manual_stop(self):
        """Cancel manual motion immediately; the worker sends both axis Stops."""
        with self.lock:
            self.manual_request = None
            self.target = None
            self.rate_request = None
            self.armed = False
        self.pause_event.set()
        self.event('manual_stop_requested')

    def stop(self):
        self.stop_event.set()
        self.pause_event.set()

    def status(self):
        with self.lock:
            now = time.monotonic()
            feedback_live = (self.connection == 'live' and all(
                now - self.position_times.get(axis, 0) <= self.feedback_timeout
                for axis in AXES))
            return {'experimental': True, 'connection': self.connection, 'motion': self.motion,
                    'feedback_live': feedback_live,
                    'last_error': self.last_error, 'reconnections': self.reconnections,
                    'local_reference_set': self.offsets is not None, 'armed': self.armed,
                    'envelope_degrees': self.envelope,
                    'feedback_timeout_seconds': self.feedback_timeout,
                    'configured_dispatch_rate_hz': self.dispatch_rate_hz,
                    'configured_feedback_poll_rate_hz': self.feedback_poll_rate_hz,
                    'target_change_threshold_degrees': self.target_change_threshold_degrees,
                    'stop_on_divergence': self.stop_on_divergence,
                    'configured_cycle_interval_ms': 1000 * self.cycle_interval,
                    'conservative_default_dispatch_rate_hz': 4,
                    'rate_warning': self.rate_warning,
                    'cycle_count': self.cycle_count,
                    'cycle_overruns': self.cycle_overruns,
                    'encoder_poll_count': self.encoder_poll_count,
                    'runtime_rx_count': self.runtime_rx_count,
                    'target_confirmation': dict(self.target_confirmation),
                    'last_cycle_duration_ms': self.last_cycle_duration_ms,
                    'last_cycle_period_ms': self.last_cycle_period_ms,
                    'last_achieved_cycle_rate_hz': (1000 / self.last_cycle_period_ms
                                                    if self.last_cycle_period_ms else None),
                    'max_cycle_duration_ms': self.max_cycle_duration_ms,
                    'positions': {AXES[a]: {'encoder': count, 'raw_degrees': angle(count),
                                          'reference_degrees': ((angle(count) + self.offsets[a]) % 360 if a == 0x10 else (angle(count) + self.offsets[a] + 180) % 360 - 180) if self.offsets else None,
                                          'age_seconds': now - self.position_times.get(a, 0),
                                          'live': self.connection == 'live' and now - self.position_times.get(a, 0) <= self.feedback_timeout}
                                  for a, count in self.positions.items()},
                    'last_target': self.target, 'rate_request': self.rate_request,
                    'manual_request': self.manual_request,
                    'events': list(self.history)[-20:],
                    'log_directory': str(self.recorder.directory) if self.recorder else None}

    def send_stops(self):
        if not self.sock:
            return
        for axis in AXES:
            self.sock.sendall(motor_packet(axis, 0x24, b'\x00'))
        self.event('stops_sent')

    def send_target(self, axis, encoder):
        """Transmit one finite position target without blocking on its ACK.

        The COSMOS motor controller normally acknowledges MC_GOTO_FAST, but the
        acknowledgement can be delayed or absent under load. Waiting for the
        azimuth acknowledgement before transmitting elevation starves the second
        axis and turns one missed ACK into a three-second control stall. Both
        targets are therefore sent promptly; later asynchronous encoder samples
        provide observational motion confirmation.
        """
        packet = target_packet(axis, encoder)
        self.sock.sendall(packet)
        self.event('target_tx', axis=axis, command=0x02, encoder=encoder,
                   hex=packet.hex(), acknowledgement_required=False)

    def send_runtime_request(self, axis, command, data=b''):
        """Transmit a runtime AUX request and return immediately.

        Replies are consumed by :meth:`drain_runtime_frames`; this method never
        calls ``recv`` and is therefore safe on the motor-dispatch path.
        """
        packet = motor_packet(axis, command, data)
        self.sock.sendall(packet)
        self.event('aux_tx', axis=axis, command=command, hex=packet.hex(),
                   acknowledgement_required=False)

    def drain_runtime_frames(self, max_reads=32):
        """Consume available AUX frames without waiting or starving writes."""
        encoder_updates = []
        reads = 0
        while not self.stop_event.is_set() and reads < max_reads:
            try:
                raw = self.sock.recv(4096)
            except (BlockingIOError, socket.timeout):
                break
            if not raw:
                raise ConnectionError('Adapter closed connection')
            reads += 1
            self.runtime_rx_count += 1
            self.event('aux_rx', hex=raw.hex(), asynchronous=True)
            for frame in self.parser.feed(raw):
                if len(frame) < 6:
                    continue
                source, destination, command = frame[2], frame[3], frame[4]
                if (source in AXES and destination == 0x20 and command == 0x01
                        and len(frame[5:-1]) == 3):
                    count = int.from_bytes(frame[5:-1], 'big')
                    now = time.monotonic()
                    with self.lock:
                        self.positions[source] = count
                        self.position_times[source] = now
                    self.event('encoder', axis=source, count=count,
                               asynchronous=True)
                    encoder_updates.append((source, count))
                else:
                    self.event('aux_reply_observed', source=source,
                               destination=destination, command=command,
                               asynchronous=True)
        return encoder_updates

    def observe_target_progress(self, sent_target):
        """Update non-gating target confirmation telemetry.

        This routine never sends Stop, clears a target, changes ``armed`` or
        changes the connection. It reports progress from whatever encoder
        samples happen to be available while command dispatch continues.
        """
        if not sent_target or any(axis not in self.positions for axis in AXES):
            return
        now = time.monotonic()
        errors = {AXES[axis]: abs(relative_angle(
            sent_target[axis], self.positions[axis])) for axis in AXES}
        worst_error = max(errors.values())
        with self.lock:
            report = dict(self.target_confirmation)
            if report.get('encoders') != sent_target:
                report = {
                    'state': 'moving', 'encoders': dict(sent_target),
                    'sent_monotonic': now, 'last_progress_monotonic': now,
                    'best_error_degrees': worst_error,
                    'errors_degrees': errors,
                    'message': 'Target sent; encoder confirmation is observational.'}
            else:
                report['errors_degrees'] = errors
                best = float(report.get('best_error_degrees', worst_error))
                if worst_error + .002 < best:
                    report['best_error_degrees'] = worst_error
                    report['last_progress_monotonic'] = now
                    if report.get('state') != 'confirmed':
                        report['state'] = 'moving'
                        report['message'] = 'Encoder position is progressing toward the latest target.'
                if worst_error <= self.target_confirmation_tolerance_degrees:
                    if report.get('state') != 'confirmed':
                        self.event('target_confirmed', errors_degrees=errors,
                                   tolerance_degrees=
                                   self.target_confirmation_tolerance_degrees)
                    report['state'] = 'confirmed'
                    report['confirmed_monotonic'] = now
                    report['message'] = 'Latest finite target confirmed by encoder feedback.'
                elif (now - float(report.get('last_progress_monotonic', now)) >=
                      self.motion_observation_timeout):
                    if report.get('state') != 'lagging':
                        self.event('target_confirmation_lagging',
                                   errors_degrees=errors,
                                   observation_timeout_seconds=
                                   self.motion_observation_timeout,
                                   gating=False)
                    report['state'] = 'lagging'
                    report['message'] = (
                        'Latest target is not yet confirmed; dispatch remains active '
                        'and this observation does not gate motion.')
            report['age_seconds'] = now - float(report.get('sent_monotonic', now))
            report['gating'] = False
            self.target_confirmation = report

    def exchange(self, axis, command, data=b'', timeout=3.0):
        """Serialize AUX requests: don't overlap independent motor replies on the bus."""
        packet = target_packet(axis, int.from_bytes(data, 'big')) if command == 2 else motor_packet(axis, command, data)
        self.sock.sendall(packet)
        self.event('aux_tx', axis=axis, command=command, hex=packet.hex())
        end = time.monotonic() + timeout
        while time.monotonic() < end and not self.stop_event.is_set():
            if self.pause_event.is_set():
                self.send_stops()
                self.pause_event.clear()
                with self.lock:
                    self.motion = 'paused'
            try:
                raw = self.sock.recv(4096)
            except socket.timeout:
                continue
            if not raw:
                raise ConnectionError('Adapter closed connection')
            self.event('aux_rx', hex=raw.hex())
            for frame in self.parser.feed(raw):
                if frame[2] == axis and frame[3] == 0x20 and frame[4] == command:
                    return frame[5:-1]
        raise ConnectionError(f'AUX {axis:02X}/{command:02X} reply timeout')

    def run(self):
        retry_delay = .5
        try:
            self.lease.acquire()
            check_direct_route(self.host)
            self.recorder = Recorder('position-stream')
            while not self.stop_event.is_set():
                try:
                    with self.lock:
                        self.connection = 'connecting' if not self.origin else 'recovering'
                        self.position_times.clear()
                    self.sock = socket.create_connection((self.host, self.port), 3)
                    self.sock.settimeout(.05)
                    self.parser = FrameParser()
                    startup_end = time.monotonic() + .6
                    while time.monotonic() < startup_end and not self.stop_event.is_set():
                        try:
                            raw = self.sock.recv(4096)
                            if not raw:
                                raise ConnectionError('Adapter closed during startup')
                            self.event('startup_rx', hex=raw.hex())
                        except socket.timeout:
                            pass
                    # Match the production controller bootstrap: wait for greeting,
                    # then verify each motor before issuing any control commands.
                    for axis in AXES:
                        try:
                            self.exchange(axis, 0xfe, timeout=1.5)
                        except ConnectionError as error:
                            if 'reply timeout' not in str(error):
                                raise
                            # GET_VER is optional identity telemetry. Fresh
                            # encoder replies are the control-health signal.
                            self.event('firmware_reply_missed', axis=axis,
                                       message=str(error))
                    if self.origin:
                        self.send_stops()
                    sent_target = None
                    last_target_send = 0.0
                    sent_rate_sequence = None
                    sent_manual_sequence = None
                    target_error = {}
                    previous_dispatch_start = None
                    next_dispatch = time.monotonic()
                    next_feedback_poll = next_dispatch
                    feedback_stale_reported = False
                    # Startup identity probes above are allowed to wait because
                    # no motion authority exists yet. Runtime is different:
                    # recv must never block target/rate/Stop dispatch.
                    self.sock.settimeout(0)
                    while not self.stop_event.is_set():
                        loop_started = time.monotonic()
                        if self.pause_event.is_set():
                            self.send_stops()
                            self.pause_event.clear()
                            self.motion = 'paused'
                            sent_target = None
                            last_target_send = 0.0
                            sent_rate_sequence = None
                            sent_manual_sequence = None

                        # Parse whatever the adapter has already delivered. No
                        # read is awaited and no missing reply delays a write.
                        for axis, count in self.drain_runtime_frames():
                            # An encoder can legitimately have a very different raw value
                            # after a mount restart, manual/AUX movement, or a new session.
                            # A stale session origin must never prevent a disarmed reconnect.
                            # The envelope remains enforced for submitted live targets.
                            if (self.armed and self.envelope > 0 and self.origin
                                    and abs(relative_angle(count, self.origin[axis])) >
                                    self.envelope + .25):
                                raise ValueError('Observed travel exceeded confirmed envelope')
                            if (self.armed and sent_target and axis in target_error
                                    and abs(relative_angle(sent_target[axis], count)) > target_error[axis] + .3):
                                divergence = abs(relative_angle(sent_target[axis], count))
                                if self.stop_on_divergence:
                                    self.pause()
                                    self.send_stops()
                                    self.last_error = ('Measured divergence from commanded '
                                                       'target; disarmed for correction')
                                else:
                                    # Optical tracking owns the correction loop.
                                    # Report divergence as feedback without
                                    # tearing down motion authority; its PD and
                                    # direction-sanity logic can reduce/reverse
                                    # the next command. Explicit Stop/OFF and
                                    # travel-envelope violations still stop.
                                    target_error[axis] = divergence
                                    self.last_error = ('Measured divergence from commanded '
                                                       'target; optical correction remains armed')
                                self.event('divergence', axis=axis,
                                           stopped=self.stop_on_divergence,
                                           message=self.last_error)

                        # Gross confirmation is deliberately after-the-fact.
                        # It observes the latest samples and never gates writes.
                        self.observe_target_progress(sent_target)

                        now = time.monotonic()
                        with self.lock:
                            stale_axes = [axis for axis in AXES if
                                          now - self.position_times.get(axis, 0) >
                                          self.feedback_timeout]
                            fresh = not stale_axes
                            if not fresh:
                                self.last_error = (
                                    f'Encoder feedback stale beyond {self.feedback_timeout:g} seconds; '
                                    'socket retained and polling continues')
                                if not feedback_stale_reported:
                                    self.event('encoder_feedback_stale',
                                               timeout_seconds=self.feedback_timeout,
                                               missed_axes=stale_axes)
                                    feedback_stale_reported = True
                            elif self.last_error and self.last_error.startswith('Encoder feedback stale'):
                                self.last_error = None
                                self.event('encoder_feedback_resumed')
                                feedback_stale_reported = False
                            if fresh and self.connection != 'live':
                                self.connection = 'live'
                                self.live_since = now
                                # Rebase a disarmed/recovered session to fresh encoder
                                # readings. Any prior local reference is invalid after a
                                # disconnect because it may describe the previous power or
                                # camera session. The caller must explicitly reference again.
                                self.origin = dict(self.positions)
                                self.offsets = None
                                self.target = None
                                self.rate_request = None
                                self.manual_request = None
                                self.motion = 'idle'
                                self.last_error = None
                                retry_delay = .5
                                self.event('link_live', requires_new_target=True)
                            target = self.target
                            rate_request = self.rate_request
                            manual_request = self.manual_request

                        # Manual and continuous-rate commands are edge-triggered
                        # writes. Their ACKs, if any, are merely logged later.
                        if manual_request:
                            if now > manual_request['deadline']:
                                self.manual_stop()
                                manual_request = None
                            elif manual_request['sequence'] != sent_manual_sequence:
                                command = 0x24 if manual_request['direction'] > 0 else 0x25
                                self.send_runtime_request(
                                    manual_request['axis'], command,
                                    bytes((manual_request['rate'],)))
                                sent_manual_sequence = manual_request['sequence']
                                sent_rate_sequence = None
                                sent_target = None
                                self.motion = 'manual_jog'
                                self.event('manual_jog_started', **manual_request)
                        if (self.armed and rate_request and
                                rate_request['sequence'] != sent_rate_sequence):
                            command = 0x24 if rate_request['direction'] > 0 else 0x25
                            self.send_runtime_request(
                                rate_request['axis'], command,
                                bytes((rate_request['rate'],)))
                            sent_rate_sequence = rate_request['sequence']
                            sent_target = None
                            self.motion = 'rate_slew'
                            self.event('rate_slew_started', **rate_request)

                        # Dispatch is independently clocked. It never waits for
                        # the feedback poll scheduled below or for its reply.
                        if now >= next_dispatch:
                            if previous_dispatch_start is not None:
                                with self.lock:
                                    self.last_cycle_period_ms = 1000 * (
                                        now - previous_dispatch_start)
                            previous_dispatch_start = now
                            dispatch_started = time.monotonic()
                            next_dispatch = max(next_dispatch + self.cycle_interval,
                                                now + self.cycle_interval)
                        else:
                            dispatch_started = None

                        if (dispatch_started is not None and self.armed
                                and target and target['received'] > self.live_since):
                            changed_axes = []
                            if time.time() - target['utc'] > target['ttl']:
                                if self.motion != 'stale_target_stopped':
                                    self.send_stops()
                                    self.motion = 'stale_target_stopped'
                                    sent_target = None
                            else:
                                changed_axes = [axis for axis in AXES if (
                                    sent_target is None or abs(relative_angle(
                                        target['encoders'][axis], sent_target[axis])) >=
                                    self.target_change_threshold_degrees)]
                                # A small finite GoTo can be swallowed by gear
                                # backlash even while the socket and encoders
                                # remain healthy. Renew the current target at a
                                # bounded 1 Hz when observational confirmation
                                # reports no progress. This does not gate newer
                                # targets or wait synchronously for feedback.
                                if (not changed_axes
                                        and self.target_confirmation.get('state') == 'lagging'
                                        and now - last_target_send >= 1.0):
                                    changed_axes = list(AXES)
                                    self.event('lagging_target_renewed',
                                               sequence=target['sequence'],
                                               encoders=dict(target['encoders']))
                            if changed_axes:
                                for axis in changed_axes:
                                    if (not self.armed or self.stop_event.is_set()
                                            or time.time() - target['utc'] > target['ttl']):
                                        self.send_stops()
                                        break
                                    self.send_target(axis, target['encoders'][axis])
                                else:
                                    if sent_target is None:
                                        sent_target = dict(target['encoders'])
                                    else:
                                        sent_target.update({axis: target['encoders'][axis]
                                                            for axis in changed_axes})
                                    self.event('target_sent', sequence=target['sequence'],
                                               axes=changed_axes, encoders=dict(sent_target))
                                    last_target_send = now
                                    target_error = {
                                        a: abs(relative_angle(sent_target[a], self.positions[a]))
                                        for a in AXES}
                                    self.motion = 'target_sent'

                        if dispatch_started is not None:
                            dispatch_elapsed = time.monotonic() - dispatch_started
                            with self.lock:
                                self.cycle_count += 1
                                self.last_cycle_duration_ms = 1000 * dispatch_elapsed
                                self.max_cycle_duration_ms = max(
                                    self.max_cycle_duration_ms,
                                    self.last_cycle_duration_ms)
                                if dispatch_elapsed > self.cycle_interval:
                                    self.cycle_overruns += 1

                        # Poll packets are also back-to-back scheduled writes.
                        # Their replies are parsed on a later loop iteration.
                        now = time.monotonic()
                        if now >= next_feedback_poll:
                            for axis in AXES:
                                self.send_runtime_request(axis, 0x01)
                            self.encoder_poll_count += 1
                            next_feedback_poll = max(
                                next_feedback_poll + self.feedback_poll_interval,
                                now + self.feedback_poll_interval)

                        # A short cooperative wait prevents busy-spinning while
                        # keeping command latency independent of AUX reply time.
                        next_due = min(next_dispatch, next_feedback_poll)
                        self.stop_event.wait(max(0, min(.01, next_due - time.monotonic())))
                except ValueError as error:
                    self.last_error = str(error)
                    self.event('travel_fault', message=str(error))
                    self.armed = False
                    break
                except (OSError, ConnectionError) as error:
                    self.last_error = str(error)
                    self.reconnections += 1
                    self.event('link_lost', message=str(error))
                finally:
                    try:
                        self.send_stops()
                    except OSError as error:
                        self.event('stop_unconfirmed', message=str(error))
                    if self.sock:
                        self.sock.close()
                        self.sock = None
                    with self.lock:
                        self.connection = 'recovering'
                        self.motion = 'idle'
                self.stop_event.wait(retry_delay)
                retry_delay = min(2, retry_delay * 2)
        except Exception as error:
            self.last_error = str(error)
            self.event('controller_error', message=str(error))
        finally:
            self.connection = 'stopped'
            self.motion = 'stopped'
            self.armed = False
            self.lease.release()
            if self.recorder:
                self.recorder.finish(self.status())
