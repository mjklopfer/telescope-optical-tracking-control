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
not simultaneously on the serial wire. The conservative default loop rate is 4 Hz.
Rates above 4 Hz (intervals below 250 ms) are experimental and unverified; actual
throughput is also limited by four serialized AUX exchanges during a target cycle.
Incoming samples may arrive faster, but that does not make hardware actuation
equally fast. Missing feedback triggers Stop attempts, TCP close and bounded-rate
reconnection. Reconnection requires both fresh encoders and a NEW target received
after the link is live. Last known position remains labeled with age; it is never
treated as live feedback. Encoder references are local transforms, NOT a shared
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
    def __init__(self, host, port=2000, envelope_degrees=5, feedback_timeout=1.5,
                 dispatch_rate_hz=4):
        self.host, self.port = validate_endpoint(host, port)
        if (not 0 <= envelope_degrees <= 90 or not .3 <= feedback_timeout <= 2
                or not math.isfinite(dispatch_rate_hz) or not 1 <= dispatch_rate_hz <= 20):
            raise ValueError('Optional envelope 0 (disabled)–90 degrees; feedback timeout 0.3–2 seconds; dispatch rate 1–20 Hz')
        self.envelope = envelope_degrees
        self.feedback_timeout = feedback_timeout
        self.dispatch_rate_hz = float(dispatch_rate_hz)
        self.cycle_interval = 1 / self.dispatch_rate_hz
        self.rate_warning = (None if self.dispatch_rate_hz <= 4 else
                             ('STRESS TEST: requested rates above 10 Hz are unlikely to be sustained by four serialized AUX exchanges; '
                              'watch achieved cycle timing and overruns.' if self.dispatch_rate_hz > 10 else
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
        self.history = deque(maxlen=80)

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
                                 ('STRESS TEST: requested rates above 10 Hz are unlikely to be sustained by four serialized AUX exchanges; '
                                  'watch achieved cycle timing and overruns.' if dispatch_rate_hz > 10 else
                                  'EXPERIMENTAL: rates above the conservative 4 Hz baseline '
                                  '(intervals below 250 ms) are unverified and may cause delayed or missing AUX replies.'))
        self.event('dispatch_rate_changed', dispatch_rate_hz=dispatch_rate_hz,
                   warning=self.rate_warning)

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
            self.target = None
            self.rate_request = None
            keep_armed = self.armed
        self.pause_event.set()
        self.event('motion_frozen', armed=keep_armed)

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

    def stop(self):
        self.stop_event.set()
        self.pause_event.set()

    def status(self):
        with self.lock:
            now = time.monotonic()
            return {'experimental': True, 'connection': self.connection, 'motion': self.motion,
                    'last_error': self.last_error, 'reconnections': self.reconnections,
                    'local_reference_set': self.offsets is not None, 'armed': self.armed,
                    'envelope_degrees': self.envelope,
                    'configured_dispatch_rate_hz': self.dispatch_rate_hz,
                    'configured_cycle_interval_ms': 1000 * self.cycle_interval,
                    'conservative_default_dispatch_rate_hz': 4,
                    'rate_warning': self.rate_warning,
                    'cycle_count': self.cycle_count,
                    'cycle_overruns': self.cycle_overruns,
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
                    'events': list(self.history)[-20:],
                    'log_directory': str(self.recorder.directory) if self.recorder else None}

    def send_stops(self):
        if not self.sock:
            return
        for axis in AXES:
            self.sock.sendall(motor_packet(axis, 0x24, b'\x00'))
        self.event('stops_sent')

    def exchange(self, axis, command, data=b'', timeout=1.2):
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
                        self.exchange(axis, 0xfe)
                    if self.origin:
                        self.send_stops()
                    sent_target = None
                    sent_rate_sequence = None
                    target_error = {}
                    previous_cycle_start = None
                    while not self.stop_event.is_set():
                        cycle = time.monotonic()
                        if previous_cycle_start is not None:
                            with self.lock:
                                self.last_cycle_period_ms = 1000 * (cycle - previous_cycle_start)
                        previous_cycle_start = cycle
                        if self.pause_event.is_set():
                            self.send_stops()
                            self.pause_event.clear()
                            self.motion = 'paused'
                            sent_target = None
                            sent_rate_sequence = None
                        for axis in AXES:
                            data = self.exchange(axis, 1)
                            if len(data) != 3:
                                raise ConnectionError('Invalid encoder payload')
                            count = int.from_bytes(data, 'big')
                            with self.lock:
                                self.positions[axis] = count
                                self.position_times[axis] = time.monotonic()
                            self.event('encoder', axis=axis, count=count)
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
                                self.pause()
                                self.send_stops()
                                self.last_error = 'Measured divergence from commanded target; disarmed for correction'
                                self.event('divergence', axis=axis, message=self.last_error)
                        with self.lock:
                            fresh = all(time.monotonic() - self.position_times.get(a, 0) <= self.feedback_timeout for a in AXES)
                            if not fresh:
                                raise ConnectionError('Encoder pair too old')
                            if self.connection != 'live':
                                self.connection = 'live'
                                self.live_since = time.monotonic()
                                # Rebase a disarmed/recovered session to fresh encoder
                                # readings. Any prior local reference is invalid after a
                                # disconnect because it may describe the previous power or
                                # camera session. The caller must explicitly reference again.
                                self.origin = dict(self.positions)
                                self.offsets = None
                                self.target = None
                                self.rate_request = None
                                self.motion = 'idle'
                                self.last_error = None
                                retry_delay = .5
                                self.event('link_live', requires_new_target=True)
                            target = self.target
                            rate_request = self.rate_request
                        if (self.armed and rate_request and
                                rate_request['sequence'] != sent_rate_sequence):
                            command = 0x24 if rate_request['direction'] > 0 else 0x25
                            self.exchange(rate_request['axis'], command,
                                          bytes((rate_request['rate'],)))
                            sent_rate_sequence = rate_request['sequence']
                            sent_target = None
                            self.motion = 'rate_slew'
                            self.event('rate_slew_started', **rate_request)
                        if self.armed and target and target['received'] > self.live_since:
                            if time.time() - target['utc'] > target['ttl']:
                                if self.motion != 'stale_target_stopped':
                                    self.send_stops()
                                    self.motion = 'stale_target_stopped'
                                    sent_target = None
                            elif target['encoders'] != sent_target:
                                for axis in AXES:
                                    if (not self.armed or self.stop_event.is_set()
                                            or time.time() - target['utc'] > target['ttl']):
                                        self.send_stops()
                                        break
                                    self.exchange(axis, 2, target['encoders'][axis].to_bytes(3, 'big'))
                                else:
                                    self.event('target_sent', sequence=target['sequence'], encoders=target['encoders'])
                                    sent_target = dict(target['encoders'])
                                    target_error = {a: abs(relative_angle(sent_target[a], self.positions[a])) for a in AXES}
                                    self.motion = 'target_sent'
                        elapsed = time.monotonic() - cycle
                        with self.lock:
                            self.cycle_count += 1
                            self.last_cycle_duration_ms = 1000 * elapsed
                            self.max_cycle_duration_ms = max(self.max_cycle_duration_ms,
                                                             self.last_cycle_duration_ms)
                            if elapsed > self.cycle_interval:
                                self.cycle_overruns += 1
                        self.stop_event.wait(max(0, self.cycle_interval - elapsed))
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
