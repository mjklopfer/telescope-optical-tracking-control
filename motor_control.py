"""Thread-owned Celestron AUX motor control and encoder telemetry.

Priority Stop, expiring holds and bounded jogs limit commanded movement. A failed
transport can prevent Stop delivery; successful writes alone do not prove motion."""
import math
import queue
import socket
import threading
import time
from celestron_probe import FrameParser, Recorder
from setup_backend import validate_endpoint, check_direct_route
from endpoint_lock import EndpointLease

AXES = {0x10: 'Azimuth', 0x11: 'Elevation'}
COUNTS_PER_TURN = 1 << 24


def motor_packet(axis, command, data=b''):
    """Validate a supported motor request and append its AUX checksum."""
    if axis not in AXES or command not in (1, 0xfe, 0x24, 0x25):
        raise ValueError('Unsupported motor request')
    if command in (0x24, 0x25):
        if len(data) != 1 or not 0 <= data[0] <= 9:
            raise ValueError('Motor rate must be 0–9')
    elif data:
        raise ValueError('Read requests do not take data')
    body = bytes((3 + len(data), 0x20, axis, command)) + data
    return b'\x3b' + body + bytes((-sum(body) & 255,))


def angle(encoder):
    """Convert a raw 24-bit encoder count into degrees."""
    return encoder * 360 / COUNTS_PER_TURN


def relative_angle(encoder, origin):
    """Express the shortest signed encoder difference in degrees."""
    return relative_counts(encoder, origin) * 360 / COUNTS_PER_TURN


def relative_counts(encoder, origin):
    """Return the shortest signed count difference across wraparound."""
    return (encoder - origin + COUNTS_PER_TURN // 2) % COUNTS_PER_TURN - COUNTS_PER_TURN // 2


def version(data):
    """Format a controller firmware payload for display."""
    if len(data) == 2:
        return f'{data[0]}.{data[1]}'
    if len(data) == 4:
        return f'{data[0]}.{data[1]}.{int.from_bytes(data[2:], "big")}'
    return data.hex() or 'Unknown'


class MotorController:
    def __init__(self, host, port, callback, socket_factory=socket.create_connection,
                 recorder_factory=Recorder, route_check=check_direct_route, idle_release=30,
                 heartbeat_enabled=True, heartbeat_interval=1.0, interface_mode="legacy"):
        """Validate the destination and create command queues, watchdog and telemetry state."""
        if idle_release not in (0, 10, 20, 30, 60, 120, 300):
            raise ValueError('Idle release must be Never (0), 10, 20, 30, 60, 120, or 300 seconds')
        heartbeat_interval = float(heartbeat_interval)
        if not math.isfinite(heartbeat_interval) or not 1 <= heartbeat_interval <= 300:
            raise ValueError('Direct motor keep-alive interval must be 1–300 seconds')
        self.heartbeat_enabled = bool(heartbeat_enabled)
        self.heartbeat_interval = heartbeat_interval
        self.idle_release = idle_release
        self.last_control_activity = time.monotonic()
        if interface_mode not in ('legacy', 'modern'):
            raise ValueError('Motor control supports legacy or modern hardware mode, not proxy diagnostics mode.')
        self.interface_mode = interface_mode
        self.host, self.port = validate_endpoint(host, port)
        self.callback, self.socket_factory = callback, socket_factory
        self.recorder_factory, self.route_check = recorder_factory, route_check
        self.connected = threading.Event()
        self.stopping = threading.Event()
        self.disconnecting = threading.Event()
        self.command_pending = threading.Event()
        self.jog_in_progress = threading.Event()
        self.commands = queue.Queue(maxsize=1)
        self.position_reads = queue.Queue(maxsize=1)
        self.thread = None
        self.sock = None
        self.recorder = None
        self.active_axis = None
        self.holding = False
        self.lease_deadline = 0
        self.parser = FrameParser()
        self.origins = {}
        self.firmware = {}
        self.positions = {}
        self.position_times = {}
        self.endpoint_lease = EndpointLease(self.host, self.port)
        self.report = {'host': self.host, 'port': self.port, 'jogs': [], 'errors': [], 'final_stops': []}

    def emit(self, event, **fields):
        """Publish a service event for logging and the GUI event queue."""
        if self.recorder:
            self.recorder.record(event, **fields)
        self.callback({'event': event, **fields})

    def start(self):
        """Start background processing without blocking the caller."""
        if self.thread is not None:
            raise RuntimeError('Controller has already been started')
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def jog(self, axis, direction, rate, seconds, hold=False):
        """Queue a bounded movement request after validating its rate and duration."""
        if not self.connected.is_set() or self.disconnecting.is_set():
            raise RuntimeError('Connect to the motors first')
        if axis not in AXES or direction not in (-1, 1) or not isinstance(rate, int) or not 1 <= rate <= 9:
            raise ValueError('Select an axis, direction, and speed 1–9')
        # Cached idle samples may be stale when keep-alive is OFF/slow. The
        # worker obtains both encoders immediately before any motion command.
        if not .1 <= seconds <= 5:
            raise ValueError('Jog duration must be 0.1–5 seconds')
        if self.command_pending.is_set() or self.jog_in_progress.is_set() or self.stopping.is_set():
            raise RuntimeError('Wait for the current jog or stop to finish')
        self.last_control_activity = time.monotonic()
        self.holding = hold
        self.lease_deadline = time.monotonic() + .4 if hold else 0
        self.command_pending.set()
        self.commands.put_nowait((axis, direction, rate, seconds))

    def request_position(self, axis):
        """Queue a read on the owning worker; never access the socket from Tk."""
        if axis not in AXES:
            raise ValueError('Select Azimuth or Elevation')
        if not self.connected.is_set() or self.disconnecting.is_set():
            raise RuntimeError('Connect Motors before reading positions')
        if self.command_pending.is_set() or self.jog_in_progress.is_set() or self.stopping.is_set():
            raise RuntimeError('Wait for movement or Stop to finish before reading')
        try:
            self.position_reads.put_nowait(axis)
        except queue.Full:
            raise RuntimeError('Wait for the current position read to finish') from None

    def stop(self):
        """Request priority stopping through the service’s owning thread."""
        self.last_control_activity = time.monotonic()
        self.holding = False
        self.stopping.set()

    def hold(self, axis, direction, rate):
        """Begin manual movement that must be renewed while the control stays held."""
        self.jog(axis, direction, rate, .4, hold=True)

    def renew_hold(self):
        """Refresh the manual-hold watchdog without issuing a new direction."""
        if self.holding and self.command_pending.is_set() and not self.stopping.is_set():
            self.lease_deadline = time.monotonic() + .4

    def disconnect(self):
        """Ask the owning thread to stop movement and close its connection."""
        self.disconnecting.set()
        self.stopping.set()

    def send(self, axis, command, data=b''):
        """Write one supported motor frame through the service-owned socket."""
        frame = motor_packet(axis, command, data)
        self.sock.sendall(frame)
        self.emit('motor_tx', axis=axis, command=command, hex=frame.hex())

    def stop_both(self):
        """Send rate-zero Stop commands for both motor axes."""
        errors = []
        for axis in AXES:
            try:
                self.send(axis, 0x24, b'\x00')
                self.report['final_stops'].append({'axis': axis, 'sent': True})
            except Exception as e:
                errors.append(f'{AXES[axis]}: {e}')
                self.report['final_stops'].append({'axis': axis, 'sent': False, 'error': str(e)})
        if errors:
            raise ConnectionError('Stop delivery failed: ' + '; '.join(errors))
        self.active_axis = None
        self.holding = False
        while True:
            try:
                self.commands.get_nowait()
            except queue.Empty:
                break
        self.command_pending.clear()
        self.emit('motor_stopped', message='Stop sent to both axes; delivery is not encoder verification.')

    def handle_stop(self):
        """Process a pending priority Stop and reset manual movement state."""
        if self.stopping.is_set():
            # Stop precedes every read/jog and discards any queued movement.
            self.stop_both()
            self.stopping.clear()

    def drain_stop_replies(self, timeout=.8):
        """Drain outstanding replies before closing the legacy TCP/UART bridge.

        This runs only during connection cleanup, never before sending Stop.
        Replies acknowledge a command exchange; encoder samples verify rest.
        """
        replies = set()
        last_rx = time.monotonic()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                raw = self.sock.recv(4096)
            except socket.timeout:
                if len(replies) == len(AXES) and time.monotonic() - last_rx >= .1:
                    break
                continue
            if not raw:
                break
            self.recorder.record('motor_cleanup_rx', hex=raw.hex())
            last_rx = time.monotonic()
            for frame in self.parser.feed(raw):
                if frame[2] in AXES and frame[3:5] == b'\x20\x24':
                    replies.add(frame[2])
        self.report['cleanup_stop_replies'] = {str(axis): axis in replies for axis in AXES}
        # A half-close allows the adapter to see a clean end of the output
        # stream after pending UART replies have been consumed.
        if hasattr(self.sock, 'shutdown'):
            try:
                self.sock.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    def request(self, axis, command, timeout=1.2):
        """Send a controller query and wait within its reply deadline."""
        self.handle_stop()
        if self.disconnecting.is_set():
            return None
        self.send(axis, command)
        requested_at = time.monotonic()
        received_bytes = 0
        end = requested_at + timeout
        while time.monotonic() < end:
            if self.active_axis is not None and self.holding and time.monotonic() >= self.lease_deadline:
                self.stopping.set()
            self.handle_stop()
            if self.disconnecting.is_set():
                return None
            try:
                data = self.sock.recv(4096)
            except socket.timeout:
                continue
            if not data:
                raise ConnectionError('Adapter closed the connection')
            received_bytes += len(data)
            self.recorder.record('motor_rx', hex=data.hex())
            for frame in self.parser.feed(data):
                if frame[2] == axis and frame[3] == 0x20 and frame[4] == command:
                    self.emit('motor_reply', axis=axis, command=command, hex=frame.hex(),
                              elapsed_seconds=time.monotonic() - requested_at, received_bytes=received_bytes)
                    return frame[5:-1]
        self.emit('motor_request_timeout', axis=axis, command=command,
                  elapsed_seconds=time.monotonic() - requested_at, received_bytes=received_bytes)
        return None

    def read_positions(self, attempts=1, publish=True, axes=None):
        """Read both encoder positions and publish validated telemetry."""
        samples = {}
        for axis in (AXES if axes is None else axes):
            if self.disconnecting.is_set():
                return samples
            data = None
            for _ in range(attempts):
                data = self.request(axis, 1)
                if data is not None and len(data) == 3:
                    break
            if data is None or len(data) != 3:
                self.emit('motor_telemetry_missing', axis=axis)
                continue
            encoder = int.from_bytes(data, 'big')
            samples[axis] = encoder
            previous = self.positions.get(axis)
            self.positions[axis] = encoder
            self.position_times[axis] = time.monotonic()
            self.origins.setdefault(axis, encoder)
            if publish:
                self.emit('motor_values', axis=axis, encoder=encoder, angle=angle(encoder),
                      relative=relative_angle(encoder, self.origins[axis]),
                      change=None if previous is None else relative_angle(encoder, previous),
                      firmware=self.firmware.get(axis, 'Unknown'), monotonic=self.position_times[axis])
        return samples

    def perform_jog(self, axis, direction, rate, seconds):
        """Run the queued bounded motion, stop, and measure final encoder feedback."""
        if self.stopping.is_set() or self.disconnecting.is_set():
            self.handle_stop()
            return
        held = self.holding
        # Match the successful standalone test: sample immediately before motion,
        # not from an earlier idle poll or the connection origin.
        before = self.read_positions(attempts=2)
        if self.disconnecting.is_set() or not self.command_pending.is_set():
            return
        if len(before) != len(AXES):
            raise ConnectionError('Fresh encoder replies from both axes are required; movement was not sent')
        entry = {'axis': axis, 'direction': direction, 'rate': rate, 'requested_seconds': seconds, 'hold': held,
                 'start_encoder': before[axis]}
        self.report['jogs'].append(entry)
        self.active_axis = axis  # Set before sending, including failed/partial sends.
        started = time.monotonic()
        try:
            self.send(axis, 0x24 if direction > 0 else 0x25, bytes((rate,)))
            self.emit('motor_jog_started', **entry)
            deadline = started + seconds
            next_position = started + .4
            while self.active_axis is not None and not self.disconnecting.is_set():
                limit = self.lease_deadline if held else deadline
                if time.monotonic() >= limit:
                    break
                if self.stopping.wait(min(.02, max(0, limit - time.monotonic()))):
                    break
                if held and time.monotonic() >= next_position:
                    if len(self.read_positions()) != len(AXES):
                        raise ConnectionError('Encoder feedback lost during movement; sending Stop')
                    next_position = time.monotonic() + .4
        finally:
            # No telemetry/ack wait before stop; disconnect/error also reaches here.
            self.stop_both()
            self.stopping.clear()
            entry['elapsed_seconds'] = time.monotonic() - started
        self.emit('motor_jog_finished', **entry)
        samples = self.read_positions(attempts=2)
        first = samples.get(axis)
        self.disconnecting.wait(.25)
        settled = self.read_positions(attempts=2, axes=(axis,))
        after = settled.get(axis)
        # High slew rates can decelerate past the first sample pair. Keep the
        # movement gate closed and verify settling within a bounded interval.
        settling = []
        deadline = time.monotonic() + 3
        if first is not None and after is not None and abs(relative_angle(after, first)) > .03:
            self.emit('motor_settling', axis=axis,
                      message='Stop sent; waiting for encoder samples to settle (up to 3 seconds).')
            self.stop_both()
            while not self.disconnecting.is_set() and time.monotonic() < deadline:
                first = after
                self.disconnecting.wait(.25)
                after = self.read_positions(axes=(axis,)).get(axis)
                settling.append(after)
                if after is None or abs(relative_angle(after, first)) <= .03:
                    break
        entry['settling_samples'] = settling
        measured_counts = relative_counts(after, before[axis]) if after is not None else None
        measured = relative_angle(after, before[axis]) if after is not None else None
        residual = relative_angle(after, first) if after is not None and first is not None else None
        stationary = abs(residual) <= .03 if residual is not None else None
        message = ('Movement unverified — no ending encoder reply' if measured is None else
                   f'{before[axis]} → {after} counts; Δ {measured_counts:+d} counts / {measured:+.5f}°')
        if measured_counts == 0:
            message = 'No encoder movement measured: ' + message
        message += ('; stop not verified' if stationary is None else
                    '; stationary samples verified' if stationary else '; still moving after Stop')
        result = dict(axis=axis, start_encoder=before[axis], end_encoder=after,
                      measured_counts=measured_counts, measured_degrees=measured,
                      residual_degrees=residual, stationary_verified=stationary, message=message)
        entry.update({k: v for k, v in result.items() if k not in ('axis', 'message')})
        if stationary is True and len(samples) == len(AXES) and not self.disconnecting.is_set():
            # Consumers may submit the next jog upon receiving a successful
            # result. All encoder verification is complete at this point.
            self.jog_in_progress.clear()
        self.emit('motor_motion_result', **result)
        if self.disconnecting.is_set():
            return
        if stationary is False:
            self.stop_both()
            raise ConnectionError('Encoder samples show continued movement after Stop; repeated Stop')
        if len(samples) != len(AXES) or after is None:
            raise ConnectionError('Encoder feedback lost; movement controls disabled')

    def run(self):
        """Own motor I/O, process queued motion and priority Stop, then drain and close safely."""
        try:
            self.recorder = self.recorder_factory('motor-control')
        except Exception as e:
            self.callback({'event': 'motor_error', 'message': f'Cannot create session log: {e}'})
            self.callback({'event': 'motor_disconnected', 'log_directory': ''})
            return
        try:
            self.endpoint_lease.acquire()
            self.route_check(self.host)
            self.emit('motor_connecting', message=f'Connecting to {self.host}:{self.port}',
                      log_directory=str(self.recorder.directory))
            self.sock = self.socket_factory((self.host, self.port), 3)
            self.sock.settimeout(.05)
            # Let the RN-171 finish its greeting before the first AUX request.
            end = time.monotonic() + (.6 if self.interface_mode == 'legacy' else 0)
            while time.monotonic() < end and not self.disconnecting.is_set():
                try:
                    greeting = self.sock.recv(4096)
                except socket.timeout:
                    continue
                if not greeting:
                    raise ConnectionError('Adapter closed during startup')
                self.recorder.record('motor_startup_rx', hex=greeting.hex())
            for axis in AXES:
                reply = self.request(axis, 0xfe)
                self.firmware[axis] = version(reply) if reply is not None else 'No reply'
            if self.disconnecting.is_set():
                return
            samples = self.read_positions(attempts=2, publish=False)
            if len(samples) != len(AXES):
                raise ConnectionError('Wi-Fi socket opened, but motor encoder replies are missing. '
                                      'Motor connection NOT verified; movement controls remain disabled.')
            self.connected.set()
            self.emit('motor_connected', host=self.host, port=self.port)
            self.last_control_activity = time.monotonic()
            self.emit('motor_connection_policy', idle_release=self.idle_release,
                      heartbeat_enabled=self.heartbeat_enabled, heartbeat_interval=self.heartbeat_interval)
            next_poll = time.monotonic() + self.heartbeat_interval
            while not self.disconnecting.is_set():
                self.handle_stop()
                if (self.idle_release and not self.command_pending.is_set()
                        and time.monotonic() - self.last_control_activity >= self.idle_release):
                    self.report['idle_release_seconds'] = self.idle_release
                    self.emit('motor_idle_release', seconds=self.idle_release)
                    break
                try:
                    jog = self.commands.get_nowait()
                except queue.Empty:
                    jog = None
                if jog:
                    # Stop clears queued motion immediately, but the current
                    # jog remains busy through its ending/settled encoder reads.
                    self.jog_in_progress.set()
                    try:
                        self.perform_jog(*jog)
                    finally:
                        self.jog_in_progress.clear()
                        self.last_control_activity = time.monotonic()
                    next_poll = time.monotonic() + self.heartbeat_interval
                elif not self.position_reads.empty():
                    axis = self.position_reads.get_nowait()
                    self.emit('motor_manual_position_read', axis=axis,
                              message=f'Requested fresh {AXES[axis]} position from this control app.')
                    self.read_positions(axes=(axis,))
                elif self.heartbeat_enabled and time.monotonic() >= next_poll:
                    if len(self.read_positions()) != len(AXES):
                        if self.disconnecting.is_set():
                            break
                        self.connected.clear()
                        self.emit('motor_feedback_retry', message='Encoder reply delayed; movement disabled while retrying.')
                        self.stop_both()
                        # FIELD INVESTIGATION -- COSMOS 90GT, 2026-11-16:
                        # WiFly RN-171 2.40-CEL / motor firmware 5.24 sometimes
                        # delays or batches encoder replies. Earlier code made
                        # a single missed 1.2-second idle request fatal. Failures
                        # occurred at ~11, 14, 27, 44 and 65 seconds, not a fixed
                        # 30-second lease. Readback: IdleTimer=0, FLAGS=0x7,
                        # TCPMODE=0, FlushSize=1420, FlushTimer=10, UART=19200.
                        # RSSI=-67 dBm, RTRY=2 and RTRYfail/RXdrop/RXerr=0 were
                        # snapshots, not evidence excluding transient RF faults.
                        #
                        # A two-attempt per-axis retry failed after ~173 seconds:
                        # a ~5-second gap ended in an 88-byte burst with replies
                        # for both axes, after the first was marked missing.
                        # Retrying whole pairs avoids preserving that earlier
                        # failure. A stationary 180-second live run recovered
                        # 20 episodes (30 request timeouts) with no disconnect
                        # errors. This verifies tolerance in that run; it does
                        # NOT establish or repair the underlying transport cause.
                        #
                        # Controlled failure: block movement, send zero-rate
                        # Stops, require both replies in one recovery round,
                        # then close/release if recovery fails. Never reconnect
                        # automatically or resume previous motion. AUX replies
                        # have no transaction IDs; matching late replies can be
                        # stale. Active-motion feedback loss has its own prompt
                        # stop path; these idle retries must not extend a slew.
                        #
                        # The default 30-second user-idle release is a separate
                        # interface-sharing policy. Telemetry does not renew it.
                        # Check it during recovery; an in-flight read and Stop
                        # cleanup may delay closure beyond the nominal deadline.
                        # The recovery window likewise admits a final bounded
                        # request pair, so total elapsed time can exceed 10 s.
                        #
                        # TODO(root cause): correlate timestamped TCP capture
                        # with AUX serial traffic and WiFly counters; compare
                        # direct AP / infrastructure and a second host; isolate
                        # RF loss, adapter buffering, firmware scheduling and
                        # client receive handling. Preserve raw session logs.
                        # Do not change undocumented WiFly flags or claim a
                        # firmware idle timer without evidence from those tests.
                        recovery_deadline = time.monotonic() + 10
                        recovered = {}
                        idle_expired = False
                        while time.monotonic() < recovery_deadline and not self.disconnecting.is_set():
                            if (self.idle_release and time.monotonic() - self.last_control_activity
                                    >= self.idle_release):
                                idle_expired = True
                                break
                            recovered = self.read_positions()
                            if len(recovered) == len(AXES):
                                break
                            self.disconnecting.wait(.1)
                        if self.disconnecting.is_set():
                            break
                        if idle_expired:
                            self.report['idle_release_seconds'] = self.idle_release
                            self.emit('motor_idle_release', seconds=self.idle_release)
                            break
                        if len(recovered) != len(AXES):
                            raise ConnectionError('Motor feedback timeout. Connection released. Reconnect to continue moving.')
                        self.connected.set()
                        self.emit('motor_feedback_restored', message='Both encoder replies restored; movement enabled.')
                    next_poll = time.monotonic() + self.heartbeat_interval
                self.disconnecting.wait(.02)
        except Exception as e:
            self.report['errors'].append(str(e))
            self.emit('motor_error', message=str(e))
        finally:
            self.connected.clear()
            if self.sock:
                try:
                    self.stop_both()
                    self.drain_stop_replies()
                except Exception as e:
                    self.report['errors'].append(str(e))
                    self.callback({'event': 'motor_error', 'message': str(e)})
                try:
                    self.sock.close()
                except Exception as e:
                    self.report['errors'].append(str(e))
                    self.callback({'event': 'motor_error', 'message': f'Cannot close motor socket: {e}'})
            self.endpoint_lease.release()
            directory = str(self.recorder.directory)
            try:
                self.recorder.finish(self.report)
            except Exception as e:
                self.callback({'event': 'motor_error', 'message': f'Cannot save session report: {e}'})
            self.callback({'event': 'motor_disconnected', 'log_directory': directory,
                           'stop_replies': self.report.get('cleanup_stop_replies', {})})
