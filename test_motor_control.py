"""Motor service tests use an in-memory AUX adapter, never physical hardware."""
from pathlib import Path
import queue
import socket
import threading
import time
import unittest
from motor_control import MotorController, motor_packet, angle, relative_angle, version


class FakeRecorder:
    directory = Path('/tmp/simulated-motor-test')
    def __init__(self, kind):
        self.events = []
    def record(self, kind, **fields):
        self.events.append((kind, fields))
    def finish(self, report):
        self.report = report


class Adapter:
    def __init__(self):
        self.incoming = queue.Queue()
        self.sent = []
        self.positions = {0x10: 0, 0x11: 0}
        self.timeout = .05
        self.closed = False
        self.closed_with_pending_bytes = None
        self.fail_start = False

    def settimeout(self, seconds):
        self.timeout = seconds

    def sendall(self, frame):
        self.sent.append((time.monotonic(), frame))
        axis, command = frame[3], frame[4]
        if command in (0x24, 0x25):
            if frame[5] and self.fail_start:
                raise OSError('Simulated failed start send')
            if frame[5]:
                self.positions[axis] = (self.positions[axis] + (12800 if command == 0x24 else -12800)) % (1 << 24)
            data = b''
        else:
            data = b'\x05\x18' if command == 0xfe else self.positions[axis].to_bytes(3, 'big')
        body = bytes((len(data) + 3, axis, 0x20, command)) + data
        reply = b'\x3b' + body + bytes((-sum(body) & 255,))
        self.incoming.put(reply[:3])
        self.incoming.put(reply[3:])

    def recv(self, size):
        try:
            return self.incoming.get(timeout=self.timeout)
        except queue.Empty:
            raise socket.timeout() from None

    def close(self):
        self.closed_with_pending_bytes = not self.incoming.empty()
        self.closed = True


class MotorServiceTests(unittest.TestCase):
    def setUp(self):
        self.adapter = Adapter()
        self.events = queue.Queue()
        self.controller = MotorController('192.0.2.231', 2000, self.events.put,
                                          socket_factory=lambda endpoint, timeout: self.adapter,
                                          recorder_factory=FakeRecorder, route_check=lambda host: None)
        self.controller.start()
        self.wait_event('motor_values')

    def tearDown(self):
        self.controller.disconnect()
        self.controller.thread.join(3)
        self.assertFalse(self.controller.thread.is_alive())

    def wait_event(self, kind, timeout=3):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                event = self.events.get(timeout=.05)
            except queue.Empty:
                continue
            if event['event'] == kind:
                return event
        self.fail('No event ' + kind)

    def test_explicit_position_read_with_periodic_polling_off(self):
        controller = self.controller
        controller.heartbeat_enabled = False
        before = len(self.adapter.sent)
        activity = controller.last_control_activity
        self.adapter.positions[0x11] = 123456
        controller.request_position(0x11)
        self.wait_event('motor_manual_position_read')
        event = self.wait_event('motor_values')
        self.assertEqual((event['axis'], event['encoder']), (0x11, 123456))
        self.assertEqual(controller.last_control_activity, activity)
        self.assertTrue(all(frame[4] == 1 for _, frame in self.adapter.sent[before:]))

    def test_idle_delayed_reply_recovers_without_disconnect(self):
        original = self.adapter.sendall
        delayed = False
        timer = None
        retry_states = []
        original_emit = self.controller.emit
        def capture_emit(kind, **fields):
            if kind == 'motor_feedback_retry':
                retry_states.append(self.controller.connected.is_set())
            original_emit(kind, **fields)
        self.controller.emit = capture_emit
        def delayed_send(frame):
            nonlocal delayed, timer
            if frame[3:5] == b'\x11\x01' and not delayed:
                delayed = True
                timer = threading.Timer(1.4, original, args=(frame,))
                timer.start()
            else:
                original(frame)
        self.adapter.sendall = delayed_send
        try:
            self.wait_event('motor_feedback_retry', timeout=4)
            self.assertEqual(retry_states, [False])
            self.wait_event('motor_feedback_restored', timeout=5)
            self.assertTrue(self.controller.connected.is_set())
            self.assertFalse(self.controller.report['errors'])
            self.assertFalse(any(f[4] in (0x24, 0x25) and f[5] for _, f in self.adapter.sent))
        finally:
            if timer:
                timer.join(2)

    def test_persistent_idle_loss_disconnects_with_no_nonzero_motion(self):
        original = self.adapter.sendall
        self.adapter.sendall = lambda frame: None if frame[4] == 1 else original(frame)
        self.wait_event('motor_feedback_retry', timeout=5)
        self.wait_event('motor_disconnected', timeout=15)
        self.assertTrue(any('Motor feedback timeout' in e for e in self.controller.report['errors']))
        self.assertFalse(self.controller.connected.is_set())
        self.assertFalse(any(f[4] in (0x24, 0x25) and f[5] for _, f in self.adapter.sent))

    def test_keep_alive_off_stops_idle_reads_but_does_not_disable_connection(self):
        self.assertTrue(self.controller.connected.wait(2))
        self.controller.heartbeat_enabled = False
        time.sleep(.1)
        before = len(self.adapter.sent)
        time.sleep(1.2)
        self.assertEqual(len(self.adapter.sent), before)
        self.assertTrue(self.controller.connected.is_set())

    def test_never_release_ignores_elapsed_idle_time(self):
        self.assertTrue(self.controller.connected.wait(2))
        self.controller.idle_release = 0
        self.controller.last_control_activity = time.monotonic() - 600
        time.sleep(.2)
        self.assertTrue(self.controller.connected.is_set())
        self.assertTrue(self.controller.thread.is_alive())

    def test_direct_policy_validation(self):
        for seconds in (0, 10, 20, 30, 60, 120, 300):
            c = MotorController('192.0.2.231', 2000, lambda e: None,
                                idle_release=seconds, heartbeat_interval=5)
            self.assertEqual(c.heartbeat_interval, 5)
        for seconds in (0, -1, 301, float('nan')):
            with self.assertRaises(ValueError):
                MotorController('192.0.2.231', 2000, lambda e: None, heartbeat_interval=seconds)

    def test_idle_release_is_explicit_and_not_feedback_failure(self):
        self.controller.idle_release = 10
        self.controller.last_control_activity = time.monotonic() - 11
        self.wait_event('motor_idle_release')
        self.wait_event('motor_disconnected')
        self.assertFalse(self.controller.report['errors'])
        self.assertEqual(self.controller.report['idle_release_seconds'], 10)

    def test_read_values_and_timed_negative_jog_stop(self):
        self.controller.jog(0x10, -1, 6, .1)
        self.wait_event('motor_jog_finished')
        self.assertTrue(any(f[4:6] == b'\x25\x06' for _, f in self.adapter.sent))
        stops = [f for _, f in self.adapter.sent if f[4:6] == b'\x24\x00']
        self.assertEqual({f[3] for f in stops}, {0x10, 0x11})
        value = self.wait_event('motor_values')
        self.assertEqual(value['firmware'], '5.24')
        self.assertLess(value['relative'], 0)
        result = self.wait_event('motor_motion_result')
        self.assertLess(result['measured_degrees'], 0)
        self.assertEqual(result['measured_counts'], -12800)
        self.assertEqual(result['start_encoder'], 0)
        self.assertEqual(result['end_encoder'], (1 << 24) - 12800)
        self.assertTrue(result['stationary_verified'])

    def test_stop_interrupts_jog_and_does_not_queue_extra_motion(self):
        self.controller.jog(0x11, 1, 6, 2)
        self.wait_event('motor_jog_started')
        requested = time.monotonic()
        self.controller.stop()
        self.wait_event('motor_jog_finished')
        self.assertLess(time.monotonic() - requested, .6)
        self.assertEqual(sum(f[4] in (0x24, 0x25) and f[5] > 0 for _, f in self.adapter.sent), 1)

    def test_keep_alive_off_jog_refreshes_both_stale_axes_before_motion(self):
        self.assertTrue(self.controller.connected.wait(2))
        self.controller.heartbeat_enabled = False
        self.controller.position_times = {16: 0, 17: 0}
        before = len(self.adapter.sent)
        self.controller.jog(16, 1, 2, .1)
        self.wait_event('motor_motion_result')
        sent = [f for _, f in self.adapter.sent[before:]]
        start = next(i for i, f in enumerate(sent) if f[4] in (0x24, 0x25) and f[5])
        self.assertEqual({f[3] for f in sent[:start] if f[4] == 1}, {16, 17})

    def test_stale_jog_missing_other_axis_sends_no_movement(self):
        self.assertTrue(self.controller.connected.wait(2))
        self.controller.heartbeat_enabled = False
        self.controller.position_times = {16: 0, 17: 0}
        original = self.adapter.sendall
        def missing_other_axis(frame):
            if frame[3:5] == bytes((17, 1)):
                self.adapter.sent.append((time.monotonic(), frame))
                return
            original(frame)
        self.adapter.sendall = missing_other_axis
        self.controller.jog(16, 1, 2, .1)
        self.wait_event('motor_error', timeout=6)
        self.assertFalse(any(f[4] in (0x24, 0x25) and f[5] for _, f in self.adapter.sent))

    def test_jog_uses_new_baseline_instead_of_cached_idle_position(self):
        self.adapter.positions[0x10] = 100000
        self.controller.jog(0x10, 1, 6, .1)
        result = self.wait_event('motor_motion_result')
        self.assertEqual(result['start_encoder'], 100000)
        self.assertEqual(result['end_encoder'], 112800)
        self.assertEqual(result['measured_counts'], 12800)
        self.assertTrue(result['stationary_verified'])

    def test_post_stop_drift_is_reported_and_repeats_stop(self):
        original = self.adapter.sendall
        moving = False
        def drift(frame):
            nonlocal moving
            if frame[4] in (0x24, 0x25) and frame[5] > 0:
                moving = True
            if moving and frame[3:5] == b'\x10\x01':
                self.adapter.positions[0x10] += 3000
            original(frame)
        self.adapter.sendall = drift
        self.controller.jog(0x10, 1, 6, .1)
        result = self.wait_event('motor_motion_result', timeout=6)
        self.assertFalse(result['stationary_verified'])
        self.assertGreater(result['residual_degrees'], .03)
        self.wait_event('motor_disconnected')

    def test_brief_post_stop_deceleration_settles_without_disconnect(self):
        original = self.adapter.sendall
        ending_reads = 0
        moved = False
        def decelerate(frame):
            nonlocal moved, ending_reads
            if frame[4] in (0x24, 0x25) and frame[5] > 0:
                moved = True
            if moved and frame[3:5] == b'\x10\x01':
                ending_reads += 1
                if ending_reads <= 2:
                    self.adapter.positions[0x10] += 4000
            original(frame)
        self.adapter.sendall = decelerate
        self.controller.jog(0x10, 1, 9, .1)
        result = self.wait_event('motor_motion_result')
        self.assertTrue(result['stationary_verified'])
        self.assertTrue(self.controller.connected.is_set())
        self.assertTrue(self.controller.report['jogs'][0]['settling_samples'])
        self.controller.jog(0x11, -1, 9, .1)
        self.assertTrue(self.wait_event('motor_motion_result')['stationary_verified'])

    def test_failed_motion_send_still_attempts_both_stops(self):
        self.adapter.fail_start = True
        self.controller.jog(0x10, 1, 6, .1)
        self.wait_event('motor_disconnected')
        self.assertTrue(self.adapter.closed)
        last = [f for _, f in self.adapter.sent][-2:]
        self.assertEqual({f[3] for f in last}, {0x10, 0x11})
        self.assertTrue(all(f[4:6] == b'\x24\x00' for f in last))

    def test_disconnect_stops_both_axes_before_socket_close(self):
        self.controller.disconnect()
        self.wait_event('motor_disconnected')
        self.assertTrue(self.adapter.closed)
        self.assertFalse(self.adapter.closed_with_pending_bytes)
        self.assertEqual(self.controller.report['cleanup_stop_replies'], {'16': True, '17': True})
        self.assertTrue(all(f[4:6] == b'\x24\x00' for _, f in self.adapter.sent[-2:]))

    def test_invalid_or_overlapping_jogs_rejected(self):
        for args in ((0x12, 1, 6, .5), (0x10, 0, 6, .5), (0x10, 1, 10, .5), (0x10, 1, 6, 20)):
            with self.assertRaises(ValueError):
                self.controller.jog(*args)
        self.controller.jog(0x10, 1, 6, .2)
        with self.assertRaises(RuntimeError):
            self.controller.jog(0x11, 1, 6, .2)

    def test_new_jog_rejected_until_post_stop_measurements_complete(self):
        measuring = threading.Event()
        resume = threading.Event()
        original = self.controller.read_positions
        def pause_ending_read(*args, **kwargs):
            if self.controller.report['jogs'] and self.controller.active_axis is None:
                measuring.set()
                resume.wait(2)
            return original(*args, **kwargs)
        self.controller.read_positions = pause_ending_read
        self.controller.jog(0x10, 1, 6, .1)
        try:
            self.assertTrue(measuring.wait(2))
            self.assertFalse(self.controller.command_pending.is_set())
            with self.assertRaises(RuntimeError):
                self.controller.jog(0x11, 1, 6, .1)
        finally:
            resume.set()
        self.wait_event('motor_motion_result')
        self.controller.jog(0x11, -1, 6, .1)
        result = self.wait_event('motor_motion_result')
        self.assertEqual(result['axis'], 0x11)
        self.assertEqual(result['measured_counts'], -12800)

    def test_failed_socket_close_still_releases_endpoint_and_disconnects(self):
        original = self.adapter.close
        def fail_close():
            original()
            raise OSError('Simulated close failure')
        self.adapter.close = fail_close
        self.controller.disconnect()
        self.wait_event('motor_disconnected')
        self.assertIsNone(self.controller.endpoint_lease.file)
        self.assertTrue(any('close failure' in error for error in self.controller.report['errors']))

    def test_failed_report_save_still_notifies_disconnected(self):
        def fail_save(report):
            raise OSError('Simulated full disk')
        self.controller.recorder.finish = fail_save
        self.controller.disconnect()
        error = self.wait_event('motor_error')
        self.assertIn('Cannot save session report', error['message'])
        self.wait_event('motor_disconnected')
        self.assertTrue(self.adapter.closed)
        self.assertIsNone(self.controller.endpoint_lease.file)

    def test_hold_without_heartbeat_stops_automatically(self):
        self.controller.hold(0x10, 1, 6)
        self.wait_event('motor_jog_started')
        result = self.wait_event('motor_jog_finished')
        self.assertTrue(result['hold'])
        self.assertLess(result['elapsed_seconds'], .8)
        self.assertFalse(self.controller.holding)

    def test_hold_heartbeat_keeps_driving_until_release(self):
        self.controller.hold(0x11, -1, 6)
        self.wait_event('motor_jog_started')
        end = time.monotonic() + .65
        while time.monotonic() < end:
            self.controller.renew_hold()
            time.sleep(.08)
        self.assertTrue(self.controller.command_pending.is_set())
        self.controller.stop()
        self.wait_event('motor_jog_finished')
        self.assertFalse(self.controller.command_pending.is_set())


class MotorMathTests(unittest.TestCase):
    def test_tcp_only_adapter_never_enables_motor_controls(self):
        class SilentAdapter(Adapter):
            def sendall(self, frame):
                self.sent.append((time.monotonic(), frame))
                self.incoming.put(b'*CLOS*')
        adapter = SilentAdapter()
        events = []
        controller = MotorController('192.0.2.231', 2000, events.append,
                                     socket_factory=lambda *args: adapter,
                                     recorder_factory=FakeRecorder, route_check=lambda host: None)
        controller.start()
        controller.thread.join(12)
        self.assertFalse(controller.thread.is_alive())
        self.assertFalse(controller.connected.is_set())
        self.assertNotIn('motor_connected', [e['event'] for e in events])
        self.assertTrue(any('NOT verified' in e.get('message', '') for e in events))
        self.assertFalse(any(f[4] in (0x24, 0x25) and f[5] for _, f in adapter.sent))

    def test_packets_and_angles(self):
        for command in (1, 0xfe, 0x24, 0x25):
            frame = motor_packet(0x10, command, b'\x06' if command in (0x24, 0x25) else b'')
            self.assertEqual(sum(frame[1:]) % 256, 0)
        self.assertEqual(angle(1 << 22), 90)
        self.assertAlmostEqual(relative_angle(10, (1 << 24) - 10), 20 * 360 / (1 << 24))
        self.assertEqual(version(b'\x05\x18'), '5.24')


if __name__ == '__main__':
    unittest.main()


class TimedJogRangeTests(unittest.TestCase):
    def test_five_second_command_is_bounded(self):
        controller = MotorController('192.0.2.231', 2000, lambda event: None)
        controller.connected.set()
        controller.jog(16, 1, 2, 5)
        self.assertEqual(controller.commands.get_nowait(), (16, 1, 2, 5))
        controller.command_pending.clear()
        for seconds in (5.01, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                controller.jog(16, 1, 2, seconds)
