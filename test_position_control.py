import time
import unittest
from unittest.mock import patch
from position_control import PositionController, target_packet
from test_motor_control import Adapter, FakeRecorder
from tracking_stream import TrackingStream


class PositionTests(unittest.TestCase):
    def test_target_packet_scoped_and_checksummed(self):
        packet = target_packet(0x10, 12345)
        self.assertEqual(packet[3:5], b'\x10\x02')
        self.assertEqual(sum(packet[1:]) & 255, 0)
        for axis, count in ((0xb0, 1), (0x10, -1), (0x11, 1 << 24)):
            with self.assertRaises(ValueError):
                target_packet(axis, count)

    def test_dispatch_rate_defaults_and_warning(self):
        default = PositionController('192.0.2.230')
        self.assertEqual(default.dispatch_rate_hz, 4)
        self.assertEqual(default.feedback_timeout, 10)
        self.assertEqual(default.cycle_interval, .25)
        self.assertIsNone(default.rate_warning)
        status = default.status()
        self.assertTrue(status['stop_on_divergence'])
        optical = PositionController('192.0.2.230', stop_on_divergence=False)
        self.assertFalse(optical.status()['stop_on_divergence'])
        self.assertEqual(status['cycle_overruns'], 0)
        self.assertEqual(status['configured_cycle_interval_ms'], 250)
        self.assertEqual(status['configured_feedback_poll_rate_hz'], 4)
        self.assertEqual(status['feedback_timeout_seconds'], 10)
        self.assertEqual(status['target_change_threshold_degrees'], .001)
        default.set_feedback_timeout(30)
        self.assertEqual(default.status()['feedback_timeout_seconds'], 30)
        for timeout in (2.9, 60.1, float('nan')):
            with self.assertRaises(ValueError):
                default.set_feedback_timeout(timeout)
        faster = PositionController('192.0.2.230', dispatch_rate_hz=8)
        self.assertEqual(faster.cycle_interval, .125)
        self.assertIn('unverified', faster.rate_warning)
        stress = PositionController('192.0.2.230', dispatch_rate_hz=20)
        self.assertIn('STRESS TEST', stress.rate_warning)
        stress.set_dispatch_rate(4)
        self.assertIsNone(stress.rate_warning)
        self.assertEqual(stress.cycle_interval, .25)
        for rate in (0, 20.1, float('nan')):
            with self.assertRaises(ValueError):
                PositionController('192.0.2.230', dispatch_rate_hz=rate)
        for threshold in (-.001, .101, float('nan')):
            with self.assertRaises(ValueError):
                PositionController('192.0.2.230', target_change_threshold_degrees=threshold)
        default.set_feedback_poll_rate(8)
        self.assertEqual(default.status()['configured_feedback_poll_rate_hz'], 8)
        for rate in (0, 20.1, float('nan')):
            with self.assertRaises(ValueError):
                default.set_feedback_poll_rate(rate)

    def test_worker_gates_motion_and_uses_latest_target(self):
        adapter = Adapter()
        original = adapter.sendall
        def send(frame):
            if frame[4] == 2:
                adapter.positions[frame[3]] = int.from_bytes(frame[5:-1], 'big')
            original(frame)
        adapter.sendall = send
        controller = PositionController('192.0.2.230')
        with patch('position_control.socket.create_connection', return_value=adapter), \
             patch('position_control.Recorder', FakeRecorder), patch('position_control.check_direct_route'):
            controller.start()
            try:
                deadline = time.monotonic() + 3
                while controller.connection != 'live' and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertEqual(controller.connection, 'live')
                controller.reference(0, 0)
                controller.submit(.05, .05, 1, time.time())
                time.sleep(.3)
                self.assertFalse(any(frame[4] == 2 for _, frame in adapter.sent))
                controller.arm()
                controller.submit(.1, .1, 2, time.time())
                deadline = time.monotonic() + 2
                while not any(frame[4] == 2 for _, frame in adapter.sent) and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(any(frame[4] == 2 for _, frame in adapter.sent))
                controller.slew_rate(0x10, 1, 1)
                deadline = time.monotonic() + 2
                while (not any(frame[3] == 0x10 and frame[4] == 0x24 and frame[5] == 1
                               for _, frame in adapter.sent)
                       and time.monotonic() < deadline):
                    time.sleep(.01)
                self.assertTrue(any(frame[3] == 0x10 and frame[4] == 0x24 and frame[5] == 1
                                    for _, frame in adapter.sent))
                controller.freeze()
                self.assertTrue(controller.armed)
                with self.assertRaises(ValueError):
                    controller.submit(10, 0, 3, time.time())
                with self.assertRaises(ValueError):
                    controller.submit(.2, .2, 1, time.time())
                controller.pause()
                self.assertFalse(controller.armed)
            finally:
                controller.stop()
                controller.thread.join(3)
        self.assertFalse(controller.thread.is_alive())

    def test_freeze_stops_target_but_preserves_armed_link(self):
        controller = PositionController('192.0.2.230')
        controller.armed = True
        controller.target = {'test': True}
        self.assertTrue(controller.freeze())
        self.assertTrue(controller.armed)
        self.assertIsNone(controller.target)
        self.assertTrue(controller.pause_event.is_set())
        self.assertEqual(controller.history[-1]['event'], 'motion_frozen')
        controller.pause_event.clear()
        controller.motion = 'paused'
        before = len(controller.history)
        self.assertFalse(controller.freeze())
        self.assertFalse(controller.pause_event.is_set())
        self.assertEqual(len(controller.history), before)

    def test_target_confirmation_is_observational_and_never_gates(self):
        controller = PositionController('192.0.2.230')
        target = {0x10: 10000, 0x11: 20000}
        controller.positions = {0x10: 0, 0x11: 0}
        controller.armed = True
        controller.target = {'still': 'active'}
        controller.observe_target_progress(target)
        self.assertEqual(controller.target_confirmation['state'], 'moving')
        self.assertFalse(controller.target_confirmation['gating'])
        self.assertTrue(controller.armed)
        self.assertEqual(controller.target, {'still': 'active'})

        controller.positions = dict(target)
        controller.observe_target_progress(target)
        self.assertEqual(controller.target_confirmation['state'], 'confirmed')
        self.assertFalse(controller.target_confirmation['gating'])
        self.assertTrue(controller.armed)
        self.assertEqual(controller.target, {'still': 'active'})

    def test_missing_azimuth_target_ack_does_not_starve_elevation(self):
        adapter = Adapter()
        original = adapter.sendall

        def omit_target_ack(frame):
            # Apply both targets to the simulated motors, but deliberately do
            # not acknowledge MC_GOTO_FAST. Real encoder reads still reply.
            if frame[4] == 0x02:
                adapter.sent.append((time.monotonic(), frame))
                adapter.positions[frame[3]] = int.from_bytes(frame[5:-1], 'big')
                return
            original(frame)

        adapter.sendall = omit_target_ack
        controller = PositionController('192.0.2.230')
        with patch('position_control.socket.create_connection', return_value=adapter), \
             patch('position_control.Recorder', FakeRecorder), patch('position_control.check_direct_route'):
            controller.start()
            try:
                deadline = time.monotonic() + 3
                while controller.connection != 'live' and time.monotonic() < deadline:
                    time.sleep(.01)
                controller.reference(0, 0)
                controller.arm()
                controller.submit(.1, -.1, 1, time.time())
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    axes = {frame[3] for _, frame in adapter.sent if frame[4] == 0x02}
                    if axes == {0x10, 0x11}:
                        break
                    time.sleep(.01)
                self.assertEqual(axes, {0x10, 0x11})
                self.assertEqual(controller.reconnections, 0)
            finally:
                controller.stop()
                controller.thread.join(3)

    def test_delayed_encoder_replies_do_not_gate_next_axis_targets(self):
        adapter = Adapter()
        controller = PositionController('192.0.2.230')
        with patch('position_control.socket.create_connection', return_value=adapter), \
             patch('position_control.Recorder', FakeRecorder), patch('position_control.check_direct_route'):
            controller.start()
            try:
                deadline = time.monotonic() + 3
                while controller.connection != 'live' and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertEqual(controller.connection, 'live')
                controller.reference(0, 0)
                controller.arm()

                original = adapter.sendall
                def suppress_encoder_replies(frame):
                    if frame[4] == 0x01:
                        adapter.sent.append((time.monotonic(), frame))
                        return
                    if frame[4] == 0x02:
                        adapter.positions[frame[3]] = int.from_bytes(frame[5:-1], 'big')
                    original(frame)
                adapter.sendall = suppress_encoder_replies

                # Force the observational watchdog stale without waiting for
                # the production timeout. Staleness must not gate dispatch.
                controller.feedback_timeout = .01
                time.sleep(.03)
                self.assertFalse(controller.status()['feedback_live'])

                submitted = time.monotonic()
                controller.submit(.2, -.2, 1, time.time())
                deadline = submitted + .6
                target_frames = []
                while time.monotonic() < deadline:
                    target_frames = [(when, frame) for when, frame in adapter.sent
                                     if frame[4] == 0x02]
                    if len(target_frames) >= 2:
                        break
                    time.sleep(.005)
                self.assertEqual([frame[3] for _, frame in target_frames[-2:]],
                                 [0x10, 0x11])
                self.assertLess(target_frames[-1][0] - submitted, .4)
                self.assertLess(target_frames[-1][0] - target_frames[-2][0], .02)
                self.assertEqual(controller.reconnections, 0)
            finally:
                controller.stop()
                controller.thread.join(3)

    def test_manual_jog_requires_only_live_encoders_not_reference_or_arm(self):
        adapter = Adapter()
        controller = PositionController('192.0.2.230')
        with patch('position_control.socket.create_connection', return_value=adapter), \
             patch('position_control.Recorder', FakeRecorder), patch('position_control.check_direct_route'):
            controller.start()
            try:
                deadline = time.monotonic() + 3
                while controller.connection != 'live' and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertEqual(controller.connection, 'live')
                self.assertIsNone(controller.offsets)
                self.assertFalse(controller.armed)
                controller.manual_jog(0x10, 1, 2, .5)
                deadline = time.monotonic() + 2
                while (not any(frame[3] == 0x10 and frame[4] == 0x24 and frame[5] == 2
                               for _, frame in adapter.sent)
                       and time.monotonic() < deadline):
                    time.sleep(.01)
                self.assertTrue(any(frame[3] == 0x10 and frame[4] == 0x24 and frame[5] == 2
                                    for _, frame in adapter.sent))
                controller.manual_stop()
                deadline = time.monotonic() + 2
                while (not any(frame[4] == 0x24 and frame[5] == 0
                               for _, frame in adapter.sent[-8:])
                       and time.monotonic() < deadline):
                    time.sleep(.01)
                self.assertTrue(any(frame[4] == 0x24 and frame[5] == 0
                                    for _, frame in adapter.sent[-8:]))
            finally:
                controller.stop()
                controller.thread.join(3)
        self.assertFalse(controller.thread.is_alive())

    def test_subthreshold_target_changes_are_coalesced_per_axis(self):
        adapter = Adapter()
        controller = PositionController('192.0.2.230',
                                        target_change_threshold_degrees=.01)
        with patch('position_control.socket.create_connection', return_value=adapter), \
             patch('position_control.Recorder', FakeRecorder), patch('position_control.check_direct_route'):
            controller.start()
            try:
                deadline = time.monotonic() + 3
                while controller.connection != 'live' and time.monotonic() < deadline:
                    time.sleep(.01)
                controller.reference(0, 0)
                controller.arm()
                controller.submit(.1, .1, 1, time.time())
                deadline = time.monotonic() + 2
                while len([f for _, f in adapter.sent if f[4] == 0x02]) < 2 and time.monotonic() < deadline:
                    time.sleep(.01)
                first_count = len([f for _, f in adapter.sent if f[4] == 0x02])
                controller.submit(.105, .105, 2, time.time())
                time.sleep(.4)
                self.assertEqual(len([f for _, f in adapter.sent if f[4] == 0x02]),
                                 first_count)
                controller.submit(.12, .105, 3, time.time())
                deadline = time.monotonic() + 2
                while len([f for _, f in adapter.sent if f[4] == 0x02]) == first_count and time.monotonic() < deadline:
                    time.sleep(.01)
                new_frames = [f for _, f in adapter.sent if f[4] == 0x02][first_count:]
                self.assertEqual([frame[3] for frame in new_frames], [0x10])
            finally:
                controller.stop()
                controller.thread.join(3)

    def test_stream_rejects_stale_overrate_and_stops_on_fault(self):
        tracker = TrackingStream()
        now = time.time()
        tracker.submit(dict(sequence=1, utc=now, azimuth_deg=359.9, elevation_deg=40), now)
        tracker.submit(dict(sequence=2, utc=now+.1, azimuth_deg=.1, elevation_deg=40), now+.1)
        tracker.step(.02, now+.1)
        self.assertLessEqual(abs(tracker.rates[0]), 2.5)
        with self.assertRaises(ValueError):
            tracker.submit(dict(sequence=3, utc=now+.2, azimuth_deg=30, elevation_deg=40), now+.2)
        tracker.step(.02, now+.2)
        self.assertEqual(tracker.rates, [0, 0])
        self.assertEqual(tracker.state, 'REJECTED_TARGET')
        tracker.submit(dict(sequence=4, utc=now+.3, azimuth_deg=.2, elevation_deg=40), now+.3)
        tracker.step(.02, now+1)
        self.assertEqual(tracker.state, 'STALE_STREAM_STOPPED')


if __name__ == '__main__':
    unittest.main()
