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
        self.assertEqual(default.cycle_interval, .25)
        self.assertIsNone(default.rate_warning)
        status = default.status()
        self.assertEqual(status['cycle_overruns'], 0)
        self.assertEqual(status['configured_cycle_interval_ms'], 250)
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
        controller.freeze()
        self.assertTrue(controller.armed)
        self.assertIsNone(controller.target)
        self.assertTrue(controller.pause_event.is_set())
        self.assertEqual(controller.history[-1]['event'], 'motion_frozen')

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
