import time
import unittest
from unittest.mock import MagicMock, patch

try:
    import cv2
    import numpy as np
    from tools.vision_dashboard import (BrightObjectTracker, VisionDashboardEngine,
                                        create_feature_tracker,
                                        hil_marker_from_status, hil_target_from_pixel,
                                        learn_pixel_mount_transform,
                                        learned_mount_delta, normalize_camera_controls,
                                        wrapped_encoder_delta)
    VISION_DEPS_AVAILABLE = True
except ImportError:
    cv2 = np = BrightObjectTracker = VisionDashboardEngine = create_feature_tracker = hil_marker_from_status = hil_target_from_pixel = normalize_camera_controls = None
    VISION_DEPS_AVAILABLE = False


@unittest.skipUnless(VISION_DEPS_AVAILABLE, 'optional OpenCV vision dependencies are not installed')
class VisionDashboardTests(unittest.TestCase):
    def setUp(self):
        # Unit tests must never enumerate or open the Pi's physical CSI camera
        # while the live tracker owns it.  Hardware availability is covered by
        # the explicit operator-run camera probe, not this isolated suite.
        self.camera_inventory = patch(
            'tools.vision_dashboard.enumerate_cameras', return_value=[])
        self.camera_inventory.start()

    def tearDown(self):
        self.camera_inventory.stop()

    @staticmethod
    def frame(points):
        image = np.zeros((200, 300), dtype=np.uint8)
        for x, y, radius, value in points:
            cv2.circle(image, (x, y), radius, value, -1)
        return image

    def test_multiple_objects_keep_ids_and_can_be_selected(self):
        tracker = BrightObjectTracker(threshold=100, match_distance=30)
        first = tracker.detect(self.frame(((50, 60, 6, 240), (220, 140, 9, 220))))
        self.assertEqual(len(first), 2)
        left = min(first, key=lambda item: item['x'])
        right = max(first, key=lambda item: item['x'])
        self.assertEqual(len(left['bbox']), 4)
        tracker.select_at(left['x'], left['y'])
        self.assertEqual(tracker.selected_id, left['track_id'])

        second = tracker.detect(self.frame(((54, 62, 6, 240), (216, 137, 9, 220))))
        self.assertEqual(min(second, key=lambda item: item['x'])['track_id'],
                         left['track_id'])
        tracker.select(right['track_id'])
        third = tracker.detect(self.frame(((56, 64, 6, 240), (214, 135, 9, 220))))
        self.assertTrue(next(item for item in third
                             if item['track_id'] == right['track_id'])['selected'])

    def test_threshold_and_missing_selection_guards(self):
        tracker = BrightObjectTracker()
        with self.assertRaises(ValueError):
            tracker.set_threshold(0)
        with self.assertRaises(ValueError):
            tracker.select(99)
        with self.assertRaises(ValueError):
            tracker.select_at(10, 10)

    def test_dark_objects_are_detected_and_polarity_resets_selection(self):
        image = np.full((200, 300), 245, dtype=np.uint8)
        cv2.circle(image, (80, 90), 8, 10, -1)
        tracker = BrightObjectTracker(threshold=100, polarity='dark')
        objects = tracker.detect(image)
        self.assertEqual(len(objects), 1)
        self.assertAlmostEqual(objects[0]['x'], 80, delta=1)
        tracker.select(objects[0]['track_id'])
        tracker.set_polarity('bright')
        self.assertIsNone(tracker.selected_id)
        with self.assertRaises(ValueError):
            tracker.set_polarity('invalid')

    def test_hardware_loop_pixel_and_encoder_mapping(self):
        azimuth, elevation = hil_target_from_pixel(300, 0, 400, 200, 4)
        self.assertAlmostEqual(azimuth, 181.8)
        self.assertAlmostEqual(elevation, 3.6)
        marker = hil_marker_from_status({
            'positions': {
                'Azimuth': {'reference_degrees': azimuth},
                'Elevation': {'reference_degrees': elevation},
            }}, 400, 200, 4)
        self.assertAlmostEqual(marker['x'], 300)
        self.assertAlmostEqual(marker['y'], 0)
        self.assertIsNone(hil_marker_from_status({'positions': {}}, 400, 200, 4))

    def test_learned_camera_transform_handles_rotation_and_inversion(self):
        calibration = learn_pixel_mount_transform((0, -10), (20, 0), (.02, .02))
        azimuth, elevation = learned_mount_delta(20, -10, calibration)
        self.assertAlmostEqual(azimuth, -.02)
        self.assertAlmostEqual(elevation, -.02)
        self.assertEqual(wrapped_encoder_delta(3, (1 << 24) - 2), 5)
        self.assertEqual(wrapped_encoder_delta((1 << 24) - 2, 3), -5)
        coarse = learn_pixel_mount_transform((40, 5), (-3, 35), (.5, .48))
        self.assertEqual(coarse['measured_step_degrees'], [.5, .48])

    def test_learned_camera_transform_rejects_weak_or_parallel_probes(self):
        with self.assertRaises(ValueError):
            learn_pixel_mount_transform((1, 0), (0, 8), .01)
        with self.assertRaises(ValueError):
            learn_pixel_mount_transform((10, 0), (20, 0), .01)

    def test_selected_blinking_track_keeps_identity_by_predicted_motion(self):
        tracker = BrightObjectTracker(threshold=100, match_distance=35, max_missed=10)
        first = tracker.detect(self.frame(((40, 80, 5, 240), (240, 80, 9, 230))))
        selected = min(first, key=lambda item: item['x'])
        tracker.select(selected['track_id'])
        tracker.detect(self.frame(((50, 80, 5, 240), (240, 80, 9, 230))))
        tracker.detect(self.frame(((240, 80, 9, 230),)))
        tracker.detect(self.frame(((240, 80, 9, 230),)))
        returned = tracker.detect(self.frame(((78, 80, 5, 180), (240, 80, 9, 230))))
        reacquired = min(returned, key=lambda item: item['x'])
        self.assertEqual(reacquired['track_id'], selected['track_id'])
        self.assertTrue(reacquired['selected'])

    def test_synthetic_blink_timing_validation(self):
        engine = VisionDashboardEngine(source='synthetic')
        engine.set_synthetic_blink(True, .5, 1, .5, 4)
        self.assertTrue(engine.synthetic_blink['enabled'])
        with self.assertRaises(ValueError):
            engine.set_synthetic_blink(True, 1, 2, 1, 4)

    def test_camera_controls_are_normalized_and_enforced(self):
        controls = normalize_camera_controls({
            'ExposureValue': (-8.0, 8.0, 0.0),
            'ExposureTime': (1, 66666, 20000),
            'AnalogueGain': (1.0, 16.0, 1.0),
            'ScalerCrop': ((0, 0, 0, 0), (10, 20, 30, 40), None),
        })
        self.assertEqual(controls['ExposureTime']['maximum'], 66666)
        self.assertEqual(controls['ScalerCrop']['maximum'], [10, 20, 30, 40])
        engine = VisionDashboardEngine(source='synthetic')
        engine.camera_control_ranges = controls
        engine.set_camera_exposure(False, 8, 66666, 16)
        self.assertEqual(engine.camera_manual_exposure_us, 66666)
        self.assertEqual(engine.camera_manual_gain, 16)
        with self.assertRaisesRegex(ValueError, 'gain 1–16'):
            engine.set_camera_exposure(False, 0, 10000, 16.1)

    def test_bridge_service_restart_reconnects_motor_link(self):
        engine = VisionDashboardEngine(source='synthetic', hil_host='127.0.0.1')
        engine.hil_controller = MagicMock()
        engine.hil_controller.status.return_value = {'connection': 'live'}
        completed = MagicMock(returncode=0, stdout='', stderr='')
        with (patch('tools.vision_dashboard.subprocess.run', return_value=completed) as run,
              patch.object(engine, '_poll_bridge_management',
                           return_value={'verified_proxy': True}),
              patch.object(engine, 'reconnect_bridge_client') as reconnect):
            engine.restart_bridge_service()
            engine.bridge_service_restart_thread.join(2)
        self.assertFalse(engine.bridge_service_restart_in_progress)
        self.assertEqual(engine.bridge_service_restart_count, 1)
        self.assertIn('recovered successfully', engine.bridge_service_restart_result)
        reconnect.assert_called_once_with()
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], [
            '/usr/bin/systemctl', '--user', 'restart', 'celestron-bridge.service'])

    def test_feature_roi_mode_and_bounds(self):
        try:
            self.assertIsNotNone(create_feature_tracker('CSRT', lock_size=True))
        except RuntimeError:
            self.skipTest('OpenCV contrib CSRT tracker is not installed')
        engine = VisionDashboardEngine(source='synthetic')
        engine.set_tracking_mode('feature-roi', 'CSRT')
        self.assertTrue(engine.feature_lock_size)
        engine.telemetry.update(width=640, height=480)
        engine.set_feature_roi(.2, .3, .1, .15)
        self.assertEqual(len(engine.feature_roi_pending), 4)
        with self.assertRaises(ValueError):
            engine.set_feature_roi(.95, .3, .1, .15)

    def test_severely_clipped_feature_is_not_a_credible_lock(self):
        engine = VisionDashboardEngine(source='synthetic', ui_mode='live',
                                       width=640, height=480)
        engine.telemetry.update(width=640, height=480)
        clipped = {'x': 2.5, 'y': 31.0, 'bbox': [-36, 2, 77, 58]}
        self.assertIsNone(engine._filter_selected(clipped))
        mildly_clipped = {'x': 20, 'y': 31, 'bbox': [-5, 2, 50, 58]}
        self.assertIsNotNone(engine._filter_selected(mildly_clipped))

    def test_feature_similarity_rejects_blank_or_unrelated_roi(self):
        engine = VisionDashboardEngine(source='synthetic', ui_mode='live')
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(image, (20, 20), (39, 39), 255, 2)
        engine.feature_reference_patch = image[20:40, 20:40].copy()
        self.assertGreater(engine._feature_similarity(image, (20, 20, 20, 20)), .95)
        self.assertEqual(engine._feature_similarity(image, (50, 50, 20, 20)), 0.0)

    def test_feature_credibility_rejects_correlation_loss_and_teleport(self):
        engine = VisionDashboardEngine(source='synthetic', ui_mode='live')
        engine.feature_similarity_baseline = .90
        engine.feature_last_confirmed = (30, 30, 10.0)
        self.assertFalse(engine._feature_candidate_is_credible(
            (22, 22, 16, 16), .20, now=10.1))
        self.assertFalse(engine._feature_candidate_is_credible(
            (250, 150, 16, 16), .95, now=10.1))
        self.assertTrue(engine._feature_candidate_is_credible(
            (26, 24, 16, 16), .80, now=10.1))

    def test_lost_feature_requires_two_consistent_matches_before_reacquiring(self):
        engine = VisionDashboardEngine(source='synthetic', ui_mode='live')
        reference = np.zeros((18, 22), dtype=np.uint8)
        cv2.line(reference, (2, 2), (19, 15), 240, 2)
        cv2.circle(reference, (6, 13), 3, 160, -1)
        frame = np.full((120, 180), 18, dtype=np.uint8)
        frame[62:80, 103:125] = reference
        feature_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        engine.feature_reference_patch = reference.copy()
        engine.feature_selected_id = 1
        engine.feature_reacquiring = True
        tracker = MagicMock()
        tracker.init.return_value = True
        with patch('tools.vision_dashboard.create_feature_tracker',
                   return_value=tracker):
            self.assertIsNone(engine._attempt_feature_reacquisition(
                frame, feature_frame))
            detection = engine._attempt_feature_reacquisition(
                frame, feature_frame)
        self.assertIsNotNone(detection)
        self.assertAlmostEqual(detection['x'], 114, delta=1)
        self.assertAlmostEqual(detection['y'], 71, delta=1)
        self.assertFalse(engine.feature_reacquiring)
        self.assertIs(engine.feature_tracker, tracker)
        self.assertEqual(tracker.init.call_count, 1)

    def test_live_mode_click_lock_and_hold_anchor(self):
        engine = VisionDashboardEngine(source='synthetic', ui_mode='demo')
        engine.telemetry.update(width=640, height=480)
        engine.set_operating_mode('live')
        self.assertEqual(engine.ui_mode, 'live')
        self.assertEqual(engine.desired_source, 'v4l2')
        engine.set_aim_mode('hold')
        engine.set_feature_point(.25, .75, .12)
        self.assertEqual(engine.tracking_mode, 'feature-roi')
        self.assertEqual(engine.aim_point, (.25, .75))
        self.assertEqual(engine.feature_roi_pending, (122, 331, 77, 58))
        engine.set_image_orientation('reversed', 'normal')
        self.assertTrue(engine.image_horizontal_reversed)
        engine.reset_live_view()
        self.assertIsNone(engine.feature_selected_id)
        self.assertEqual(engine.aim_mode, 'center')
        self.assertFalse(engine.image_horizontal_reversed)
        self.assertTrue(engine.camera_reopen_requested)

    def test_repeated_live_clicks_do_not_shrink_frame_dimensions(self):
        try:
            create_feature_tracker('CSRT', lock_size=True)
        except RuntimeError:
            self.skipTest('OpenCV contrib CSRT tracker is not installed')
        engine = VisionDashboardEngine(source='synthetic', ui_mode='live',
                                       width=640, height=480, fps=10)
        try:
            engine.set_feature_point(.30, .45, .12)
            engine.start()
            for _click in range(3):
                previous_frame = engine.snapshot().get('frame', -1)
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    snapshot = engine.snapshot()
                    if snapshot.get('frame', -1) >= previous_frame + 2:
                        break
                    time.sleep(.02)
                self.assertEqual(snapshot.get('width'), 640)
                self.assertEqual(snapshot.get('height'), 480)
                engine.set_feature_point(.30, .45, .12)
        finally:
            engine.stop()

    def test_outer_loop_presets_and_envelope_clamp(self):
        engine = VisionDashboardEngine(source='synthetic', hil_envelope_degrees=5)
        engine.set_hil_control_strategy('balanced')
        engine.hil_target_rate = {'azimuth_deg_s': 5, 'elevation_deg_s': 5}
        command = engine._apply_outer_loop((184.5, 4.5), {
            'azimuth_degrees': 180, 'elevation_degrees': 0})
        self.assertLessEqual(abs((command[0] - 180 + 180) % 360 - 180), 4.75)
        self.assertLessEqual(abs(command[1]), 4.75)
        with self.assertRaises(ValueError):
            engine.set_hil_control_strategy('custom', 2, .2)
        engine.set_hil_control_strategy('custom', 0, 0, .1)
        held = engine._apply_outer_loop((180.05, .05), {
            'azimuth_degrees': 180, 'elevation_degrees': 0})
        self.assertAlmostEqual(held[0], 180)
        self.assertAlmostEqual(held[1], 0)


if __name__ == '__main__':
    unittest.main()
