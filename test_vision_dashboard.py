import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    import cv2
    import numpy as np
    from tools.vision_dashboard import (BrightObjectTracker, VisionDashboardEngine,
                                        create_feature_tracker,
                                        estimate_backlash_breakpoint,
                                        hil_marker_from_status, hil_target_from_pixel,
                                        learn_pixel_mount_transform,
                                        learned_mount_delta, transfer_mount_delta,
                                        normalize_camera_controls,
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

    def test_dashboard_avoids_internal_engineering_jargon(self):
        dashboard = (Path(__file__).parent / 'tools' /
                     'vision_dashboard_static' / 'index.html').read_text(
                         encoding='utf-8').lower()
        for phrase in ('artificial optical plant', 'optical plant',
                       'hardware-loop test', 'outer-loop', 'plant offset'):
            self.assertNotIn(phrase, dashboard)
        self.assertIn("exposurevalue').oninput", dashboard)
        self.assertIn("e.target.dataset.pending='1'", dashboard)
        self.assertIn("['manualexposure','manualgain'].foreach", dashboard)
        self.assertIn('manual exposure to apply', dashboard)
        self.assertIn('editable settings, not readouts', dashboard)
        self.assertIn('mirror left / right', dashboard)
        self.assertIn('flip top / bottom', dashboard)
        self.assertIn('maximum tracking slew rate', dashboard)
        self.assertIn('rate 1 — approximately 0.0023°/s', dashboard)
        self.assertIn('rate 9 — approximately 2.91°/s', dashboard)
        self.assertIn('zero encoder marker at center', dashboard)
        self.assertIn('manual — center here', dashboard)
        self.assertIn("rate:number(document.getelementbyid('manualjogspeed').value)", dashboard)

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

    def test_transfer_mount_delta_inverts_known_artificial_plant(self):
        coefficients = {
            'azimuth_x_px_per_degree': 180,
            'azimuth_y_px_per_degree': 25,
            'elevation_x_px_per_degree': -18,
            'elevation_y_px_per_degree': -150,
        }
        azimuth, elevation = transfer_mount_delta(81, -62.5, coefficients)
        self.assertAlmostEqual(azimuth, .5)
        self.assertAlmostEqual(elevation, .5)
        with self.assertRaisesRegex(ValueError, 'singular'):
            transfer_mount_delta(1, 1, {
                'azimuth_x_px_per_degree': 1,
                'azimuth_y_px_per_degree': 2,
                'elevation_x_px_per_degree': 2,
                'elevation_y_px_per_degree': 4,
            })

    def test_learned_camera_transform_rejects_weak_or_parallel_probes(self):
        with self.assertRaises(ValueError):
            learn_pixel_mount_transform((1, 0), (0, 8), .01)
        with self.assertRaises(ValueError):
            learn_pixel_mount_transform((10, 0), (20, 0), .01)

    def test_backlash_breakpoint_uses_repeatable_flat_then_linear_trials(self):
        trials = []
        for offset in (-.002, .0, .003):
            samples = []
            for index in range(11):
                encoder = index * .02
                noise = (.15, -.10, .05)[index % 3]
                pixels = max(0, (encoder - (.06 + offset)) * 120) + noise
                samples.append((encoder, max(0, pixels)))
            trials.append(samples)
        result = estimate_backlash_breakpoint(trials)
        self.assertEqual(result['accepted_trials'], 3)
        self.assertAlmostEqual(result['degrees'], .06, delta=.025)
        self.assertLess(result['spread_degrees'], .015)

    def test_backlash_breakpoint_rejects_hunting_jump_trial(self):
        good = [(index * .02, max(0, (index * .02 - .06) * 120))
                for index in range(11)]
        jumped = [(index * .02, value) for index, value in enumerate(
            (0, .1, .2, 18, 2, 17, 3, 19, 4, 20, 5))]
        with self.assertRaisesRegex(ValueError, '3 consistent'):
            estimate_backlash_breakpoint([good, good, jumped])

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

    def test_initial_snapshot_is_safe_before_first_camera_frame(self):
        engine = VisionDashboardEngine(
            source='picamera2', device='picamera2:0', ui_mode='live')
        snapshot = engine.snapshot()
        self.assertEqual(snapshot['frame'], -1)
        self.assertEqual(snapshot['state'], 'STARTING')
        self.assertEqual(snapshot['objects'], [])
        self.assertIsNone(snapshot['selected_track_id'])
        self.assertFalse(snapshot['recording'])

        dashboard = (Path(__file__).parent / 'tools' /
                     'vision_dashboard_static' / 'index.html').read_text(
                         encoding='utf-8')
        self.assertIn('Array.isArray(data.objects)?data.objects:[]', dashboard)
        self.assertIn('<img id="video" alt="Live tracking camera">', dashboard)
        self.assertIn("window.addEventListener('load'", dashboard)
        self.assertIn('CAMERA UNAVAILABLE — RESUME DISABLED', dashboard)

    def test_hybrid_mode_keeps_camera_and_supplies_known_target(self):
        engine = VisionDashboardEngine(
            source='picamera2', device='picamera2:0', ui_mode='live',
            width=640, height=480)
        engine.set_operating_mode('hybrid')
        self.assertEqual(engine.ui_mode, 'hybrid')
        self.assertEqual(engine.desired_source, 'picamera2')
        frame = np.zeros((480, 640), dtype=np.uint8)
        target = engine._hybrid_target(frame, 10)
        self.assertTrue(target['synthetic'])
        self.assertEqual(target['track_id'], 1)
        self.assertGreater(frame.max(), 0)

    def test_transfer_function_uses_real_encoder_delta_to_move_dot(self):
        engine = VisionDashboardEngine(
            source='synthetic', ui_mode='transfer', width=640, height=480,
            hil_host='127.0.0.1')
        engine.hil_controller = MagicMock()
        engine._transfer_path = MagicMock(return_value=(0.0, 0.0))
        origin = {
            'connection': 'live', 'positions': {
                'Azimuth': {'raw_degrees': 100.0},
                'Elevation': {'raw_degrees': 30.0}}}
        moved = {
            'connection': 'live', 'positions': {
                'Azimuth': {'raw_degrees': 101.0},
                'Elevation': {'raw_degrees': 32.0}}}
        engine.hil_controller.status.side_effect = [origin, moved]
        first = engine._transfer_target(np.zeros((480, 640), dtype=np.uint8))
        second = engine._transfer_target(np.zeros((480, 640), dtype=np.uint8))
        expected_x = 320 - (180 * 1 + -18 * 2)
        expected_y = 240 - (25 * 1 + -150 * 2)
        self.assertAlmostEqual(first['x'], 320)
        self.assertAlmostEqual(first['y'], 240)
        self.assertAlmostEqual(second['x'], expected_x)
        self.assertAlmostEqual(second['y'], expected_y)
        self.assertEqual(engine.transfer_state['encoder_delta_degrees'], [1.0, 2.0])

    def test_transfer_function_rejects_singular_matrix(self):
        engine = VisionDashboardEngine(source='synthetic', ui_mode='transfer')
        with self.assertRaisesRegex(ValueError, 'singular'):
            engine.set_transfer_function(100, 50, 200, 100)
        engine.set_transfer_function(180, 25, -18, -150)
        self.assertEqual(engine.transfer_function['elevation_y_px_per_degree'], -150)

    def test_transfer_mode_has_separate_mapping_profile_key(self):
        engine = VisionDashboardEngine(source='synthetic', ui_mode='live')
        camera_key = engine._orientation_profile_key()
        engine.ui_mode = 'transfer'
        transfer_key = engine._orientation_profile_key()
        self.assertNotEqual(camera_key, transfer_key)
        self.assertTrue(transfer_key.startswith('transfer-function-test|'))

    def test_transfer_mode_keeps_physical_camera_capture_active(self):
        engine = VisionDashboardEngine(
            source='picamera2', device='picamera2:0', ui_mode='live')
        engine.set_operating_mode('transfer')
        self.assertEqual(engine.ui_mode, 'transfer')
        self.assertEqual(engine.desired_source, 'picamera2')

    def test_transfer_path_pans_once_over_thirty_minutes(self):
        engine = VisionDashboardEngine(
            source='synthetic', ui_mode='transfer', width=640, height=480)
        with patch('tools.vision_dashboard.time.monotonic', return_value=1000):
            engine.transfer_path_started = 1000
            start = engine._transfer_path()
        with patch('tools.vision_dashboard.time.monotonic', return_value=1900):
            middle = engine._transfer_path()
        with patch('tools.vision_dashboard.time.monotonic', return_value=2800):
            end = engine._transfer_path()
        self.assertAlmostEqual(start[0], -102.4)
        self.assertAlmostEqual(middle[0], 0.0)
        self.assertAlmostEqual(end[0], 102.4)
        self.assertAlmostEqual(start[1], 0.0)
        self.assertAlmostEqual(middle[1], 38.4)
        self.assertAlmostEqual(end[1], 0.0, places=5)

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
        engine = VisionDashboardEngine(
            source='synthetic', hil_host='127.0.0.1',
            hil_route_target='192.168.1.231', bridge_recovery_quiet_seconds=0)
        engine.hil_controller = MagicMock()
        engine.hil_controller.thread = None
        recovered = MagicMock()
        recovered.status.return_value = {'connection': 'live'}
        completed = MagicMock(returncode=0, stdout='', stderr='')
        with (patch('tools.vision_dashboard.subprocess.run', return_value=completed) as run,
              patch.object(engine, '_poll_bridge_management',
                           return_value={'verified_proxy': True}),
              patch.object(engine, '_refresh_bridge_host', return_value=False),
              patch.object(engine, '_new_position_controller', return_value=recovered)):
            engine.restart_bridge_service()
            engine.bridge_service_restart_thread.join(2)
        self.assertFalse(engine.bridge_service_restart_in_progress)
        self.assertEqual(engine.bridge_service_restart_count, 1)
        self.assertIn('recovered successfully', engine.bridge_service_restart_result)
        self.assertEqual(engine.hil_controller, recovered)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].args[0], [
            '/usr/bin/systemctl', '--user', 'stop', 'celestron-bridge.service'])
        self.assertEqual(run.call_args_list[1].args[0], [
            '/usr/bin/systemctl', '--user', 'start', 'celestron-bridge.service'])

    def test_bridge_host_refresh_uses_current_route_source(self):
        engine = VisionDashboardEngine(
            source='synthetic', hil_host='192.168.1.163',
            hil_route_target='192.168.1.231')
        with patch('tools.vision_dashboard.route_source_address',
                   return_value='192.168.1.149'):
            self.assertTrue(engine._refresh_bridge_host())
        self.assertEqual(engine.hil_host, '192.168.1.149')

    def test_failed_client_still_exposes_bridge_recovery(self):
        engine = VisionDashboardEngine(
            source='synthetic', hil_host='192.168.1.149',
            hil_route_target='192.168.1.231')
        engine.hil_controller = None
        engine.hil_error = 'previous recovery failed'
        status = engine._hil_update(None, 640, 480)
        self.assertTrue(status['available'])
        self.assertEqual(status['connection'], 'disconnected')
        self.assertIn('Recover Bridge', status['description'])

    def test_manual_jog_does_not_require_target_mapping_reference_or_arm(self):
        engine = VisionDashboardEngine(source='synthetic', hil_host='127.0.0.1')
        engine.hil_controller = MagicMock()
        engine.hil_controller.status.return_value = {'connection': 'live'}
        engine.manual_jog('azimuth', 1, 3)
        engine.hil_controller.manual_jog.assert_called_once_with(0x10, 1, 3, 2.0)
        engine.manual_stop()
        engine.hil_controller.manual_stop.assert_called_once_with()

    def test_zero_encoder_marker_rebases_relative_reference_without_disarming(self):
        engine = VisionDashboardEngine(source='synthetic', hil_host='127.0.0.1')
        engine.hil_controller = MagicMock()
        engine.hil_controller.status.side_effect = [
            {'connection': 'live', 'motion': 'target_sent'},
            {'connection': 'live', 'motion': 'paused'}]
        engine.hil_enabled = True
        engine.hil_current_target = {'azimuth_degrees': 181, 'elevation_degrees': 1}
        engine.hil_target_rate = {'azimuth_deg_s': .1, 'elevation_deg_s': .1}
        engine.zero_encoder_marker()
        engine.hil_controller.freeze.assert_called_once_with()
        engine.hil_controller.reference.assert_called_once_with(180, 0)
        self.assertTrue(engine.hil_enabled)
        self.assertIsNone(engine.hil_current_target)
        self.assertIsNone(engine.hil_target_rate)
        self.assertEqual(engine.hil_last_locked_marker['azimuth_degrees'], 180.0)
        self.assertFalse(engine.hil_reference_rebasing)

    def test_feedback_interruption_retains_tracking_request(self):
        engine = VisionDashboardEngine(source='synthetic', hil_host='127.0.0.1')
        engine.hil_controller = MagicMock()
        engine.hil_controller.status.return_value = {
            'connection': 'recovering', 'motion': 'idle', 'positions': {},
            'reconnections': 1, 'armed': False}
        engine.hil_enabled = True
        status = engine._hil_update(None, 640, 480)
        self.assertTrue(engine.hil_enabled)
        self.assertTrue(status['enabled'])
        self.assertIn('Tracking remains ON', status['error'])
        engine.hil_controller.freeze.assert_called_once_with()

    def test_snapshot_overlays_live_motor_state_on_stale_camera_frame(self):
        engine = VisionDashboardEngine(source='synthetic', hil_host='127.0.0.1')
        engine.hil_controller = MagicMock()
        engine.hil_controller.status.return_value = {
            'connection': 'live', 'feedback_live': True,
            'motion': 'target_sent', 'armed': True,
            'positions': {'Azimuth': {'live': True},
                          'Elevation': {'live': True}},
            'cycle_count': 12, 'cycle_overruns': 0,
            'last_achieved_cycle_rate_hz': 4.0, 'last_error': None}
        engine.hil_enabled = True
        engine.telemetry = {
            'frame': 10, 'state': 'CAMERA RECOVERY',
            'motor_output': 'DISABLED',
            'hardware_loop': {'enabled': False, 'motion': 'paused'}}
        snapshot = engine.snapshot()
        self.assertEqual(snapshot['motor_output'], 'ENABLED')
        self.assertTrue(snapshot['hardware_loop']['enabled'])
        self.assertTrue(snapshot['hardware_loop']['feedback_live'])
        self.assertEqual(snapshot['hardware_loop']['motion'], 'target_sent')

    def test_warm_link_with_stale_encoders_warns_without_gating(self):
        engine = VisionDashboardEngine(source='synthetic', hil_host='127.0.0.1')
        engine.hil_controller = MagicMock()
        engine.hil_controller.status.return_value = {
            'connection': 'live', 'feedback_live': False, 'motion': 'paused',
            'positions': {
                'Azimuth': {'reference_degrees': 180.0},
                'Elevation': {'reference_degrees': 0.0}},
            'reconnections': 0, 'armed': True,
            'local_reference_set': True}
        engine.hil_controller.freeze.return_value = False
        engine.hil_enabled = True
        engine.orientation_calibration = {
            'valid': True,
            'inverse_degrees_per_pixel': [[.001, 0], [0, .001]]}
        status = engine._hil_update({'x': 330, 'y': 240}, 640, 480)
        self.assertTrue(status['enabled'])
        self.assertFalse(status['feedback_live'])
        self.assertIn('does not gate target dispatch', status['error'])
        engine.hil_controller.freeze.assert_not_called()

    def test_reconnect_restores_last_optical_reference(self):
        engine = VisionDashboardEngine(source='synthetic', hil_host='127.0.0.1')
        engine.hil_controller = MagicMock()
        positions = {
            'Azimuth': {'reference_degrees': None},
            'Elevation': {'reference_degrees': None}}
        restored = {
            'connection': 'live', 'motion': 'paused', 'positions': positions,
            'reconnections': 1, 'armed': True, 'local_reference_set': False}
        live_positions = {
            'Azimuth': {'reference_degrees': 181.25},
            'Elevation': {'reference_degrees': 42.5}}
        live = dict(restored, positions=live_positions, local_reference_set=True)
        engine.hil_controller.status.side_effect = [restored, live]
        engine.hil_enabled = True
        engine.hil_last_locked_marker = {
            'azimuth_degrees': 181.25, 'elevation_degrees': 42.5}
        engine._hil_update(None, 640, 480)
        engine.hil_controller.reference.assert_called_once_with(181.25, 42.5)
        self.assertTrue(engine.hil_enabled)

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
        source = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        np.testing.assert_array_equal(
            engine._apply_image_orientation(source), np.array([[2, 1], [4, 3]]))
        engine.set_hil_max_tracking_slew(3)
        self.assertEqual(engine.hil_max_tracking_slew_deg_s, 3)
        with self.assertRaisesRegex(ValueError, '0.001–5'):
            engine.set_hil_max_tracking_slew(6)
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

    def test_live_target_can_be_reselected_while_motor_output_remains_armed(self):
        engine = VisionDashboardEngine(source='synthetic', ui_mode='live',
                                       width=640, height=480)
        engine.telemetry.update(width=640, height=480)
        engine.hil_enabled = True
        engine.feature_selected_id = 1
        engine.hil_previous_target = (180.0, 10.0)
        engine.set_aim_mode('center')
        engine.set_feature_point(.60, .40, .12)
        self.assertTrue(engine.hil_enabled)
        self.assertIsNone(engine.feature_selected_id)
        self.assertEqual(engine.feature_roi_pending, (346, 163, 77, 58))
        self.assertIsNone(engine.hil_previous_target)
        self.assertIn('initializing feature lock', engine.hil_error)

    def test_manual_lock_hold_freezes_then_reseeds_without_disarming(self):
        engine = VisionDashboardEngine(source='synthetic', ui_mode='live',
                                       width=640, height=480)
        engine.telemetry.update(width=640, height=480)
        engine.hil_enabled = True
        engine.hil_controller = MagicMock()
        engine.set_manual_lock_hold(True, .25, .75, .12)
        self.assertTrue(engine.hil_enabled)
        engine.hil_controller.freeze.assert_called()
        self.assertTrue(engine.manual_lock_hold['active'])
        engine.set_manual_lock_hold(False, .30, .70, .12)
        self.assertTrue(engine.hil_enabled)
        self.assertFalse(engine.manual_lock_hold['active'])
        self.assertTrue(engine.manual_lock_hold['release_pending'])
        time.sleep(1.1)
        self.assertIsNone(engine.manual_lock_hold)
        self.assertEqual(engine.feature_roi_pending, (154, 307, 77, 58))
        self.assertEqual(engine.aim_mode, 'center')

    def test_initial_slow_trajectory_regresses_pixel_drift_to_mount_rate(self):
        engine = VisionDashboardEngine(source='synthetic', ui_mode='live')
        engine.orientation_calibration = {
            'valid': True,
            'inverse_degrees_per_pixel': [[.01, 0], [0, .01]]}
        engine.hil_initial_trajectory_enabled = True
        engine.hil_initial_trajectory_window = 2.0
        for index in range(21):
            timestamp = 100 + index * .1
            engine.feature_samples.append(
                (timestamp, 200 + 2 * (timestamp - 100),
                 150 - 1 * (timestamp - 100)))
        engine._update_initial_trajectory({'x': 204, 'y': 148}, 102.0)
        trajectory = engine.hil_initial_trajectory
        self.assertTrue(trajectory['valid'])
        self.assertAlmostEqual(
            trajectory['mount_rate']['azimuth_deg_s'], -.02, places=3)
        self.assertAlmostEqual(
            trajectory['mount_rate']['elevation_deg_s'], .01, places=3)

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
        engine.set_hil_control_strategy('custom', 0, 0, 0, .1)
        held = engine._apply_outer_loop((180.05, .05), {
            'azimuth_degrees': 180, 'elevation_degrees': 0})
        self.assertAlmostEqual(held[0], 180)
        self.assertAlmostEqual(held[1], 0)

    def test_balanced_pd_response_reduces_correction_and_brakes_closing_error(self):
        engine = VisionDashboardEngine(source='synthetic')
        engine.set_hil_control_strategy('balanced')
        first = engine._damped_mount_correction(.20, -.10, 10.0)
        second = engine._damped_mount_correction(.10, -.05, 10.25)
        self.assertGreater(first[0], 0)
        self.assertLess(first[1], 0)
        self.assertGreaterEqual(second[0], 0)
        self.assertLess(second[0], first[0])
        self.assertLessEqual(abs(second[1]), abs(first[1]))

    def test_agc_lock_freezes_current_camera_metadata_and_unlocks(self):
        engine = VisionDashboardEngine(source='synthetic')
        engine.source = 'picamera2'
        engine.capture = MagicMock()
        engine.camera_control_ranges = {
            'ExposureValue': {'minimum': -8, 'maximum': 8},
            'ExposureTime': {'minimum': 1, 'maximum': 66666},
            'AnalogueGain': {'minimum': 1, 'maximum': 16},
        }
        engine.camera_metadata = {'ExposureTime': 321, 'AnalogueGain': 1.75}
        engine.hil_enabled = True
        engine.set_camera_agc_lock(True)
        self.assertFalse(engine.camera_auto_exposure)
        self.assertEqual(engine.camera_manual_exposure_us, 321)
        self.assertEqual(engine.camera_manual_gain, 1.75)
        engine.capture.set_controls.assert_called_with(
            {'AeEnable': False, 'ExposureTime': 321, 'AnalogueGain': 1.75})
        engine.set_camera_agc_lock(False)
        self.assertTrue(engine.camera_auto_exposure)
        engine.capture.set_controls.assert_called_with(
            {'AeEnable': True, 'ExposureValue': engine.camera_exposure_value})

    def test_gain_lock_holds_gain_and_software_adjusts_exposure(self):
        engine = VisionDashboardEngine(source='synthetic')
        engine.source = 'picamera2'
        engine.capture = MagicMock()
        engine.camera_control_ranges = {
            'ExposureValue': {'minimum': -8, 'maximum': 8},
            'ExposureTime': {'minimum': 1, 'maximum': 66666},
            'AnalogueGain': {'minimum': 1, 'maximum': 16},
        }
        engine.camera_metadata = {'ExposureTime': 1000, 'AnalogueGain': 2.5}
        engine.set_camera_gain_lock(True)
        self.assertTrue(engine.camera_gain_locked)
        self.assertFalse(engine.camera_auto_exposure)
        engine.capture.reset_mock()
        engine._adjust_locked_gain_exposure(
            np.full((40, 40), 30, dtype=np.uint8), now=1.0)
        controls = engine.capture.set_controls.call_args.args[0]
        self.assertFalse(controls['AeEnable'])
        self.assertGreater(controls['ExposureTime'], 1000)
        self.assertEqual(controls['AnalogueGain'], 2.5)
        engine.set_camera_gain_lock(False)
        self.assertFalse(engine.camera_gain_locked)
        self.assertTrue(engine.camera_auto_exposure)


if __name__ == '__main__':
    unittest.main()
