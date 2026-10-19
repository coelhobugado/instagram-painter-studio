"""Regressões das correções da camada de execução (ADB, worker e orquestrador)."""
import queue
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from studio import config
from studio.adb_manager import ADBManager, CommandResult
from studio.core.models import CanvasTransform
from studio.device.persistent_touch import PersistentTouchExecutor, TouchChannelType
from studio.device.profile_loader import DeviceProfileLoader
from studio.device.quality_gate import QualityGateResult
from studio.device.ui_timing import UITimingManager
from studio.execution_orchestrator import ExecutionOrchestrator
from studio.execution_worker import ExecutionWorker
from studio.planning.motion_compiler import ExecutionBatch, MotionEvent, MotionStroke, TouchAction


class StrokeCommandTests(unittest.TestCase):
    def test_continuous_mode_sends_one_gesture_per_stroke(self):
        commands = ADBManager.build_stroke_commands([(100, 400), (200, 400), (200, 500)], continuous=True)
        self.assertEqual(commands[0], "input motionevent DOWN 100 400\n")
        self.assertEqual(commands[-1], "input motionevent UP 200 500\n")
        self.assertTrue(all("motionevent MOVE" in c for c in commands[1:-1]))
        self.assertFalse(any("swipe" in c for c in commands))

    def test_swipe_mode_remains_default(self):
        self.assertFalse(config.CONTINUOUS_STROKES)
        commands = ADBManager.build_stroke_commands([(100, 400), (150, 400)])
        self.assertEqual(commands, ["input swipe 100 400 150 400 55\n"])

    def test_degenerate_stroke_has_no_gesture(self):
        self.assertEqual(ADBManager.build_motion_commands([(10, 10), (11, 10)]), [])


class AdbManagerTests(unittest.TestCase):
    def test_run_cmd_keeps_returncode_and_stays_a_triple(self):
        manager = ADBManager(adb_path="not-used", device_target="OFFLINE")
        fake = Mock(returncode=1, stdout="", stderr="")
        with patch("studio.adb_manager.subprocess.run", return_value=fake):
            result = manager.run_cmd(["shell", "true"])
        ok, out, err = result
        self.assertFalse(ok)
        self.assertEqual(result.returncode, 1)
        self.assertIn("1", err)
        self.assertIsInstance(result, CommandResult)

    def test_suspend_blocker_alone_does_not_mean_screen_on(self):
        manager = ADBManager(adb_path="not-used", device_target="OFFLINE")
        manager.run_cmd = Mock(return_value=(True, "mHoldingDisplaySuspendBlocker=true\nmWakefulness=Asleep", ""))
        self.assertFalse(manager.is_screen_on())
        manager.run_cmd = Mock(return_value=(True, "mWakefulness=Awake", ""))
        self.assertTrue(manager.is_screen_on())

    def test_screenshot_uses_single_exec_out_call(self):
        import cv2
        manager = ADBManager(adb_path="not-used", device_target="OFFLINE")
        ok, png = cv2.imencode(".png", np.full((4, 3, 3), 7, dtype=np.uint8))
        fake = Mock(returncode=0, stdout=png.tobytes())
        manager.run_cmd = Mock()
        with patch("studio.adb_manager.subprocess.run", return_value=fake) as run:
            image = manager.capture_screenshot()
        self.assertEqual(image.shape, (4, 3, 3))
        self.assertIn("exec-out", run.call_args.args[0])
        manager.run_cmd.assert_not_called()

    def test_profile_follows_connected_model_when_available(self):
        manager = ADBManager(adb_path="not-used", device_target="OFFLINE")
        manager._select_profile_for_model("MODELO-INEXISTENTE")
        self.assertEqual(manager.profile_loader.profile_name, config.DEFAULT_DEVICE_PROFILE)
        self.assertTrue(DeviceProfileLoader.profile_exists("SM-A736B"))

    def test_fallback_coordinates_match_calibrated_profile(self):
        loader = DeviceProfileLoader("SM-A736B")
        missing = DeviceProfileLoader("PERFIL-AUSENTE")
        self.assertEqual(missing.get_safe_bounds(), loader.get_safe_bounds())
        self.assertEqual(missing.get_slider_thickness(), loader.get_slider_thickness())
        self.assertEqual(missing.get_palette_swipe("next_page", config.UI_PALETTE_NEXT_SWIPE),
                         loader.get_palette_swipe("next_page", config.UI_PALETTE_NEXT_SWIPE))

    def test_close_shell_waits_for_sender_lock(self):
        manager = ADBManager(adb_path="not-used", device_target="OFFLINE")
        shell = Mock()
        manager._interactive_shell = shell
        manager._shell_output = queue.Queue()
        with manager._shell_lock:
            closer = threading.Thread(target=manager.close_interactive_shell, daemon=True)
            closer.start()
            time.sleep(0.05)
            self.assertIs(manager._interactive_shell, shell)
        closer.join(2)
        self.assertIsNone(manager._interactive_shell)


class PersistentTouchTests(unittest.TestCase):
    def make_stroke(self):
        return MotionStroke(events=[
            MotionEvent(TouchAction.DOWN, 100, 100, 0, 1.0),
            MotionEvent(TouchAction.UP, 200, 100, 8, 0.0),
        ])

    def test_abort_event_cancels_batches(self):
        adb = Mock()
        adb._abort_event = threading.Event()
        adb._abort_event.set()
        executor = PersistentTouchExecutor(fallback_adb=adb, enable_mock=True)
        executor.connect()
        batch = ExecutionBatch(batch_id=1, strokes=[self.make_stroke()], estimated_duration_ms=8)
        self.assertFalse(executor.execute_batch(batch))
        adb.execute_stroke_batch.assert_not_called()

    def test_live_channel_is_reused(self):
        executor = PersistentTouchExecutor(fallback_adb=Mock(), enable_mock=False)
        executor.is_connected = True
        executor.channel_type = TouchChannelType.PERSISTENT_SHELL
        executor._shell_proc = Mock(stdin=Mock())
        executor._shell_proc.poll.return_value = None
        executor.connect = Mock()
        ok, _ = executor.ensure_connected()
        self.assertTrue(ok)
        executor.connect.assert_not_called()


class WorkerBatchingTests(unittest.TestCase):
    @patch("studio.execution_worker.SessionTelemetry.export_json")
    def test_strokes_are_sent_in_batches(self, _export):
        adb = Mock()
        adb.timing = UITimingManager(is_mock=True)
        adb.capture_screenshot.return_value = np.zeros((40, 40, 3), dtype=np.uint8)
        adb.execute_stroke_batch.return_value = True
        plan = {
            "total_strokes": 5,
            "quality_gate": QualityGateResult(passed=True, summary="ok"),
            "custom_bounds": {"left": 0, "right": 500, "top": 0, "bottom": 500},
            "layers": [{"name": "L", "thickness": "min", "touch_info": {"target_x": 1, "target_y": 1},
                        "strokes": [[(100, 100), (200, 100)]] * 5}],
        }
        finished = []
        worker = ExecutionWorker(adb, plan, {"on_finished": lambda *r: finished.append(r)},
                                 stroke_batch_size=4)
        worker.state_machine.validate_safety_for_drawing = Mock(return_value=(True, "OK"))
        worker.run()
        self.assertEqual([len(c.args[0]) for c in adb.execute_stroke_batch.call_args_list], [4, 1])
        self.assertTrue(finished[0][0])

    def test_quality_gate_is_checked_before_any_capture(self):
        adb = Mock()
        adb.timing = UITimingManager(is_mock=True)
        plan = {"layers": [{"strokes": [[(1, 1), (5, 5)]]}]}
        finished = []
        ExecutionWorker(adb, plan, {"on_finished": lambda *r: finished.append(r)}).run()
        self.assertFalse(finished[0][0])
        adb.capture_screenshot.assert_not_called()


class OrchestratorTests(unittest.TestCase):
    def test_artifact_subset_uses_original_layer_reference(self):
        orchestrator = ExecutionOrchestrator(Mock(), enable_mock=True)
        orchestrator.validate_safety = Mock(return_value=(True, "OK", 1.0))
        seen = []

        def fake_layer(plan, layer_index, **kwargs):
            seen.append((plan["cumulative_layer_images"][0][0, 0, 0], kwargs.get("pre_validated_screen") is not None))
            return {"success": True}

        orchestrator.execute_layer = fake_layer
        cum = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(3)]
        artifact = Mock(is_quality_gate_approved=True, cumulative_layer_images=cum,
                        layers=[{"strokes": [[(0, 0), (5, 5)]]}] * 3, transform=None)
        result = orchestrator.execute_artifact(artifact, layer_indices=[2, 0])
        self.assertTrue(result["success"])
        self.assertEqual([v for v, _ in seen], [2, 0])
        self.assertEqual([pre for _, pre in seen], [True, False])

    def test_layer_strokes_are_transformed_and_clamped_like_worker(self):
        from studio.device.fake_adb import FakeADB
        adb = FakeADB()
        orchestrator = ExecutionOrchestrator(adb)
        orchestrator.validate_safety = Mock(return_value=(True, "OK", 1.0))
        orchestrator.evaluate_quality_gate = Mock(return_value=QualityGateResult(passed=True, summary="ok"))
        sent = []
        orchestrator.touch_executor.execute_batch = lambda b: sent.extend(b.strokes) or True
        transform = CanvasTransform(offset_x=100, offset_y=400, scale=1.0)
        plan = {"layers": [{"strokes": [[(0, 0), (50, 0)], [(0, 10), (5000, 10)]],
                            "touch_info": {"target_x": 411, "target_y": 2184}}],
                "transform": transform}
        result = orchestrator.execute_layer(plan, 0, verify_fidelity=False)
        self.assertTrue(result["success"], result)
        xs = [e.x for s in sent for e in s.events]
        bounds = adb.profile_loader.get_safe_bounds()
        ys = [e.y for s in sent for e in s.events]
        self.assertEqual(min(xs), 100)  # deslocamento do canvas aplicado
        self.assertLessEqual(max(xs), bounds["right"])  # traço longo recortado na zona segura
        self.assertGreaterEqual(min(ys), bounds["top"])


if __name__ == "__main__":
    unittest.main()
