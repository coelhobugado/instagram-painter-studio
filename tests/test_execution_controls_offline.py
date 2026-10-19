"""Pausa, retomada e parada sem iniciar ADB nem tocar no celular."""
import threading
import time
import queue
import io
import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np

from studio.adb_manager import ADBManager
from studio.device.ui_timing import UITimingManager
from studio.execution_worker import ExecutionWorker
from studio.device.quality_gate import QualityGateResult
from studio.core.models import CanvasTransform
from studio.app_gui import InstagramPainterStudioGUI
from studio.device.persistent_touch import PersistentTouchExecutor, TouchChannelType
from studio.planning.motion_compiler import MotionStroke, MotionEvent, TouchAction
import main


class OfflineExecutionControls(unittest.TestCase):
    def test_uniform_background_uses_geometric_color_confirmation(self):
        before = np.zeros((100, 100, 3), dtype=np.uint8)
        after = before.copy()
        after[20:70, 20:70] = (205, 205, 205)
        expected = np.full((50, 50, 3), 205, dtype=np.uint8)
        layer = {"purpose": "background", "render_color_rgb": (205, 205, 205)}
        transform = CanvasTransform(offset_x=20, offset_y=20, scale=1.0)
        verify = ExecutionWorker._verify_background_fill

        passed, _ = verify(before, after, expected, layer, transform)
        self.assertTrue(passed)
        passed, reason = verify(before, before, expected, layer, transform)
        self.assertFalse(passed)
        self.assertIn("fundo visível", reason)
        wrong = before.copy()
        wrong[20:70, 20:70] = (30, 30, 200)
        self.assertFalse(verify(before, wrong, expected, layer, transform)[0])
        self.assertTrue(verify(after, after, expected, layer, transform, resumed=True)[0])

    def test_line_strokes_are_checked_against_their_screen_positions(self):
        before = np.full((120, 120, 3), (30, 42, 60), dtype=np.uint8)
        after = before.copy()
        expected = np.zeros((70, 70, 3), dtype=np.uint8)
        layer = {"purpose": "line_art_centerline", "render_color_rgb": (85, 85, 85),
                 "strokes": [[(10, 10), (60, 10)], [(10, 30), (60, 30)]]}
        transform = CanvasTransform(offset_x=20, offset_y=20, scale=1.0)
        for stroke in layer["strokes"]:
            import cv2
            cv2.polylines(after, [np.asarray(stroke, np.int32).reshape(-1, 1, 2)
                                  + np.array([[[20, 20]]], dtype=np.int32)],
                          False, (85, 85, 85), 10)
        verify = ExecutionWorker._verify_strokes_at_planned_positions
        self.assertTrue(verify(before, after, expected, layer, transform)[0])
        self.assertFalse(verify(before, before, expected, layer, transform)[0])
        shifted = before.copy()
        shifted[20:100, 20:100] = (85, 85, 85)
        self.assertFalse(verify(before, shifted, expected, layer, transform)[0])

    def test_line_verification_applies_canvas_scale(self):
        import cv2
        before = np.full((180, 180, 3), (30, 42, 60), dtype=np.uint8)
        after = before.copy()
        expected = np.zeros((100, 100, 3), dtype=np.uint8)
        layer = {"purpose": "line_art_centerline", "render_color_rgb": (255, 255, 255),
                 "strokes": [[(10, 10), (90, 10)], [(10, 40), (90, 40)]]}
        transform = CanvasTransform(offset_x=25, offset_y=30, scale=0.75)
        for stroke in layer["strokes"]:
            points = np.asarray([transform.local_to_screen(pt) for pt in stroke], np.int32)
            cv2.polylines(after, [points.reshape(-1, 1, 2)], False, (255, 255, 255), 5)
        verify = ExecutionWorker._verify_strokes_at_planned_positions
        self.assertTrue(verify(before, after, expected, layer, transform)[0])
        self.assertFalse(verify(before, before, expected, layer, transform)[0])

    def test_background_verification_applies_canvas_scale(self):
        before = np.zeros((180, 180, 3), dtype=np.uint8)
        after = before.copy()
        expected = np.full((100, 100, 3), 205, dtype=np.uint8)
        transform = CanvasTransform(offset_x=25, offset_y=30, scale=0.75)
        after[30:105, 25:100] = (205, 205, 205)
        layer = {"purpose": "background", "render_color_rgb": (205, 205, 205)}
        self.assertTrue(ExecutionWorker._verify_background_fill(
            before, after, expected, layer, transform)[0])

    def make_worker(self, draw):
        adb = Mock()
        adb.timing = UITimingManager(is_mock=True)
        adb.capture_screenshot.return_value = np.zeros((40, 40, 3), dtype=np.uint8)
        adb.execute_stroke_batch.side_effect = draw
        plan = {
            "total_strokes": 3,
            "quality_gate": QualityGateResult(passed=True, summary="Plano de teste aprovado"),
            "custom_bounds": {"left": 0, "right": 500, "top": 0, "bottom": 500},
            "layers": [{"name": "Teste", "brush": "solid", "thickness": "min",
                        "touch_info": {"target_x": 411, "target_y": 2184},
                        "strokes": [[(100, 100), (200, 100)]] * 3}],
        }
        finished = []
        # Lote de 1 traço: estes testes medem a pausa entre traços individuais.
        worker = ExecutionWorker(adb, plan, {"on_finished": lambda *result: finished.append(result)},
                                 stroke_batch_size=1)
        worker.state_machine.validate_safety_for_drawing = Mock(return_value=(True, "OK"))
        return worker, adb, finished

    @patch("studio.execution_worker.SessionTelemetry.export_json")
    def test_pause_prevents_second_stroke_until_resume(self, _export):
        entered = threading.Event()
        release = threading.Event()
        calls = []

        def draw(strokes):
            calls.append(strokes)
            if len(calls) == 1:
                entered.set()
                self.assertTrue(release.wait(3))
            return True

        worker, adb, finished = self.make_worker(draw)
        worker.start()
        self.assertTrue(entered.wait(3))
        worker.pause()
        release.set()
        time.sleep(0.15)
        self.assertEqual(len(calls), 1)
        self.assertTrue(worker.is_alive())
        worker.resume()
        worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(calls), 3)
        self.assertEqual(finished, [(True, "Pintura concluída com sucesso!")])

    @patch("studio.execution_worker.SessionTelemetry.export_json")
    def test_stop_returns_immediately_and_prevents_more_strokes(self, _export):
        entered = threading.Event()
        release = threading.Event()
        calls = []

        def draw(strokes):
            calls.append(strokes)
            entered.set()
            self.assertTrue(release.wait(3))
            return True

        worker, adb, finished = self.make_worker(draw)
        worker.start()
        self.assertTrue(entered.wait(3))
        started = time.monotonic()
        worker.stop()
        self.assertLess(time.monotonic() - started, 0.2)
        release.set()
        worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(calls), 1)
        self.assertEqual(adb.emergency_kill.call_count, 1)
        self.assertFalse(finished[0][0])

    @patch("studio.execution_worker.SessionTelemetry.export_json")
    def test_pause_after_last_layer_does_not_report_success(self, _export):
        worker, _adb, finished = self.make_worker(lambda _strokes: True)
        paused = threading.Event()

        def pause_after_capture(_image):
            worker.pause()
            paused.set()

        worker.cb_screencap = pause_after_capture
        worker.start()
        self.assertTrue(paused.wait(3))
        time.sleep(0.1)
        self.assertEqual(finished, [])
        self.assertTrue(worker.is_alive())
        worker.resume()
        worker.join(3)
        self.assertEqual(finished, [(True, "Pintura concluída com sucesso!")])

    @patch("studio.execution_worker.SessionTelemetry.export_json")
    @patch("studio.execution_worker.CanvasRegistrationEngine.align_screenshot_to_target")
    def test_failed_final_registration_reports_result_without_pausing(self, align, _export):
        worker, _adb, finished = self.make_worker(lambda _strokes: True)
        worker.plan["cumulative_layer_images"] = [np.zeros((40, 40, 3), dtype=np.uint8)]
        worker.verify_each_layer = True
        align.return_value = (None, False)

        worker.start()
        worker.join(3)
        if worker.is_alive():
            worker.stop()
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(finished), 1)
        self.assertFalse(finished[0][0])
        self.assertIn("alinhar", finished[0][1])
        self.assertEqual(worker.total_completed_strokes, 3)
        self.assertFalse(worker._is_paused)

    @patch("studio.execution_worker.SessionTelemetry.export_json")
    @patch("studio.execution_worker.CanvasRegistrationEngine.align_screenshot_to_target")
    def test_failed_final_fidelity_reports_result_without_pausing(self, align, _export):
        worker, _adb, finished = self.make_worker(lambda _strokes: True)
        saved_positions = []
        worker.cb_checkpoint = lambda layer, stroke: saved_positions.append((layer, stroke))
        expected = np.zeros((40, 40, 3), dtype=np.uint8)
        worker.plan["cumulative_layer_images"] = [expected]
        worker.verify_each_layer = True
        align.return_value = (expected, True)
        worker._verify_layer_fidelity = Mock(return_value=(False, "cor divergente"))

        worker.start()
        worker.join(3)
        if worker.is_alive():
            worker.stop()
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(finished), 1)
        self.assertFalse(finished[0][0])
        self.assertIn("cor divergente", finished[0][1])
        self.assertEqual(worker.total_completed_strokes, 3)
        self.assertFalse(worker._is_paused)
        self.assertEqual(saved_positions, [(0, 0)])

    @patch("studio.execution_worker.SessionTelemetry.export_json")
    @patch("studio.execution_worker.CanvasRegistrationEngine.align_screenshot_to_target")
    def test_line_art_accepts_physical_strokes_when_wallpaper_edges_confuse_alignment(self, align, _export):
        worker, _adb, finished = self.make_worker(lambda _strokes: True)
        expected = np.zeros((40, 40, 3), dtype=np.uint8)
        worker.plan["layers"][0]["purpose"] = "line_art_centerline"
        worker.plan["cumulative_layer_images"] = [expected]
        worker.verify_each_layer = True
        align.return_value = (expected, True)
        worker._verify_layer_fidelity = Mock(return_value=(False, "alinhamento de bordas 3.0%"))
        worker._verify_strokes_at_planned_positions = Mock(return_value=(True, "traços confirmados"))

        worker.start()
        worker.join(3)

        self.assertEqual(finished, [(True, "Pintura concluída com sucesso!")])
        worker._verify_strokes_at_planned_positions.assert_called_once()

    def test_manager_construction_does_not_start_adb(self):
        with patch("studio.adb_manager.subprocess.run") as run:
            manager = ADBManager(adb_path="not-used", device_target="OFFLINE")
        run.assert_not_called()
        self.assertFalse(manager.is_connected)

    def test_emergency_kill_never_uses_broad_process_kill(self):
        manager = ADBManager(adb_path="not-used", device_target="OFFLINE")
        manager.run_cmd = Mock(return_value=(True, "", ""))
        manager.close_interactive_shell = Mock()
        self.assertTrue(manager.emergency_kill()[0])
        manager.run_cmd.assert_not_called()
        self.assertFalse(manager.execute_stroke_batch([[(100, 100), (200, 100)]]))
        manager.reset_abort()
        self.assertFalse(manager._abort_event.is_set())

    def test_gui_emergency_button_requests_stop_only_once(self):
        view = SimpleNamespace(
            log=Mock(), worker=Mock(), adb=Mock(),
            btn_start=Mock(), btn_pause=Mock(), btn_resume=Mock(), lbl_status=Mock(),
            _is_painting=lambda: True,
        )
        view.worker.is_alive.return_value = True
        InstagramPainterStudioGUI.action_emergency_kill(view)
        view.worker.stop.assert_called_once()
        view.adb.emergency_kill.assert_not_called()
        view.btn_pause.configure.assert_called_with(state="disabled")

    def test_automatic_pause_updates_resume_button(self):
        view = SimpleNamespace(
            after=lambda _delay, callback: callback(), worker=Mock(),
            btn_pause=Mock(), btn_resume=Mock(), lbl_status=Mock(),
            _is_painting=lambda: True,
        )
        view.worker.is_alive.return_value = True
        InstagramPainterStudioGUI._on_worker_pause_state(view, True)
        view.btn_pause.configure.assert_called_with(state="disabled")
        view.btn_resume.configure.assert_called_with(state="normal")

    @patch("studio.app_gui.PersistentCheckpointManager.clear_checkpoint")
    @patch("studio.app_gui.messagebox.showinfo")
    def test_gui_announces_success_after_worker_finishes(self, showinfo, clear_checkpoint):
        view = SimpleNamespace(
            after=lambda _delay, callback: callback(),
            _set_input_controls_locked=Mock(),
            _can_start_current_plan=lambda: True,
            action_refresh_screencap=Mock(),
            btn_start=Mock(), btn_pause=Mock(), btn_resume=Mock(),
            progress_bar=Mock(), lbl_progress_text=Mock(), lbl_status=Mock(),
            checkpoint={"layer": 0, "stroke": 3}, adb=Mock(),
        )

        InstagramPainterStudioGUI._on_worker_finished(view, True, "Pintura concluída com sucesso!")

        showinfo.assert_called_once()
        view.lbl_status.configure.assert_called_with(text="Pintura concluída")
        view.progress_bar.set.assert_called_once_with(1.0)
        self.assertEqual(view.checkpoint, {"layer": 0, "stroke": 0})
        clear_checkpoint.assert_called_once()
        # O Parar liga um freio no ADB; terminada a execução ele precisa ser liberado
        view.adb.reset_abort.assert_called_once()
        self.assertTrue(view._worker_finished)

    def test_stop_interrupts_wait_for_shell_ack(self):
        manager = ADBManager(adb_path="not-used", device_target="OFFLINE")
        shell = Mock()
        shell.stdin = io.StringIO()
        shell.stdout = io.StringIO()
        shell.poll.return_value = None
        manager._interactive_shell = shell
        manager._shell_output = queue.Queue()
        manager.get_interactive_shell = Mock(return_value=shell)
        result = []
        sender = threading.Thread(
            target=lambda: result.append(manager._send_to_interactive_shell(["input swipe 1 2 3 4 55\n"], timeout_s=5)),
            daemon=True,
        )
        sender.start()
        time.sleep(0.05)
        manager.emergency_kill()
        sender.join(1)
        self.assertFalse(sender.is_alive())
        self.assertEqual(result, [False])
        self.assertIn("input swipe", shell.stdin.getvalue())

    def test_stop_during_palette_navigation_sends_no_more_touches(self):
        manager = ADBManager(adb_path="not-used", device_target="OFFLINE")
        sent = []

        def run(args, **_kwargs):
            sent.append(args)
            manager._abort_event.set()
            return True, "", ""

        manager.run_cmd = Mock(side_effect=run)
        with self.assertRaisesRegex(RuntimeError, "interrompida"):
            manager.select_palette_color({"palette_page": 3, "target_x": 636, "target_y": 2184})
        self.assertEqual(len(sent), 1)

    def test_canvas_open_reports_failed_tap(self):
        manager = ADBManager(adb_path="not-used", device_target="OFFLINE")
        manager.run_cmd = Mock(return_value=(False, "", "offline"))
        ok, message = manager.open_instagram_canvas()
        self.assertFalse(ok)
        self.assertIn("offline", message)
        manager.run_cmd.assert_called_once()

    def test_socket_without_ack_does_not_repeat_stroke(self):
        fallback = Mock()
        executor = PersistentTouchExecutor(fallback_adb=fallback, enable_mock=False)
        executor.sock = Mock()
        executor.sock.recv.return_value = b""
        executor.is_connected = True
        executor.channel_type = TouchChannelType.SOCKET
        stroke = MotionStroke(events=[
            MotionEvent(TouchAction.DOWN, 100, 100, 0, 1.0),
            MotionEvent(TouchAction.UP, 200, 100, 50, 0.0),
        ])
        self.assertFalse(executor.execute_motion_stroke(stroke))
        fallback.execute_stroke_batch.assert_not_called()

    def test_main_starts_from_project_directory_without_loading_device(self):
        launch = Mock()
        with patch.dict(sys.modules, {"studio.app_gui": SimpleNamespace(launch_app=launch)}):
            with patch("main.os.chdir") as change_dir:
                self.assertEqual(main.main(), 0)
        change_dir.assert_called_once_with(Path(main.__file__).resolve().parent)
        launch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
