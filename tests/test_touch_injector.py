"""Injetor app_process: reamostragem, protocolo, roteamento e ajustes de interface (offline)."""
import math
import os
import queue
import threading
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from studio import config
from studio.adb_manager import ADBManager
from studio.device import touch_injector
from studio.device.persistent_touch import PersistentTouchExecutor
from studio.device.profile_loader import DeviceProfileLoader
from studio.device.touch_injector import TouchInjectorClient, resample_stroke


class ResampleStrokeTests(unittest.TestCase):
    def test_straight_line_keeps_ends_and_respects_max_step(self):
        pts = resample_stroke([(100, 500), (700, 500)], max_step_px=24)
        self.assertEqual(pts[0], (100, 500))
        self.assertEqual(pts[-1], (700, 500))
        steps = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
        self.assertLessEqual(max(steps), 24 + 1)
        # ~600 px a 24 px por quadro, mais as rampas das pontas.
        self.assertLess(len(pts), 40)

    def test_ends_ramp_up_and_down(self):
        pts = resample_stroke([(0, 0), (400, 0)], max_step_px=24)
        self.assertLessEqual(math.dist(pts[0], pts[1]), touch_injector.RAMP_STEPS_PX[0] + 1)
        self.assertLessEqual(math.dist(pts[-2], pts[-1]), touch_injector.RAMP_STEPS_PX[0] + 1)

    def test_corner_vertex_is_preserved(self):
        pts = resample_stroke([(0, 0), (200, 0), (200, 200)], max_step_px=24)
        self.assertIn((200, 0), pts)

    def test_dense_collinear_points_are_compacted(self):
        dense = [(x, 300) for x in range(100, 901)]
        pts = resample_stroke(dense, max_step_px=24)
        self.assertLess(len(pts), len(dense) / 10)

    def test_closed_ring_stays_closed(self):
        # Anel fechado (início = fim): antes o último trecho era cortado e o anel ficava aberto
        ring = [(159, 471), (159, 645), (335, 645), (318, 641), (275, 621), (248, 603), (210, 570),
                (182, 525), (169, 501), (159, 471)]
        pts = resample_stroke(ring, max_step_px=24)
        self.assertEqual(pts[0], (159, 471))
        self.assertEqual(pts[-1], (159, 471))
        steps = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
        self.assertLessEqual(max(steps), 25)

    def test_degenerate_stroke(self):
        self.assertEqual(len(resample_stroke([(5, 5), (5, 5)])), 2)


class _FakeInjectorProcess:
    """Imita o app_process: responde READY e um OK por comando."""

    def __init__(self, *args, **kwargs):
        self.commands = []
        self._out = queue.Queue()
        self._out.put("READY\n")
        self._alive = True
        fake = self

        class _Stdin:
            closed = False

            def write(self, data):
                for line in data.splitlines():
                    fake.commands.append(line)
                    if line == "Q":
                        fake._alive = False
                    fake._out.put(f"OK {len(fake.commands)}\n")
                    if not fake._alive:
                        fake._out.put(None)

            def flush(self):
                pass

        class _Stdout:
            def __iter__(self):
                while True:
                    item = fake._out.get()
                    if item is None:
                        return
                    yield item

        self.stdin = _Stdin()
        self.stdout = _Stdout()

    def poll(self):
        return None if self._alive else 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self._alive = False


class TouchInjectorClientTests(unittest.TestCase):
    def setUp(self):
        self.procs = []

        def popen(*args, **kwargs):
            proc = _FakeInjectorProcess()
            self.procs.append(proc)
            return proc

        self.popen = patch.object(touch_injector.subprocess, "Popen", side_effect=popen)
        self.popen.start()
        self.addCleanup(self.popen.stop)
        self.client = TouchInjectorClient("adb", "serial")
        self.client.deploy = Mock(return_value=True)

    def test_start_sets_wait_for_finish_mode(self):
        self.assertTrue(self.client.start())
        self.assertEqual(self.procs[0].commands, ["M 2"])

    def test_stroke_tap_and_drag_protocol(self):
        self.client.start()
        self.assertTrue(self.client.stroke([(10, 20), (30.4, 40.6)]))
        self.assertTrue(self.client.tap(5, 6))
        self.assertTrue(self.client.drag((486, 2184), (700, 2200), frames=2, hold_ms=50, settle_frames=1))
        cmds = self.procs[0].commands
        self.assertEqual(cmds[1], "S 0 10 20 30 41")
        self.assertEqual(cmds[2], "T 5 6")
        # Espera parada (3 quadros), arrasto em 2 quadros e 1 quadro de acomodação.
        self.assertEqual(cmds[3], "S 16.7 486 2184 486 2184 486 2184 593 2192 700 2200 700 2200")

    def test_close_sends_quit(self):
        self.client.start()
        self.client.close()
        self.assertEqual(self.procs[0].commands[-1], "Q")
        self.assertFalse(self.client.is_alive())

    def test_jar_is_shipped_with_the_project(self):
        self.assertTrue(os.path.exists(touch_injector.JAR_LOCAL_PATH))


def _offline_adb():
    adb = object.__new__(ADBManager)
    adb.adb_path = "adb"
    adb.device_target = "serial"
    adb.profile_loader = DeviceProfileLoader()
    adb.timing = Mock()
    adb.run_cmd = Mock(return_value=(True, "", ""))
    adb._abort_event = threading.Event()
    adb.touch_transport = "injector"
    adb.is_connected = True
    adb._injector_failed = False
    adb._palette_page = None
    adb.active_token = None
    adb.close_interactive_shell = Mock()
    return adb


class ADBManagerInjectorRoutingTests(unittest.TestCase):
    def setUp(self):
        self.adb = _offline_adb()
        self.injector = Mock()
        self.injector.is_alive.return_value = True
        self.injector.stroke.return_value = True
        self.injector.tap.return_value = True
        self.injector.drag.return_value = True
        self.injector.last_error = ""
        self.adb._injector = self.injector

    def test_strokes_go_through_injector_as_single_gestures(self):
        self.assertTrue(self.adb.execute_stroke_batch([[(100, 500), (600, 500)], [(100, 600), (100, 900)]]))
        self.assertEqual(self.injector.stroke.call_count, 2)
        self.adb.run_cmd.assert_not_called()
        first = self.injector.stroke.call_args_list[0].args[0]
        self.assertEqual((first[0], first[-1]), ((100, 500), (600, 500)))

    def test_injector_failure_is_not_retried_via_shell(self):
        self.injector.stroke.return_value = False
        self.assertFalse(self.adb.execute_stroke_batch([[(100, 500), (600, 500)]]))
        self.adb.run_cmd.assert_not_called()

    def test_long_press_color_holds_finger_on_anchor(self):
        self.adb._palette_page = 1
        self.adb.capture_screenshot = Mock(return_value=_screen_with_page_dots(1))
        self.adb.select_palette_color({"is_long_press": True, "palette_page": 1, "base_x": 486, "base_y": 2184,
                                       "target_x": 720, "target_y": 2240})
        args, kwargs = self.injector.drag.call_args
        self.assertEqual(args, ((486, 2184), (720, 2240)))
        self.assertGreaterEqual(kwargs["hold_ms"], config.PALETTE_LONG_PRESS_MS)

    def test_emergency_kill_closes_injector(self):
        self.adb.emergency_kill()
        self.injector.close.assert_called_once()
        self.assertIsNone(self.adb._injector)

    def test_unavailable_injector_falls_back_to_input_once(self):
        self.adb._injector = None
        with patch("studio.adb_manager.TouchInjectorClient") as client_cls:
            client_cls.return_value.start.return_value = False
            client_cls.return_value.last_error = "sem app_process"
            self.adb._send_to_interactive_shell = Mock(return_value=True)
            self.adb.get_interactive_shell = Mock(return_value=object())
            self.assertTrue(self.adb.execute_stroke_batch([[(100, 500), (600, 500)]]))
            self.assertTrue(self.adb.execute_stroke_batch([[(100, 500), (600, 500)]]))
            self.assertEqual(client_cls.return_value.start.call_count, 1)

    def test_persistent_executor_uses_injector(self):
        executor = PersistentTouchExecutor(fallback_adb=self.adb)
        stroke = Mock()
        stroke.events = [Mock(), Mock()]
        stroke.to_points_list.return_value = [(100, 500), (600, 500)]
        self.assertTrue(executor.execute_motion_stroke(stroke))
        self.injector.stroke.assert_called_once()


def _screen_with_page_dots(active):
    img = np.full((2400, 1080, 3), 20, np.uint8)
    for index, (x, y) in enumerate(config.UI_PALETTE_PAGE_DOTS, start=1):
        shade = 255 if index == active else 138
        cv2.circle(img, (x, y), 4, (shade, shade, shade), -1)
    return img


class PalettePageCheckTests(unittest.TestCase):
    def test_detects_active_page(self):
        for page in (1, 2, 3):
            self.assertEqual(ADBManager.detect_palette_page(_screen_with_page_dots(page)), page)
        self.assertIsNone(ADBManager.detect_palette_page(np.zeros((2400, 1080, 3), np.uint8)))

    def test_swipe_that_did_not_turn_the_page_is_corrected(self):
        # Medido no A73: o deslize para a página 2 às vezes não vira a página.
        adb = _offline_adb()
        injector = Mock()
        injector.is_alive.return_value = True
        injector.drag.return_value = True
        injector.tap.return_value = True
        injector.last_error = ""
        adb._injector = injector
        adb._palette_page = 1
        adb.capture_screenshot = Mock(side_effect=[_screen_with_page_dots(1), _screen_with_page_dots(2)])
        adb.select_palette_color({"palette_page": 2, "target_x": 936, "target_y": 2184})
        # Um deslize planejado + um de correção, depois o toque na cor.
        self.assertEqual(injector.drag.call_count, 2)
        injector.tap.assert_called_once_with(936, 2184)
        self.assertEqual(adb._palette_page, 2)

    def test_unreadable_page_fails_closed(self):
        adb = _offline_adb()
        injector = Mock()
        injector.is_alive.return_value = True
        injector.drag.return_value = True
        adb._injector = injector
        adb._palette_page = 1
        adb.capture_screenshot = Mock(return_value=np.zeros((2400, 1080, 3), np.uint8))
        with self.assertRaisesRegex(RuntimeError, "não identificada"):
            adb.select_palette_color({"palette_page": 1, "target_x": 411, "target_y": 2184})
        injector.tap.assert_not_called()


class SolidInkLineArtTests(unittest.TestCase):
    def test_filled_blob_becomes_rings_and_keeps_its_hole(self):
        from studio.planning.line_art_engine import LineArtEngine
        mask = np.zeros((200, 200), np.uint8)
        cv2.circle(mask, (100, 100), 45, 255, -1)
        cv2.circle(mask, (100, 100), 12, 0, -1)          # brilho do olho
        cv2.line(mask, (10, 10), (190, 10), 255, 3)      # linha fina não é área sólida
        solid = LineArtEngine.extract_solid_ink(mask, brush_width=10)
        self.assertEqual(int(solid[10, 100]), 0)
        self.assertGreater(int(solid[100, 140]), 0)
        rings = LineArtEngine.generate_solid_fill_strokes(solid, brush_width=10)
        self.assertGreaterEqual(len(rings), 3)
        drawn = np.zeros_like(mask)
        for ring in rings:
            cv2.polylines(drawn, [np.array(ring, np.int32).reshape(-1, 1, 2)], False, 255, 10)
        self.assertEqual(int(drawn[100, 100]), 0)          # o furo continua aberto
        inside = cv2.erode(solid, np.ones((3, 3), np.uint8)) > 0
        self.assertGreater(np.count_nonzero(drawn[inside]) / np.count_nonzero(inside), 0.9)


def _screen_with_thumb(y, x=39):
    img = np.full((2400, 1080, 3), 30, np.uint8)
    cv2.line(img, (x, 880), (x, 1500), (200, 200, 200), 2)
    if y is not None:
        cv2.circle(img, (x, y), 19, (255, 255, 255), -1)
    return img


class SliderThumbTests(unittest.TestCase):
    def test_detects_thumb_center(self):
        for y in (900, 1200, 1497):
            self.assertAlmostEqual(ADBManager.detect_slider_thumb_y(_screen_with_thumb(y)), y, delta=2)

    def test_missing_thumb(self):
        self.assertIsNone(ADBManager.detect_slider_thumb_y(_screen_with_thumb(None)))

    def test_drag_starts_on_thumb_and_corrects_overshoot(self):
        adb = _offline_adb()
        injector = Mock()
        injector.is_alive.return_value = True
        injector.drag.return_value = True
        adb._injector = injector
        # Bolinha em 1320; o primeiro arrasto passa do alvo (1180), o segundo acerta.
        adb.capture_screenshot = Mock(side_effect=[_screen_with_thumb(1320), _screen_with_thumb(1180),
                                                   _screen_with_thumb(1201)])
        self.assertTrue(adb.set_brush_thickness("med"))
        starts = [c.args[0] for c in injector.drag.call_args_list]
        self.assertEqual(starts, [(39, 1320), (39, 1180)])

    def test_unconfirmed_thickness_fails_closed(self):
        adb = _offline_adb()
        injector = Mock()
        injector.is_alive.return_value = True
        injector.drag.return_value = True
        adb._injector = injector
        adb.capture_screenshot = Mock(return_value=_screen_with_thumb(1450))
        with self.assertRaisesRegex(RuntimeError, "não confirmada"):
            adb.set_brush_thickness("thick")


class MeasuredColorLutTests(unittest.TestCase):
    def test_profile_lut_comes_from_measured_selector(self):
        import json
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "profiles", "SM-A736B", "color_lut.json")
        with open(path, encoding="utf-8") as f:
            lut = json.load(f)
        base = [e for e in lut["entries"] if not e["is_long_press"]]
        spectrum = [e for e in lut["entries"] if e["is_long_press"]]
        self.assertEqual(len(base), 22)
        self.assertGreater(len(spectrum), 3000)
        b = lut["selector_bounds"]
        for e in spectrum:
            self.assertTrue(b["x_min"] <= e["target_x"] <= b["x_max"])
            self.assertTrue(b["y_top"] <= e["target_y"] <= b["y_bottom"])
            self.assertEqual((e["base_x"], e["base_y"]), (486, 2184))


if __name__ == "__main__":
    unittest.main()
