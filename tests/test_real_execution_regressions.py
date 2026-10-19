"""Regressões do caminho que envia cores e traços ao Android."""
import math
import threading
import unittest
from unittest.mock import Mock
import numpy as np

from studio.adb_manager import ADBManager
from studio.device.profile_loader import DeviceProfileLoader
from studio.vision.perceptual_color import PerceptualColorEngine


class RealExecutionRegressions(unittest.TestCase):
    def setUp(self):
        # Evita iniciar ADB: exercita somente a composição dos comandos.
        self.adb = object.__new__(ADBManager)
        self.adb.profile_loader = DeviceProfileLoader()
        self.adb.timing = Mock()
        self.adb.run_cmd = Mock(return_value=(True, "", ""))
        self.adb._abort_event = threading.Event()

    def test_long_press_keeps_finger_down_until_color_is_selected(self):
        touch = {"is_long_press": True, "base_x": 636, "base_y": 2184,
                 "target_x": 720, "target_y": 2155}
        self.adb.select_palette_color(touch)
        command = self.adb.run_cmd.call_args.args[0][1]
        self.assertIn("DOWN 636 2184", command)
        self.assertIn("sleep 0.5", command)
        self.assertIn("MOVE 720 2155", command)
        self.assertIn("UP 720 2155", command)
        self.assertLess(command.index("DOWN"), command.index("MOVE"))

    def test_color_selection_failure_is_visible(self):
        self.adb.run_cmd.return_value = (False, "", "disconnected")
        with self.assertRaisesRegex(RuntimeError, "disconnected"):
            self.adb.select_palette_color({"target_x": 411, "target_y": 2184})

    def test_brush_visual_check_retries_one_transient_frame(self):
        self.adb.capture_screenshot = Mock(return_value=np.zeros((2400, 1080, 3), dtype=np.uint8))
        self.adb._slider_changed_at_targets = Mock(side_effect=[False, True])
        self.assertTrue(self.adb.prepare_brush_thickness("min"))
        self.assertEqual(self.adb._slider_changed_at_targets.call_count, 2)
        self.assertEqual(self.adb.run_cmd.call_count, 4)

    def test_palette_navigates_to_second_and_third_pages(self):
        for page, x in ((2, 936), (3, 636)):
            with self.subTest(page=page):
                self.adb.run_cmd.reset_mock()
                self.adb._palette_page = None
                self.adb.select_palette_color({"palette_page": page, "target_x": x, "target_y": 2184})
                commands = [call.args[0][1] for call in self.adb.run_cmd.call_args_list]
                # Página desconhecida: volta à primeira com um deslize de folga.
                self.assertEqual(commands[:3], ["input swipe 450 2184 800 2184 350"] * 3)
                self.assertEqual(commands[3:-1], ["input swipe 850 2184 500 2184 350"] * (page - 1))
                self.assertEqual(commands[-1], f"input tap {x} 2184")

    def test_palette_remembers_page_and_swipes_only_the_difference(self):
        self.adb._palette_page = None
        self.adb.select_palette_color({"palette_page": 3, "target_x": 636, "target_y": 2184})
        self.adb.run_cmd.reset_mock()
        self.adb.select_palette_color({"palette_page": 2, "target_x": 936, "target_y": 2184})
        commands = [call.args[0][1] for call in self.adb.run_cmd.call_args_list]
        self.assertEqual(commands, ["input swipe 450 2184 800 2184 350", "input tap 936 2184"])
        self.adb.run_cmd.reset_mock()
        self.adb.select_palette_color({"palette_page": 2, "target_x": 486, "target_y": 2184})
        commands = [call.args[0][1] for call in self.adb.run_cmd.call_args_list]
        self.assertEqual(commands, ["input tap 486 2184"])

    def test_failed_color_selection_forgets_palette_page(self):
        self.adb._palette_page = 2
        self.adb.run_cmd.return_value = (False, "", "disconnected")
        with self.assertRaises(RuntimeError):
            self.adb.select_palette_color({"palette_page": 2, "target_x": 411, "target_y": 2184})
        self.assertIsNone(self.adb._palette_page)

    def test_dark_color_is_real_measured_swatch(self):
        # O seletor de nuances da página 2 chega abaixo da bolinha #262626;
        # preto absoluto não existe no Instagram.
        match = PerceptualColorEngine().match_color((0, 0, 0))
        self.assertEqual(match["palette_page"], 2)
        self.assertLessEqual(max(match["matched_rgb"]), 38)
        self.assertNotEqual(match["matched_hex"], "#000000")

    def test_measured_detail_keeps_its_palette_page_and_anchor(self):
        match = PerceptualColorEngine().match_color((140, 140, 140))
        self.assertEqual(match["palette_page"], 3)
        self.assertTrue(match["is_long_press"])
        # Nuance medida no seletor: toque longo na bolinha-âncora e soltura dentro da faixa.
        self.assertEqual((match["base_x"], match["base_y"]), (486, 2184))
        self.assertTrue(375 <= match["target_x"] <= 974 and 2133 <= match["target_y"] <= 2274)

    def test_compaction_preserves_ends_and_bounds_event_count(self):
        points = [(x, 400) for x in range(100, 601)]
        compact = self.adb._compact_stroke(points)
        self.assertEqual(compact[0], points[0])
        self.assertEqual(compact[-1], points[-1])
        self.assertLess(len(compact), len(points) / 10)
        self.assertEqual(len(compact), 2)

    def test_whole_straight_line_is_one_android_gesture(self):
        self.adb._send_to_interactive_shell = Mock(return_value=True)
        self.adb.get_interactive_shell = Mock(return_value=object())
        self.adb.execute_stroke_batch([[(100, 400), (150, 400)]])
        commands = "".join(self.adb._send_to_interactive_shell.call_args.args[0])
        self.assertEqual(commands, "input swipe 100 400 150 400 55\n")

    def test_resolution_uses_profile_screen_resolution(self):
        profile = DeviceProfileLoader()
        profile.set_target_resolution(540, 1200)
        self.assertEqual(profile.scale_point(1080, 2400), (540, 1200))


if __name__ == "__main__":
    unittest.main()
