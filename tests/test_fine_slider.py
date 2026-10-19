# -*- coding: utf-8 -*-
"""Posição mais fina da caneta sólida (abaixo do trilho), alcançada pela troca neon -> sólida."""
import unittest
from unittest.mock import MagicMock

from studio.adb_manager import ADBManager
from studio.device.brush_width import slider_y_for_width


class TestFineSlider(unittest.TestCase):
    def _adb(self, thumbs):
        adb = ADBManager.__new__(ADBManager)
        adb.select_pen_tool = MagicMock()
        adb.set_brush_thickness = MagicMock()
        adb.capture_screenshot = MagicMock(return_value=None)
        adb.detect_slider_thumb_y = MagicMock(side_effect=thumbs)
        adb.timing = MagicMock()
        return adb

    def test_curve_reaches_below_track(self):
        self.assertGreater(slider_y_for_width(3.0), 1550)
        self.assertEqual(slider_y_for_width(10.0), 1497)

    def test_switches_through_neon(self):
        adb = self._adb([1572])
        self.assertTrue(adb._set_below_track_thickness(39, 1470, 1496, "w3"))
        self.assertEqual([c.args[0] for c in adb.select_pen_tool.call_args_list], ["neon", "solid"])
        adb.set_brush_thickness.assert_called_once_with("min")

    def test_already_there_does_nothing(self):
        adb = self._adb([])
        self.assertTrue(adb._set_below_track_thickness(39, 1572, 1496, "w3"))
        adb.select_pen_tool.assert_not_called()

    def test_falls_back_to_track_minimum_with_warning(self):
        adb = self._adb([1497])
        self.assertTrue(adb._set_below_track_thickness(39, 1470, 1496, "w3"))
        self.assertEqual([c.args[0] for c in adb.set_brush_thickness.call_args_list], ["min", "min"])
        self.assertIn("Traço fino indisponível", adb.last_thickness_note)

    def test_fails_closed_when_thumb_disappears(self):
        adb = self._adb([None])
        with self.assertRaises(RuntimeError):
            adb._set_below_track_thickness(39, 1470, 1496, "w3")


if __name__ == "__main__":
    unittest.main()
