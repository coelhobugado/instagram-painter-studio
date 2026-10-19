# -*- coding: utf-8 -*-
"""Traço fino (~4 px, caneta sólida abaixo do trilho) nas linhas finas dos modos de desenho."""
import unittest
from unittest.mock import patch

from studio import config
from studio.device.brush_width import FINE_LEVEL, FINE_DRAWN_PX, drawn_width_px, is_fine_level
from studio.device.fake_adb import FakeADB
from studio.execution_worker import ExecutionWorker
from studio.feedback.error_map import ErrorMapEngine
from studio.painter_engine import PainterEngine
from studio.planning.area_fill import estimate_paint_time, FINE_SWITCH_EXTRA_S
from studio.planning.line_art_engine import LineArtEngine


class TestWidthModel(unittest.TestCase):
    def test_drawn_width(self):
        self.assertTrue(is_fine_level(FINE_LEVEL))
        self.assertFalse(is_fine_level("min"))
        self.assertEqual(drawn_width_px(FINE_LEVEL, 3.0), FINE_DRAWN_PX)
        self.assertGreater(drawn_width_px("min", 10.0), 10.0)

    def test_width_class_uses_fine_for_thin_lines(self):
        self.assertEqual(LineArtEngine._width_class(3.0, 10.0, fine=True), 3)
        self.assertEqual(LineArtEngine._width_class(7.0, 10.0, fine=True), 3)
        self.assertEqual(LineArtEngine._width_class(13.0, 10.0, fine=True), 10)
        self.assertEqual(LineArtEngine._width_class(3.0, 10.0, fine=False), 10)
        self.assertEqual(LineArtEngine._line_level(3, 10.0), FINE_LEVEL)
        self.assertEqual(LineArtEngine._line_level(10, 10.0), "min")

    def test_entering_fine_costs_more_time(self):
        stroke = [[(0, 0), (100, 0)]]
        normal = estimate_paint_time([{"strokes": stroke, "thickness": "min", "color_rgb": (1, 1, 1)}])
        fine = estimate_paint_time([{"strokes": stroke, "thickness": FINE_LEVEL, "color_rgb": (1, 1, 1)}])
        self.assertAlmostEqual(fine - normal, FINE_SWITCH_EXTRA_S, places=1)


class TestFinePlans(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = PainterEngine()
        cls.lines = cls.engine.generate_line_art_plan("inputs/flores.jpg", colorize=True)

    def _line_layers(self, plan):
        return [l for l in plan["layers"] if l.get("purpose") == "line_art_centerline"]

    def test_line_art_thin_lines_use_fine_stroke(self):
        fine = [l for l in self._line_layers(self.lines) if l["thickness"] == FINE_LEVEL]
        self.assertTrue(fine)
        self.assertTrue(all(l["drawn_width_px"] == FINE_DRAWN_PX for l in fine))
        self.assertTrue(self.lines["quality_gate"].passed)

    def test_fine_layers_come_last(self):
        kinds = [is_fine_level(l["thickness"]) for l in self.lines["layers"]]
        self.assertEqual(kinds, sorted(kinds))

    def test_flat_outline_and_details_fine_and_last(self):
        plan = self.engine.generate_flat_illustration_plan("inputs/personagem.jpg", num_colors=8,
                                                           include_outline=True, include_details=True)
        outline = [l for l in plan["layers"] if "Contorno" in l["name"]]
        self.assertTrue(outline and outline[0]["thickness"] == FINE_LEVEL)
        kinds = [is_fine_level(l["thickness"]) for l in plan["layers"]]
        self.assertEqual(kinds, sorted(kinds))
        self.assertTrue(all(l["name"].startswith(f"Camada {i + 1}:") for i, l in enumerate(plan["layers"])))

    def test_fine_lines_can_be_disabled(self):
        with patch.object(config, "FINE_LINES", False):
            plan = self.engine.generate_line_art_plan("inputs/flores.jpg", colorize=True)
        self.assertFalse(any(is_fine_level(l["thickness"]) for l in plan["layers"]))

    def test_fake_adb_execution_matches_preview(self):
        fake = FakeADB()
        done = {}
        ExecutionWorker(fake, self.lines, {"on_finished": lambda ok, msg: done.update(ok=ok)}).run()
        self.assertTrue(done.get("ok"))
        self.assertIn(f"set_brush_thickness_{FINE_LEVEL}", fake.executed_commands)
        sim = self.lines["simulation_image"]
        t = self.lines["transform"]
        h, w = sim.shape[:2]
        crop = fake.capture_screenshot()[t.offset_y:t.offset_y + h, t.offset_x:t.offset_x + w]
        _, metrics = ErrorMapEngine.compute_error_map(sim, crop)
        self.assertGreater(metrics["ssim"], 0.95)


if __name__ == "__main__":
    unittest.main()
