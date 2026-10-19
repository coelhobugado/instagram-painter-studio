# -*- coding: utf-8 -*-
"""Modo Texto: diagramação ajustada ao canvas, pingos, ordem de escrita e plano aprovado."""
import os
import unittest

import numpy as np

from studio.painter_engine import PainterEngine
from studio.planning import text_engine
from studio.planning.text_engine import TextEngine, TextFitError

ARIAL = os.path.join(text_engine.FONTS_DIR, "arial.ttf")
BOUNDS = {"left": 80, "top": 380, "right": 1000, "bottom": 1980}


@unittest.skipUnless(os.path.exists(ARIAL), "Arial não instalada")
class TestTextLayout(unittest.TestCase):
    def test_largest_size_that_fits(self):
        layout = TextEngine.fit_layout("Oi", ARIAL, 900, 1600, brush_px=10)
        bigger = TextEngine._measure("Oi", ARIAL, layout.font_size + 1, 890, 0.15)
        self.assertLessEqual(layout.width, 890)
        self.assertLessEqual(layout.height, 1590)
        self.assertTrue(bigger is None or bigger[3] > 890 or bigger[4] > 1590)

    def test_wraps_long_text(self):
        layout = TextEngine.fit_layout("um dois tres quatro cinco seis sete oito", ARIAL, 900, 1600, brush_px=10)
        self.assertGreater(len(layout.lines), 1)
        self.assertEqual(" ".join(layout.lines).split(), "um dois tres quatro cinco seis sete oito".split())

    def test_refuses_text_below_legible_size(self):
        with self.assertRaises(TextFitError):
            TextEngine.fit_layout("palavra " * 200, ARIAL, 900, 1600, brush_px=10)

    def test_max_cap_height(self):
        layout = TextEngine.fit_layout("Oi", ARIAL, 900, 1600, brush_px=10, max_cap_px=80)
        self.assertLessEqual(layout.cap_height, 80)
        self.assertGreater(layout.cap_height, 75)


@unittest.skipUnless(os.path.exists(ARIAL), "Arial não instalada")
class TestTextPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = PainterEngine()

    def _strokes(self, plan):
        return [s for layer in plan["layers"] for s in layer["strokes"]]

    def test_i_keeps_its_dot(self):
        plan = self.engine.generate_text_plan("i", font="Arial", custom_bounds=BOUNDS, max_cap_px=80)
        strokes = self._strokes(plan)
        self.assertEqual(len(strokes), 2, "haste e pingo")
        tops = sorted(min(p[1] for p in s) for s in strokes)
        self.assertLess(tops[0] + 20, tops[1], "pingo acima da haste")

    def test_dots_survive_on_thicker_brush(self):
        plan = self.engine.generate_text_plan("ii!", font="Arial", custom_bounds=BOUNDS, max_cap_px=112,
                                              stroke="font")
        self.assertGreater(plan["text_layout"]["brush_px"], 10)
        self.assertEqual(len(self._strokes(plan)), 6)

    def test_writes_left_to_right(self):
        plan = self.engine.generate_text_plan("I l I", font="Arial", custom_bounds=BOUNDS, max_cap_px=80)
        xs = [np.mean([p[0] for p in s]) for s in self._strokes(plan)]
        self.assertEqual(xs, sorted(xs))

    def test_brush_follows_font_weight(self):
        kw = dict(custom_bounds=BOUNDS, stroke="font")
        small = self.engine.generate_text_plan("Oi", font="Arial", max_cap_px=80, **kw)
        regular = self.engine.generate_text_plan("Oi", font="Arial", max_cap_px=160, **kw)
        bold = self.engine.generate_text_plan("Oi", font="Arial Negrito", max_cap_px=160, **kw)
        self.assertEqual(small["layers"][0]["thickness"], "min")
        self.assertGreater(bold["text_layout"]["brush_px"], regular["text_layout"]["brush_px"])
        self.assertTrue(bold["layers"][0]["thickness"].startswith("w"))

    def test_slider_is_set_below_the_drawn_width(self):
        """Letras saem ~4 px mais largas que o nominal do slider (medido no A73)."""
        plan = self.engine.generate_text_plan("Oi", font="Arial Negrito", custom_bounds=BOUNDS, max_cap_px=160,
                                              stroke="font")
        layer = plan["layers"][0]
        drawn = plan["text_layout"]["brush_px"]
        self.assertEqual(layer["drawn_width_px"], drawn)
        self.assertAlmostEqual(float(layer["thickness"][1:]), drawn - text_engine.LETTER_WIDTH_GAIN_PX, delta=1.0)

    def test_plan_is_approved_and_inside_canvas(self):
        plan = self.engine.generate_text_plan("Oi, tudo bem?", font="Arial", custom_bounds=BOUNDS)
        self.assertEqual(plan["mode"], "text")
        self.assertTrue(plan["quality_gate"].passed)
        t = plan["transform"]
        for s in self._strokes(plan):
            for x, y in s:
                self.assertTrue(80 <= x + t.offset_x <= 1000 and 380 <= y + t.offset_y <= 1980)

    def test_fine_stroke_is_default(self):
        plan = self.engine.generate_text_plan("Oi", font="Arial Negrito", custom_bounds=BOUNDS, max_cap_px=160)
        layer = plan["layers"][0]
        self.assertEqual(layer["thickness"], text_engine.FINE_SLIDER_LEVEL)
        self.assertEqual(layer["drawn_width_px"], text_engine.FINE_DRAWN_PX)
        self.assertTrue(plan["quality_gate"].passed)

    def test_fine_stroke_fits_text_that_font_stroke_refuses(self):
        text = "Feliz aniversário! " * 20
        with self.assertRaises(TextFitError):
            self.engine.generate_text_plan(text, font="Arial", custom_bounds=BOUNDS, stroke="font")
        plan = self.engine.generate_text_plan(text, font="Arial", custom_bounds=BOUNDS)
        self.assertGreaterEqual(plan["text_layout"]["cap_height_px"], text_engine.MIN_CAP_PX)

    def test_ink_color(self):
        plan = self.engine.generate_text_plan("A", font="Arial", custom_bounds=BOUNDS, ink_rgb=(230, 40, 40))
        self.assertEqual(plan["layers"][0]["target_color_rgb"], (230, 40, 40))


class TestCornerSplit(unittest.TestCase):
    def test_right_angle_becomes_two_strokes(self):
        t = [(0.0, 0.0), (30.0, 0.0), (30.0, 80.0)]  # travessão -> haste do "T"
        pieces = TextEngine.split_at_corners(t)
        self.assertEqual(len(pieces), 2)
        self.assertEqual(pieces[0][-1], pieces[1][0])

    def test_smooth_curve_stays_whole(self):
        circle = [(40 * np.cos(a), 40 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 60)]
        self.assertEqual(len(TextEngine.split_at_corners(circle)), 1)


class TestFineStrokeCleanup(unittest.TestCase):
    def test_retraced_segment_is_removed(self):
        stem = [(50.0, 10.0), (50.0, 90.0)]
        bowl_back_up_stem = [(10.0, 60.0), (30.0, 90.0), (50.0, 90.0), (50.0, 50.0)]
        out = TextEngine.remove_retraced([stem, bowl_back_up_stem], (120, 120))
        self.assertEqual(out[0], stem)
        rest = out[1:]
        self.assertTrue(rest)
        self.assertTrue(all(min(p[0] for p in s) < 47 for s in rest), "o repasse pela haste saiu")

    def test_crossing_is_not_split(self):
        stem = [(50.0, 10.0), (50.0, 90.0)]
        crossbar = [(30.0, 40.0), (70.0, 40.0)]
        out = TextEngine.remove_retraced([stem, crossbar], (120, 120))
        self.assertEqual(len(out), 2)

    def test_cedilla_pieces_join_below_baseline_only(self):
        baseline = [100.0]
        a = [(50.0, 98.0), (49.0, 108.0)]
        b = [(52.0, 110.0), (58.0, 116.0), (45.0, 120.0)]
        self.assertEqual(len(TextEngine.join_small_fragments([a, b], 30.0, baseline)), 1)
        above_a = [(50.0, 40.0), (49.0, 50.0)]
        above_b = [(52.0, 52.0), (58.0, 58.0)]
        self.assertEqual(len(TextEngine.join_small_fragments([above_a, above_b], 30.0, baseline)), 2)

    def test_spike_removed(self):
        path = [(0.0, 0.0), (5.0, 8.0), (0.0, 11.0), (7.0, 9.0), (12.0, 14.0)]
        self.assertNotIn((0.0, 11.0), TextEngine._remove_spikes(path))


@unittest.skipUnless(os.path.exists(ARIAL), "Arial não instalada")
class TestTextMCP(unittest.TestCase):
    def test_plan_text_tool(self):
        from studio.mcp.server import TOOL_HANDLERS
        res = TOOL_HANDLERS["ig_plan_text"]({"text": "Oi", "font": "Arial", "max_cap_px": 80})
        self.assertTrue(res["success"])
        self.assertEqual(res["mode"], "text")
        self.assertTrue(res["quality_gate_passed"])
        self.assertEqual(res["text_layout"]["lines"], ["Oi"])

    def test_plan_text_tool_reports_overflow(self):
        from studio.mcp.server import TOOL_HANDLERS
        res = TOOL_HANDLERS["ig_plan_text"]({"text": "palavra " * 200})
        self.assertFalse(res["success"])
        self.assertIn("mínimo legível", res["error"])


if __name__ == "__main__":
    unittest.main()
