# -*- coding: utf-8 -*-
"""Livro de colorir: cores de cada forma e contornos grudados em áreas coloridas."""
import math
import unittest

import numpy as np

from studio.painter_engine import PainterEngine


class TestColoringBookRegions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = PainterEngine().generate_line_art_plan("inputs/personagem.jpg", colorize=True, fill_regions=True)

    def _layers(self, purpose, dark=None):
        out = []
        for l in self.plan["layers"]:
            if l["purpose"] != purpose:
                continue
            if dark is not None and (sum(l["color_rgb"]) < 120) != dark:
                continue
            out.append(l)
        return out

    def test_black_outline_is_continuous_line(self):
        """O contorno preto não pode virar sobras curtas de preenchimento (saía pontilhado)."""
        self.assertFalse(self._layers("line_art_fill", dark=True))
        outline = [s for l in self._layers("line_art_centerline", dark=True) for s in l["strokes"]]
        self.assertTrue(outline)
        lengths = [sum(math.dist(a, b) for a, b in zip(s, s[1:])) for s in outline]
        self.assertGreater(float(np.median(lengths)), 40.0)

    def test_small_yellow_star_keeps_its_color(self):
        """Com k fixo, a estrela amarela era agrupada com o vermelho."""
        yellow = [l for l in self.plan["layers"]
                  if l["color_rgb"][0] > 200 and l["color_rgb"][1] > 160 and l["color_rgb"][2] < 120]
        self.assertTrue(yellow)
        self.assertGreater(sum(l["stroke_count"] for l in yellow), 3)

    def test_plan_is_approved(self):
        self.assertTrue(self.plan["quality_gate"].passed)


if __name__ == "__main__":
    unittest.main()
