# -*- coding: utf-8 -*-
"""Ilustração chapada: cores pequenas e distintas, miolos cobertos e bordas sem halo."""
import math
import unittest

import cv2
import numpy as np

from studio.painter_engine import PainterEngine
from studio.planning.area_fill import fill_region
from studio.planning.flat_illustration import AREA_LEVELS


class TestAreaFillTaps(unittest.TestCase):
    def test_small_core_tap_survives_sanitizer(self):
        """O toque no miolo de um anel tinha 1 px e era descartado (< 2,5 px), deixando um buraco."""
        mask = np.zeros((300, 300), np.uint8)
        cv2.fillPoly(mask, [np.array([(150, 40), (250, 120), (215, 250), (85, 250), (50, 120)], np.int32)], 255)
        result = fill_region(mask, AREA_LEVELS, thin_level=("min", 10.0))
        for strokes in result.values():
            for s in strokes:
                self.assertGreaterEqual(sum(math.dist(a, b) for a, b in zip(s, s[1:])), 2.5)
        widths = dict(AREA_LEVELS)
        painted = np.zeros_like(mask)
        for lvl, strokes in result.items():
            for s in strokes:
                cv2.polylines(painted, [np.int32(s).reshape(-1, 1, 2)], False, 255, int(widths[lvl]))
        self.assertGreater(np.count_nonzero(painted[mask > 0]) / np.count_nonzero(mask), 0.97)


class TestPortraitDetails(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = PainterEngine().generate_flat_illustration_plan("inputs/retrato.jpg", num_colors=10)

    def test_small_distinct_colors_survive(self):
        """Broche vermelho e boca rosa viravam a cor da pele (faltavam grupos de cor)."""
        rgbs = [l["color_rgb"] for l in self.plan["layers"]]
        self.assertTrue(any(r > 180 and g < 90 and b < 90 for r, g, b in rgbs), "broche vermelho")
        self.assertTrue(any(r > 160 and 80 < g < 140 and 100 < b < 160 for r, g, b in rgbs), "boca rosa")

    def test_no_unpainted_hole_in_collar(self):
        sim = self.plan["simulation_image"]
        h, w = sim.shape[:2]
        dark = (sim.max(axis=2) < 40).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(dark)
        holes = [stats[i] for i in range(1, n)
                 if stats[i][4] > 200 and stats[i][1] > h * 0.6 and w * 0.3 < stats[i][0] < w * 0.7]
        self.assertFalse(holes)

    def test_plan_is_approved(self):
        self.assertTrue(self.plan["quality_gate"].passed)


if __name__ == "__main__":
    unittest.main()
