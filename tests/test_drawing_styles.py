"""Estilos de desenho: Traçado colorido, Livro de colorir e Ilustração chapada (offline)."""
import os
import tempfile
import unittest

import cv2
import numpy as np

from studio.painter_engine import PainterEngine
from studio.planning.area_fill import fill_region, estimate_paint_time


def _write(img):
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    cv2.imwrite(path, img)
    return path


def _layer_rgb(layer):
    return tuple(int(v) for v in layer["render_color_rgb"])


class AreaFillTests(unittest.TestCase):
    def test_fill_stays_inside_and_covers(self):
        mask = np.zeros((300, 300), np.uint8)
        cv2.rectangle(mask, (40, 60), (260, 240), 255, -1)
        cv2.circle(mask, (150, 150), 25, 0, -1)  # furo
        result = fill_region(mask, (("thick", 104.0), ("med", 58.0), ("min", 10.0)))
        drawn = np.zeros_like(mask)
        widths = {"thick": 104, "med": 58, "min": 10}
        for lvl, strokes in result.items():
            for s in strokes:
                cv2.polylines(drawn, [np.asarray(s, np.int32).reshape(-1, 1, 2)], False, 255, widths[lvl])
        outside = np.count_nonzero(drawn[mask == 0])
        self.assertLess(outside, 0.01 * np.count_nonzero(mask))           # quase nada vaza
        self.assertEqual(int(drawn[150, 150]), 0)                         # o furo fica aberto
        self.assertGreater(np.count_nonzero(drawn[mask > 0]) / np.count_nonzero(mask), 0.95)

    def test_paint_time_counts_switches(self):
        layer = {"strokes": [[(0, 0), (180, 0)]], "render_color_rgb": (1, 2, 3), "thickness": "min"}
        one = estimate_paint_time([layer])
        two = estimate_paint_time([layer, dict(layer, render_color_rgb=(9, 9, 9))])
        self.assertGreater(two - one, 0.9)  # troca de cor custa ~1 s


class ColoredStylesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = PainterEngine()

    def test_crossing_lines_keep_their_own_colors(self):
        img = np.full((400, 400, 3), 255, np.uint8)
        cv2.circle(img, (150, 200), 90, (40, 40, 220), 5, cv2.LINE_AA)     # vermelho
        cv2.circle(img, (260, 200), 90, (200, 110, 30), 5, cv2.LINE_AA)    # azul
        path = _write(img)
        try:
            plan = self.engine.generate_line_art_plan(path, colorize=True)
        finally:
            os.remove(path)
        lines = [l for l in plan["layers"] if l["purpose"] == "line_art_centerline"]
        reds = [l for l in lines if _layer_rgb(l)[0] > 150 and _layer_rgb(l)[2] < 120]
        blues = [l for l in lines if _layer_rgb(l)[2] > 150 and _layer_rgb(l)[0] < 120]
        self.assertTrue(reds and blues, [l["name"] for l in lines])
        red_len = sum(len(s) for l in reds for s in l["strokes"])
        blue_len = sum(len(s) for l in blues for s in l["strokes"])
        self.assertGreater(min(red_len, blue_len) / max(red_len, blue_len), 0.5)

    def test_coloring_book_paints_regions_between_lines(self):
        img = np.full((400, 400, 3), 255, np.uint8)
        cv2.rectangle(img, (80, 80), (320, 320), (60, 180, 60), -1)          # área verde
        cv2.rectangle(img, (80, 80), (320, 320), (0, 0, 0), 6, cv2.LINE_AA)  # contorno preto
        path = _write(img)
        try:
            plain = self.engine.generate_line_art_plan(path, colorize=True)
            book = self.engine.generate_line_art_plan(path, colorize=True, fill_regions=True)
        finally:
            os.remove(path)
        self.assertEqual(book.get("style"), "coloring_book")
        fills = [l for l in book["layers"] if l["purpose"] == "line_art_fill"]
        self.assertTrue(any(_layer_rgb(l)[1] > _layer_rgb(l)[0] + 40 for l in fills))  # alguma área verde
        sim = book["simulation_image"]
        h, w = sim.shape[:2]
        center = sim[h // 2 - 10:h // 2 + 10, w // 2 - 10:w // 2 + 10].reshape(-1, 3).mean(axis=0)
        self.assertGreater(center[1], center[2] + 40)  # o miolo saiu verde na prévia
        self.assertIsNotNone(plain)

    def test_flat_illustration_has_flat_editable_colors(self):
        img = np.zeros((300, 300, 3), np.uint8)
        img[:] = (230, 220, 210)
        cv2.circle(img, (150, 150), 100, (60, 90, 200), -1)
        cv2.circle(img, (150, 150), 40, (40, 160, 60), -1)
        path = _write(img)
        try:
            plan = self.engine.generate_flat_illustration_plan(path, num_colors=4, include_details=False)
        finally:
            os.remove(path)
        self.assertEqual(plan["mode"], "flat_illustration")
        self.assertGreaterEqual(len(plan["palette_colors"]), 3)
        self.assertTrue(all("palette_index" in l for l in plan["layers"]))
        self.assertTrue(plan["editable_regions"])
        self.assertTrue(getattr(plan["quality_gate"], "passed", False))


if __name__ == "__main__":
    unittest.main()
