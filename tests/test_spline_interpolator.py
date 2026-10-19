# -*- coding: utf-8 -*-
"""
Testes Unitários para o SplineInterpolator (Pilar 3: Suavização Caligráfica C¹).
"""
import unittest
import numpy as np
from studio.planning.spline_interpolator import SplineInterpolator

class TestSplineInterpolator(unittest.TestCase):
    def test_spline_preserves_endpoints(self):
        """Verifica se a interpolação Catmull-Rom preserva exatamente o início e o fim."""
        pts = [(10, 10), (50, 80), (120, 30), (200, 150)]
        smoothed = SplineInterpolator.catmull_rom_spline(pts, num_samples_per_segment=5)
        self.assertGreater(len(smoothed), len(pts))
        self.assertEqual(smoothed[0], pts[0])
        self.assertEqual(smoothed[-1], pts[-1])

    def test_spline_straight_line_invariance(self):
        """Verifica se traços que já são perfeitamente retos não sofrem distorções espúrias."""
        straight = [(10, 20), (50, 20), (100, 20), (180, 20)]
        result = SplineInterpolator.smooth_stroke_with_safety(straight)
        # Deve retornar o traço original ou pontos todos com y=20
        for pt in result:
            self.assertEqual(pt[1], 20)

    def test_spline_mask_safety_concave(self):
        """
        Verifica se a spline respeita uma máscara com concavidade/furo.
        Se uma spline tentar 'cortar caminho' por fora da máscara, deve fazer fallback seguro.
        """
        mask = np.zeros((200, 200), dtype=np.uint8)
        # Região em formato de 'L':
        mask[20:180, 20:60] = 255
        mask[140:180, 20:180] = 255

        # Traço contido no 'L' contornando a quina:
        stroke = [(40, 30), (40, 160), (160, 160)]
        smoothed = SplineInterpolator.smooth_stroke_with_safety(stroke, mask=mask)

        # Cada ponto retornado DEVE estar estritamente dentro da máscara
        for px, py in smoothed:
            self.assertTrue(0 <= px < 200 and 0 <= py < 200, f"Ponto fora dos limites: ({px}, {py})")
            self.assertEqual(mask[py, px], 255, f"Ponto violou a máscara: ({px}, {py})")

    def test_smooth_all_strokes(self):
        """Verifica processamento em lote de múltiplos traços."""
        strokes = [
            [(10, 10), (20, 30), (40, 20)],
            [(5, 5), (15, 15)]
        ]
        res = SplineInterpolator.smooth_all_strokes(strokes)
        self.assertEqual(len(res), 2)
        self.assertGreaterEqual(len(res[0]), 3)

if __name__ == "__main__":
    unittest.main()
