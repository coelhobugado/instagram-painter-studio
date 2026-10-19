# -*- coding: utf-8 -*-
"""
Testes Unitários para as novas estratégias de preenchimento (Streamlines e Chiaroscuro).
"""
import unittest
import numpy as np
from studio.planning.fill_strategies import FillStrategyEngine

class TestFillStrategiesV2(unittest.TestCase):
    def test_streamline_tensor_strategy(self):
        """Verifica a estratégia 'streamline_tensor' em uma máscara com gradiente."""
        h, w = 100, 100
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[20:80, 20:80] = 255
        bbox = (20, 20, 60, 60)

        strokes = FillStrategyEngine.generate_region_strokes(
            mask, bbox, strategy="streamline_tensor", brush_spacing=6
        )
        self.assertGreater(len(strokes), 0)
        for s in strokes:
            self.assertGreaterEqual(len(s), 2)
            for px, py in s:
                self.assertTrue(0 <= px < w and 0 <= py < h)
                self.assertEqual(mask[py, px], 255)

    def test_chiaroscuro_cross_strategy(self):
        """Verifica a estratégia 'chiaroscuro_cross' gerando malha sobreposta."""
        h, w = 100, 100
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[20:80, 20:80] = 255
        bbox = (20, 20, 60, 60)

        strokes = FillStrategyEngine.generate_region_strokes(
            mask, bbox, strategy="chiaroscuro_cross", brush_spacing=8, texture_angle=45.0
        )
        self.assertGreater(len(strokes), 0)
        for s in strokes:
            self.assertGreaterEqual(len(s), 2)
            for px, py in s:
                self.assertTrue(0 <= px < w and 0 <= py < h)
                self.assertEqual(mask[py, px], 255)

if __name__ == "__main__":
    unittest.main()
