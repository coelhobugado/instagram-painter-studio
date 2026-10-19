# -*- coding: utf-8 -*-
"""
Testes Unitários para StructureTensorEngine (Pilar 2: Streamlines Curvilíneas).
"""
import unittest
import numpy as np
import cv2
from studio.vision.structure_tensor import StructureTensorEngine

class TestStructureTensorEngine(unittest.TestCase):
    def test_orientation_field_circle(self):
        """Em uma imagem com gradiente circular, o campo tangencial deve ser perpendicular ao raio."""
        h, w = 100, 100
        y, x = np.ogrid[:h, :w]
        # Distância radial do centro (50, 50)
        dist = np.sqrt((x - 50)**2 + (y - 50)**2).astype(np.float32)
        circle_img = np.clip(dist * 4.0, 0, 255).astype(np.uint8)

        flow_dx, flow_dy, coherence = StructureTensorEngine.compute_orientation_field(circle_img)

        self.assertEqual(flow_dx.shape, (h, w))
        self.assertEqual(flow_dy.shape, (h, w))
        self.assertEqual(coherence.shape, (h, w))

        # No ponto (50, 70), o raio aponta para baixo (0, 1). O fluxo tangencial deve ser horizontal (+/- 1, 0)
        fx = flow_dx[70, 50]
        fy = flow_dy[70, 50]
        self.assertAlmostEqual(abs(fx), 1.0, delta=0.25)
        self.assertAlmostEqual(abs(fy), 0.0, delta=0.25)

    def test_generate_curvilinear_streamlines(self):
        """Gera streamlines dentro de uma máscara e verifica contenção de limites."""
        h, w = 120, 120
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[20:100, 20:100] = 255

        # Fluxo horizontal constante
        flow_dx = np.ones((h, w), dtype=np.float32)
        flow_dy = np.zeros((h, w), dtype=np.float32)

        streamlines = StructureTensorEngine.generate_curvilinear_streamlines(
            mask, flow_dx, flow_dy, bbox=(20, 20, 80, 80), spacing=8, step_size=3.0
        )

        self.assertGreater(len(streamlines), 0)
        for line in streamlines:
            self.assertGreaterEqual(len(line), 2)
            for px, py in line:
                self.assertTrue(0 <= px < w and 0 <= py < h)
                self.assertEqual(mask[py, px], 255)

if __name__ == "__main__":
    unittest.main()
