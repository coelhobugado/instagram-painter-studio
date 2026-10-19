"""
Testes Unitários do Motor de Cor Perceptual CIE Lab e CIEDE2000.
"""
import unittest
import numpy as np
from studio.vision.perceptual_color import PerceptualColorEngine

class TestPerceptualColor(unittest.TestCase):
    def setUp(self):
        self.engine = PerceptualColorEngine()

    def test_rgb_to_lab_roundtrip(self):
        """Valida precisão de ida e volta entre RGB e Lab."""
        test_rgbs = [(255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (215, 165, 130)]
        for r, g, b in test_rgbs:
            L, a, b_val = self.engine.rgb_to_lab(r, g, b)
            r2, g2, b2 = self.engine.lab_to_rgb(L, a, b_val)
            self.assertLess(abs(r - r2), 3)
            self.assertLess(abs(g - g2), 3)
            self.assertLess(abs(b - b2), 3)

    def test_ciede2000_identical_colors(self):
        """Duas cores idênticas devem ter Delta E exatamente igual a 0."""
        lab = (50.0, 25.0, -15.0)
        de = self.engine.delta_e_ciede2000(lab, lab)
        self.assertAlmostEqual(de, 0.0, places=4)

    def test_ciede2000_distinct_colors(self):
        """Cores distintas devem apresentar Delta E positivo coerente."""
        white = self.engine.rgb_to_lab(255, 255, 255)
        black = self.engine.rgb_to_lab(0, 0, 0)
        de = self.engine.delta_e_ciede2000(white, black)
        self.assertGreater(de, 50.0)

    def test_lut_sample_count(self):
        """Valida que a LUT possui amostras suficientes especificadas e documentadas."""
        self.assertGreaterEqual(len(self.engine.entries), 161)

    def test_match_skin_tone(self):
        """Valida correspondência precisa para tons de pele na escala Fitzpatrick."""
        # Tom de pele morena clara
        match = self.engine.match_color((215, 165, 130))
        # Os tons de pele ficam na página 2 da paleta (bolinhas e nuances).
        self.assertEqual(match["palette_page"], 2)
        self.assertLess(match["delta_e"], 5.0)

if __name__ == "__main__":
    unittest.main()
