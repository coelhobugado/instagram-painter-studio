# -*- coding: utf-8 -*-
"""
Testes Unitários para o SemanticFeatureExtractor (Pilar 1: Isolamento Semântico e Destaque Anatômico).
"""
import unittest
import numpy as np
import cv2
from studio.vision.semantic_features import SemanticFeatureExtractor

class TestSemanticFeatureExtractor(unittest.TestCase):
    def test_extract_specular_highlights(self):
        """Verifica a identificação de micro-ilhas de brilho especular em uma imagem simulada de olho."""
        img = np.full((100, 100), 50, dtype=np.uint8)  # Fundo escuro (esclera/íris)
        # Cria um catchlight brilhante de 3x3 no centro (50, 50)
        img[49:52, 49:52] = 250

        specular = SemanticFeatureExtractor.extract_specular_highlights(img, min_val=220, max_island_area=50)

        self.assertEqual(specular.shape, (100, 100))
        self.assertGreater(np.count_nonzero(specular), 0)
        self.assertEqual(specular[50, 50], 255)
        # O fundo escuro não deve estar na máscara
        self.assertEqual(specular[10, 10], 0)

    def test_specular_preservation_mask(self):
        """Verifica se o brilho especular é corretamente recortado da camada base."""
        base_mask = np.full((100, 100), 255, dtype=np.uint8)
        specular = np.zeros((100, 100), dtype=np.uint8)
        specular[45:55, 45:55] = 255

        safe_mask = SemanticFeatureExtractor.create_specular_preservation_mask(
            base_mask, specular, dilation_radius=2
        )

        # O centro protegido deve ter valor 0
        self.assertEqual(safe_mask[50, 50], 0)
        # As bordas externas devem permanecer 255
        self.assertEqual(safe_mask[10, 10], 255)

    def test_micro_detail_strokes_generation(self):
        """Verifica a extração de micro-traços anatômicos a partir de bordas."""
        img = np.zeros((100, 100), dtype=np.uint8)
        # Desenha uma linha fina de alto contraste (simulando linha de pálpebra/cílio)
        cv2.line(img, (20, 50), (80, 50), 255, 1)

        strokes = SemanticFeatureExtractor.generate_micro_detail_strokes(img, min_stroke_length=10.0)
        self.assertGreater(len(strokes), 0)
        for s in strokes:
            self.assertGreaterEqual(len(s), 2)

if __name__ == "__main__":
    unittest.main()
