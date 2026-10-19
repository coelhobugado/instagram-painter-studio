"""
Suíte de testes de qualidade pictórica, resolução física e preservação de espaço negativo (P3).
Atende às metas das Etapas 0, 1 e 2 do Plano de Evolução da Qualidade do Desenho:
- O caso sentinela 'anime_face' mantém olhos e boca abertos na simulação calibrada.
- A taxa de pixels escuros não sofre ink bleed artificial.
- Poda de esporões de esqueleto elimina espinhos parasitas sem perder topologia.
- Particionamento adaptativo multi-camada respeita a física real do hardware.
"""

import os
import unittest
import numpy as np
import cv2

from studio.painter_engine import PainterEngine
from studio.planning.pictorial_quality import (
    NegativeSpaceAnalyzer,
    SkeletonOptimizer,
    AdaptiveLineArtPartitioner,
    NegativeSpaceRegion,
)
from studio.device.calibration import HardwareCalibrationEngine


class TestP3PictorialQuality(unittest.TestCase):
    """Testes de Qualidade de Desenho, Resolução Física e Espaço Negativo (P3)."""

    @classmethod
    def setUpClass(cls):
        cls.engine = PainterEngine()
        cls.canonical_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "benchmarks", "canonical"
        )
        cls.anime_path = os.path.join(cls.canonical_dir, "anime_face.png")
        cls.flower_path = os.path.join(cls.canonical_dir, "flower_lineart.png")

    def test_negative_space_preservation_anime_face(self):
        """Valida que o caso sentinela anime_face não fecha os olhos na simulação física."""
        self.assertTrue(os.path.exists(self.anime_path), f"Imagem não encontrada: {self.anime_path}")

        plan = self.engine.generate_line_art_plan(self.anime_path)
        self.assertIsNotNone(plan)

        sim_canvas = plan["simulation_image"]
        h, w = sim_canvas.shape[:2]
        gray_sim = cv2.cvtColor(sim_canvas, cv2.COLOR_BGR2GRAY)

        # 1. Taxa de pixels escuros não pode explodir (antigamente dava 12.88%, a ref tem 2.56%)
        dark_ratio = float((gray_sim < 128).mean() * 100)
        self.assertLess(dark_ratio, 6.0, f"Taxa de tinta escura ({dark_ratio:.2f}%) excedeu o limite máximo seguro de 6.0%")

        # 2. Espaço negativo interno do olho esquerdo (x=96..135, y=140..161) deve permanecer ABERTO
        # Na imagem original de 300x300, a cavidade do olho tem ~516 pixels brancos
        eye_roi = gray_sim[140:161, 96:135]
        white_pixels_in_eye = int((eye_roi > 180).sum())
        self.assertGreater(white_pixels_in_eye, 300,
                           f"Cavidade do olho fechou! Apenas {white_pixels_in_eye} pixels brancos detectados.")

        # 3. Pelo menos uma camada de traço deve usar espessura física 'min' para preservar detalhes
        thicknesses = [l.get("thickness") for l in plan["layers"] if l.get("purpose") == "line_art_centerline"]
        self.assertIn("min", thicknesses, "Detalhes do rosto devem ser atribuídos ao pincel 'min' de hardware.")

    def test_skeleton_optimizer_prune_spurs(self):
        """Valida que SkeletonOptimizer remove ramos parasitas curtos sem romper a linha mestra."""
        # Cria uma linha horizontal com pequenos esporões perpendiculares
        skel = np.zeros((50, 50), dtype=np.uint8)
        # Linha horizontal principal (y=25, x=10..40)
        skel[25, 10:41] = 255
        # Esporão 1: comprimento 2px em x=20 (y=23, 24)
        skel[23:25, 20] = 255
        # Esporão 2: comprimento 3px em x=30 (y=26, 27, 28)
        skel[26:29, 30] = 255

        pts_before = int(np.count_nonzero(skel))
        pruned = SkeletonOptimizer.prune_spurs(skel, max_spur_length=4)
        pts_after = int(np.count_nonzero(pruned))

        self.assertLess(pts_after, pts_before, "A poda deve remover pixels dos esporões espúrios.")
        # A linha principal deve permanecer contínua de x=10 até x=40
        self.assertEqual(int(np.count_nonzero(pruned[25, 10:41])), 31, "A linha mestra não pode sofrer quebras.")
        # Os esporões devem ter sido eliminados
        self.assertEqual(int(pruned[23, 20]), 0)
        self.assertEqual(int(pruned[28, 30]), 0)

    def test_hatch_decimation_preserves_air(self):
        """Valida que linhas de hachura muito próximas (< 6px) são filtradas para evitar empastamento."""
        # Cria 5 traços paralelos verticais a cada 3px (x = 10, 13, 16, 19, 22)
        strokes = [
            [(10, y) for y in range(10, 30)],
            [(13, y) for y in range(10, 30)],
            [(16, y) for y in range(10, 30)],
            [(19, y) for y in range(10, 30)],
            [(22, y) for y in range(10, 30)],
        ]
        filtered = SkeletonOptimizer.filter_dense_parallel_strokes(strokes, min_separation_px=8.0, shape=(50, 50))
        self.assertLess(len(filtered), len(strokes), "Linhas hiperdensas devem ser decimadas para deixar frestas de ar.")
        self.assertGreaterEqual(len(filtered), 2, "Traços representativos estruturais devem ser conservados.")

    def test_adaptive_line_art_partitioning(self):
        """Valida o particionamento em camadas hierárquicas ('min', 'med', 'thick')."""
        calib = HardwareCalibrationEngine()
        partitioner = AdaptiveLineArtPartitioner(calib)

        stroke_fine = [(10, 10), (10, 50)]
        stroke_med = [(50, 10), (50, 50)]
        stroke_thick = [(90, 10), (90, 50)]

        # Larguras desejadas: 4px (fino), 40px (médio), 95px (grosso).
        # Níveis físicos medidos: 10 / 58 / 104 px.
        widths = [[4.0, 4.0], [40.0, 40.0], [95.0, 95.0]]
        strokes = [stroke_fine, stroke_med, stroke_thick]

        buckets = partitioner.partition_strokes(strokes, widths)
        self.assertEqual(len(buckets["min"]["strokes"]), 1)
        self.assertEqual(len(buckets["med"]["strokes"]), 1)
        self.assertEqual(len(buckets["thick"]["strokes"]), 1)

    def test_flower_lineart_preserves_features(self):
        """Valida que o line art botânico da tulipa mantém o traço fino calibrado em 'min'."""
        if not os.path.exists(self.flower_path):
            self.skipTest(f"Arquivo não disponível: {self.flower_path}")

        plan = self.engine.generate_line_art_plan(self.flower_path)
        self.assertIsNotNone(plan)
        thicknesses = [l.get("thickness") for l in plan["layers"] if l.get("purpose") == "line_art_centerline"]
        self.assertIn("min", thicknesses, "O line art floral deve usar a caneta 'min' para evitar borramento de pétalas.")


if __name__ == "__main__":
    unittest.main()
