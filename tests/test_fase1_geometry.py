# -*- coding: utf-8 -*-
"""
Testes Unitários da FASE 1: Geometria de Caminhos e Topologia Perfeita.
Conforme especificado no INSTAGRAM_PAINTER_ROADMAP_APICE.md:
- 1.1: Zhang-Suen Thinning (linha 1px, 2px, T-junction, X-junction, loop, componentes).
- 1.2: SkeletonGraph (endpoints, junctions, arestas, resolução angular).
- 1.3: StrokeSchedulerV2 can_chain (zero false bridges, validação de tangentes e corredor).
- 1.4: PathSanitizer split (zero pontes artificiais).
"""
import unittest
import numpy as np
import cv2
from studio.planning.skeleton_graph import zhang_suen_thinning, SkeletonGraph
from studio.planning.scheduler_v2 import StrokeSchedulerV2
from studio.planning.path_sanitizer import PathSanitizer

class TestFase1Geometry(unittest.TestCase):

    def test_1_1_zhang_suen_1px_line_preserved(self):
        """1.1: Garante que uma linha de 1 pixel não desaparece."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[25, 10:40] = 255
        skel = zhang_suen_thinning(mask)
        self.assertEqual(np.count_nonzero(skel), 30)

    def test_1_1_zhang_suen_2px_line_becomes_centerline(self):
        """1.1: Garante que linha de 2px vira centerline de 1px contínua."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[24:26, 10:40] = 255 # 2px de espessura
        skel = zhang_suen_thinning(mask)
        # Deve ter largura de 1px (cerca de 30 pixels)
        count = np.count_nonzero(skel)
        self.assertGreaterEqual(count, 28)
        self.assertLessEqual(count, 32)

    def test_1_1_t_junction_preservation(self):
        """1.1 & 1.2: Garante que uma junção em T é preservada com todos os 3 braços."""
        mask = np.zeros((60, 60), dtype=np.uint8)
        # Barra horizontal (20, 10) a (20, 50)
        mask[20, 10:50] = 255
        # Haste vertical de (20, 30) a (50, 30)
        mask[20:50, 30] = 255

        skel = zhang_suen_thinning(mask)
        graph = SkeletonGraph(skel)

        # Deve encontrar 3 endpoints e 1 junction
        endpoints = [n for n in graph.nodes.values() if n.node_type == "endpoint"]
        junctions = [n for n in graph.nodes.values() if n.node_type == "junction"]

        self.assertEqual(len(endpoints), 3)
        self.assertEqual(len(junctions), 1)
        self.assertEqual(junctions[0].degree, 3)

        strokes = graph.extract_continuous_strokes(approx_epsilon=0.0)
        self.assertGreaterEqual(len(strokes), 2)

    def test_1_1_x_junction_preservation(self):
        """1.1 & 1.2: Garante que um cruzamento em X é preservado com 4 braços."""
        mask = np.zeros((60, 60), dtype=np.uint8)
        for i in range(15, 45):
            mask[i, i] = 255
            mask[i, 59 - i] = 255

        skel = zhang_suen_thinning(mask)
        graph = SkeletonGraph(skel)

        endpoints = [n for n in graph.nodes.values() if n.node_type == "endpoint"]
        self.assertGreaterEqual(len(endpoints), 4)

        strokes = graph.extract_continuous_strokes(approx_epsilon=0.5)
        # Resolução angular deve produzir 2 traços cruzados contínuos
        self.assertGreaterEqual(len(strokes), 2)

    def test_1_1_closed_loop_preservation(self):
        """1.1 & 1.2: Garante que um loop circular fechado permanece fechado."""
        mask = np.zeros((80, 80), dtype=np.uint8)
        cv2.circle(mask, (40, 40), 25, 255, 2)
        skel = zhang_suen_thinning(mask)
        graph = SkeletonGraph(skel)

        strokes = graph.extract_continuous_strokes(approx_epsilon=0.5)
        self.assertGreater(len(strokes), 0)
        # O loop deve ter primeiro e último pontos conectados ou muito próximos
        loop_stroke = strokes[0]
        dist = np.hypot(loop_stroke[0][0] - loop_stroke[-1][0], loop_stroke[0][1] - loop_stroke[-1][1])
        self.assertLessEqual(dist, 2.0)

    def test_1_3_can_chain_alignment_and_distance(self):
        """1.3: Garante que apenas traços próximos e alinhados são encadeados (zero false bridges)."""
        # Traço 1: horizontal (0, 10) -> (20, 10)
        s1 = [(0, 10), (10, 10), (20, 10)]
        # Traço 2: continua horizontalmente (23, 10) -> (40, 10) [distância 3px, ângulo 0°]
        s2 = [(23, 10), (30, 10), (40, 10)]
        self.assertTrue(StrokeSchedulerV2.can_chain(s1, s2, max_gap=5.0))

        # Traço 3: muito distante (60, 10) [distância 40px]
        s3 = [(60, 10), (70, 10)]
        self.assertFalse(StrokeSchedulerV2.can_chain(s1, s3, max_gap=5.0))

        # Traço 4: próximo mas em ângulo oposto/perpendicular brusco (22, 20) -> (22, 40)
        s4 = [(22, 10), (22, 20), (22, 30)]
        # Tangente s1 é (1, 0), tangente s4 é (0, 1) -> dot product = 0 < 0.70
        self.assertFalse(StrokeSchedulerV2.can_chain(s1, s4, max_gap=5.0, min_cos_tangent=0.70))

    def test_1_3_can_chain_with_corridor_mask(self):
        """1.3: Garante que o segmento entre pontas não pode atravessar área fora da máscara permitida."""
        allowed_mask = np.zeros((100, 100), dtype=np.uint8)
        # Máscara tem dois blocos separados por um buraco vazio de 4px
        allowed_mask[10:30, 10:30] = 255
        allowed_mask[10:30, 35:55] = 255 # buraco em x=30..34

        s1 = [(15, 20), (28, 20)] # termina em x=28
        s2 = [(36, 20), (45, 20)] # começa em x=36

        # A ponte cruzaria x=29..35 onde a máscara é 0 -> deve ser rejeitada!
        self.assertFalse(StrokeSchedulerV2.can_chain(s1, s2, allowed_mask=allowed_mask, max_gap=10.0))

        # Agora preenchendo o corredor na máscara
        allowed_mask[20, 28:37] = 255
        self.assertTrue(StrokeSchedulerV2.can_chain(s1, s2, allowed_mask=allowed_mask, max_gap=10.0))

    def test_1_4_path_sanitizer_split(self):
        """1.4: Garante que sanitização divide traço em múltiplos sem conectar sobre o vazio."""
        stroke = [
            (200, 500), (210, 505),
            (float("nan"), 600),
            (300, 700), (310, 705)
        ]
        splits = PathSanitizer.sanitize_stroke_split(stroke)
        self.assertEqual(len(splits), 2)
        self.assertEqual(splits[0], [(200, 500), (210, 505)])
        self.assertEqual(splits[1], [(300, 700), (310, 705)])

if __name__ == "__main__":
    unittest.main()
