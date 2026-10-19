"""
Testes Unitários do Agendador de Trajetórias e Otimizador TSP / 2-Opt.
"""
import unittest
import numpy as np
from studio.planning.scheduler_v2 import StrokeSchedulerV2
from studio.tsp_optimizer import optimize_paths_tsp

class TestScheduler(unittest.TestCase):
    def test_tsp_reduces_aerial_distance(self):
        """Valida que a ordenação TSP reduz o custo acumulado de deslocamento aéreo."""
        # Cria traços espalhados
        strokes = [
            [(100, 100), (150, 150)],
            [(900, 900), (950, 950)],
            [(160, 160), (200, 200)],
            [(850, 850), (890, 890)]
        ]

        # Distância na ordem não-otimizada
        def total_aerial_dist(paths):
            d = 0
            for i in range(len(paths) - 1):
                p1 = paths[i][-1]
                p2 = paths[i + 1][0]
                d += np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            return d

        dist_before = total_aerial_dist(strokes)
        optimized = optimize_paths_tsp(strokes)
        dist_after = total_aerial_dist(optimized)

        self.assertLess(dist_after, dist_before)

    def test_2opt_scheduler_integrity(self):
        """Garante que o scheduler 2-opt preserve todos os traços sem perda de nós."""
        strokes = [
            [(100, 100), (200, 100)],
            [(200, 200), (100, 200)],
            [(100, 300), (200, 300)],
            [(200, 400), (100, 400)],
            [(100, 500), (200, 500)]
        ]
        res = StrokeSchedulerV2.optimize_layer_strokes(strokes, enable_2opt=True)
        self.assertEqual(len(res), len(strokes))

if __name__ == "__main__":
    unittest.main()
