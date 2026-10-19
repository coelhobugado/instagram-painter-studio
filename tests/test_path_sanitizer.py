"""
Testes Unitários do Sanitizador de Caminhos e Safe Bounds (Seção 47/48/80).
"""
import unittest
import math
from studio.planning.path_sanitizer import PathSanitizer
from studio import config

class TestPathSanitizer(unittest.TestCase):
    def test_safe_bounds_clamping(self):
        """Garante que nenhum ponto fique fora dos limites no modo DEBUG_CLAMP e rejeição no modo STRICT."""
        unsafe_stroke = [
            (-50, 100),       # Fora à esquerda e no topo
            (540, 1150),      # Ponto seguro central
            (1500, 3000)      # Fora à direita e no rodapé
        ]
        # Modo DEBUG_CLAMP (diagnóstico)
        from studio.planning.path_sanitizer import SanitizerMode
        clean_clamp = PathSanitizer.sanitize_stroke(unsafe_stroke, mode=SanitizerMode.DEBUG_CLAMP)
        self.assertEqual(len(clean_clamp), 3)
        for x, y in clean_clamp:
            self.assertGreaterEqual(x, config.CANVAS_X_MIN)
            self.assertLessEqual(x, config.CANVAS_X_MAX)
            self.assertGreaterEqual(y, config.CANVAS_Y_MIN)
            self.assertLessEqual(y, config.CANVAS_Y_MAX)

        # Modo STRICT (produção - Seção 5 do Plano de Correções)
        clean_strict = PathSanitizer.sanitize_stroke(unsafe_stroke, mode=SanitizerMode.STRICT)
        # Pontos inseguros rejeitados, restando menos de 2 pontos -> stroke descartado com 0 clamp
        self.assertEqual(len(clean_strict), 0)
        self.assertEqual(PathSanitizer.last_stats["clamped_points"], 2) # os 2 anteriores do debug clamp

    def test_nan_and_inf_rejection(self):
        """Garante que pontos inválidos não gerem conexões artificiais (A -> INVALID -> B não vira A -> B)."""
        corrupted_stroke = [
            (500, 1000),
            (float("nan"), 1100),
            (520, float("inf")),
            (530, 1200)
        ]
        # Como há apenas 1 ponto isolado antes e 1 após os inválidos, nenhuma linha falsa deve ser criada
        clean = PathSanitizer.sanitize_stroke(corrupted_stroke)
        self.assertEqual(len(clean), 0)

    def test_split_on_invalid_points(self):
        """Fase 1.4: Valida que a sanitização divide o traço em sub-traços (SPLIT) em vez de conectar os extremos."""
        stroke_with_gap = [
            (500, 1000), (510, 1010),      # Sub-traço 1
            (float("nan"), 1050),          # Ponto inválido no meio
            (530, 1200), (540, 1210)       # Sub-traço 2
        ]
        splits = PathSanitizer.sanitize_stroke_split(stroke_with_gap)
        self.assertEqual(len(splits), 2)
        self.assertEqual(splits[0], [(500, 1000), (510, 1010)])
        self.assertEqual(splits[1], [(530, 1200), (540, 1210)])

    def test_empty_and_single_point_strokes(self):
        """Traços com menos de 2 pontos válidos devem ser eliminados."""
        self.assertEqual(PathSanitizer.sanitize_stroke([]), [])
        self.assertEqual(PathSanitizer.sanitize_stroke([(500, 500)]), [])

if __name__ == "__main__":
    unittest.main()
