"""
Testes unitários para Fases 11 e 12.
Cobre:
- Fase 11: LineArtEngine aprimorado com width classes, offset strokes, ordenação pictórica e gate de topologia
- Fase 12: Otimização de memória com ROIMask, SharedLabelMap, ChunkedColorMatcher e MemoryProfileTracker
"""
import unittest
import numpy as np
import cv2
from studio.planning.line_art_engine import LineArtEngine
from studio.vision.memory_optimizer import (
    ROIMask,
    SharedLabelMap,
    ChunkedColorMatcher,
    MemoryProfileTracker,
)


class TestFases11e12LineArtAndMemory(unittest.TestCase):
    def test_fase11_width_classes_and_offset_strokes(self):
        """Valida classificação de espessura e geração de offset strokes para linhas espessas."""
        self.assertEqual(LineArtEngine.classify_width(2.5), "thin")
        self.assertEqual(LineArtEngine.classify_width(8.0), "medium")
        self.assertEqual(LineArtEngine.classify_width(18.0), "thick")

        # Teste de offset strokes
        linear_stroke = [(10, 10), (10, 50), (10, 100)]
        multi_strokes = LineArtEngine.generate_offset_strokes(linear_stroke, offset_px=2.5)
        self.assertEqual(len(multi_strokes), 3)  # original, +offset, -offset
        # O traço original permanece idêntico
        self.assertEqual(multi_strokes[0], linear_stroke)
        # O offset deve deslocar em x (já que a linha é vertical)
        self.assertNotEqual(multi_strokes[1][0][0], linear_stroke[0][0])

    def test_fase11_stroke_ordering_and_topology_gate(self):
        """Valida a ordenação pictórica (contornos -> estruturas -> detalhes) e o gate métrico."""
        strokes = [
            [(0, 0), (5, 5)],                          # Curto: detalhe / hachura
            [(10, 10), (10, 150), (10, 300)],          # Longo: contorno principal
            [(50, 50), (50, 90), (50, 120)],           # Médio: estrutura interna
        ]
        widths = [[2.0, 2.0], [5.0, 5.0, 5.0], [3.5, 3.5, 3.5]]

        ordered_s, ordered_w, purposes = LineArtEngine.order_line_art_strokes(strokes, widths)
        self.assertEqual(len(ordered_s), 3)
        self.assertEqual(purposes[0], "contour")             # Longo primeiro
        self.assertEqual(purposes[1], "internal_structure")  # Médio depois
        self.assertIn(purposes[2], ("detail", "hatch"))      # Curto por último

        # Gate de validação métrica com máscara sintética
        mask = np.zeros((400, 400), dtype=np.uint8)
        cv2.line(mask, (50, 50), (350, 50), 255, 4)
        synthetic_stroke = [[(50, 50), (200, 50), (350, 50)]]
        metrics = LineArtEngine.verify_line_art_topology_and_metrics(synthetic_stroke, mask)
        self.assertEqual(metrics["false_bridges"], 0)
        self.assertGreaterEqual(metrics["topology_preservation"], 0.90)
        self.assertTrue(metrics["passed"])

    def test_fase12_roi_mask_compression_and_shared_label_map(self):
        """Valida que ROIMask economiza memória e que SharedLabelMap extrai bboxes compactas."""
        full_mask = np.zeros((1000, 1000), dtype=np.uint8)
        # Objeto pequeno 20x20 no centro
        full_mask[500:520, 400:420] = 255

        roi = ROIMask.from_full_mask(full_mask)
        self.assertIsNotNone(roi)
        self.assertEqual(roi.bbox, (400, 500, 20, 20))
        # Memória de 20x20 = 400 bytes vs 1.000.000 bytes do full-frame
        self.assertEqual(roi.memory_bytes, 400)
        self.assertLess(roi.memory_bytes, full_mask.nbytes / 1000)

        # Reconstrução perfeita
        reconstructed = roi.to_full_mask((1000, 1000))
        self.assertTrue(np.array_equal(full_mask, reconstructed))

        # SharedLabelMap
        label_map, num_labels = SharedLabelMap.from_binary_components(full_mask)
        self.assertGreaterEqual(num_labels, 2)  # Background (0) + componente (1)
        comp_roi = label_map.get_roi_mask(1)
        self.assertIsNotNone(comp_roi)
        self.assertEqual(comp_roi.bbox, (400, 500, 20, 20))

    def test_fase12_chunked_color_matcher_and_memory_tracker(self):
        """Valida processamento de cores em blocos e perfilador de RAM."""
        pixels = np.array([
            [255, 0, 0],    # Azul puro BGR
            [0, 255, 0],    # Verde puro
            [0, 0, 255],    # Vermelho puro
            [250, 5, 5],    # Próximo de azul
        ], dtype=np.uint8)

        palette = np.array([
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        ], dtype=np.uint8)

        indices = ChunkedColorMatcher.match_pixels_chunked(pixels, palette, chunk_size=2)
        self.assertEqual(len(indices), 4)
        self.assertEqual(indices[0], 0)  # Azul -> paleta 0
        self.assertEqual(indices[1], 1)  # Verde -> paleta 1
        self.assertEqual(indices[2], 2)  # Vermelho -> paleta 2
        self.assertEqual(indices[3], 0)  # Próximo de azul -> paleta 0

        # MemoryProfileTracker
        tracker = MemoryProfileTracker()
        tracker.start()
        tracker.end_planning(total_strokes=15)
        tracker.end_simulation(sim_time_s=0.12, event_count=60)
        
        prof = tracker.profile.to_dict()
        self.assertGreater(prof["ram_peak_mb"], 0.0)
        self.assertGreaterEqual(prof["planning_time_s"], 0.0)
        self.assertEqual(prof["total_strokes"], 15)
        self.assertEqual(prof["event_count"], 60)


if __name__ == "__main__":
    unittest.main()
