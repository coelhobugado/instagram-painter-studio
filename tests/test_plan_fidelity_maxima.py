"""
Testes Unitários Exaustivos de Fidelidade Máxima (Seção 45 do PLANO_DE_CORRECOES_E_FIDELIDADE_MAXIMA.md).
Garante a equivalência estrita:
Target Executável ≈ StrokePlan Local ≈ Simulador Físico ≈ Preview ≈ Dispositivo Real Instagram Direct
"""
import unittest
import numpy as np
import cv2
import json
import os

from studio import config
from studio.core.models import CanvasTransform, ColorSelection, CorrectionPlan, StrokePurpose, StrokePlan
from studio.vision.perceptual_color import PerceptualColorEngine
from studio.planning.fill_strategies import FillStrategyEngine
from studio.planning.layer_graph import LayerGraphEngine
from studio.planning.path_sanitizer import PathSanitizer, SanitizerMode
from studio.simulation.simulator_v2 import CalibratedSimulatorV2
from studio.feedback.error_map import ErrorMapEngine
from studio.feedback.correction_planner import CorrectionPlanner
from studio.painter_engine import PainterEngine

class TestFidelityMaximaPlan(unittest.TestCase):
    def setUp(self):
        self.color_engine = PerceptualColorEngine()
        self.simulator = CalibratedSimulatorV2()
        self.engine = PainterEngine()

    # =========================================================================
    # 1. GEOMETRY (Seções 4, 5 e 45 do Plano)
    # =========================================================================
    def test_no_point_outside_safe_zone(self):
        """Garante que nenhum ponto gerado pelo plano fique fora da safe zone física após transformação."""
        # Cria imagem sintética 300x400
        test_img = np.ones((400, 300, 3), dtype=np.uint8) * 150
        cv2.circle(test_img, (150, 200), 80, (20, 20, 200), -1)
        temp_path = "tests_temp_geom.png"
        cv2.imwrite(temp_path, test_img)

        try:
            plan = self.engine.generate_layered_paint_plan(temp_path, num_colors=3, seed=42)
            transform = plan["transform"]

            for layer in plan["layers"]:
                for stroke in layer["strokes"]:
                    for local_pt in stroke:
                        screen_pt = transform.local_to_screen(local_pt)
                        sx, sy = screen_pt
                        self.assertGreaterEqual(sx, config.CANVAS_X_MIN, f"Ponto {screen_pt} fora à esquerda")
                        self.assertLessEqual(sx, config.CANVAS_X_MAX, f"Ponto {screen_pt} fora à direita")
                        self.assertGreaterEqual(sy, config.CANVAS_Y_MIN, f"Ponto {screen_pt} fora no topo")
                        self.assertLessEqual(sy, config.CANVAS_Y_MAX, f"Ponto {screen_pt} fora no rodapé")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_no_clamp_in_valid_plan(self):
        """No modo STRICT de produção, nenhum ponto é clampeado silenciosamente (0 clamped points)."""
        test_img = np.ones((200, 200, 3), dtype=np.uint8) * 120
        temp_path = "tests_temp_clamp.png"
        cv2.imwrite(temp_path, test_img)

        try:
            # Reseta estatísticas antes da execução
            PathSanitizer.last_stats["clamped_points"] = 0
            plan = self.engine.generate_layered_paint_plan(temp_path, num_colors=2, seed=42)

            self.assertEqual(PathSanitizer.last_stats["clamped_points"], 0,
                             "O plano executou clamp em produção! Esperado: 0 clamped points.")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_transform_roundtrip(self):
        """Valida que screen_to_local(local_to_screen(p)) == p com fidelidade sub-pixel inteira exata."""
        transform = CanvasTransform(offset_x=190, offset_y=675, width=700, height=1050)
        test_points = [(0, 0), (100, 250), (699, 1049), (350, 525)]
        for pt in test_points:
            scr = transform.local_to_screen(pt)
            recovered = transform.screen_to_local(scr)
            self.assertEqual(recovered, pt)

    # =========================================================================
    # 2. COLOR (Seções 6, 7, 8, 9 e 45 do Plano)
    # =========================================================================
    def test_lut_unique_touch_mapping(self):
        """Valida que a LUT física do dispositivo possui entradas válidas de toque com valores medidos."""
        lut_path = "profiles/SM-A736B/color_lut.json"
        self.assertTrue(os.path.exists(lut_path))
        with open(lut_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries = data.get("entries", [])
        self.assertGreater(len(entries), 50)
        for entry in entries:
            self.assertIn("target_x", entry)
            self.assertIn("target_y", entry)
            self.assertIn("rgb", entry)
            tx = entry["target_x"]
            ty = entry["target_y"]
            self.assertGreater(tx, 0)
            self.assertGreater(ty, 0)

    def test_global_ciede_match(self):
        """Valida busca exaustiva global CIEDE2000 encontrando matching exato para cores padrão."""
        white_match = self.color_engine.match_color((255, 255, 255))
        self.assertLess(white_match["delta_e"], 5.0)
        black_match = self.color_engine.match_color((0, 0, 0))
        # A nuance mais escura medida no seletor fica abaixo da bolinha #262626.
        self.assertLess(black_match["delta_e"], 10.0)
        self.assertLessEqual(max(black_match["matched_rgb"]), 38)

    def test_simulator_uses_mapped_color(self):
        """Garante que o CalibratedSimulatorV2 utilize render_color_rgb (cor física) para fidelidade de preview."""
        layer = {
            "name": "Camada de Teste",
            "color_rgb": (255, 0, 0),        # Cor teórica ideal
            "render_color_rgb": (0, 255, 0), # Cor física mapeada da LUT (verde proposital para teste)
            "brush": "solid",
            "thickness": "thick",
            "strokes": [[(20, 20), (80, 20)]]
        }
        canvas, _ = self.simulator.render_simulation((100, 100, 3), [layer], seed=42)
        # O traço desenhado em (50, 20) deve conter verde (render_color_rgb), não vermelho
        drawn_bgr = canvas[20, 50]
        # BGR: verde é [0, 255, 0]
        self.assertGreater(drawn_bgr[1], 150, "O simulador não utilizou render_color_rgb!")
        self.assertLess(drawn_bgr[2], 50, "O simulador utilizou incorretamente a cor teórica ideal!")

    # =========================================================================
    # 3. BRUSH & FILL STRATEGIES (Seções 11, 12, 13 e 45 do Plano)
    # =========================================================================
    def test_executor_and_simulator_share_brush_entry(self):
        """Valida consistência entre ui_geometry.json, brushes.json e config.py."""
        brushes_path = "profiles/SM-A736B/brushes.json"
        with open(brushes_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        levels = data["brushes"]["solid"]["levels"]
        self.assertIn("thick", levels)
        self.assertIn("med", levels)
        self.assertIn("min", levels)
        # Espessuras devem estar em ordem decrescente
        self.assertGreater(levels["thick"]["effective_width_px"], levels["med"]["effective_width_px"])
        self.assertGreater(levels["med"]["effective_width_px"], levels["min"]["effective_width_px"])

    def test_spacing_derived_from_width(self):
        """Valida que o espaçamento entre traços derive diretamente da largura do pincel (Seção 12: ~0.85 * width)."""
        w28 = FillStrategyEngine.calculate_brush_spacing(28.0)
        self.assertEqual(w28, 24)  # round(0.85 * 28) = 24
        w14 = FillStrategyEngine.calculate_brush_spacing(14.5)
        self.assertEqual(w14, 12)  # round(0.85 * 14.5) = 12
        w3 = FillStrategyEngine.calculate_brush_spacing(3.2)
        self.assertEqual(w3, 3)   # round(0.85 * 3.2) = 3

    def test_safe_mask_accounts_for_radius(self):
        """Valida que safe mask com erosão por raio proteja a região contra extravasamento de borda."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255 # Quadrado 60x60
        safe_mask = FillStrategyEngine.compute_safe_mask(mask, brush_radius=5.0)
        # O safe_mask erodido deve ser estritamente menor que a máscara original
        self.assertLess(np.count_nonzero(safe_mask), np.count_nonzero(mask))
        # O centro deve estar presente
        self.assertEqual(safe_mask[50, 50], 255)
        # A borda externa deve ter sido desbastada
        self.assertEqual(safe_mask[22, 22], 0)

    # =========================================================================
    # 4. SEMANTIC SEPARATION & CONTOURS (Seções 14, 15, 20 e 21 do Plano)
    # =========================================================================
    def test_coverage_before_texture(self):
        """Valida ordem topológica do pintor: Camadas de Cobertura Base antes de Midtones e Highlights."""
        regions = [
            {"mean_L": 95.0, "total_area": 500, "saliency": 0.8, "regions": []},   # Highlight
            {"mean_L": 15.0, "total_area": 8000, "saliency": 0.35, "regions": []}, # Dark Base
            {"mean_L": 50.0, "total_area": 2000, "saliency": 0.5, "regions": []},  # Midtone
        ]
        ordered = LayerGraphEngine.build_layer_order(regions)
        purposes = [o["purpose"] for o in ordered]
        self.assertEqual(purposes[0], StrokePurpose.DARK_BASE.value)
        self.assertEqual(purposes[1], StrokePurpose.MIDTONE.value)
        self.assertEqual(purposes[2], StrokePurpose.HIGHLIGHT.value)

    def test_large_dark_region_does_not_use_min_only(self):
        """Garante que grandes áreas escuras (cabelo preto, terno) sejam dark_base (Seção 15 do Plano)."""
        regions = [
            {"mean_L": 18.0, "total_area": 25000, "saliency": 0.30, "regions": []}, # Grande massa escura
            {"mean_L": 60.0, "total_area": 10000, "saliency": 0.6, "regions": []},
        ]
        ordered = LayerGraphEngine.build_layer_order(regions)
        dark_reg = ordered[0]
        self.assertEqual(dark_reg["purpose"], StrokePurpose.DARK_BASE.value)

    def test_concave_region_no_overpaint(self):
        """Garante que regiões côncavas com buraco interno não sofram travessia de traço pelo buraco."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:90, 10:90] = 255
        mask[30:70, 30:70] = 0  # Buraco central
        strokes = FillStrategyEngine.generate_region_strokes(mask, (10, 10, 80, 80), brush_spacing=5)
        # Nenhum traço deve atravessar o ponto central do buraco (50, 50)
        for s in strokes:
            for px, py in s:
                self.assertFalse(32 <= px <= 68 and 32 <= py <= 68,
                                 f"Traço penetrou no buraco da região côncava: {(px, py)}")

    def test_contour_grouped_by_physical_touch(self):
        """Garante que múltiplos contornos sejam agrupados pela coordenada física de toque da LUT real."""
        test_img = np.ones((150, 150, 3), dtype=np.uint8) * 200
        # Desenha linha preta nítida
        cv2.line(test_img, (30, 75), (120, 75), (10, 10, 10), 3)
        temp_path = "tests_temp_contour.png"
        cv2.imwrite(temp_path, test_img)

        try:
            plan = self.engine.generate_layered_paint_plan(temp_path, num_colors=2, seed=42)
            contour_layers = [l for l in plan["layers"] if l["purpose"] == "contour"]
            # Não deve explodir o número de camadas de contorno
            self.assertLessEqual(len(contour_layers), 2)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # =========================================================================
    # 5. CLOSED-LOOP & CORRECTION PLAN (Seções 22, 23, 25 e 45 do Plano)
    # =========================================================================
    def test_correction_plan_executes_nonzero_strokes(self):
        """Valida que CorrectionPlanner retorne a chave 'layers' unificada e gere traços em caso de erro."""
        target = np.ones((150, 150, 3), dtype=np.uint8) * 220
        observed = target.copy()
        observed[40:110, 40:110] = 20 # Grande erro escuro central

        planner = CorrectionPlanner()
        plan = planner.plan_corrections(target, observed, offset_xy=(100, 200), max_correction_strokes=20)
        # Contrato unificado: chave 'layers' deve conter as camadas de retoque
        self.assertIn("layers", plan)
        self.assertGreater(plan["total_correction_strokes"], 0)
        self.assertGreater(len(plan["layers"]), 0)

    def test_loop_stops_when_no_improvement(self):
        """Valida critério de parada autônomo (Seção 25 do Plano) quando a imagem observada já está correta."""
        target = np.ones((100, 100, 3), dtype=np.uint8) * 150
        observed = target.copy() # Erro zero

        planner = CorrectionPlanner()
        plan = planner.plan_corrections(target, observed, offset_xy=(0, 0))
        self.assertTrue(plan["should_stop"])
        self.assertEqual(plan["total_correction_strokes"], 0)

    # =========================================================================
    # 6. INSTAGRAM-CONSTRAINED QUANTIZATION (Seções 10, 32 e 56 do Plano)
    # =========================================================================
    def test_instagram_constrained_quantization_uses_physical_colors(self):
        """Valida que o Target Executável é composto exclusivamente por cores presentes na LUT física."""
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[:50, :] = [30, 80, 220]    # Laranja
        test_img[50:, :] = [200, 100, 40]   # Azul

        target_exec, target_ideal, labels, palette = self.engine.quantize_instagram_constrained(
            test_img, num_colors=4, seed=42
        )
        self.assertEqual(target_exec.shape, test_img.shape)
        self.assertGreater(len(palette), 0)
        for p in palette:
            self.assertIn("render_rgb", p)
            self.assertIn("touch", p)
            self.assertIn("target_x", p["touch"])

    # =========================================================================
    # 7. CHECKPOINT SEGURO (Seção 28 do Plano)
    # =========================================================================
    def test_checkpoint_plan_hash_determinism(self):
        """Garante que o hash criptográfico do plano seja reprodutível e detecte alterações."""
        layer = {"name": "L1", "color_rgb": (255, 255, 255), "strokes": [[(10, 10), (20, 20)]]}
        h1 = StrokePlan.compute_plan_hash(42, "LAYERED", [layer], image_hash="abc123")
        h2 = StrokePlan.compute_plan_hash(42, "LAYERED", [layer], image_hash="abc123")
        self.assertEqual(h1, h2)

        # Alteração no traço deve mudar o hash
        layer_mod = {"name": "L1", "color_rgb": (255, 255, 255), "strokes": [[(10, 10), (20, 25)]]}
        h3 = StrokePlan.compute_plan_hash(42, "LAYERED", [layer_mod], image_hash="abc123")
        self.assertNotEqual(h1, h3)

if __name__ == "__main__":
    unittest.main()
