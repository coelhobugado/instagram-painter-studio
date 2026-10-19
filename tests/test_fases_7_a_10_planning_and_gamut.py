"""
Testes unitários para Fases 7, 8, 9 e 10.
Cobre:
- Fase 7: PaintingPlan, PaintingRegion, vocabulário de 10 StrokeStrategy e racionalidade auditável
- Fase 8: Orçamento e LOD Progressivo (0 a 3, presets FAST/BALANCED/HIGH/ULTRA)
- Fase 9: Função Objetivo Global, pesos por imagem e Ganho por Segundo
- Fase 10: Quantização física real no gamut e detecção de cores inalcançáveis
"""
import unittest
import numpy as np
from studio.planning.painting_plan import (
    PaintingPlan,
    PaintingRegion,
    StrokeStrategy,
    LODLevel,
    PRESET_BUDGETS_SECONDS,
)
from studio.planning.global_objective import GlobalObjectiveLoss, LossWeights
from studio.color_palette import PhysicalGamutMatcher, ColorPaletteManager


class TestFases7a10PlanningAndGamut(unittest.TestCase):
    def test_fase7_stroke_vocabulary_and_auditability(self):
        """Valida que todos os 10 vocabulários de stroke existem e que cada região tem racionalidade auditável."""
        expected_strategies = [
            "MASS_FILL", "FLOW_FILL", "CONTOUR", "CENTERLINE",
            "WIDTH_VARYING_INK", "HATCH", "STIPPLE", "DETAIL",
            "POINT_HIGHLIGHT", "NEGATIVE_STROKE"
        ]
        for name in expected_strategies:
            self.assertIn(name, StrokeStrategy.__members__)

        plan = PaintingPlan(image_class="portrait", time_budget_s=180.0)
        region_face = PaintingRegion(
            id="face_skin",
            semantic_type="skin",
            importance=0.92,
            strategy=StrokeStrategy.FLOW_FILL,
            lod_min=LODLevel.LOD_0_SILHOUETTE_MASS,
            protected_regions=["eyes", "mouth"],
            time_cost_s=25.0,
            stroke_count=45,
            target_color_rgb=(230, 185, 155),
            rationale="Preenchimento volumétrico suave da face respeitando fluxo muscular."
        )
        plan.add_region(region_face)

        self.assertEqual(len(plan.regions), 1)
        r_dict = region_face.to_dict()
        self.assertEqual(r_dict["strategy"], "FLOW_FILL")
        self.assertIn("respeitando fluxo", r_dict["rationale"])
        self.assertEqual(r_dict["stroke_count"], 45)

    def test_fase8_lod_progressive_and_budget_pruning(self):
        """Valida filtragem por LOD (0 a 3) e poda por orçamento preservando silhueta estrutural."""
        plan = PaintingPlan(image_class="photo_general", time_budget_s=PRESET_BUDGETS_SECONDS["FAST"])
        
        # Região LOD 0 (massa dominante)
        plan.add_region(PaintingRegion(
            id="bg_mass", semantic_type="background", importance=0.8,
            strategy=StrokeStrategy.MASS_FILL, lod_min=0, time_cost_s=20.0, stroke_count=10
        ))
        # Região LOD 1 (estrutura intermediária)
        plan.add_region(PaintingRegion(
            id="main_body", semantic_type="subject", importance=0.9,
            strategy=StrokeStrategy.FLOW_FILL, lod_min=1, time_cost_s=15.0, stroke_count=15
        ))
        # Região LOD 2 (contornos)
        plan.add_region(PaintingRegion(
            id="body_contour", semantic_type="edges", importance=0.85,
            strategy=StrokeStrategy.CONTOUR, lod_min=2, time_cost_s=12.0, stroke_count=20
        ))
        # Região LOD 3 (microdetalhes caros)
        plan.add_region(PaintingRegion(
            id="tiny_highlights", semantic_type="highlights", importance=0.4,
            strategy=StrokeStrategy.POINT_HIGHLIGHT, lod_min=3, time_cost_s=30.0, stroke_count=40
        ))

        # Teste de filtro por LOD 1 (deve conter apenas LOD 0 e LOD 1)
        lod1_regions = plan.get_regions_for_lod(max_lod=1)
        self.assertEqual(len(lod1_regions), 2)
        self.assertTrue(all(r.lod_min <= 1 for r in lod1_regions))

        # Teste de poda para orçamento FAST (45s)
        pruned_fast = plan.prune_to_budget(max_seconds=45.0)
        total_time = sum(r.time_cost_s for r in pruned_fast)
        self.assertLessEqual(total_time, 48.0)
        # O microdetalhe caro de 30s deve ter sido podado
        pruned_ids = [r.id for r in pruned_fast]
        self.assertIn("bg_mass", pruned_ids)
        self.assertNotIn("tiny_highlights", pruned_ids)

    def test_fase9_global_objective_loss_and_gain_per_second(self):
        """Valida cálculo de Loss ponderada, score de ganho/segundo e explicação auditável."""
        # Pesos adaptativos
        weights_portrait = GlobalObjectiveLoss.get_weights_for_class("portrait")
        self.assertGreater(weights_portrait.w_feature, 0.30)

        weights_lineart = GlobalObjectiveLoss.get_weights_for_class("line_art")
        self.assertGreater(weights_lineart.w_edge, 0.40)

        # Ganho por segundo
        cost = GlobalObjectiveLoss.compute_execution_cost(stroke_duration_s=0.4, is_color_switch=True)
        self.assertGreater(cost, 1.8)  # Penalidade da troca de cor
        gain_s = GlobalObjectiveLoss.compute_gain_per_second(predicted_loss_reduction=0.15, execution_cost_s=cost)
        self.assertGreater(gain_s, 0.0)

        # Explicação de escolha do stroke
        stroke_cand1 = {"id": "stroke_A", "gain_per_second": 0.25, "predicted_loss_reduction": 0.5, "execution_cost": 2.0}
        stroke_cand2 = {"id": "stroke_B", "gain_per_second": 0.10, "predicted_loss_reduction": 0.2, "execution_cost": 2.0}
        explanation = GlobalObjectiveLoss.explain_stroke_choice(stroke_cand1, [stroke_cand1, stroke_cand2])
        self.assertIn("stroke_A", explanation)
        self.assertIn("supera stroke_B", explanation)

    def test_fase10_physical_gamut_quantization_and_unreproducible_warning(self):
        """Valida quantização no gamut da LUT e aviso de cor inalcançável."""
        matcher = PhysicalGamutMatcher(unreproducible_threshold=25.0)

        # Cor presente na paleta (ex: pele ou amarelo)
        res_yellow = matcher.match_target((0, 255, 255))
        self.assertFalse(res_yellow["is_unreproducible"])
        self.assertIsNone(res_yellow["warning"])

        # Cor ultrassaturada bizarra inalcançável (limiar baixo para teste)
        strict_matcher = PhysicalGamutMatcher(unreproducible_threshold=2.0)
        res_exotic = strict_matcher.match_target((12, 189, 45))
        self.assertTrue(res_exotic["is_unreproducible"])
        self.assertIsNotNone(res_exotic["warning"])
        self.assertIn("Aviso de Gamut", res_exotic["warning"])

        # Quantização direta de imagem pequena
        test_img = np.ones((40, 40, 3), dtype=np.uint8) * 200
        quant_img, matches, unrep_cnt = matcher.quantize_image_direct(test_img, max_colors=4)
        self.assertEqual(quant_img.shape, test_img.shape)
        self.assertGreater(len(matches), 0)


if __name__ == "__main__":
    unittest.main()
