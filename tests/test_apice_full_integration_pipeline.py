"""
Teste de Integração Completo de Ponta a Ponta (E2E Pipeline Integrado).
Verifica a integração real entre todos os módulos de Fases 0 a 20:
- PainterEngine (Line-Art, Layered Paint e Stencil)
- SceneUnderstandingEngine (Classificação e Importância)
- PaintingPlan & Vocabulário de Strokes (Fases 7 e 8)
- MotionCompiler (Fase 2)
- SimulatorV2 (Fase 5)
- QualityGate (Fase 20)
- PaintingInspectorManager (Fase 19)
- ExecutionWorker com FakeADB (Fase 3, Fase 6 e Fase 17)
"""
import unittest
import os
import tempfile
import cv2
import numpy as np
from studio.painter_engine import PainterEngine
from studio.device.fake_adb import FakeADB, FakeADBManager
from studio.execution_worker import ExecutionWorker
from studio.device.quality_gate import QualityGateResult
from studio.planning.motion_compiler import CompiledMotionPlan
from studio.planning.painting_plan import PaintingPlan
from studio.planning.inspector_model import PaintingInspectorManager


class TestApiceFullIntegrationPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_img_path = os.path.join(self.temp_dir.name, "test_sample.png")
        
        # Cria imagem de teste com fundo claro e traços com detalhes faciais simples
        img = np.ones((300, 300, 3), dtype=np.uint8) * 245
        cv2.circle(img, (150, 150), 60, (50, 50, 200), -1)  # Círculo avermelhado
        cv2.circle(img, (130, 140), 8, (255, 255, 255), -1)  # Olho esquerdo
        cv2.circle(img, (170, 140), 8, (255, 255, 255), -1)  # Olho direito
        cv2.line(img, (100, 250), (200, 250), (20, 20, 20), 4) # Linha inferior
        cv2.imwrite(self.test_img_path, img)

        self.engine = PainterEngine()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_line_art_plan_full_integration(self):
        """Valida que o plano de Line-Art integra todos os artefatos de Fases 2, 7, 13, 19 e 20."""
        plan = self.engine.generate_line_art_plan(self.test_img_path, seed=42)
        
        # 1. Metadados básicos
        self.assertEqual(plan["mode"], "line_art")
        self.assertGreater(plan["total_strokes"], 0)
        self.assertIn("plan_hash", plan)

        # 2. Integração Semântica (Fase 13)
        self.assertEqual(plan["image_class"], "line_art")
        self.assertIn("importance_map", plan)
        self.assertEqual(plan["importance_map"].shape, plan["fit_image"].shape[:2])

        # 3. Integração PaintingPlan (Fases 7 e 8)
        self.assertIn("painting_plan", plan)
        self.assertIsInstance(plan["painting_plan"], PaintingPlan)
        self.assertGreater(len(plan["painting_plan"].regions), 0)

        # 4. Integração Motion Compiler (Fase 2)
        self.assertIn("compiled_motion_plan", plan)
        self.assertIsInstance(plan["compiled_motion_plan"], CompiledMotionPlan)
        self.assertGreater(plan["compiled_motion_plan"].total_events, 0)

        # 5. Integração Quality Gate (Fase 20)
        self.assertIn("quality_gate", plan)
        self.assertIsInstance(plan["quality_gate"], QualityGateResult)

        # 6. Integração Inspector Model (Fase 19)
        self.assertIn("inspector", plan)
        self.assertIsInstance(plan["inspector"], PaintingInspectorManager)
        self.assertGreater(len(plan["editable_regions"]), 0)
        region_id = plan["editable_regions"][0]["id"]
        y, x = np.argwhere(plan["region_map"] == region_id)[0]
        detail = plan["inspector"].inspect_at_point(int(x), int(y))
        self.assertIsNotNone(detail)
        self.assertEqual(detail.region_id, f"region_{region_id}")

    def test_layered_paint_plan_full_integration(self):
        """Valida que o plano de Pintura em Camadas integra todos os artefatos com sucesso."""
        plan = self.engine.generate_layered_paint_plan(self.test_img_path, num_colors=4, seed=42, optimize_in_simulation=False)

        self.assertEqual(plan["mode"], "layered_paint")
        self.assertIn("image_class", plan)
        self.assertIn("painting_plan", plan)
        self.assertIn("compiled_motion_plan", plan)
        self.assertIn("quality_gate", plan)
        self.assertIn("inspector", plan)
        self.assertIn("triad_diagnostics", plan)

        # Tríade de diagnósticos de erro da Fase 5.3
        triad = plan["triad_diagnostics"]
        self.assertIn("error_original_target", triad)
        self.assertIn("error_target_prediction", triad)
        self.assertIn("diagnosis", triad)

    def test_execution_worker_lifecycle_with_telemetry_and_closed_loop(self):
        """Valida a execução de ponta a ponta no ExecutionWorker com FakeADB e telemetria gravada."""
        plan = self.engine.generate_line_art_plan(self.test_img_path, seed=42)
        fake_adb = FakeADBManager()

        logs = []
        progresses = []
        finished_status = []

        callbacks = {
            "on_progress": lambda cur_l, tot_l, cur_s, tot_s, pct, msg: progresses.append(pct),
            "on_log": lambda msg: logs.append(msg),
            "on_finished": lambda ok, reason: finished_status.append((ok, reason)),
        }

        worker = ExecutionWorker(
            adb_manager=fake_adb,
            plan=plan,
            callbacks=callbacks,
            verify_each_layer=True,
            enable_closed_loop=True,
        )

        worker.start()
        worker.join(timeout=25.0)

        # Valida que o worker concluiu a execução
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(finished_status), 1)
        self.assertTrue(finished_status[0][0])  # Sucesso!
        self.assertIn("concluída", finished_status[0][1])

        # Valida que a telemetria foi instanciada e gravada (Fase 17)
        self.assertTrue(hasattr(worker, "telemetry"))
        self.assertGreater(worker.telemetry.execution_time_s, 0.0)
        self.assertTrue(os.path.exists("outputs/session_telemetry.json"))


if __name__ == "__main__":
    unittest.main()
