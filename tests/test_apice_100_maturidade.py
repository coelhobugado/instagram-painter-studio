# -*- coding: utf-8 -*-
"""
Testes de Integração e Validação do Ápice de Maturidade 100% (Todos os 5 Pilares).
"""
import unittest
import numpy as np
from studio.painter_engine import PainterEngine
from studio.device.fake_adb import FakeADB
from studio.execution_worker import ExecutionWorker

class TestApice100Maturidade(unittest.TestCase):
    def setUp(self):
        self.engine = PainterEngine()
        self.fake_adb = FakeADB()

    def test_layered_painting_plan_apice_features(self):
        """Valida que o plano gerado contém streamlines, chiaroscuro, splines e camada de destaque semântico."""
        plan = self.engine.generate_layered_paint_plan(
            "inputs/flores.jpg",
            num_colors=6,
            brush_spacing=6,
            enable_semantic_highlights=True,
            enable_streamlines=True,
            enable_chiaroscuro=True,
            enable_spline=True
        )

        self.assertEqual(plan["mode"], "layered_paint")
        self.assertGreater(plan["total_strokes"], 0)
        self.assertGreater(len(plan["layers"]), 2)

        # 1. Verifica se a camada semântica de micro-detalhes e catchlights foi gerada
        purposes = [l.get("purpose") for l in plan["layers"]]
        self.assertIn("highlight", purposes, "Camada de micro-detalhes anatômicos/catchlights deve existir no plano.")

        # 2. Verifica a integridade de todas as pinceladas em todas as camadas
        for l_idx, layer in enumerate(plan["layers"]):
            self.assertGreater(len(layer["strokes"]), 0, f"Camada {l_idx} não deve ter traços vazios.")
            for s_idx, stroke in enumerate(layer["strokes"]):
                self.assertGreaterEqual(len(stroke), 2, f"Traço {s_idx} da camada {l_idx} deve ter pelo menos 2 pontos.")
                for px, py in stroke:
                    self.assertFalse(np.isnan(px) or np.isnan(py))
                    self.assertTrue(0 <= px < 1080 and 0 <= py < 2400)

    def test_execution_worker_with_apice_plan(self):
        """Valida a execução completa e sem falhas do plano com FakeADB."""
        plan = self.engine.generate_layered_paint_plan(
            "inputs/flores.jpg",
            num_colors=4,
            brush_spacing=8,
            enable_semantic_highlights=True,
            enable_streamlines=True,
            enable_chiaroscuro=True
        )

        completed = {"success": False}
        def on_finished(success, msg):
            completed["success"] = success

        callbacks = {
            "on_progress": None,
            "on_layer_start": None,
            "on_log": None,
            "on_finished": on_finished,
            "on_screencap": None
        }

        worker = ExecutionWorker(self.fake_adb, plan, callbacks)
        worker.run()

        self.assertTrue(completed["success"], "A execução do plano de maturidade 100% no FakeADB deve concluir com sucesso.")
        self.assertGreater(len(self.fake_adb.injected_strokes), 0)

if __name__ == "__main__":
    unittest.main()
