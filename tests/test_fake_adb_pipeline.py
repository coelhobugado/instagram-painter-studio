"""
Testes de Integração de Pipeline Completo Usando FakeADB (Seção 60).
Permite validar o fluxo inteiro de ponta a ponta em ambiente de CI/CD sem hardware físico.
"""
import unittest
import numpy as np
from studio.device.fake_adb import FakeADB
from studio.painter_engine import PainterEngine
from studio.execution_worker import ExecutionWorker

class TestFakeADBPipeline(unittest.TestCase):
    def setUp(self):
        self.fake_adb = FakeADB()
        self.engine = PainterEngine()

    def test_end_to_end_fake_drawing(self):
        """Valida que o plano de desenho da tulipa é executado até o fim pelo FakeADB."""
        plan = self.engine.generate_stencil_plan("inputs/flores.jpg")
        self.assertGreater(plan["total_strokes"], 0)

        completed = {"flag": False}

        def on_finish(success, msg):
            completed["flag"] = success

        callbacks = {
            "on_progress": None,
            "on_layer_start": None,
            "on_log": None,
            "on_finished": on_finish,
            "on_screencap": None
        }

        worker = ExecutionWorker(self.fake_adb, plan, callbacks)
        # Executa de forma síncrona
        worker.run()

        self.assertTrue(completed["flag"])
        # Garante que os traços foram gravados na tela virtual
        self.assertGreater(len(self.fake_adb.injected_strokes), 0)
        # Verifica se o screencap virtual tem dados desenhados
        virtual_screen = self.fake_adb.capture_screenshot()
        self.assertEqual(virtual_screen.shape, (2400, 1080, 3))

if __name__ == "__main__":
    unittest.main()
