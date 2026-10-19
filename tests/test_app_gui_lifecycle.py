"""
Testes Unitários para a Interface Gráfica (AppGUI) e Ciclo de Callbacks.
Valida que callbacks de captura ao vivo (screencap), mapa de erro (error_map),
logs assíncronos, checkpoints e mudança dinâmica de cor não provoquem AttributeError nem falhas no Tkinter.
"""
import unittest
import numpy as np
import cv2
import tkinter as tk
from unittest.mock import patch
from studio.app_gui import InstagramPainterStudioGUI

class TestAppGUILifecycle(unittest.TestCase):
    def setUp(self):
        # Nenhum teste deve abrir diálogos modais reais na tela.
        msg_patch = patch("studio.app_gui.messagebox")
        msg_patch.start()
        self.addCleanup(msg_patch.stop)
        try:
            self.app = InstagramPainterStudioGUI()
            self.app.withdraw()  # Oculta a janela durante os testes
            self.app.update()
        except tk.TclError as e:
            err_msg = str(e).lower()
            if any(k in err_msg for k in ("no display", "cannot connect", "display name", "x11")):
                self.skipTest(f"Ambiente gráfico headless sem display disponível: {e}")
            raise

    def tearDown(self):
        if hasattr(self, "app") and self.app:
            try:
                self.app.update()
                self.app.destroy()
            except Exception:
                pass

    def test_closing_window_does_not_raise(self):
        """Fechar a janela não pode parar no meio (antes: "can't delete Tcl command")."""
        self.app.geometry("1200x700")
        self.app.update()
        app, self.app = self.app, None
        app.destroy()

    def test_lbl_canvas_live_attribute_exists(self):
        """Garante que o atributo lbl_canvas_live existe na instância e não lança AttributeError."""
        self.assertTrue(hasattr(self.app, "lbl_canvas_live"))
        # Acesso direto não pode estourar exceção
        val = self.app.lbl_canvas_live
        # A etapa "Pintar" mostra a tela do celular ao vivo durante a execução.
        self.assertIsNotNone(val)

    def test_on_worker_screencap_handles_image(self):
        """Garante que _on_worker_screencap processa imagem sem AttributeError."""
        dummy_screencap = np.zeros((1920, 1080, 3), dtype=np.uint8)
        dummy_screencap[100:200, 100:200] = (0, 255, 0)
        
        # Dispara o callback como o worker faria
        self.app._on_worker_screencap(dummy_screencap)
        # Processa eventos do timer Tkinter (self.after)
        self.app.update()
        
        self.assertIsNotNone(self.app.latest_screencap_raw)
        self.assertEqual(self.app.latest_screencap_raw.shape, (1920, 1080, 3))

    def test_on_worker_screencap_handles_none(self):
        """Garante que _on_worker_screencap trata imagem None com segurança fail-safe."""
        self.app._on_worker_screencap(None)
        self.app.update()

    def test_on_worker_error_map_updates_ui(self):
        """Garante que _on_worker_error_map atualiza métricas e mapa de calor sem erro."""
        dummy_heatmap = np.zeros((300, 300, 3), dtype=np.uint8)
        metrics = {
            "mean_delta_e": 2.45,
            "p95_delta_e": 5.12,
            "ssim": 0.941
        }
        self.app._on_worker_error_map(dummy_heatmap, metrics)
        self.app.update()
        
        lbl_text = self.app.lbl_error_metrics.cget("text")
        self.assertIn("2.45", lbl_text)
        self.assertIn("5.12", lbl_text)
        self.assertIn("0.941", lbl_text)

    def test_on_worker_log_appends_cleanly(self):
        """Garante que mensagens de log da thread worker são inseridas sem colisão de closure."""
        test_msg = "Mensagem de teste do worker de execução"
        self.app._on_worker_log(test_msg)
        self.app.update()
        
        log_content = self.app.log_box.get("1.0", "end")
        self.assertIn(test_msg, log_content)

    def test_populate_palette_inspector_renders_color_cards(self):
        """Garante que o Inspetor de Paleta renderiza os cards com botões de mudar cor."""
        mock_palette = [
            {"name": "Vermelho", "hex": "#FF0000", "rgb": (255, 0, 0), "matched_hex": "#E62328", "delta_e": 1.2},
            {"name": "Azul", "hex": "#0000FF", "rgb": (0, 0, 255), "matched_hex": "#2882F0", "delta_e": 0.8}
        ]
        self.app._populate_palette_inspector(mock_palette)
        self.app.update()
        
        children = self.app.palette_scroll.winfo_children()
        # Deve ter o cabeçalho e os 2 cards de cor
        self.assertGreaterEqual(len(children), 2)

    def test_set_input_controls_locked_toggles_states(self):
        """Garante que _set_input_controls_locked bloqueia e desbloqueia controles sem AttributeError."""
        # Bloqueia controles ao iniciar pintura
        self.app._set_input_controls_locked(True)
        self.app.update()
        for b in self.app.mode_buttons.values():
            self.assertEqual(b.cget("state"), "disabled")
        if hasattr(self.app, "btn_preview"):
            self.assertEqual(self.app.btn_preview.cget("state"), "disabled")

        # Desbloqueia controles ao finalizar ou interromper pintura
        self.app._set_input_controls_locked(False)
        self.app.update()
        for b in self.app.mode_buttons.values():
            self.assertEqual(b.cget("state"), "normal")
        if hasattr(self.app, "btn_preview"):
            self.assertEqual(self.app.btn_preview.cget("state"), "normal")

if __name__ == "__main__":
    unittest.main()
