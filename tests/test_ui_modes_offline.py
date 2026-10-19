"""Verifica a seleção de estratégias sem conectar ou comandar um celular."""
import unittest
from unittest.mock import patch
import tkinter as tk
from types import SimpleNamespace

from studio.app_gui import InstagramPainterStudioGUI


class _OfflineADB:
    is_connected = False

    def discover_devices(self):
        return []


class _ImmediateThread:
    def __init__(self, target, **_kwargs):
        self.target = target

    def start(self):
        self.target()


class TestUIModesOffline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adb_patch = patch("studio.app_gui.ADBManager", _OfflineADB)
        cls.adb_patch.start()
        # Nenhum teste deve abrir diálogos modais reais na tela.
        cls.msg_patch = patch("studio.app_gui.messagebox")
        cls.msg_patch.start()
        try:
            cls.app = InstagramPainterStudioGUI()
            cls.app.withdraw()
            cls.app.update()
        except tk.TclError as exc:
            cls.adb_patch.stop()
            cls.msg_patch.stop()
            msg = str(exc).lower()
            if any(k in msg for k in ("no display", "cannot connect", "display name", "x11")):
                raise unittest.SkipTest(f"Ambiente gráfico headless sem display disponível: {exc}")
            raise

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "app"):
            try:
                cls.app.destroy()
            except Exception:
                pass
        cls.adb_patch.stop()
        cls.msg_patch.stop()

    def test_each_manual_mode_uses_its_own_engine(self):
        calls = []

        def fake_plan(mode):
            def generate(*_args, **kwargs):
                calls.append((mode, kwargs.get("colorize"), kwargs.get("fill_regions")))
                return {"mode": mode}
            return generate

        self.app.engine.generate_line_art_plan = fake_plan("line_art")
        self.app.engine.generate_flat_illustration_plan = fake_plan("flat_illustration")

        def finish(_plan, _revision):
            self.app._generating_preview = False

        # Sem imagem selecionada a GUI recusa gerar a prévia.
        self.app.selected_image_path = "inputs/flores.jpg"
        with patch("studio.app_gui.threading.Thread", _ImmediateThread),              patch.object(self.app, "_post_generate_preview", side_effect=finish):
            for selected in ("LINHAS", "COLORIR", "CAMADAS"):
                self.app._select_mode(selected)
                self.app.action_generate_preview()
                self.app.update()

        self.assertEqual(calls, [("line_art", True, False), ("line_art", True, True),
                                 ("flat_illustration", None, None)])

    def test_text_mode_generates_preview_without_image(self):
        old_path = self.app.selected_image_path
        try:
            self.app.selected_image_path = None
            self.app._select_mode("TEXTO")
            self.app.txt_text_input.delete("1.0", "end")
            self.assertFalse(self.app._has_source())
            self.assertEqual(self.app.btn_start.cget("state"), "disabled")

            self.app.txt_text_input.insert("1.0", "Oi, tudo bem?")
            self.app.opt_text_font.set("Arial")
            self.app._mark_plan_dirty()
            self.assertTrue(self.app._has_source())
            with patch("studio.app_gui.threading.Thread", _ImmediateThread):
                self.app.action_generate_preview()
                self.app.update()

            self.assertEqual(self.app.current_plan["mode"], "text")
            self.assertFalse(self.app._plan_dirty)
            self.assertTrue(self.app.current_plan["quality_gate"].passed)
            self.assertIn("Texto", self.app.lbl_preview_mode.cget("text"))
            self.assertEqual(self.app.btn_start.cget("text"), "Conecte o celular")
        finally:
            self.app.selected_image_path = old_path
            self.app._select_mode("AUTO")

    def test_auto_mode_picks_style_from_image(self):
        from studio.planning.auto_mode import AutoModeDecisor
        import cv2
        self.assertEqual(AutoModeDecisor.choose_drawing_style(cv2.imread("inputs/flores.jpg")), "LINHAS")
        self.assertEqual(AutoModeDecisor.choose_drawing_style(cv2.imread("inputs/personagem.jpg")), "COLORIR")
        self.assertEqual(AutoModeDecisor.choose_drawing_style(cv2.imread("inputs/retrato.jpg")), "CAMADAS")

    def test_changing_mode_invalidates_existing_preview(self):
        self.app.current_plan = {"mode": "line_art"}
        self.app._plan_dirty = False
        self.app.btn_start.configure(state="normal")

        self.app._select_mode("CAMADAS")

        self.assertTrue(self.app._plan_dirty)
        # O botão vira "Atualizar prévia" (ação válida), mas a pintura fica bloqueada
        self.assertEqual(self.app.btn_start.cget("text"), "Atualizar prévia")
        self.assertFalse(self.app._can_start_current_plan())
        self.assertIn("novamente", self.app.lbl_plan_summary.cget("text"))

    def test_result_from_old_settings_is_discarded(self):
        previous_plan = self.app.current_plan
        revision = self.app._plan_revision
        self.app._select_mode("STENCIL")
        self.app._post_generate_preview({"mode": "line_art"}, revision)

        self.assertIs(self.app.current_plan, previous_plan)
        self.assertTrue(self.app._plan_dirty)
        self.assertFalse(self.app._can_start_current_plan())

    def test_start_requires_approved_preview_and_connection(self):
        old_plan, old_dirty = self.app.current_plan, self.app._plan_dirty
        try:
            self.app.current_plan = {"quality_gate": SimpleNamespace(passed=True)}
            self.app._plan_dirty = False
            self.app._editing_plan = False
            self.app._generating_preview = False
            self.app._update_start_availability()
            self.assertFalse(self.app._can_start_current_plan())
            self.assertEqual(self.app.btn_start.cget("text"), "Conecte o celular")
            # Clicar leva até a seção de conexão; nunca inicia a pintura
            with patch.object(self.app, "_show_step") as show_step,                  patch.object(self.app, "action_start_painting") as start:
                self.app.action_primary()
            show_step.assert_called_once_with("device")
            start.assert_not_called()
            self.app.adb.is_connected = True
            self.app._update_start_availability()
            self.assertTrue(self.app._can_start_current_plan())
            self.assertEqual(self.app.btn_start.cget("state"), "normal")
        finally:
            self.app.adb.is_connected = False
            self.app.current_plan, self.app._plan_dirty = old_plan, old_dirty
            self.app._update_start_availability()

    def test_primary_button_regenerates_stale_preview(self):
        old_plan, old_dirty, old_path = self.app.current_plan, self.app._plan_dirty, self.app.selected_image_path
        try:
            self.app.current_plan = {"quality_gate": SimpleNamespace(passed=True)}
            self.app.selected_image_path = "inputs/flores.jpg"
            self.app._plan_dirty = True
            self.app._update_start_availability()
            self.assertEqual(self.app.btn_start.cget("state"), "normal")
            with patch.object(self.app, "action_generate_preview") as generate,                  patch.object(self.app, "action_start_painting") as start:
                self.app.action_primary()
            generate.assert_called_once()
            start.assert_not_called()
        finally:
            self.app.current_plan, self.app._plan_dirty, self.app.selected_image_path = old_plan, old_dirty, old_path
            self.app._update_start_availability()

    def test_new_area_regenerates_existing_preview(self):
        old_plan, old_path = self.app.current_plan, self.app.selected_image_path
        try:
            self.app.current_plan = {"quality_gate": SimpleNamespace(passed=True)}
            self.app.selected_image_path = "inputs/flores.jpg"
            with patch.object(self.app, "action_generate_preview") as generate:
                self.app._reset_roi_selection()
                self.app.after(200, self.app.quit)
                self.app.mainloop()
            generate.assert_called_once()
        finally:
            self.app.current_plan, self.app.selected_image_path = old_plan, old_path

    def test_header_stops_showing_painting_once_worker_reports_end(self):
        # O executor avisa o fim ainda dentro da própria thread (is_alive() = True)
        worker = SimpleNamespace(is_alive=lambda: True, verify_each_layer=False)
        self.app.worker = worker
        try:
            self.app._worker_finished = False
            self.assertTrue(self.app._is_painting())
            self.app._on_worker_finished(False, "Execução interrompida.")
            self.app.update()
            self.assertFalse(self.app._is_painting())
            self.assertNotIn("Pintando", self.app.connection_pill._text.cget("text"))
        finally:
            self.app.worker = None

    def test_automatic_mode_can_skip_background(self):
        self.app._select_mode("AUTO")
        self.app.var_skip_background.set(True)
        self.assertFalse(self.app._get_line_color_overrides()[2])
        self.app.var_skip_background.set(False)


if __name__ == "__main__":
    unittest.main()
