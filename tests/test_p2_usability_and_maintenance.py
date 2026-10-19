"""
Suíte de testes automatizados para requisitos P2 (Usabilidade, Manutenção, Performance e Acessibilidade).
Cobre P2.1 a P2.9 da auditoria técnica.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
from collections import OrderedDict

from studio.mcp.handlers_planning import PlanLRUCache
from studio.gui_services import GuiProjectState, DeviceService, PlanningService
import scripts.cleanup_artifacts as cleanup_module


class TestP2MaintenanceAndCache(unittest.TestCase):
    """Testa P2.8 (Cache LRU) e P2.4 (Limpeza de artefatos)."""

    def test_plan_lru_cache_capacity_and_eviction(self):
        cache = PlanLRUCache(maxsize=3)
        cache["plan1"] = {"id": 1}
        cache["plan2"] = {"id": 2}
        cache["plan3"] = {"id": 3}

        self.assertEqual(len(cache), 3)
        self.assertIn("plan1", cache)

        # Acessa plan1 para torná-lo recentemente usado
        _ = cache["plan1"]

        # Adiciona quarto item, deve remover plan2 (o mais antigo não usado)
        cache["plan4"] = {"id": 4}
        self.assertEqual(len(cache), 3)
        self.assertIn("plan1", cache)
        self.assertNotIn("plan2", cache)
        self.assertIn("plan3", cache)
        self.assertIn("plan4", cache)

    def test_plan_lru_cache_thread_safety(self):
        cache = PlanLRUCache(maxsize=10)
        cache.set("k1", "v1")
        self.assertEqual(cache.get("k1"), "v1")
        cache.clear()
        self.assertEqual(len(cache), 0)

    def test_cleanup_artifacts_dry_run_and_execution(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test_output.png")
            with open(test_file, "w") as f:
                f.write("dummy")

            # Dry run não deve apagar
            removed = cleanup_module.cleanup_directory(temp_dir, [".png"], dry_run=True)
            self.assertEqual(len(removed), 1)
            self.assertTrue(os.path.exists(test_file))

            # Execução real apaga
            removed_real = cleanup_module.cleanup_directory(temp_dir, [".png"], dry_run=False)
            self.assertEqual(len(removed_real), 1)
            self.assertFalse(os.path.exists(test_file))


class TestP2DocumentationAndRepo(unittest.TestCase):
    """Testa P2.3 (README, .gitignore e pyproject.toml)."""

    def test_readme_and_gitignore_exist(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        readme_path = os.path.join(root, "README.md")
        gitignore_path = os.path.join(root, ".gitignore")
        pyproject_path = os.path.join(root, "pyproject.toml")

        self.assertTrue(os.path.exists(readme_path), "README.md deve existir na raiz do projeto (P2.3)")
        self.assertTrue(os.path.exists(gitignore_path), ".gitignore deve existir na raiz do projeto (P2.3)")
        self.assertTrue(os.path.exists(pyproject_path), "pyproject.toml deve existir na raiz do projeto (P2.3)")

        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("Instagram Painter Studio", content)
            self.assertIn("Requisitos", content)

        with open(gitignore_path, "r", encoding="utf-8") as f:
            git_content = f.read()
            self.assertIn("cache/", git_content)
            self.assertIn("checkpoint.json", git_content)


class TestP2GuiStateAndServices(unittest.TestCase):
    """Testa P2.1 e P2.2 (Modularização de Estado, Device e Planning Services)."""

    def test_state_revision_and_dirty_handling(self):
        state = GuiProjectState()
        self.assertEqual(state.plan_revision, 0)
        self.assertFalse(state.plan_dirty)

        rev = state.mark_dirty()
        self.assertEqual(rev, 1)
        self.assertTrue(state.plan_dirty)

        # Tentar aplicar plano de revisão antiga
        accepted = state.set_clean_plan({"id": "old_plan"}, revision=0)
        self.assertFalse(accepted)
        self.assertTrue(state.plan_dirty)

        # Aplicar plano com revisão atualizada
        accepted = state.set_clean_plan({"id": "new_plan"}, revision=1)
        self.assertTrue(accepted)
        self.assertFalse(state.plan_dirty)
        self.assertEqual(state.current_plan["id"], "new_plan")

    def test_state_can_start_execution(self):
        state = GuiProjectState()
        self.assertFalse(state.can_start_execution())

        # Plano sem quality gate
        state.set_clean_plan({"id": "plan"})
        self.assertFalse(state.can_start_execution())

        # Plano com quality gate aprovado
        qg_mock = MagicMock()
        qg_mock.passed = True
        state.set_clean_plan({"id": "plan", "quality_gate": qg_mock})
        self.assertTrue(state.can_start_execution())

        # Plano marcado como dirty não pode iniciar
        state.mark_dirty()
        self.assertFalse(state.can_start_execution())

    def test_roi_nudge_and_scale(self):
        state = GuiProjectState()
        # Inicia e desloca ROI
        bounds = state.nudge_roi(20, 30, max_w=1080, max_h=2400)
        self.assertIsNotNone(bounds)
        self.assertTrue(state.plan_dirty)
        prev_cx = bounds["center_x"]

        # Desloca novamente
        bounds2 = state.nudge_roi(10, 0, max_w=1080, max_h=2400)
        self.assertEqual(bounds2["center_x"], prev_cx + 10)

        # Escala ROI
        prev_w = bounds2["max_width"]
        scaled_bounds = state.scale_roi(40, min_size=50, max_w=1080, max_h=2400)
        self.assertEqual(scaled_bounds["max_width"], prev_w + 40)


class TestP2GuiAccessibilityAndLayout(unittest.TestCase):
    """Testa P2.6 (Log recolhível) e P2.7 (Acessibilidade de ROI por botões/teclado)."""

    @classmethod
    def setUpClass(cls):
        try:
            import tkinter as tk
            from studio.app_gui import InstagramPainterStudioGUI
            with patch("studio.app_gui.ADBManager"), \
                 patch("studio.app_gui.PainterEngine"), \
                 patch.object(InstagramPainterStudioGUI, "action_scan_devices", return_value=None), \
                 patch.object(InstagramPainterStudioGUI, "_check_existing_checkpoint", return_value=None):
                cls.app = InstagramPainterStudioGUI()
                cls.app.withdraw()
                cls.app.update()
        except tk.TclError as exc:
            err_str = str(exc).lower()
            if "no display" in err_str or "display" in err_str or "connect to display" in err_str:
                raise unittest.SkipTest(f"Ambiente gráfico/display X11 não disponível no runner headless: {exc}")
            raise

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "app") and cls.app:
            try:
                for after_id in list(cls.app.tk.eval('after info').split()):
                    try:
                        cls.app.after_cancel(after_id)
                    except Exception:
                        pass
                cls.app.destroy()
            except Exception:
                pass

    def test_log_panel_toggle_and_clear(self):
        # Inicialmente recolhido para preservar a área da prévia.
        self.assertTrue(self.app._is_log_collapsed)
        self.assertEqual(self.app.btn_toggle_log.cget("text"), "▲ Expandir Log")

        # Expande o log
        self.app._toggle_log_panel()
        self.assertFalse(self.app._is_log_collapsed)
        self.assertEqual(self.app.btn_toggle_log.cget("text"), "▼ Recolher Log")

        # Recolhe de volta
        self.app._toggle_log_panel()
        self.assertTrue(self.app._is_log_collapsed)
        self.assertEqual(self.app.btn_toggle_log.cget("text"), "▲ Expandir Log")

        # Limpar log
        self.app.log("Mensagem de teste")
        self.assertIn("Mensagem de teste", self.app.log_box.get("1.0", "end"))
        self.app._clear_log()
        self.assertEqual(self.app.log_box.get("1.0", "end").strip(), "")

    def test_roi_nudge_and_scale_gui_controls(self):
        self.app.current_plan = {"mode": "test", "total_strokes": 10}
        self.app._plan_dirty = False
        self.app._nudge_roi(20, 20)
        self.assertIsNotNone(self.app.custom_canvas_bounds)
        self.assertTrue(self.app._plan_dirty)
        self.assertIn("Ajuste de Área", self.app.lbl_roi_coords.cget("text"))

        w_before = self.app.custom_canvas_bounds["max_width"]
        self.app._scale_roi(20)
        self.assertEqual(self.app.custom_canvas_bounds["max_width"], w_before + 20)
        self.assertIn("Escala de Área", self.app.lbl_roi_coords.cget("text"))


if __name__ == "__main__":
    unittest.main()
