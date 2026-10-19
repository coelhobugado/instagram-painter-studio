# -*- coding: utf-8 -*-
"""
Testes Unitários e de Protocolo do Servidor MCP do Instagram Painter Studio.
"""
import unittest
import json
import os
from unittest.mock import MagicMock, patch

from studio.mcp.server import MCPServer, TOOLS_CATALOG, TOOL_HANDLERS
from studio.mcp.handlers_planning import PLANS_CACHE

class TestMCPServer(unittest.TestCase):
    def setUp(self):
        self.server = MCPServer()

    def test_tools_catalog_completeness(self):
        """Valida que o catálogo contém as 13 ferramentas com schemas válidos."""
        self.assertEqual(len(TOOLS_CATALOG), 13)
        tool_names = [t["name"] for t in TOOLS_CATALOG]

        expected_tools = [
            "ig_device_status",
            "ig_verify_canvas_safety",
            "ig_emergency_kill",
            "ig_select_tool",
            "ig_set_thickness",
            "ig_select_color",
            "ig_action_undo",
            "ig_plan_artwork",
            "ig_plan_text",
            "ig_preview_layer",
            "ig_execute_layer",
            "ig_execute_freehand_path",
            "ig_capture_screen_diff"
        ]
        for exp in expected_tools:
            self.assertIn(exp, tool_names, f"Ferramenta esperada {exp} não encontrada no catálogo.")
            self.assertIn(exp, TOOL_HANDLERS, f"Handler para {exp} não registrado.")

    def test_tool_select_tool(self):
        """Valida a ferramenta ig_select_tool."""
        handler = TOOL_HANDLERS["ig_select_tool"]
        with patch("studio.mcp.handlers_canvas.get_adb") as mock_adb_getter:
            mock_adb = MagicMock()
            mock_adb_getter.return_value = mock_adb

            # Caso de sucesso
            res = handler({"pen_type": "neon"})
            self.assertTrue(res["success"])
            self.assertEqual(res["selected_tool"], "neon")
            mock_adb.select_pen_tool.assert_called_with("neon")

            # Caso de erro (ferramenta inexistente)
            err_res = handler({"pen_type": "spray_can"})
            self.assertFalse(err_res["success"])
            self.assertIn("inválida", err_res["error"])

    def test_tool_set_thickness(self):
        """Valida a ferramenta ig_set_thickness."""
        handler = TOOL_HANDLERS["ig_set_thickness"]
        with patch("studio.mcp.handlers_canvas.get_adb") as mock_adb_getter:
            mock_adb = MagicMock()
            mock_adb_getter.return_value = mock_adb

            res = handler({"mode": "thick"})
            self.assertTrue(res["success"])
            self.assertEqual(res["thickness_mode"], "thick")
            mock_adb.set_brush_thickness.assert_called_with("thick")

    def test_tool_select_color(self):
        """Valida a ferramenta ig_select_color."""
        handler = TOOL_HANDLERS["ig_select_color"]
        with patch("studio.mcp.handlers_canvas.get_adb") as mock_adb_getter:
            mock_adb = MagicMock()
            mock_adb_getter.return_value = mock_adb

            # Amarelo da paleta
            res = handler({"color_name": "amarelo"})
            self.assertTrue(res["success"])
            self.assertEqual(res["selected_color"], "Amarelo")
            mock_adb.select_palette_color.assert_called()

            # Preto Puro
            res_black = handler({"color_name": "preto"})
            self.assertTrue(res_black["success"])
            self.assertEqual(res_black["selected_color"], "Preto Puro")

    def test_tool_plan_and_preview_artwork(self):
        """Valida o ciclo de planejamento e geração de prévia via MCP."""
        plan_handler = TOOL_HANDLERS["ig_plan_artwork"]
        preview_handler = TOOL_HANDLERS["ig_preview_layer"]

        sample_img = "inputs/flores.jpg"
        if not os.path.exists(sample_img):
            self.skipTest("Arquivo de amostra inputs/flores.jpg não encontrado.")

        # Planeja arte em modo stencil
        plan_res = plan_handler({
            "image_path": sample_img,
            "style": "stencil"
        })
        self.assertTrue(plan_res["success"])
        self.assertIn("plan_id", plan_res)
        plan_id = plan_res["plan_id"]
        self.assertIn(plan_id, PLANS_CACHE)

        # Gera prévia da arte planejada
        prev_res = preview_handler({"plan_id": plan_id})
        self.assertTrue(prev_res["success"])
        self.assertTrue(os.path.exists(prev_res["preview_path"]))

    def test_tool_emergency_kill(self):
        """Valida o acionamento do kill switch via MCP."""
        handler = TOOL_HANDLERS["ig_emergency_kill"]
        with patch("studio.mcp.handlers_device.get_adb") as mock_adb_getter:
            mock_adb = MagicMock()
            mock_adb.emergency_kill.return_value = (True, "Kill switch acionado")
            mock_adb_getter.return_value = mock_adb

            res = handler({})
            self.assertTrue(res["success"])
            self.assertIn("Kill switch", res["message"])

if __name__ == "__main__":
    unittest.main()
