# -*- coding: utf-8 -*-
"""
Testes de framing JSON-RPC do servidor MCP: linhas em branco, JSON malformado,
validação de argumentos e persistência das dimensões do CanvasTransform.
"""
import io
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from studio.mcp.server import MCPServer, validate_arguments, TOOL_SCHEMAS
from studio.core.models import CanvasTransform
from studio.storage.execution_artifact import ExecutionArtifact


def _run_server(raw_input: str):
    stdin, stdout = io.StringIO(raw_input), io.StringIO()
    with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
        MCPServer().run()
    return [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]


class TestMCPFraming(unittest.TestCase):
    def test_blank_lines_are_skipped(self):
        ping = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"})
        responses = _run_server("\n\n" + ping + "\n")
        self.assertEqual(responses, [{"jsonrpc": "2.0", "id": 1, "result": {}}])

    def test_malformed_json_returns_parse_error_and_continues(self):
        ping = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"})
        responses = _run_server("{bad json\n" + ping + "\n")
        self.assertEqual(len(responses), 2)
        self.assertEqual(responses[0]["error"]["code"], -32700)
        self.assertIsNone(responses[0]["id"])
        self.assertEqual(responses[1], {"jsonrpc": "2.0", "id": 2, "result": {}})

    def test_content_length_framing(self):
        body = json.dumps({"jsonrpc": "2.0", "id": 3, "method": "ping"})
        responses = _run_server(f"Content-Length: {len(body)}\r\n\r\n{body}")
        self.assertEqual(responses, [{"jsonrpc": "2.0", "id": 3, "result": {}}])

    def test_initialize_echoes_supported_protocol_version(self):
        init = json.dumps({"jsonrpc": "2.0", "id": 4, "method": "initialize",
                           "params": {"protocolVersion": "2024-11-05"}})
        responses = _run_server(init + "\n")
        self.assertEqual(responses[0]["result"]["protocolVersion"], "2024-11-05")

    def test_invalid_arguments_return_invalid_params(self):
        call = json.dumps({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                           "params": {"name": "ig_execute_layer", "arguments": {"plan_id": "x", "layer_index": "0"}}})
        responses = _run_server(call + "\n")
        self.assertEqual(responses[0]["error"]["code"], -32602)
        self.assertIn("layer_index", responses[0]["error"]["message"])

    def test_validate_arguments_required_and_types(self):
        schema = TOOL_SCHEMAS["ig_execute_layer"]
        self.assertIn("layer_index", validate_arguments(schema, {"plan_id": "p"}))
        self.assertIsNotNone(validate_arguments(schema, {"plan_id": "p", "layer_index": True}))
        self.assertIsNone(validate_arguments(schema, {"plan_id": "p", "layer_index": 1, "verify_fidelity": False}))
        self.assertIn("batch_size", TOOL_SCHEMAS["ig_execute_freehand_path"]["properties"])


class TestArtifactTransformDimensions(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_transform_width_height_round_trip(self):
        artifact = ExecutionArtifact(
            plan_id="p1", fingerprint="f1", mode="stencil", image_class="logo",
            transform=CanvasTransform(offset_x=10, offset_y=20, width=300, height=400, scale=1.5),
            layers=[], quality_gate={}, profile_metrics={}, source_image="img.png",
            estimated_time_s=1.0, total_strokes=0,
        )
        base = os.path.join(self.temp_dir, "plan")
        artifact.save(base)
        loaded = ExecutionArtifact.load(base)
        self.assertEqual(loaded.transform, artifact.transform)
        self.assertEqual([f for f in os.listdir(self.temp_dir) if f.startswith(".tmp_")], [])


if __name__ == "__main__":
    unittest.main()
