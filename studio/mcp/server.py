# -*- coding: utf-8 -*-
"""
Servidor MCP Principal (stdio) para o Instagram Painter Studio.
Implementa os endpoints JSON-RPC: initialize, tools/list, tools/call.
"""
import sys
import json
import logging
import threading
import concurrent.futures
from importlib import metadata
from typing import Dict, Any, List, Optional
import numpy as np

def _json_default(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

from .protocol import MCPProtocol, MCPParseError, PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS
from .handlers_device import (
    handle_device_status,
    handle_verify_canvas_safety,
    handle_emergency_kill,
)
from .handlers_canvas import (
    handle_select_tool,
    handle_set_thickness,
    handle_select_color,
    handle_action_undo,
)
from .handlers_planning import (
    handle_plan_artwork,
    handle_plan_text,
    handle_preview_layer,
)
from .handlers_execution import (
    handle_execute_freehand_path,
    handle_execute_layer,
    handle_capture_screen_diff,
)

# Configuração de log para stderr (sem poluir o stdout do stdio MCP)
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [MCP] %(levelname)s: %(message)s"
)
logger = logging.getLogger("InstagramPainterMCP")

# Catálogo oficial de ferramentas MCP com JSON Schemas
TOOLS_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "ig_device_status",
        "description": "Verifica o status do celular Android (conexão ADB, tela ligada, aplicativo em primeiro plano, resolução e latência).",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "ig_verify_canvas_safety",
        "description": "Executa Servo-Visão fail-closed na tela do celular para garantir que o Instagram Direct Canvas está ativo e seguro para desenhar.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "ig_emergency_kill",
        "description": "Aciona o Kill Switch imediato no celular, cancelando qualquer script ou processo de toque ativo.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "ig_select_tool",
        "description": "Seleciona a ferramenta de desenho na barra superior do Instagram Direct.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pen_type": {
                    "type": "string",
                    "enum": ["solid", "arrow", "marker", "neon", "eraser"],
                    "description": "Tipo da caneta de desenho (padrão: solid)."
                }
            },
            "required": ["pen_type"]
        }
    },
    {
        "name": "ig_set_thickness",
        "description": "Ajusta o slider lateral de espessura de traço no Instagram.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["min", "med", "thick"],
                    "description": "Espessura desejada: min (fino), med (médio), thick (grosso)."
                }
            },
            "required": ["mode"]
        }
    },
    {
        "name": "ig_select_color",
        "description": "Seleciona uma cor na paleta do Instagram (toque simples ou toque longo para preto puro).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "color_name": {
                    "type": "string",
                    "description": "Nome da cor: branco, azul, verde, amarelo, laranja, vermelho, rosa, roxo, preto."
                }
            },
            "required": ["color_name"]
        }
    },
    {
        "name": "ig_action_undo",
        "description": "Toca no botão Desfazer (Undo) no Instagram para remover os últimos traços desenhados.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "default": 1,
                    "description": "Quantidade de vezes que o botão Undo deve ser pressionado."
                }
            }
        }
    },
    {
        "name": "ig_plan_artwork",
        "description": "Processa uma imagem de entrada com Visão Computacional, decompondo-a em camadas vetoriais estruturadas.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Caminho absoluto ou relativo do arquivo de imagem."
                },
                "style": {
                    "type": "string",
                    "enum": ["auto", "layered_paint", "line_art", "stencil"],
                    "default": "auto",
                    "description": "Estilo de vetorização artística."
                },
                "max_colors": {
                    "type": "integer",
                    "default": 4,
                    "description": "Número máximo de cores para pintura em camadas."
                },
                "time_budget_s": {
                    "type": "number",
                    "default": 600.0,
                    "description": "Orçamento de tempo limite para execução em segundos."
                }
            },
            "required": ["image_path"]
        }
    },
    {
        "name": "ig_plan_text",
        "description": ("Planeja a escrita de um texto com uma fonte TrueType, no maior tamanho que cabe na área "
                        "do canvas, com quebra de linha automática. Falha se o texto só couber abaixo do tamanho "
                        "legível para o pincel."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Texto a escrever; uma quebra de linha no texto força nova linha."},
                "font": {"type": "string", "default": "Arial",
                         "description": "Nome da fonte (ex.: Arial, Comic Sans, Segoe Script), arquivo da pasta de fontes ou caminho."},
                "ink_rgb": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3,
                            "default": [255, 255, 255], "description": "Cor da tinta (RGB)."},
                "align": {"type": "string", "enum": ["center", "left", "right"], "default": "center"},
                "max_cap_px": {"type": "number", "description": "Altura máxima da maiúscula em px de tela (opcional)."},
                "stroke": {"type": "string", "enum": ["fine", "font"], "default": "fine",
                           "description": "fine: traço fino de caneta (~4 px); font: espessura segue o peso da fonte."},
                "brush_px": {"type": "number", "description": "Com stroke=font: largura do traço em px; omitido, segue a haste da fonte."},
                "time_budget_s": {"type": "number", "default": 600.0}
            },
            "required": ["text"]
        }
    },
    {
        "name": "ig_preview_layer",
        "description": "Gera uma imagem de simulação de uma camada específica ou da arte final para o agente inspecionar visualmente.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "plan_id": {
                    "type": "string",
                    "description": "ID do plano retornado previamente por ig_plan_artwork."
                },
                "layer_index": {
                    "type": "integer",
                    "description": "Índice da camada (opcional; se omitido, gera a arte completa)."
                }
            },
            "required": ["plan_id"]
        }
    },
    {
        "name": "ig_execute_layer",
        "description": "Executa uma camada inteira de um plano artístico via ExecutionOrchestrator industrial (QualityGate, Servo-Visão, MotionCompiler a 120Hz, PersistentTouchExecutor, auditoria Closed-Loop e Telemetria).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "plan_id": {
                    "type": "string",
                    "description": "ID do plano gerado por ig_plan_artwork."
                },
                "layer_index": {
                    "type": "integer",
                    "description": "Índice numérico da camada a executar."
                },
                "batch_size": {
                    "type": "integer",
                    "default": 8,
                    "description": "Tamanho do microlote de traços por injeção atômica."
                },
                "verify_fidelity": {
                    "type": "boolean",
                    "default": True,
                    "description": "Ignorado: a auditoria pós-execução (registro subpixel e CIEDE2000) é sempre aplicada por política fail-closed."
                }
            },
            "required": ["plan_id", "layer_index"]
        }
    },
    {
        "name": "ig_execute_freehand_path",
        "description": "Desenha uma lista de traços livres ou trajetórias matemáticas (Bezier/Spline) diretamente no canvas do celular.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "strokes": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "integer"}
                        }
                    },
                    "description": "Lista de polilinhas: [ [ [x1, y1], [x2, y2], ... ], ... ]"
                },
                "thickness": {
                    "type": "string",
                    "enum": ["min", "med", "thick"],
                    "default": "min"
                },
                "color_name": {
                    "type": "string",
                    "default": "preto"
                },
                "batch_size": {
                    "type": "integer",
                    "default": 8,
                    "description": "Tamanho do microlote de traços por injeção atômica."
                }
            },
            "required": ["strokes"]
        }
    },
    {
        "name": "ig_capture_screen_diff",
        "description": "Captura a tela atual do celular e computa o mapa de calor e métricas de divergência contra o plano esperado.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "plan_id": {
                    "type": "string",
                    "description": "ID do plano opcional para calcular métricas de desvio (Delta E, SSIM)."
                }
            }
        }
    }
]

# Roteador de execução de ferramentas
TOOL_HANDLERS = {
    "ig_device_status": handle_device_status,
    "ig_verify_canvas_safety": handle_verify_canvas_safety,
    "ig_emergency_kill": handle_emergency_kill,
    "ig_select_tool": handle_select_tool,
    "ig_set_thickness": handle_set_thickness,
    "ig_select_color": handle_select_color,
    "ig_action_undo": handle_action_undo,
    "ig_plan_artwork": handle_plan_artwork,
    "ig_plan_text": handle_plan_text,
    "ig_preview_layer": handle_preview_layer,
    "ig_execute_layer": handle_execute_layer,
    "ig_execute_freehand_path": handle_execute_freehand_path,
    "ig_capture_screen_diff": handle_capture_screen_diff,
}

# Ferramentas que tocam fisicamente o celular: nunca rodam em paralelo entre si.
# ig_emergency_kill fica de fora de propósito para interromper uma execução em andamento.
DEVICE_TOOLS = frozenset({
    "ig_select_tool",
    "ig_set_thickness",
    "ig_select_color",
    "ig_action_undo",
    "ig_execute_layer",
    "ig_execute_freehand_path",
})

SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

_JSON_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}

TOOL_SCHEMAS = {tool["name"]: tool["inputSchema"] for tool in TOOLS_CATALOG}


def _server_version() -> str:
    try:
        return metadata.version("instagram-painter-studio")
    except metadata.PackageNotFoundError:
        return "3.1.0"


def validate_arguments(schema: Dict[str, Any], arguments: Any) -> Optional[str]:
    """Validação mínima (campos obrigatórios e tipos de primeiro nível) contra o inputSchema."""
    if not isinstance(arguments, dict):
        return "'arguments' deve ser um objeto."
    for name in schema.get("required", []):
        if name not in arguments:
            return f"Argumento obrigatório ausente: '{name}'."
    for name, value in arguments.items():
        expected = schema.get("properties", {}).get(name, {}).get("type")
        py_type = _JSON_TYPES.get(expected)
        if py_type is None:
            continue
        # bool é subclasse de int em Python, mas não é integer/number em JSON
        if (isinstance(value, bool) and expected != "boolean") or not isinstance(value, py_type):
            return f"Argumento '{name}' deve ser do tipo {expected}."
    return None


class MCPServer:
    def __init__(self):
        self.running = True
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="mcp_worker")
        self.device_lock = threading.Lock()

    def run(self):
        logger.info("Servidor MCP Instagram Painter iniciado em stdio.")
        try:
            while self.running:
                try:
                    msg = MCPProtocol.read_message()
                except MCPParseError as e:
                    logger.warning(f"Entrada JSON-RPC inválida: {e}")
                    MCPProtocol.send_error(None, PARSE_ERROR, "Parse error", str(e))
                    continue
                if msg is None:
                    break
                try:
                    self._dispatch(msg)
                except Exception as e:
                    logger.error(f"Erro no loop principal do MCP: {e}", exc_info=True)
        finally:
            # Aguarda as respostas em andamento antes de encerrar o processo
            self.executor.shutdown(wait=True)

    def _execute_tool_call(self, tool_name: str, handler, arguments: Dict[str, Any], req_id: Any):
        try:
            if tool_name in DEVICE_TOOLS:
                with self.device_lock:
                    res_data = handler(arguments)
            else:
                res_data = handler(arguments)
            text_content = json.dumps(res_data, ensure_ascii=False, indent=2, default=_json_default)
            MCPProtocol.send_response(req_id, {
                "content": [
                    {"type": "text", "text": text_content}
                ],
                "isError": not res_data.get("success", True) if isinstance(res_data, dict) and "success" in res_data else False
            })
        except Exception as e:
            logger.error(f"Erro na execução da ferramenta {tool_name}: {e}", exc_info=True)
            MCPProtocol.send_response(req_id, {
                "content": [
                    {"type": "text", "text": f"Erro interno na ferramenta: {str(e)}"}
                ],
                "isError": True
            })

    def _dispatch(self, msg: Dict[str, Any]):
        method = msg.get("method")
        req_id = msg.get("id")
        params = msg.get("params") or {}

        if not isinstance(method, str):
            MCPProtocol.send_error(req_id, INVALID_REQUEST, "Invalid Request: campo 'method' ausente.")
            return

        if method == "initialize":
            requested = params.get("protocolVersion")
            version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else SUPPORTED_PROTOCOL_VERSIONS[0]
            result = {
                "protocolVersion": version,
                "capabilities": {
                    "tools": {"listChanged": False}
                },
                "serverInfo": {
                    "name": "instagram-painter-mcp",
                    "version": _server_version()
                }
            }
            MCPProtocol.send_response(req_id, result)

        elif method == "notifications/initialized":
            logger.info("Cliente MCP inicializado e conectado com sucesso.")

        elif method == "tools/list":
            result = {"tools": TOOLS_CATALOG}
            MCPProtocol.send_response(req_id, result)

        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments")
            if arguments is None:
                arguments = {}
            handler = TOOL_HANDLERS.get(tool_name)

            if not handler:
                MCPProtocol.send_error(req_id, METHOD_NOT_FOUND, f"Ferramenta desconhecida '{tool_name}'")
                return

            schema = TOOL_SCHEMAS.get(tool_name)
            error = validate_arguments(schema, arguments) if schema else None
            if error:
                MCPProtocol.send_error(req_id, INVALID_PARAMS, error)
                return

            if tool_name == "ig_emergency_kill":
                # P0.6: Emergency Kill executado de forma imediata na thread do loop, sem esperar
                self._execute_tool_call(tool_name, handler, arguments, req_id)
            else:
                # Demais ferramentas rodadas em worker thread para não bloquear o loop de leitura
                self.executor.submit(self._execute_tool_call, tool_name, handler, arguments, req_id)

        elif method == "ping":
            MCPProtocol.send_response(req_id, {})

        else:
            if req_id is not None:
                MCPProtocol.send_error(req_id, METHOD_NOT_FOUND, f"Método '{method}' não suportado")

def main():
    """Ponto de entrada do servidor MCP (``mcp_server.py``, ``python -m studio.mcp.server`` ou ``instagram-painter-mcp``)."""
    import os
    # Planos, prévias e cache usam caminhos relativos à raiz do projeto.
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    # JSON-RPC sobre stdio exige UTF-8, independente da code page do Windows.
    for stream in (sys.stdin, sys.stdout):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    MCPServer().run()


if __name__ == "__main__":
    main()
