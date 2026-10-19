# -*- coding: utf-8 -*-
"""
Protocolo de Transporte JSON-RPC 2.0 para o Servidor MCP via stdio.
Compatível com a especificação oficial do Model Context Protocol (Anthropic/Google).
Suporta leitura com framing por linha ou cabeçalho Content-Length.
"""
import sys
import json
import threading
from typing import Dict, Any, Optional

# Códigos de erro padrão JSON-RPC 2.0
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602


class MCPParseError(Exception):
    """Entrada recebida não é JSON válido (o servidor responde -32700 e continua lendo)."""


class MCPProtocol:
    _write_lock = threading.Lock()

    @staticmethod
    def read_message() -> Optional[Dict[str, Any]]:
        """
        Lê uma mensagem JSON-RPC da entrada padrão (stdin).

        Retorna None somente em EOF. Linhas em branco são ignoradas e entradas
        malformadas levantam MCPParseError, para que o loop do servidor continue.
        """
        while True:
            line = sys.stdin.readline()
            if not line:
                return None

            line = line.strip()
            if line:
                break

        # Suporte a cabeçalho Content-Length
        if line.lower().startswith("content-length:"):
            try:
                length = int(line.split(":", 1)[1].strip())
            except ValueError as e:
                raise MCPParseError(f"Cabeçalho Content-Length inválido: {line!r}") from e
            # Consome linha em branco separadora
            sys.stdin.readline()
            payload = sys.stdin.read(length)
            if len(payload) < length:
                return None
        else:
            # Suporte a JSON-RPC por linha
            payload = line

        try:
            msg = json.loads(payload)
        except json.JSONDecodeError as e:
            raise MCPParseError(str(e)) from e
        if not isinstance(msg, dict):
            raise MCPParseError("Mensagem JSON-RPC deve ser um objeto.")
        return msg

    @staticmethod
    def _write(msg: Dict[str, Any]):
        out_str = json.dumps(msg, ensure_ascii=False)
        with MCPProtocol._write_lock:
            sys.stdout.write(out_str + "\n")
            sys.stdout.flush()

    @staticmethod
    def send_response(req_id: Any, result: Any):
        """Envia uma resposta JSON-RPC bem-sucedida para stdout de forma thread-safe."""
        MCPProtocol._write({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": result
        })

    @staticmethod
    def send_error(req_id: Any, code: int, message: str, data: Any = None):
        """Envia uma resposta de erro JSON-RPC para stdout de forma thread-safe."""
        err_dict = {"code": code, "message": message}
        if data is not None:
            err_dict["data"] = data
        MCPProtocol._write({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": err_dict
        })
