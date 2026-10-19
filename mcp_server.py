#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ponto de Entrada do Servidor MCP Instagram Painter Studio.
Executável via CLI ou registrado no Antigravity MCP runner.
"""
import sys
import os

# Garante que a raiz do repositório está no PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from studio.mcp.server import main

if __name__ == "__main__":
    main()
