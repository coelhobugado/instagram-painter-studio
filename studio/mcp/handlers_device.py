# -*- coding: utf-8 -*-
"""
Handlers MCP para Diagnóstico de Dispositivo e Segurança Fail-Closed.
"""
import os
import time
from typing import Dict, Any

from ..adb_manager import ADBManager
from ..device.ui_state_machine import ServoVisionDetector, UIState

_SHARED_ADB = None

def get_adb() -> ADBManager:
    global _SHARED_ADB
    if _SHARED_ADB is None:
        _SHARED_ADB = ADBManager()
        _SHARED_ADB.connect()
    return _SHARED_ADB

def set_adb(adb: Any):
    """Permite injetar uma instância customizada ou mock de ADB (ex: FakeADB nos testes)."""
    global _SHARED_ADB
    _SHARED_ADB = adb

def handle_device_status(args: Dict[str, Any]) -> Dict[str, Any]:
    """Retorna o status físico completo do celular conectado."""
    adb = get_adb()
    is_conn = adb.is_connected
    if not is_conn:
        is_conn, msg = adb.connect()
    else:
        msg = "Conectado"

    screen_on = adb.is_screen_on() if is_conn else False
    ig_foreground = adb.is_instagram_foreground() if is_conn else False
    ping_ok, latency = adb.ping() if is_conn else (False, "N/A")

    return {
        "connected": is_conn,
        "device_target": adb.device_target,
        "device_model": adb.device_info.get("model", "Desconhecido"),
        "resolution": adb.device_info.get("resolution", "1080x2400"),
        "screen_on": screen_on,
        "instagram_foreground": ig_foreground,
        "latency": latency,
        "status_message": msg
    }

def handle_verify_canvas_safety(args: Dict[str, Any]) -> Dict[str, Any]:
    """Executa servo-visão fail-closed na tela atual para garantir integridade do canvas."""
    adb = get_adb()
    if not adb.is_connected or not adb.is_screen_on():
        return {
            "is_valid": False,
            "reason": "Dispositivo desconectado ou tela apagada.",
            "confidence": 0.0
        }

    screenshot = adb.capture_screenshot()
    if screenshot is None:
        return {
            "is_valid": False,
            "reason": "Falha ao capturar a tela do celular.",
            "confidence": 0.0
        }

    detector = ServoVisionDetector()
    val_res = detector.detect_and_validate(screenshot)

    return {
        "is_valid": bool(val_res.is_valid),
        "reason": str(val_res.reason),
        "confidence": round(float(val_res.confidence), 3),
        "dimensions_valid": bool(val_res.dimensions_valid),
        "safe_bounds_valid": bool(val_res.safe_bounds_valid),
        "toolbar_detected": bool(val_res.toolbar_present),
        "palette_detected": bool(val_res.palette_present)
    }

def handle_emergency_kill(args: Dict[str, Any]) -> Dict[str, Any]:
    """Aciona o Kill Switch imediato no Android."""
    adb = get_adb()
    ok, msg = adb.emergency_kill()
    return {
        "success": ok,
        "message": msg,
        "timestamp": time.time()
    }
