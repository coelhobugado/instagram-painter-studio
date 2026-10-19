# -*- coding: utf-8 -*-
"""
Handlers MCP para Controle da Interface de Pintura do Instagram Direct (Ferramentas, Cores, Espessura, Undo).
"""
from typing import Dict, Any
from .handlers_device import get_adb

# Mapeamento canônico de cores da paleta rápida do Instagram Direct (Samsung Galaxy A73)
PALETTE_MAP = {
    "branco": {"name": "Branco", "target_x": 411, "target_y": 2185, "is_long_press": False},
    "azul": {"name": "Azul Claro", "target_x": 486, "target_y": 2185, "is_long_press": False},
    "azul_claro": {"name": "Azul Claro", "target_x": 486, "target_y": 2185, "is_long_press": False},
    "verde": {"name": "Verde", "target_x": 561, "target_y": 2185, "is_long_press": False},
    "amarelo": {"name": "Amarelo", "target_x": 636, "target_y": 2185, "is_long_press": False},
    "laranja": {"name": "Laranja", "target_x": 711, "target_y": 2185, "is_long_press": False},
    "vermelho": {"name": "Vermelho Coral", "target_x": 786, "target_y": 2185, "is_long_press": False},
    "rosa": {"name": "Rosa", "target_x": 861, "target_y": 2185, "is_long_press": False},
    "roxo": {"name": "Roxo", "target_x": 936, "target_y": 2185, "is_long_press": False},
    "preto": {"name": "Preto Puro", "is_black": True},
}

def handle_select_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Seleciona a ferramenta de desenho na barra superior."""
    adb = get_adb()
    pen = args.get("pen_type", "solid").lower().strip()
    valid_pens = ["solid", "arrow", "marker", "neon", "eraser"]
    if pen not in valid_pens:
        return {"success": False, "error": f"Ferramenta inválida '{pen}'. Opções: {valid_pens}"}

    adb.select_pen_tool(pen)
    return {"success": True, "selected_tool": pen}

def handle_set_thickness(args: Dict[str, Any]) -> Dict[str, Any]:
    """Ajusta o slider lateral de espessura de traço."""
    adb = get_adb()
    mode = args.get("mode", "med").lower().strip()
    valid_modes = ["min", "med", "thick"]
    if mode not in valid_modes:
        return {"success": False, "error": f"Modo de espessura inválido '{mode}'. Opções: {valid_modes}"}

    adb.set_brush_thickness(mode)
    return {"success": True, "thickness_mode": mode}

def handle_select_color(args: Dict[str, Any]) -> Dict[str, Any]:
    """Seleciona uma cor na paleta do Instagram."""
    adb = get_adb()
    color_name = args.get("color_name", "").lower().strip()

    if color_name in PALETTE_MAP:
        touch_info = PALETTE_MAP[color_name]
    elif args.get("is_black"):
        touch_info = {"name": "Preto Puro", "is_black": True}
    elif "target_x" in args and "target_y" in args:
        touch_info = {
            "name": color_name or "Personalizada",
            "target_x": int(args["target_x"]),
            "target_y": int(args["target_y"]),
            "is_long_press": bool(args.get("is_long_press", False)),
            "base_x": args.get("base_x"),
            "base_y": args.get("base_y")
        }
    else:
        return {
            "success": False,
            "error": f"Cor '{color_name}' não reconhecida na paleta rápida. Opções: {list(PALETTE_MAP.keys())}"
        }

    try:
        adb.select_palette_color(touch_info)
        return {"success": True, "selected_color": touch_info["name"]}
    except Exception as e:
        return {"success": False, "error": str(e)}

def handle_action_undo(args: Dict[str, Any]) -> Dict[str, Any]:
    """Aciona o botão Desfazer (Undo) no Instagram Direct."""
    adb = get_adb()
    count = max(1, int(args.get("count", 1)))
    undo_coord = adb.profile_loader.get_ui_anchor("button_undo", (199, 2193))
    ux, uy = adb.profile_loader.scale_point(undo_coord[0], undo_coord[1])

    for _ in range(count):
        if hasattr(adb, "run_cmd"):
            adb.run_cmd(["shell", f"input tap {ux} {uy}"])
        elif hasattr(adb, "executed_commands"):
            adb.executed_commands.append("button_undo")
        if hasattr(adb, "timing"):
            adb.timing.settle("undo_action")

    return {"success": True, "undo_count": count}
