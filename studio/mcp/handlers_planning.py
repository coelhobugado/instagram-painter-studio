# -*- coding: utf-8 -*-
"""
Handlers MCP para Planejamento de Arte e Pré-visualização com Visão Computacional.
"""
import os
import cv2
import numpy as np
from collections import OrderedDict
import threading
from typing import Dict, Any

from ..painter_engine import PainterEngine

class PlanLRUCache(OrderedDict):
    """Cache thread-safe com política de evicção LRU para conter consumo de memória MCP (P2.8)."""
    def __init__(self, maxsize: int = 30, max_size: int = None, *args, **kwargs):
        self.max_size = maxsize if max_size is None else max_size
        self._lock = threading.RLock()
        super().__init__(*args, **kwargs)

    def set(self, key, value):
        self[key] = value

    def clear(self):
        with self._lock:
            super().clear()

    def __getitem__(self, key):
        with self._lock:
            val = super().__getitem__(key)
            self.move_to_end(key)
            return val

    def __setitem__(self, key, value):
        with self._lock:
            super().__setitem__(key, value)
            self.move_to_end(key)
            while len(self) > self.max_size:
                self.popitem(last=False)

    def get(self, key, default=None):
        with self._lock:
            if key in self:
                self.move_to_end(key)
                return super().__getitem__(key)
            return default

    def __contains__(self, key):
        with self._lock:
            return super().__contains__(key)

# Cache limitado e thread-safe de planos gerados na sessão ativa (P2.8)
PLANS_CACHE: PlanLRUCache = PlanLRUCache(maxsize=30)
_ENGINE = None

def get_engine() -> PainterEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = PainterEngine()
    return _ENGINE

def handle_plan_artwork(args: Dict[str, Any]) -> Dict[str, Any]:
    """Gera um plano de desenho completo a partir de uma imagem de entrada."""
    image_path = args.get("image_path")
    if not image_path or not os.path.exists(image_path):
        return {"success": False, "error": f"Arquivo de imagem '{image_path}' não encontrado."}

    style = args.get("style", "auto").lower().strip()
    max_colors = int(args.get("max_colors", 4))
    time_budget_s = float(args.get("time_budget_s", 600.0))
    engine = get_engine()

    # Seleção inteligente ou explícita de estilo
    decision_payload = None
    if style == "auto":
        from ..planning.auto_mode import AutoModeDecisor
        decision = AutoModeDecisor.decide_mode(image_path)
        chosen_mode = getattr(decision, "mode", None) or getattr(decision, "recommended_mode", None) or decision.get("recommended_mode") or "layered_paint"
        if chosen_mode == "layered_paint" and hasattr(decision, "recommended_k_colors"):
            if "max_colors" not in args:
                max_colors = decision.recommended_k_colors
        decision_payload = decision.to_dict() if hasattr(decision, "to_dict") else dict(decision)
    else:
        chosen_mode = style

    if chosen_mode == "line_art":
        plan = engine.generate_line_art_plan(image_path, paint_background=False, time_budget_s=time_budget_s)
    elif chosen_mode == "stencil":
        plan = engine.generate_stencil_plan(image_path, time_budget_s=time_budget_s)
    else:
        plan = engine.generate_layered_paint_plan(image_path, num_colors=max_colors, time_budget_s=time_budget_s)

    res = _register_plan(plan)
    if decision_payload:
        res["auto_decision"] = decision_payload
    return res

def handle_plan_text(args: Dict[str, Any]) -> Dict[str, Any]:
    """Gera um plano que escreve um texto com fonte TrueType, ajustado à área do canvas."""
    from ..planning.text_engine import TextFitError
    text = str(args.get("text", ""))
    if not text.strip():
        return {"success": False, "error": "Informe o texto em 'text'."}
    ink = args.get("ink_rgb", [255, 255, 255])
    try:
        plan = get_engine().generate_text_plan(
            text, font=args.get("font", "Arial"),
            ink_rgb=tuple(int(c) for c in ink),
            brush_px=args.get("brush_px"),
            max_cap_px=args.get("max_cap_px"),
            align=args.get("align", "center"),
            time_budget_s=float(args.get("time_budget_s", 600.0)),
            stroke=args.get("stroke", "fine"))
    except (TextFitError, FileNotFoundError, ValueError) as exc:
        return {"success": False, "error": str(exc)}
    res = _register_plan(plan)
    res["text_layout"] = {k: v for k, v in plan["text_layout"].items() if k != "font_path"}
    return res

def _register_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Guarda o plano no cache e devolve o resumo das camadas."""
    plan_id = plan.get("plan_hash", f"plan_{len(PLANS_CACHE) + 1}")
    PLANS_CACHE[plan_id] = plan

    layers_summary = []
    for idx, l in enumerate(plan.get("layers", [])):
        layers_summary.append({
            "index": idx,
            "name": l.get("name", f"Camada {idx + 1}"),
            "color_rgb": list(l.get("color_rgb", (0, 0, 0))),
            "stroke_count": len(l.get("strokes", [])),
            "thickness": l.get("thickness", "min"),
            "purpose": l.get("purpose", "fill")
        })

    qgate = plan.get("quality_gate")
    qgate_passed = bool(getattr(qgate, "passed", False)) if qgate is not None else False

    res = {
        "success": True,
        "plan_id": plan_id,
        "mode": plan.get("mode"),
        "image_class": plan.get("image_class", "general"),
        "total_layers": len(plan.get("layers", [])),
        "total_strokes": plan.get("total_strokes", 0),
        "estimated_time_s": plan.get("estimated_time_s", 0.0),
        "quality_gate_passed": qgate_passed,
        "layers": layers_summary
    }
    return res

def handle_preview_layer(args: Dict[str, Any]) -> Dict[str, Any]:
    """Gera a imagem de prévia simulada de uma camada específica ou da arte cumulativa."""
    plan_id = args.get("plan_id")
    if not plan_id or plan_id not in PLANS_CACHE:
        return {"success": False, "error": f"Plano com ID '{plan_id}' não encontrado no cache."}

    plan = PLANS_CACHE[plan_id]
    cum_buffers = plan.get("cumulative_layer_images", [])
    layer_idx = args.get("layer_index")

    out_dir = os.path.join("outputs", "mcp_previews")
    os.makedirs(out_dir, exist_ok=True)

    if layer_idx is not None and 0 <= int(layer_idx) < len(cum_buffers):
        preview_img = cum_buffers[int(layer_idx)]
        filename = f"preview_{plan_id}_layer_{layer_idx}.png"
    else:
        preview_img = plan.get("simulation_image")
        filename = f"preview_{plan_id}_final.png"

    if preview_img is None:
        return {"success": False, "error": "Imagem de simulação não disponível no plano."}

    out_path = os.path.join(out_dir, filename)
    cv2.imwrite(out_path, preview_img)

    return {
        "success": True,
        "plan_id": plan_id,
        "preview_path": out_path,
        "width": int(preview_img.shape[1]),
        "height": int(preview_img.shape[0])
    }
