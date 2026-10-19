"""
Modelo de Inspeção Interativa e Edição de Regiões de Pintura.
Implementa a Fase 19 (Inspeção Detalhada e Painel de Edição de Regiões) do Roadmap.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
from .painting_plan import PaintingPlan, PaintingRegion, StrokeStrategy


@dataclass
class RegionInspectionDetail:
    """Dados completos de inspeção obtidos ao clicar em uma região (Fase 19)."""
    region_id: str
    semantic_type: str
    priority: float
    target_color_rgb: Tuple[int, int, int]
    selected_tool: str
    passes: int
    predicted_delta_e: float
    strategy: str
    stroke_count: int
    time_cost_s: float
    is_locked: bool = False
    is_disabled: bool = False
    is_protected: bool = False
    rationale: str = ""


class PaintingInspectorManager:
    """
    Gerenciador de Inspeção e Edição de Regiões (Fase 19).
    Mapeia coordenadas espaciais para regiões do plano e aplica modificações do operador humano.
    """

    def __init__(self, plan: PaintingPlan, label_map: Optional[np.ndarray] = None):
        self.plan = plan
        self.label_map = label_map
        self.region_overrides: Dict[str, Dict[str, Any]] = {}

    def inspect_at_point(self, x: int, y: int) -> Optional[RegionInspectionDetail]:
        """Identifica a região de pintura presente na coordenada (x, y)."""
        if self.label_map is None:
            return None

        h, w = self.label_map.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return None

        lbl = int(self.label_map[y, x])
        if lbl == 0:
            return None
        matched_region = next((r for r in self.plan.regions
                               if r.id in (f"region_{lbl}", f"reg_{lbl}")), None)
        return self._to_detail(matched_region) if matched_region else None

    def _to_detail(self, region: PaintingRegion) -> RegionInspectionDetail:
        overrides = self.region_overrides.get(region.id, {})
        return RegionInspectionDetail(
            region_id=region.id,
            semantic_type=region.semantic_type,
            priority=overrides.get("importance", region.importance),
            target_color_rgb=overrides.get("target_color_rgb", region.target_color_rgb),
            selected_tool=overrides.get("selected_tool", region.selected_tool),
            passes=overrides.get("passes", region.passes),
            predicted_delta_e=region.predicted_delta_e,
            strategy=overrides.get("strategy", region.strategy.value),
            stroke_count=region.stroke_count,
            time_cost_s=region.time_cost_s,
            is_locked=overrides.get("is_locked", False),
            is_disabled=overrides.get("is_disabled", False),
            is_protected=overrides.get("is_protected", False),
            rationale=region.rationale,
        )

    # Comandos de edição interativa da Fase 19
    def lock_region(self, region_id: str, locked: bool = True) -> None:
        self._ensure_override(region_id)["is_locked"] = locked

    def change_strategy(self, region_id: str, new_strategy: StrokeStrategy) -> None:
        self._ensure_override(region_id)["strategy"] = new_strategy.value

    def protect_region(self, region_id: str, protected: bool = True) -> None:
        self._ensure_override(region_id)["is_protected"] = protected

    def force_color(self, region_id: str, color_rgb: Tuple[int, int, int]) -> None:
        self._ensure_override(region_id)["target_color_rgb"] = color_rgb

    def force_brush(self, region_id: str, brush_name: str) -> None:
        self._ensure_override(region_id)["selected_tool"] = brush_name

    def disable_region(self, region_id: str, disabled: bool = True) -> None:
        self._ensure_override(region_id)["is_disabled"] = disabled

    def change_importance(self, region_id: str, new_importance: float) -> None:
        self._ensure_override(region_id)["importance"] = float(np.clip(new_importance, 0.0, 1.0))

    def _ensure_override(self, region_id: str) -> Dict[str, Any]:
        if region_id not in self.region_overrides:
            self.region_overrides[region_id] = {}
        return self.region_overrides[region_id]
