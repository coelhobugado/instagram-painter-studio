"""
Artefato Central do Plano de Pintura e LOD Progressivo.
Implementa a Fase 7 (Vocabulário de Strokes e Estratégia por Região)
e Fase 8 (Orçamento e LOD Progressivo com Presets Reais) do Roadmap.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import json


class StrokeStrategy(str, Enum):
    """Vocabulário formal de estratégias de pincelada (Fase 7.1)."""
    MASS_FILL = "MASS_FILL"
    FLOW_FILL = "FLOW_FILL"
    CONTOUR = "CONTOUR"
    CENTERLINE = "CENTERLINE"
    WIDTH_VARYING_INK = "WIDTH_VARYING_INK"
    HATCH = "HATCH"
    STIPPLE = "STIPPLE"
    DETAIL = "DETAIL"
    POINT_HIGHLIGHT = "POINT_HIGHLIGHT"
    NEGATIVE_STROKE = "NEGATIVE_STROKE"


class LODLevel(int, Enum):
    """Níveis de detalhe progressivos (Fase 8)."""
    LOD_0_SILHOUETTE_MASS = 0  # Silhueta e massas dominantes
    LOD_1_CORE_STRUCTURE = 1   # Cores principais e estrutura
    LOD_2_CONTOURS_FEATURES = 2 # Contornos e features importantes
    LOD_3_MICRO_DETAILS = 3    # Microdetalhes, highlights e textura


# Orçamentos reais em segundos por Preset (Fase 8.1)
PRESET_BUDGETS_SECONDS: Dict[str, float] = {
    "FAST": 45.0,        # ~45 segundos
    "BALANCED": 120.0,   # ~2 minutos
    "HIGH": 300.0,       # ~5 minutos
    "ULTRA": 600.0,      # ~10 minutos
}


@dataclass
class PaintingRegion:
    """Região de pintura com explicação auditável de seu propósito e custo (Fase 7.3)."""
    id: str
    semantic_type: str
    importance: float            # 0.0 a 1.0
    strategy: StrokeStrategy
    lod_min: int = 0             # 0, 1, 2 ou 3
    protected_regions: List[str] = field(default_factory=list)
    time_cost_s: float = 0.0
    stroke_count: int = 0
    target_color_rgb: Tuple[int, int, int] = (0, 0, 0)
    target_color_lab: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    selected_tool: str = "solid"
    passes: int = 1
    predicted_delta_e: float = 0.0
    is_unreproducible: bool = False
    rationale: str = ""          # Explicação auditável do porquê existe e da estratégia escolhida

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "semantic_type": self.semantic_type,
            "importance": round(self.importance, 3),
            "strategy": self.strategy.value,
            "lod_min": self.lod_min,
            "protected_regions": list(self.protected_regions),
            "time_cost_s": round(self.time_cost_s, 2),
            "stroke_count": self.stroke_count,
            "target_color_rgb": list(self.target_color_rgb),
            "target_color_lab": [round(c, 2) for c in self.target_color_lab],
            "selected_tool": self.selected_tool,
            "passes": self.passes,
            "predicted_delta_e": round(self.predicted_delta_e, 2),
            "is_unreproducible": self.is_unreproducible,
            "rationale": self.rationale,
        }


@dataclass
class PaintingPlan:
    """
    Plano de Pintura de Nível Superior.
    Separa o que queremos representar (semântica, intenções, regiões)
    de como o robô físico executa (strokes, motion, eventos de toque).
    """
    image_class: str             # portrait, photo_general, anime, line_art, logo, text_heavy, mixed
    time_budget_s: float
    regions: List[PaintingRegion] = field(default_factory=list)
    total_strokes: int = 0
    estimated_time_s: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_region(self, region: PaintingRegion) -> None:
        self.regions.append(region)
        self.total_strokes += region.stroke_count
        self.estimated_time_s += region.time_cost_s

    def get_regions_for_lod(self, max_lod: int) -> List[PaintingRegion]:
        """Filtra regiões que devem ser renderizadas até o nível de LOD especificado."""
        return [r for r in self.regions if r.lod_min <= max_lod]

    def prune_to_budget(self, max_seconds: Optional[float] = None) -> List[PaintingRegion]:
        """
        Orçamento Progressivo (Fase 8):
        Ordena regiões por prioridade (importance / custo) e seleciona as que cabem no budget.
        Garante que mesmo interrompido cedo produza uma imagem coerente.
        """
        budget = max_seconds if max_seconds is not None else self.time_budget_s
        # Ordena: menor lod_min primeiro, depois maior importância por custo
        sorted_regions = sorted(
            self.regions,
            key=lambda r: (r.lod_min, -r.importance / max(0.1, r.time_cost_s))
        )
        selected: List[PaintingRegion] = []
        accumulated_time = 0.0

        for r in sorted_regions:
            if accumulated_time + r.time_cost_s <= budget or len(selected) == 0:
                selected.append(r)
                accumulated_time += r.time_cost_s
            elif r.lod_min == 0:
                # Regiões fundamentais de silhueta (LOD 0) são sempre mantidas para evitar buracos
                selected.append(r)
                accumulated_time += r.time_cost_s

        return selected

    def to_json(self) -> str:
        return json.dumps({
            "image_class": self.image_class,
            "time_budget_s": self.time_budget_s,
            "estimated_time_s": round(self.estimated_time_s, 2),
            "total_strokes": self.total_strokes,
            "regions": [r.to_dict() for r in self.regions],
            "metadata": self.metadata,
        }, indent=2)
