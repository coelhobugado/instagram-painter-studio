"""
Modelos de Dados Fundamentais e Estruturas Imutáveis (Core Domain Models).
Em conformidade com a Seção 18 do Plano de Evolução Completo.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, List, Dict, Any, Optional
import hashlib
import json

class QualityPreset(str, Enum):
    FAST = "FAST"
    BALANCED = "BALANCED"
    HIGH = "HIGH"
    ULTRA = "ULTRA"
    AUTO = "AUTO"

class BrushType(str, Enum):
    SOLID = "solid"
    MARKER = "marker"
    NEON = "neon"
    ERASER = "eraser"

class StrokePurpose(str, Enum):
    BACKGROUND = "background"
    DARK_BASE = "dark_base"
    COLOR_MASS = "color_mass"
    MIDTONE = "midtone"
    SHADOW = "shadow"
    HIGHLIGHT = "highlight"
    TEXTURE = "texture"
    CONTOUR = "contour"
    CORRECTION = "correction"

@dataclass(frozen=True)
class Stroke:
    """Representação atômica imutável de uma pincelada física."""
    id: str
    points: Tuple[Tuple[int, int], ...]
    region_id: int
    layer_id: int
    brush: str
    width: float
    opacity_mode: str
    target_color_lab: Tuple[float, float, float]
    target_color_rgb: Tuple[int, int, int]
    selector_coords: Tuple[int, int]
    is_long_press: bool
    priority: float
    purpose: str
    dependencies: Tuple[str, ...] = ()
    estimated_cost: float = 0.0

    @property
    def point_count(self) -> int:
        return len(self.points)

@dataclass
class RegionFeatures:
    """Características perceptuais e geométricas de uma região conectada (Seção 17 do Plano de Fidelidade Máxima)."""
    id: int = 0
    mean_lab: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    lab_variance: float = 0.0
    texture_angle: float = 0.0
    edge_density: float = 0.0
    area: int = 0
    centroid: Tuple[int, int] = (0, 0)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, w, h)
    region_id: Optional[int] = None
    cluster_id: int = -1
    component_id: int = -1
    neighbors: List[int] = field(default_factory=list)
    saliency: float = 0.5
    strategy: str = "horizontal_scanline"

    def __post_init__(self):
        if self.region_id is None:
            self.region_id = self.id
        elif self.id == 0 and self.region_id != 0:
            self.id = self.region_id

@dataclass(frozen=True)
class CanvasTransform:
    """Transformação bidirecional canônica entre coordenadas do Canvas Local e tela física do celular.
    Elimina a dupla escala (Seção 4 do Plano de Correções): os pontos gerados na imagem fitted
    já estão dimensionados no espaço local [0, width-1] x [0, height-1] e apenas necessitam
    do offset físico (offset_x, offset_y) para mapeamento direto na tela."""
    offset_x: int
    offset_y: int
    width: int = 0
    height: int = 0
    scale: float = 1.0  # Metadado da escala original da imagem fonte para o canvas, NÃO reaplicada a pontos

    def local_to_screen(self, pt: Tuple[int, int]) -> Tuple[int, int]:
        return (int(round(pt[0] + self.offset_x)),
                int(round(pt[1] + self.offset_y)))

    def screen_to_local(self, pt: Tuple[int, int]) -> Tuple[int, int]:
        return (int(round(pt[0] - self.offset_x)),
                int(round(pt[1] - self.offset_y)))

@dataclass(frozen=True)
class ColorSelection:
    """Modelo canônico unificado de seleção cromática (Seção 6 do Plano de Correções)."""
    rgb: Tuple[int, int, int]
    lab: Tuple[float, float, float]
    is_long_press: bool
    base_x: Optional[int]
    base_y: Optional[int]
    target_x: int
    target_y: int
    delta_e: float
    lut_id: str
    name: str = ""
    palette_page: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rgb": list(self.rgb),
            "lab": list(self.lab),
            "is_long_press": self.is_long_press,
            "base_x": self.base_x,
            "base_y": self.base_y,
            "target_x": self.target_x,
            "target_y": self.target_y,
            "x": self.target_x,
            "y": self.target_y,
            "delta_e": self.delta_e,
            "lut_id": self.lut_id,
            "name": self.name,
            "palette_page": self.palette_page,
        }

@dataclass(frozen=True)
class ColorTouch:
    """Schema canônico estrito para acionamento de cor via toque na UI do Instagram."""
    is_long_press: bool
    base_x: Optional[int]
    base_y: Optional[int]
    target_x: int
    target_y: int
    name: str = ""
    palette_page: int = 1

@dataclass
class CorrectionPlan:
    """Contrato único de correções de circuito fechado (Seção 22 do Plano de Correções)."""
    layers: List[Dict[str, Any]]
    metrics_before: Dict[str, Any]
    expected_gain: float
    total_strokes: int
    should_stop: bool = False

@dataclass
class StrokePlan:
    """Plano de execução completo versionado e auditável."""
    plan_id: str
    plan_hash: str
    preset: QualityPreset
    target_image_shape: Tuple[int, int, int]
    layers: List[Dict[str, Any]]
    total_strokes: int
    total_points: int
    estimated_time_s: float
    seed: int
    created_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    transform: Optional[CanvasTransform] = None

    @staticmethod
    def compute_plan_hash(seed: int,
                          preset: str,
                          layers: List[Dict[str, Any]],
                          image_hash: str = "",
                          device_id: str = "SM-A736B",
                          transform: Any = None) -> str:
        """Calcula hash criptográfico SHA-256 canônico e determinístico do plano completo."""
        h = hashlib.sha256()
        h.update(f"seed:{seed}|preset:{preset}|img:{image_hash}|dev:{device_id}".encode("utf-8"))
        if transform:
            h.update(f"|trans:{transform.offset_x},{transform.offset_y},{transform.scale}".encode("utf-8"))
        for l in layers:
            name = l.get("name", "")
            rgb = tuple(l.get("color_rgb", ()))
            brush = l.get("brush", "")
            thickness = l.get("thickness", "")
            strokes = l.get("strokes", [])
            touch = l.get("touch_info") or {}
            physical = (
                touch.get("palette_page", 1), touch.get("id", touch.get("lut_id")),
                touch.get("is_long_press", False), touch.get("base_x"), touch.get("base_y"),
                touch.get("target_x"), touch.get("target_y"),
            )
            h.update(f"|L:{name}:{rgb}:{tuple(l.get('render_color_rgb', ()))}:{brush}:{thickness}:{physical}:{len(strokes)}".encode("utf-8"))
            for s in strokes:
                if s:
                    for point in s:
                        h.update(f"({int(point[0])},{int(point[1])})".encode("ascii"))
                    h.update(b";")
        return h.hexdigest()[:16]

    @staticmethod
    def compute_hash(seed: int, preset: str, strokes_data: Any) -> str:
        """Método legado mantido para compatibilidade, com fallback se strokes_data for lista de camadas."""
        if isinstance(strokes_data, list) and len(strokes_data) > 0 and isinstance(strokes_data[0], dict):
            return StrokePlan.compute_plan_hash(seed, preset, strokes_data)
        content = f"{seed}_{preset}_{str(strokes_data)}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

@dataclass
class TelemetryBatch:
    """Métricas em tempo real emitidas durante a injeção física de micro-lotes."""
    batch_index: int
    total_batches: int
    stroke_start: int
    stroke_end: int
    elapsed_ms: float
    adb_latency_ms: float
    screen_state_valid: bool
    checkpoint_saved: bool
