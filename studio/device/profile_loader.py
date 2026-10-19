"""
Carregador e Validador de Perfis de Dispositivo (Hardware Profile Loader).
Implementa a governança completa dos arquivos em profiles/SM-A736B/ (device.json, ui_geometry.json, brushes.json, etc.)
em conformidade com as Seções 1, 2 e 19 da Auditoria Real.
"""
import os
import json
import logging
from typing import Dict, Any, Tuple, Optional

from .. import config

logger = logging.getLogger(__name__)


class DeviceProfileLoader:
    def __init__(self, profile_name: str = config.DEFAULT_DEVICE_PROFILE):
        self.profile_name = profile_name
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.profile_dir = os.path.join(self.base_dir, "profiles", profile_name)

        self.device_json = self._load_file("device.json")
        self.ui_geometry = self._load_file("ui_geometry.json")
        self.brushes_json = self._load_file("brushes.json")
        self.opacity_json = self._load_file("opacity.json")
        self.version_json = self._load_file("instagram_version.json")
        self.scale_x = 1.0
        self.scale_y = 1.0
        resolution = self.device_json.get("screen_resolution", self.device_json.get("display", {}))
        self.target_width = resolution.get("width", 1080)
        self.target_height = resolution.get("height", 2400)

    def set_target_resolution(self, width: int, height: int):
        """Ajusta proporcionalmente todas as coordenadas de toque para a resolução real do aparelho."""
        if width <= 0 or height <= 0:
            return
        self.target_width = width
        self.target_height = height
        resolution = self.device_json.get("screen_resolution", self.device_json.get("display", {}))
        ref_w = resolution.get("width", 1080)
        ref_h = resolution.get("height", 2400)
        self.scale_x = width / float(ref_w)
        self.scale_y = height / float(ref_h)

    def scale_point(self, x: int, y: int) -> Tuple[int, int]:
        """Aplica o fator de escala horizontal e vertical a um ponto (X, Y)."""
        return (int(round(x * self.scale_x)), int(round(y * self.scale_y)))

    @classmethod
    def profile_exists(cls, profile_name: str) -> bool:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        return bool(profile_name) and os.path.isfile(
            os.path.join(base_dir, "profiles", profile_name, "ui_geometry.json"))

    def _load_file(self, filename: str) -> Dict[str, Any]:
        p = os.path.join(self.profile_dir, filename)
        if not os.path.exists(p):
            logger.warning("Perfil %s sem %s; usando coordenadas padrão de config.py.",
                           self.profile_name, filename)
            return {}
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Falha ao ler %s (%s); usando coordenadas padrão de config.py.", p, exc)
            return {}

    def get_safe_bounds(self) -> Dict[str, int]:
        """Retorna os limites seguros da tela física (safe_bounds) ajustados para a resolução."""
        sb = self.ui_geometry.get("safe_bounds", {})
        l, t = self.scale_point(sb.get("left", config.CANVAS_X_MIN), sb.get("top", config.CANVAS_Y_MIN))
        r, b = self.scale_point(sb.get("right", config.CANVAS_X_MAX), sb.get("bottom", config.CANVAS_Y_MAX))
        cx, cy = self.scale_point(sb.get("center_x", config.CANVAS_SAFE_CX), sb.get("center_y", config.CANVAS_SAFE_CY))
        mw = int(round(sb.get("max_width", config.CANVAS_SAFE_MAX_W) * self.scale_x))
        mh = int(round(sb.get("max_height", config.CANVAS_SAFE_MAX_H) * self.scale_y))
        return {
            "left": l,
            "right": r,
            "top": t,
            "bottom": b,
            "center_x": cx,
            "center_y": cy,
            "max_width": mw,
            "max_height": mh
        }

    def get_ui_anchor(self, key: str, default: Tuple[int, int] = (0, 0)) -> Tuple[int, int]:
        """Obtém coordenada física (X, Y) escalada de um botão ou âncora da interface."""
        anchors = self.ui_geometry.get("ui_anchors", {})
        val = anchors.get(key)
        if isinstance(val, dict) and "x" in val and "y" in val:
            return self.scale_point(int(val["x"]), int(val["y"]))
        return self.scale_point(default[0], default[1])

    def get_width_curve(self):
        """Curva medida (posição y do slider, largura px) do pincel sólido, em coordenadas da tela."""
        curve = (self.brushes_json.get("brushes", {}).get("solid", {})
                 .get("width_calibration", {}).get("slider_y_to_width_px"))
        if not curve:
            return None
        return [(self.scale_point(0, int(y))[1], float(w)) for y, w in curve]

    def get_slider_thickness(self) -> Dict[str, int]:
        """Obtém coordenadas escaladas do slider de espessura de traço."""
        anchors = self.ui_geometry.get("ui_anchors", {})
        sl = anchors.get("slider_thickness", {})
        sx, y_start = self.scale_point(sl.get("x", config.UI_SLIDER_X), sl.get("min_y_start", config.UI_SLIDER_MIN_Y_START))
        _, y_end = self.scale_point(0, sl.get("min_y_end", config.UI_SLIDER_MIN_Y_END))
        _, min_y = self.scale_point(0, sl.get("min_y", config.UI_SLIDER_MIN_Y))
        _, my = self.scale_point(0, sl.get("medium_y", config.UI_SLIDER_MED_Y))
        _, ty = self.scale_point(0, sl.get("thick_y", config.UI_SLIDER_THICK_Y))
        return {
            "x": sx,
            "min_y_start": y_start,
            "min_y_end": y_end,
            "min_y": min_y,
            "medium_y": my,
            "thick_y": ty
        }

    def get_palette_base_anchor(self, key: str, default: Tuple[int, int] = (0, 0)) -> Tuple[int, int]:
        """Obtém âncora de toque escalada da paleta inferior."""
        pb = self.ui_geometry.get("ui_anchors", {}).get("palette_base", {})
        btn = pb.get(key)
        if isinstance(btn, dict) and "x" in btn and "y" in btn:
            return self.scale_point(int(btn["x"]), int(btn["y"]))
        return self.scale_point(default[0], default[1])

    def get_palette_swipe(self, key: str, default) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Obtém um gesto (início, fim) escalado da navegação da paleta."""
        gestures = self.ui_geometry.get("ui_anchors", {}).get("palette_swipes", {})
        val = gestures.get(key)
        try:
            (sx, sy), (ex, ey) = (val["from"]["x"], val["from"]["y"]), (val["to"]["x"], val["to"]["y"])
        except (TypeError, KeyError):
            (sx, sy), (ex, ey) = default
        return self.scale_point(int(sx), int(sy)), self.scale_point(int(ex), int(ey))
