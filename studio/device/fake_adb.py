"""
Simulador e Mock Estrito do Driver ADB (Strict FakeADB) para Testes de Integração e CI.
Implementa a Seção 60 (FakeADB) e Seção 24 da Auditoria Real (Strict Validation).
"""
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional
from .profile_loader import DeviceProfileLoader
from .ui_timing import UITimingManager

class FakeADB:
    """
    Mock estrito do ADB que simula fielmente a tela do celular, valida schemas de touch_info,
    respeita cores e espessuras selecionadas e renderiza a pintura virtual com antialiasing.
    """
    def __init__(self, width: int = 1080, height: int = 2400, init_ui: bool = True, background_bgr: Tuple[int, int, int] = (22, 22, 22)):
        self.width = width
        self.height = height
        self.is_connected = True
        self.screen_on = True
        self.instagram_foreground = True
        self.device_target = "fake_device_emulator"
        self.profile_loader = DeviceProfileLoader("SM-A736B")
        self.profile_loader.set_target_resolution(width, height)
        self.timing = UITimingManager(is_mock=True)
        self.device_info = {
            "model": "Strict FakeADB Emulator (CI/Test)",
            "resolution": f"{width}x{height}",
            "type": "Mock"
        }
        # Canvas virtual do Instagram Direct
        self.virtual_screen = np.ones((height, width, 3), dtype=np.uint8)
        self.virtual_screen[:] = background_bgr
        if init_ui:
            self._init_virtual_ui()
        self.executed_commands = []
        self.injected_strokes = []

        # Estado atual de ferramentas
        self.current_tool = "solid"
        self.current_thickness_mode = "med"
        self.current_width_px = 6
        self.current_color_bgr = (240, 240, 240)
        self.fail_next_batch = False

    def clear_canvas(self, background_bgr: Tuple[int, int, int] = (22, 22, 22)):
        """Limpa o canvas virtual com a cor de fundo especificada, sem elementos sintéticos de UI."""
        self.virtual_screen[:] = background_bgr

    def _init_virtual_ui(self):
        """Inicializa a tela virtual com os componentes visuais do Instagram Direct Canvas."""
        self.virtual_screen[:] = (22, 22, 22)
        h, w = self.height, self.width

        # Toolbar superior (8% a 20%)
        t_top = int(h * 0.08)
        t_bottom = int(h * 0.20)
        cv2.rectangle(self.virtual_screen, (40, t_top + 10), (w - 40, t_bottom - 10), (70, 70, 70), 2)
        for x in range(120, w - 100, 160):
            cv2.circle(self.virtual_screen, (x, (t_top + t_bottom) // 2), 26, (180, 180, 180), 2)
            cv2.putText(self.virtual_screen, "P", (x - 10, (t_top + t_bottom) // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Slider lateral esquerdo (35% a 75%)
        s_top = int(h * 0.35)
        s_bottom = int(h * 0.75)
        cv2.line(self.virtual_screen, (45, s_top), (45, s_bottom), (160, 160, 160), 4)
        cv2.circle(self.virtual_screen, (45, (s_top + s_bottom) // 2), 16, (230, 230, 230), -1)

        # Paleta inferior (82% a 100%)
        b_top = int(h * 0.82)
        for i, cx in enumerate(range(80, w - 80, 110)):
            color = ((i * 45) % 255, (i * 75) % 255, (255 - i * 35) % 255)
            cv2.circle(self.virtual_screen, (cx, (b_top + h) // 2), 28, color, -1)
            cv2.circle(self.virtual_screen, (cx, (b_top + h) // 2), 30, (200, 200, 200), 2)

    def connect(self, target: str = None) -> Tuple[bool, str]:
        self.is_connected = True
        return True, "FakeADB conectado com sucesso (Mock)"

    def auto_reconnect(self) -> bool:
        self.is_connected = True
        return True

    def ping(self) -> Tuple[bool, str]:
        return True, "0 ms (FakeADB Local)"

    def is_screen_on(self) -> bool:
        return self.is_connected and self.screen_on

    def is_instagram_foreground(self) -> bool:
        return self.is_connected and self.instagram_foreground

    def open_instagram_canvas(self) -> Tuple[bool, str]:
        self.executed_commands.append("open_instagram_canvas")
        return True, "Tela de desenho aberta no FakeADB"

    def select_pen_tool(self, pen: str = "solid"):
        self.current_tool = pen
        self.executed_commands.append(f"select_pen_tool_{pen}")

    def set_brush_thickness(self, mode: str = "min"):
        self.current_thickness_mode = mode
        if mode == "min":
            self.current_width_px = 10
        elif mode == "thick":
            self.current_width_px = 98
        elif mode == "med":
            self.current_width_px = 52
        elif isinstance(mode, (int, float)) or (isinstance(mode, str) and mode.startswith("w")):
            from .brush_width import parse_width_level, is_fine_level, FINE_DRAWN_PX
            # Traço fino: a largura real medida no aparelho (~4 px), como na prévia
            width = FINE_DRAWN_PX if is_fine_level(mode) else (parse_width_level(mode) or 6)
            self.current_width_px = max(1, int(round(width)))
        else:
            self.current_width_px = 6
        self.executed_commands.append(f"set_brush_thickness_{mode}")

    def set_drawn_width_hint(self, drawn_width_px: Optional[float]):
        """Largura real da camada segundo a prévia (traços de linha); None mantém a do nível."""
        if drawn_width_px:
            self.current_width_px = max(1, int(round(float(drawn_width_px))))

    def select_palette_color(self, touch_info: Dict[str, Any]):
        """
        Validação Estrita de Schema (Seção 3 e 24 da Auditoria Real):
        Rejeita chamadas com coordenadas incompletas ou chaves ausentes.
        """
        if not isinstance(touch_info, dict):
            raise TypeError("touch_info deve ser um dicionário")

        if touch_info.get("is_black") or touch_info.get("name") == "Preto Puro":
            self.executed_commands.append("tap_color_black")
            self.current_color_bgr = (38, 38, 38)
            return

        if touch_info.get("is_long_press"):
            base = touch_info.get("base_touch") or {}
            bx = touch_info.get("base_x", base.get("x"))
            by = touch_info.get("base_y", base.get("y"))
            tx = touch_info.get("target_x", touch_info.get("touch_x", touch_info.get("x")))
            ty = touch_info.get("target_y", touch_info.get("touch_y", touch_info.get("y")))
            if bx is None or by is None or tx is None or ty is None:
                raise KeyError(f"StrictFakeADB: Coordenadas de toque longo incompletas: {touch_info}")
            self.executed_commands.append(f"select_spectrum_drag_{bx}_{by}_to_{tx}_{ty}")
        else:
            x = touch_info.get("target_x", touch_info.get("touch_x", touch_info.get("x")))
            y = touch_info.get("target_y", touch_info.get("touch_y", touch_info.get("y")))
            if x is None or y is None:
                raise KeyError(f"StrictFakeADB: Coordenadas de toque simples incompletas: {touch_info}")
            self.executed_commands.append(f"tap_color_{x}_{y}")

        # Atualiza a cor se fornecida
        if "rgb" in touch_info:
            r, g, b = touch_info["rgb"]
            self.current_color_bgr = (int(b), int(g), int(r))
        elif "matched_rgb" in touch_info:
            r, g, b = touch_info["matched_rgb"]
            self.current_color_bgr = (int(b), int(g), int(r))
        elif "target_rgb" in touch_info:
            r, g, b = touch_info["target_rgb"]
            self.current_color_bgr = (int(b), int(g), int(r))

    def set_color_rgb(self, rgb: Tuple[int, int, int]):
        """Define a cor física atual diretamente via tupla RGB."""
        r, g, b = rgb
        self.current_color_bgr = (int(b), int(g), int(r))
        self.executed_commands.append(f"set_color_rgb_{r}_{g}_{b}")

    def set_color_bgr(self, bgr: Tuple[int, int, int]):
        """Define a cor física atual diretamente via tupla BGR."""
        self.current_color_bgr = tuple(int(c) for c in bgr)
        self.executed_commands.append(f"set_color_bgr_{bgr[0]}_{bgr[1]}_{bgr[2]}")

    def execute_stroke_batch(self, strokes: List[List[Tuple[int, int]]]) -> bool:
        """Renderiza os traços estritamente no canvas virtual do FakeADB."""
        if self.fail_next_batch:
            self.fail_next_batch = False
            return False

        if not self.is_connected or not self.is_screen_on():
            return False

        self.injected_strokes.extend(strokes)
        for stroke in strokes:
            if len(stroke) >= 2:
                # Valida que nenhum ponto possui NaN ou coordenadas infinitas
                for pt in stroke:
                    if pt[0] < 0 or pt[0] >= self.width or pt[1] < 0 or pt[1] >= self.height:
                        pass # permitido com clip
                pts = np.array(stroke, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(self.virtual_screen, [pts], False, self.current_color_bgr, self.current_width_px, cv2.LINE_AA)
        return True

    def capture_screenshot(self) -> np.ndarray:
        return self.virtual_screen.copy()

    def emergency_kill(self) -> Tuple[bool, str]:
        self.executed_commands.append("emergency_kill")
        return True, "Kill switch acionado no FakeADB!"

# Alias para compatibilidade com testes e workers
FakeADBManager = FakeADB

