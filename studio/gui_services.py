"""
Serviços desacoplados e gerenciamento de estado para a interface gráfica (P2.1, P2.2).
Isola as regras de negócio, ciclo de vida do dispositivo e orquestração de planejamento do Tkinter.
"""

from typing import Dict, Any, Optional, Tuple, List, Callable
import threading
import time
import numpy as np


class GuiProjectState:
    """
    Gerenciador de estado centralizado do projeto e da sessão de pintura.
    Controla revisões, sujidade (dirty flags), histórico de undo/redo e limites de ROI.
    """
    def __init__(self):
        self.selected_image_path: Optional[str] = None
        self.custom_canvas_bounds: Optional[Dict[str, Any]] = None
        self.current_plan: Optional[Dict[str, Any]] = None
        self.plan_revision: int = 0
        self.plan_dirty: bool = False
        self.selected_region_id: Optional[int] = None
        self.checkpoint: Dict[str, int] = {"layer": 0, "stroke": 0}
        self.undo_plans: List[Dict[str, Any]] = []
        self.redo_plans: List[Dict[str, Any]] = []
        self.max_history_depth: int = 15

    def mark_dirty(self) -> int:
        """Marca o plano atual como desatualizado em relação às configurações e incrementa a revisão."""
        self.plan_dirty = True
        self.plan_revision += 1
        return self.plan_revision

    def set_clean_plan(self, plan: Dict[str, Any], revision: Optional[int] = None) -> bool:
        """
        Atualiza o plano ativo se a revisão coincidir.
        Retorna True se aceito, False se rejeitado por obsolescência.
        """
        if revision is not None and revision != self.plan_revision:
            self.plan_dirty = True
            return False
        self.current_plan = plan
        self.plan_dirty = False
        return True

    def can_start_execution(self) -> bool:
        """Verifica se há um plano limpo, validado pelo QualityGate e pronto para pintura física."""
        if not self.current_plan or self.plan_dirty:
            return False
        qgate = self.current_plan.get("quality_gate")
        return bool(qgate and getattr(qgate, "passed", False))

    def push_undo_state(self, plan_snapshot: Dict[str, Any]):
        """Armazena cópia do plano antes de uma edição manual."""
        self.undo_plans.append(plan_snapshot)
        if len(self.undo_plans) > self.max_history_depth:
            self.undo_plans.pop(0)
        self.redo_plans.clear()

    def pop_undo(self) -> Optional[Dict[str, Any]]:
        """Recupera o último plano da pilha de desfazer."""
        if not self.undo_plans:
            return None
        if self.current_plan:
            self.redo_plans.append(self.current_plan)
        self.current_plan = self.undo_plans.pop()
        return self.current_plan

    def pop_redo(self) -> Optional[Dict[str, Any]]:
        """Recupera o último plano da pilha de refazer."""
        if not self.redo_plans:
            return None
        if self.current_plan:
            self.undo_plans.append(self.current_plan)
        self.current_plan = self.redo_plans.pop()
        return self.current_plan

    def nudge_roi(self, dx: int, dy: int, max_w: int = 1080, max_h: int = 2400) -> Optional[Dict[str, Any]]:
        """
        Desloca a área de pintura (ROI) por teclado ou botões com limites físicos (P2.7).
        """
        if not self.custom_canvas_bounds:
            # Padrão central
            w, h = 750, 950
            left = max(0, (max_w - w) // 2)
            top = max(0, (max_h - h) // 2)
            self.custom_canvas_bounds = {
                "left": left,
                "top": top,
                "right": left + w,
                "bottom": top + h,
                "center_x": left + w // 2,
                "center_y": top + h // 2,
                "max_width": w,
                "max_height": h
            }

        bounds = self.custom_canvas_bounds
        cur_w = bounds["max_width"]
        cur_h = bounds["max_height"]

        new_left = max(0, min(max_w - cur_w, bounds["left"] + dx))
        new_top = max(0, min(max_h - cur_h, bounds["top"] + dy))
        new_right = new_left + cur_w
        new_bottom = new_top + cur_h

        self.custom_canvas_bounds = {
            "left": int(new_left),
            "top": int(new_top),
            "right": int(new_right),
            "bottom": int(new_bottom),
            "center_x": int((new_left + new_right) / 2),
            "center_y": int((new_top + new_bottom) / 2),
            "max_width": int(cur_w),
            "max_height": int(cur_h)
        }
        self.mark_dirty()
        return self.custom_canvas_bounds

    def scale_roi(self, delta: int, min_size: int = 60, max_w: int = 1080, max_h: int = 2400) -> Optional[Dict[str, Any]]:
        """
        Ajusta simetricamente a escala da área de pintura (ROI) (P2.7).
        """
        if not self.custom_canvas_bounds:
            self.nudge_roi(0, 0, max_w, max_h)

        bounds = self.custom_canvas_bounds
        cx = bounds["center_x"]
        cy = bounds["center_y"]
        new_w = max(min_size, min(max_w, bounds["max_width"] + delta))
        new_h = max(min_size, min(max_h, bounds["max_height"] + delta))

        half_w = new_w // 2
        half_h = new_h // 2

        new_left = max(0, cx - half_w)
        new_right = min(max_w, new_left + new_w)
        new_top = max(0, cy - half_h)
        new_bottom = min(max_h, new_top + new_h)

        self.custom_canvas_bounds = {
            "left": int(new_left),
            "top": int(new_top),
            "right": int(new_right),
            "bottom": int(new_bottom),
            "center_x": int((new_left + new_right) / 2),
            "center_y": int((new_top + new_bottom) / 2),
            "max_width": int(new_right - new_left),
            "max_height": int(new_bottom - new_top)
        }
        self.mark_dirty()
        return self.custom_canvas_bounds


class DeviceService:
    """
    Serviço desacoplado para chamadas e diagnósticos ADB com tratamento padronizado de erro (P2.1, P2.2).
    """
    def __init__(self, adb_driver):
        self.adb = adb_driver

    def connect(self, target: Optional[str] = None) -> Tuple[bool, str]:
        try:
            return self.adb.connect(target)
        except Exception as exc:
            return False, f"Exceção ao conectar ADB: {exc}"

    def ping(self) -> Tuple[bool, str]:
        try:
            return self.adb.ping()
        except Exception as exc:
            return False, f"Falha no ping: {exc}"

    def capture_screenshot(self) -> Optional[np.ndarray]:
        try:
            return self.adb.capture_screenshot()
        except Exception:
            return None

    def open_canvas(self) -> Tuple[bool, str]:
        try:
            return self.adb.open_instagram_canvas()
        except Exception as exc:
            return False, f"Falha ao abrir tela de desenho: {exc}"

    def test_tap(self, x: int, y: int) -> Tuple[bool, str]:
        try:
            res = self.adb.run_cmd(["shell", f"input tap {x} {y}"])
            return True, f"Toque enviado em ({x}, {y})"
        except Exception as exc:
            return False, f"Erro ao enviar toque: {exc}"

    def emergency_kill(self):
        try:
            self.adb.emergency_kill()
        except Exception:
            pass


class PlanningService:
    """
    Serviço desacoplado para geração de planos em background com controle rigoroso de revisão (P2.1, P2.2).
    """
    def __init__(self, engine):
        self.engine = engine
        self._lock = threading.Lock()
        self._active_worker: Optional[threading.Thread] = None

    def run_plan_generation(
        self,
        generator_fn: Callable[[], Dict[str, Any]],
        revision: int,
        on_success: Callable[[Dict[str, Any], int], None],
        on_error: Callable[[str, int], None]
    ):
        def worker():
            try:
                plan = generator_fn()
                on_success(plan, revision)
            except Exception as exc:
                on_error(str(exc), revision)

        thread = threading.Thread(target=worker, daemon=True)
        self._active_worker = thread
        thread.start()
