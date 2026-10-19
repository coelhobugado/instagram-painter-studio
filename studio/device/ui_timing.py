# -*- coding: utf-8 -*-
"""
Gerenciador Central de Timing e Estabilização de Interface (UI Timing Manager).
Elimina latências artificiais sobrepostas e sleeps duplicados através de uma única
camada de responsabilidade adaptativa para transições de UI.
"""
import time
from typing import Optional, Any, Dict


class UITimingManager:
    """
    Camada única responsável pelo controle de estabilização de interface do Instagram.
    Adapta automaticamente os atrasos:
    - Modo Mock / Emulador (FakeADB): 0 ms (Zero latência artificial para CI/Testes).
    - Modo Físico Real (ADBManager): Delays mínimos calibrados (elimina sobreposições).
    """

    # Delays canônicos mínimos para estabilização de animações da UI no Android (em segundos)
    DEFAULT_DELAYS: Dict[str, float] = {
        "tool_select": 0.08,           # Transição entre botões da barra superior
        "thickness_slider": 0.10,      # Acomodação do slider lateral
        "color_palette_tap": 0.08,     # Toque simples em bolinha da paleta rápida
        "color_palette_swipe": 0.45,   # Paginação da paleta e seletor de nuances (animação de ~0,4 s medida no A73)
        "undo_action": 0.08,           # Acomodação após desfazer traço
        "canvas_open_step": 0.25,      # Abertura do menu de desenho no chat
        "layer_post_draw": 0.15,       # Estabilização do buffer visual antes do screenshot
    }

    def __init__(self, is_mock: bool = False, timing_scale: float = 1.0):
        self.is_mock = is_mock
        self.timing_scale = max(0.0, float(timing_scale))
        self.delays = dict(self.DEFAULT_DELAYS)

    @classmethod
    def detect_is_mock(cls, adb_manager: Any) -> bool:
        """Determina com precisão se a instância ADB é um mock/emulador virtual."""
        if adb_manager is None:
            return True
        if getattr(adb_manager, "device_target", "") == "fake_device_emulator":
            return True
        if type(adb_manager).__name__ in ("FakeADB", "FakeADBManager"):
            return True
        if hasattr(adb_manager, "virtual_screen") and not hasattr(adb_manager, "adb_path"):
            return True
        return False

    def settle(self, action_name: str, adb_manager: Optional[Any] = None) -> float:
        """
        Aguarda o tempo estritamente necessário para estabilização da interface.
        Se for detectado ambiente mock, não dorme (0.0s).
        Retorna o tempo real esperado (em segundos).
        """
        if self.is_mock or (adb_manager is not None and self.detect_is_mock(adb_manager)):
            return 0.0

        base_delay = self.delays.get(action_name, 0.08)
        effective_delay = base_delay * self.timing_scale

        if effective_delay > 0.001:
            time.sleep(effective_delay)

        return effective_delay

    def zero_delays(self):
        """Zera todos os atrasos de interface para execução ultrarrápida contínua (0 ms)."""
        self.timing_scale = 0.0
        for k in self.delays:
            self.delays[k] = 0.0



# Instância global padrão adaptativa
_GLOBAL_TIMING = UITimingManager()


def get_ui_timing(adb_manager: Optional[Any] = None) -> UITimingManager:
    """Retorna o gerenciador de timing apropriado para o contexto de execução."""
    if adb_manager is not None and hasattr(adb_manager, "timing") and isinstance(adb_manager.timing, UITimingManager):
        return adb_manager.timing
    is_mock = UITimingManager.detect_is_mock(adb_manager)
    return UITimingManager(is_mock=is_mock)
