"""
Sistema visual do Instagram Painter Studio: cores, tipografia e variantes de botão.

Toda cor usada pela interface sai daqui. A paleta é escura e neutra para que
a prévia da pintura seja o elemento mais vivo da tela; o acento magenta/laranja
remete ao degradê do Instagram e marca apenas a ação principal de cada etapa.
"""
import sys

import customtkinter as ctk

COLORS = {
    # Superfícies, da mais funda para a mais elevada
    "bg": "#0B0B10",
    "surface": "#13131A",
    "surface_2": "#1A1A23",
    "surface_3": "#23232F",
    "surface_4": "#2D2D3B",
    "border": "#262633",
    "border_strong": "#3A3A4C",
    # Texto
    "text": "#F3F3F7",
    "text_muted": "#A3A3B8",
    "text_faint": "#6E6E85",
    # Acento (degradê do Instagram)
    "accent": "#E1306C",
    "accent_hover": "#C81E5B",
    "accent_soft": "#3A1424",
    "accent_2": "#F77737",
    "accent_3": "#833AB4",
    # Semânticas
    "success": "#22C55E",
    "success_hover": "#16A34A",
    "success_soft": "#10291A",
    "warning": "#F59E0B",
    "warning_soft": "#2E2208",
    "danger": "#EF4444",
    "danger_hover": "#DC2626",
    "danger_soft": "#3A1214",
    "info": "#38BDF8",
    "info_soft": "#0C2633",
    # Canvas de imagens
    "canvas": "#08080C",
    "selection": "#00E6FF",
}

# Degradê da marca usado no logotipo e na barra de progresso (início → fim)
BRAND_GRADIENT = ("#833AB4", "#E1306C", "#F77737")

RADIUS = {"sm": 8, "md": 10, "lg": 14, "pill": 999}
SPACING = {"xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24}

_FAMILY = "Segoe UI" if sys.platform.startswith("win") else "Helvetica"
_MONO = "Cascadia Mono" if sys.platform.startswith("win") else "Menlo"

_TYPE_SCALE = {
    "display": (20, "bold"),
    "title": (15, "bold"),
    "subtitle": (13, "bold"),
    "body": (13, "normal"),
    "small": (11, "normal"),
    "small_bold": (11, "bold"),
    "caption": (10, "bold"),
    "metric": (19, "bold"),
    "button": (13, "bold"),
    "button_sm": (12, "bold"),
}


def font(role: str = "body", mono: bool = False) -> ctk.CTkFont:
    size, weight = _TYPE_SCALE.get(role, _TYPE_SCALE["body"])
    return ctk.CTkFont(family=_MONO if mono else _FAMILY, size=size, weight=weight)


_BUTTON_VARIANTS = {
    "primary": dict(fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                    text_color="#FFFFFF", border_width=0),
    "success": dict(fg_color=COLORS["success"], hover_color=COLORS["success_hover"],
                    text_color="#06140B", border_width=0),
    "danger": dict(fg_color=COLORS["danger_soft"], hover_color=COLORS["danger_hover"],
                   text_color="#FCA5A5", border_width=1, border_color="#5B1A1E"),
    "secondary": dict(fg_color=COLORS["surface_3"], hover_color=COLORS["surface_4"],
                      text_color=COLORS["text"], border_width=0),
    "ghost": dict(fg_color="transparent", hover_color=COLORS["surface_3"],
                  text_color=COLORS["text_muted"], border_width=1, border_color=COLORS["border"]),
    "tile": dict(fg_color=COLORS["surface_2"], hover_color=COLORS["surface_3"],
                 text_color=COLORS["text"], border_width=1, border_color=COLORS["border"]),
    "tile_selected": dict(fg_color=COLORS["accent_soft"], hover_color="#4A1A2E",
                          text_color="#FFFFFF", border_width=1, border_color=COLORS["accent"]),
}
_BUTTON_SIZES = {"sm": (30, "button_sm"), "md": (36, "button"), "lg": (46, "button")}


def button_style(variant: str = "secondary", size: str = "md") -> dict:
    """Argumentos de CTkButton para uma variante; servem também em ``configure``."""
    height, role = _BUTTON_SIZES[size]
    style = dict(_BUTTON_VARIANTS[variant])
    style.update(height=height, corner_radius=RADIUS["md"], font=font(role),
                 text_color_disabled=COLORS["text_faint"])
    return style


def button(parent, text: str, command=None, variant: str = "secondary", size: str = "md", **kwargs) -> ctk.CTkButton:
    style = button_style(variant, size)
    style.update(kwargs)
    return ctk.CTkButton(parent, text=text, command=command, **style)


def variant_colors(variant: str) -> dict:
    """Só as cores de uma variante, para trocar o estado visual de um botão existente."""
    v = _BUTTON_VARIANTS[variant]
    return {k: v[k] for k in ("fg_color", "hover_color", "text_color", "border_color") if k in v}


def checkbox_style() -> dict:
    return dict(font=font("body"), text_color=COLORS["text"], fg_color=COLORS["accent"],
                hover_color=COLORS["accent_hover"], border_color=COLORS["border_strong"],
                checkmark_color="#FFFFFF", corner_radius=6, border_width=2)


def slider_style() -> dict:
    return dict(button_color=COLORS["accent"], button_hover_color=COLORS["accent_hover"],
                progress_color=COLORS["accent"], fg_color=COLORS["surface_4"], height=16)


def option_menu_style() -> dict:
    return dict(fg_color=COLORS["surface_3"], button_color=COLORS["surface_4"],
                button_hover_color=COLORS["border_strong"], text_color=COLORS["text"],
                dropdown_fg_color=COLORS["surface_2"], dropdown_hover_color=COLORS["surface_3"],
                dropdown_text_color=COLORS["text"], corner_radius=RADIUS["md"], height=36,
                font=font("body"), dropdown_font=font("body"))


def entry_style() -> dict:
    return dict(fg_color=COLORS["surface_2"], border_color=COLORS["border_strong"], text_color=COLORS["text"],
                placeholder_text_color=COLORS["text_faint"], corner_radius=RADIUS["md"], height=36,
                font=font("body"), border_width=1)


def apply_global_theme() -> None:
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")


def apply_dark_title_bar(window) -> None:
    """Barra de título escura e da cor do fundo no Windows 10/11 (ignorado em outros sistemas)."""
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        window.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        dark = ctypes.c_int(1)
        # 20 = DWMWA_USE_IMMERSIVE_DARK_MODE
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(dark), ctypes.sizeof(dark))
        bg = COLORS["bg"].lstrip("#")
        colorref = ctypes.c_int(int(bg[4:6] + bg[2:4] + bg[0:2], 16))
        # 35 = DWMWA_CAPTION_COLOR (Windows 11)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 35, ctypes.byref(colorref), ctypes.sizeof(colorref))
    except Exception:
        pass
