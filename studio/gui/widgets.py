"""Componentes visuais reutilizáveis da interface (cartões, etapas, selos, métricas, avisos)."""
import math

import customtkinter as ctk
from PIL import Image, ImageDraw

from .theme import COLORS, BRAND_GRADIENT, RADIUS, font, button_style, variant_colors


def _hex_to_rgb(value):
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def brand_logo(size: int = 64) -> Image.Image:
    """Quadrado arredondado com o degradê da marca e um traço contínuo em branco."""
    scale = 4
    s = size * scale
    stops = [_hex_to_rgb(c) for c in BRAND_GRADIENT]
    grad = Image.new("RGB", (s, s))
    px = grad.load()
    for y in range(s):
        for x in range(s):
            t = min(1.0, max(0.0, (x + (s - y)) / (2 * s)))
            seg = 0 if t < 0.5 else 1
            local = (t - 0.5 * seg) * 2
            a, b = stops[seg], stops[seg + 1]
            px[x, y] = tuple(int(a[i] + (b[i] - a[i]) * local) for i in range(3))
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, s - 1, s - 1), radius=int(s * 0.28), fill=255)
    logo = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    logo.paste(grad, (0, 0), mask)
    draw = ImageDraw.Draw(logo)
    pts = [(s * (0.2 + 0.6 * i / 40), s * (0.52 + 0.17 * math.sin(i / 40 * 2 * math.pi))) for i in range(41)]
    draw.line(pts, fill=(255, 255, 255, 255), width=int(s * 0.085), joint="curve")
    r = s * 0.045
    for cx, cy in (pts[0], pts[-1]):
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 255, 255, 255))
    return logo.resize((size, size), Image.LANCZOS)


class ActionButton(ctk.CTkButton):
    """Botão de ação que perde a cor de destaque quando desativado.

    O CTkButton mantém o fundo colorido mesmo desativado, o que faz um botão
    bloqueado parecer clicável. Aqui o estado e a variante decidem as cores.
    """

    def __init__(self, master, text="", command=None, variant="primary", size="md", **kwargs):
        self._variant = variant
        style = button_style(variant, size)
        style.update(kwargs)
        super().__init__(master, text=text, command=command, **style)
        self._sync_colors()

    def configure(self, require_redraw=False, **kwargs):
        super().configure(require_redraw=require_redraw, **kwargs)
        if "state" in kwargs:
            self._sync_colors()

    def set_variant(self, variant):
        if variant != self._variant:
            self._variant = variant
            self._sync_colors()

    def _sync_colors(self):
        enabled = str(self.cget("state")) != "disabled"
        super().configure(**variant_colors(self._variant if enabled else "secondary"))


class Card(ctk.CTkFrame):
    """Superfície elevada com título opcional; o conteúdo vai em ``card.body``."""

    def __init__(self, master, title=None, subtitle=None, padding=14, **kwargs):
        kwargs.setdefault("fg_color", COLORS["surface"])
        kwargs.setdefault("corner_radius", RADIUS["lg"])
        kwargs.setdefault("border_width", 1)
        kwargs.setdefault("border_color", COLORS["border"])
        super().__init__(master, **kwargs)
        if title:
            ctk.CTkLabel(self, text=title, font=font("subtitle"), text_color=COLORS["text"],
                         anchor="w").pack(fill="x", padx=padding, pady=(padding, 0))
        if subtitle:
            ctk.CTkLabel(self, text=subtitle, font=font("small"), text_color=COLORS["text_muted"],
                         anchor="w", justify="left", wraplength=300).pack(fill="x", padx=padding, pady=(2, 0))
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.pack(fill="both", expand=True, padx=padding, pady=padding)


def section_label(master, text):
    return ctk.CTkLabel(master, text=text.upper(), font=font("caption"), text_color=COLORS["text_faint"], anchor="w")


class Stepper(ctk.CTkFrame):
    """Trilha de etapas clicáveis: ativa (acento), concluída (✓ verde) ou pendente."""

    def __init__(self, master, steps, command=None, **kwargs):
        kwargs.setdefault("fg_color", "transparent")
        super().__init__(master, **kwargs)
        self._command = command
        self._items = {}
        self._active = None
        self._done = set()
        for index, (key, label) in enumerate(steps):
            if index:
                ctk.CTkFrame(self, width=28, height=2, fg_color=COLORS["border_strong"]).pack(side="left", padx=4)
            item = ctk.CTkFrame(self, fg_color="transparent", corner_radius=RADIUS["pill"], cursor="hand2")
            item.pack(side="left")
            dot = ctk.CTkLabel(item, text=str(index + 1), width=24, height=24, corner_radius=12,
                               fg_color=COLORS["surface_4"], text_color=COLORS["text_muted"], font=font("small_bold"))
            dot.pack(side="left", padx=(6, 6), pady=6)
            text = ctk.CTkLabel(item, text=label, font=font("small_bold"), text_color=COLORS["text_muted"])
            text.pack(side="left", padx=(0, 10))
            for widget in (item, dot, text):
                widget.bind("<Button-1>", lambda _e, k=key: self._on_click(k))
            self._items[key] = (item, dot, text, index)

    def _on_click(self, key):
        if self._command:
            self._command(key)

    def set_active(self, key):
        self._active = key
        self._paint()

    def set_done(self, key, done=True):
        (self._done.add if done else self._done.discard)(key)
        self._paint()

    def _paint(self):
        for key, (item, dot, text, index) in self._items.items():
            if key == self._active:
                item.configure(fg_color=COLORS["surface_3"])
                dot.configure(fg_color=COLORS["accent"], text_color="#FFFFFF", text=str(index + 1))
                text.configure(text_color=COLORS["text"])
            elif key in self._done:
                item.configure(fg_color="transparent")
                dot.configure(fg_color=COLORS["success_soft"], text_color=COLORS["success"], text="✓")
                text.configure(text_color=COLORS["text_muted"])
            else:
                item.configure(fg_color="transparent")
                dot.configure(fg_color=COLORS["surface_4"], text_color=COLORS["text_muted"], text=str(index + 1))
                text.configure(text_color=COLORS["text_faint"])


class StatusPill(ctk.CTkFrame):
    """Selo de estado com ponto colorido (conectado, desconectado, ocupado...)."""

    _TONES = {
        "ok": (COLORS["success"], COLORS["success_soft"]),
        "off": (COLORS["danger"], COLORS["danger_soft"]),
        "busy": (COLORS["warning"], COLORS["warning_soft"]),
        "info": (COLORS["info"], COLORS["info_soft"]),
        "idle": (COLORS["text_faint"], COLORS["surface_3"]),
    }

    def __init__(self, master, text="", tone="idle", command=None, **kwargs):
        kwargs.setdefault("corner_radius", RADIUS["pill"])
        if command is not None:
            kwargs.setdefault("cursor", "hand2")
        super().__init__(master, **kwargs)
        self._dot = ctk.CTkLabel(self, text="●", font=font("small_bold"), width=10)
        self._dot.pack(side="left", padx=(12, 6), pady=5)
        self._text = ctk.CTkLabel(self, text=text, font=font("small_bold"))
        self._text.pack(side="left", padx=(0, 14), pady=5)
        if command is not None:
            for widget in (self, self._dot, self._text):
                widget.bind("<Button-1>", lambda _e: command())
        self.set(text, tone)

    def set(self, text, tone="idle"):
        fg, bg = self._TONES.get(tone, self._TONES["idle"])
        self.configure(fg_color=bg)
        self._dot.configure(text_color=fg)
        self._text.configure(text=text, text_color=fg if tone != "idle" else COLORS["text_muted"])


class StatTile(ctk.CTkFrame):
    """Bloco de métrica: rótulo pequeno em cima, valor grande embaixo."""

    _TONES = {"neutral": COLORS["text"], "good": COLORS["success"], "warn": COLORS["warning"],
              "bad": COLORS["danger"], "muted": COLORS["text_faint"]}

    def __init__(self, master, label, value="—", **kwargs):
        kwargs.setdefault("fg_color", COLORS["surface_2"])
        kwargs.setdefault("corner_radius", RADIUS["md"])
        super().__init__(master, **kwargs)
        ctk.CTkLabel(self, text=label.upper(), font=font("caption"), text_color=COLORS["text_faint"],
                     anchor="w").pack(fill="x", padx=12, pady=(10, 0))
        self._value = ctk.CTkLabel(self, text=value, font=font("metric"), text_color=COLORS["text_faint"], anchor="w")
        self._value.pack(fill="x", padx=12, pady=(0, 10))

    def set(self, value, tone="neutral"):
        self._value.configure(text=value, text_color=self._TONES.get(tone, COLORS["text"]))


class Toast(ctk.CTkFrame):
    """Aviso flutuante não-bloqueante no canto inferior direito da janela."""

    _TONES = {"ok": COLORS["success"], "info": COLORS["info"], "warn": COLORS["warning"], "error": COLORS["danger"]}
    _ICONS = {"ok": "✓", "info": "i", "warn": "!", "error": "×"}

    def __init__(self, root, message, tone="info", duration_ms=3500):
        super().__init__(root, fg_color=COLORS["surface_3"], corner_radius=RADIUS["lg"],
                         border_width=1, border_color=self._TONES.get(tone, COLORS["info"]))
        color = self._TONES.get(tone, COLORS["info"])
        ctk.CTkLabel(self, text=self._ICONS.get(tone, "i"), width=24, height=24, corner_radius=12,
                     fg_color=color, text_color=COLORS["bg"], font=font("small_bold")).pack(side="left", padx=(12, 8), pady=12)
        ctk.CTkLabel(self, text=message, font=font("body"), text_color=COLORS["text"], justify="left",
                     wraplength=340).pack(side="left", padx=(0, 16), pady=12)
        self.place(relx=1.0, rely=1.0, x=-20, y=-96, anchor="se")
        self.lift()
        self.after(duration_ms, self._dismiss)

    def _dismiss(self):
        try:
            self.destroy()
        except Exception:
            pass
