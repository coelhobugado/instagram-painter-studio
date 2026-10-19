"""
Larguras arbitrárias do pincel sólido ("w24" = 24 px na tela).

O slider de espessura do Instagram é contínuo: medido no SM-A736B, a largura
cresce quase linearmente de ~10 px (base do trilho) a ~106 px (topo). Os níveis
nomeados "min", "med" e "thick" continuam valendo; "wNN" escolhe uma largura
exata e o ADBManager converte em posição do slider pela curva de calibração.
"""
from typing import Optional, Sequence, Tuple

# Curva medida em 2026-10-10 (profiles/SM-A736B/brushes.json, width_calibration);
# o ponto de 3 px, com a bolinha abaixo do trilho visível, em 2026-10-18.
DEFAULT_SLIDER_CURVE: Tuple[Tuple[int, float], ...] = ((1571, 3.0), (1497, 10.0), (1320, 38.0), (1183, 61.0), (1081, 74.0),
                                                       (882, 106.0))


# Traço fino: caneta sólida abaixo do trilho visível. Arrastar não chega lá;
# o ADBManager usa a troca neon (no mínimo) -> sólida. Medido em 2026-10-18:
# 2–4 px em retas e curvas (~6 px só muito devagar). Planejado como 4 px.
FINE_LEVEL = "w3"
FINE_DRAWN_PX = 4.0
# Base do trilho visível: abaixo disso só o traço fino.
TRACK_MIN_WIDTH_PX = 10.0
# Medido em 2026-10-18 (círculo r=35, haste e reta, 5 níveis do slider): a
# partir da base do trilho o traço sai ~2 px mais largo que o nominal em
# retas e ~5 px em curvas. Linhas de desenho misturam os dois (+3,5); letras
# são mais curvas (+4).
LINE_DRAWN_GAIN_PX = 3.5
LETTER_DRAWN_GAIN_PX = 4.0


def parse_width_level(level) -> Optional[float]:
    """Largura em px de um nível "wNN", ou None para níveis nomeados."""
    if isinstance(level, (int, float)):
        return float(level)
    if isinstance(level, str) and len(level) > 1 and level[0] == "w":
        try:
            return float(level[1:])
        except ValueError:
            return None
    return None


def width_level(px: float) -> str:
    return f"w{int(round(px))}"


def slider_y_for_width(px: float, curve: Sequence[Sequence[float]] = DEFAULT_SLIDER_CURVE) -> int:
    """Posição vertical do slider para a largura pedida (interpolação linear na curva medida)."""
    pts = sorted(((float(w), float(y)) for y, w in curve), key=lambda p: p[0])
    if px <= pts[0][0]:
        return int(round(pts[0][1]))
    if px >= pts[-1][0]:
        return int(round(pts[-1][1]))
    for (w0, y0), (w1, y1) in zip(pts, pts[1:]):
        if w0 <= px <= w1:
            t = (px - w0) / max(1e-9, w1 - w0)
            return int(round(y0 + (y1 - y0) * t))
    return int(round(pts[-1][1]))


def is_fine_level(level) -> bool:
    """Nível abaixo da base do trilho (traço fino)."""
    exact = parse_width_level(level)
    return exact is not None and exact < TRACK_MIN_WIDTH_PX


def drawn_width_px(level, nominal_px: float, gain_px: float = LINE_DRAWN_GAIN_PX) -> float:
    """Largura que a linha realmente tem no aparelho para o nível do slider."""
    if is_fine_level(level):
        return FINE_DRAWN_PX
    return float(nominal_px) + gain_px
