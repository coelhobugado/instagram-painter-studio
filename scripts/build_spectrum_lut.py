"""
Gera ``color_lut.json`` a partir de capturas reais da paleta do Instagram.

Entradas (em ``profiles/<modelo>/spectrum/``), recortes da tela a partir de
(x=340, y=2100), 660x200 px:
- ``palette_page{1,2,3}.png``: a fileira de bolinhas de cada página;
- ``strip_page{1,2,3}.png``: a mesma fileira capturada com o dedo ainda
  pressionado sobre uma bolinha, quando ela vira o seletor 2D de nuances
  (horizontal: matizes da página; vertical: do claro ao escuro).

No SM-A736B a cor desenhada coincidiu com o pixel da faixa a ~5–15 RGB.

Uso: python scripts/build_spectrum_lut.py [profiles/SM-A736B]
"""
import json
import os
import sys
from datetime import datetime, timezone

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from studio.vision.perceptual_color import PerceptualColorEngine  # noqa: E402

CROP_ORIGIN = (340, 2100)
SWATCH_Y = 2184
SWATCH_XS = [411, 486, 561, 636, 711, 786, 861, 936]
SWATCHES_PER_PAGE = {1: 8, 2: 8, 3: 6}
# A bolinha mantida pressionada para abrir o seletor existe nas três páginas.
LONG_PRESS_BASE = (486, SWATCH_Y)
PAGE1_NAMES = ["Branco Puro", "Azul Cobalto", "Verde Folha", "Amarelo Ouro",
               "Laranja Solar", "Vermelho Intenso", "Rosa Magenta", "Roxo Violeta"]
STRIP_MAX_X = 974
DARKEST_RELIABLE = 38
GRID_STEP_X = 8
GRID_STEP_Y = 7
EDGE_MARGIN = 6
SEPARATOR_MARGIN = 3


def color_name(rgb):
    """Nome legível em português a partir da cor (família de matiz + claridade)."""
    import colorsys
    r, g, b = (v / 255.0 for v in rgb)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    lightness = ("muito escuro" if l < 0.18 else "escuro" if l < 0.38 else "médio" if l < 0.62
                 else "claro" if l < 0.85 else "muito claro")
    if s < 0.12 or max(r, g, b) - min(r, g, b) < 0.06:
        if l > 0.96:
            return "Branco"
        if l < 0.06:
            return "Preto"
        return f"Cinza {lightness}"
    deg = h * 360
    if 15 <= deg < 45 and l < 0.55:
        family = "Marrom"
    elif 10 <= deg < 45 and l >= 0.55 and s < 0.75:
        family = "Pele"
    else:
        family = next(name for limit, name in ((15, "Vermelho"), (40, "Laranja"), (65, "Amarelo"), (160, "Verde"),
                                               (195, "Ciano"), (255, "Azul"), (290, "Roxo"), (340, "Rosa"),
                                               (361, "Vermelho")) if deg < limit)
    if family == "Pele":
        lightness = {"muito escuro": "muito escura", "escuro": "escura", "claro": "clara",
                     "muito claro": "muito clara"}.get(lightness, lightness)
    return f"{family} {lightness}"


def to_screen(x, y):
    return x + CROP_ORIGIN[0], y + CROP_ORIGIN[1]


def strip_bounds(strip_bgr, background_bgr):
    """Retângulo da faixa: maior região diferente do fundo escuro da barra."""
    diff = np.abs(strip_bgr.astype(np.int16) - np.asarray(background_bgr, np.int16)).sum(axis=2)
    mask = (diff > 30).astype(np.uint8)
    # O botão vermelho de enviar encosta na faixa pela direita e não faz parte dela.
    mask[:, STRIP_MAX_X - CROP_ORIGIN[0]:] = 0
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    best = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    x, y, w, h = stats[best, :4]
    return int(x), int(y), int(w), int(h)


def entry(entry_id, category, name, page, rgb, touch, base=None):
    r, g, b = (int(v) for v in rgb)
    lab = PerceptualColorEngine.rgb_to_lab(r, g, b)
    return {
        "id": entry_id,
        "category": category,
        "name": name,
        "is_long_press": base is not None,
        "palette_page": page,
        "touch_x": touch[0], "touch_y": touch[1],
        "target_x": touch[0], "target_y": touch[1],
        "base_x": base[0] if base else None,
        "base_y": base[1] if base else None,
        "rgb": [r, g, b],
        "hex": f"#{r:02X}{g:02X}{b:02X}",
        "lab": [round(float(v), 2) for v in lab],
    }


def build(profile_dir):
    spectrum_dir = os.path.join(profile_dir, "spectrum")
    entries = []
    bounds_by_page = {}
    for page in (1, 2, 3):
        palette = cv2.imread(os.path.join(spectrum_dir, f"palette_page{page}.png"))
        strip = cv2.imread(os.path.join(spectrum_dir, f"strip_page{page}.png"))
        if palette is None or strip is None:
            raise FileNotFoundError(f"Capturas da página {page} ausentes em {spectrum_dir}")

        for k in range(SWATCHES_PER_PAGE[page]):
            sx, sy = SWATCH_XS[k] - CROP_ORIGIN[0], SWATCH_Y - CROP_ORIGIN[1]
            bgr = np.median(palette[sy - 3:sy + 4, sx - 3:sx + 4].reshape(-1, 3), axis=0)
            name = PAGE1_NAMES[k] if page == 1 else color_name(bgr[::-1])
            entries.append(entry(f"base_p{page}_{k + 1}", "base_palette", name, page,
                                 bgr[::-1], (SWATCH_XS[k], SWATCH_Y)))

        background = strip[5, 5]
        x0, y0, w, h = strip_bounds(strip, background)
        bounds_by_page[page] = (x0, y0, w, h)
        separators = [x0 + w * i / 8.0 for i in range(1, 8)]
        # Indicador de página (três pontos) fica sobre a faixa, ao centro e embaixo.
        dots_x = (640 - CROP_ORIGIN[0], 710 - CROP_ORIGIN[0])
        dots_y = (2240 - CROP_ORIGIN[1], 2268 - CROP_ORIGIN[1])
        for y in range(y0 + EDGE_MARGIN, y0 + h - EDGE_MARGIN, GRID_STEP_Y):
            for x in range(x0 + EDGE_MARGIN, x0 + w - EDGE_MARGIN, GRID_STEP_X):
                if any(abs(x - s) <= SEPARATOR_MARGIN for s in separators):
                    continue
                if dots_x[0] <= x <= dots_x[1] and dots_y[0] <= y <= dots_y[1]:
                    continue
                bgr = np.median(strip[y - 1:y + 2, x - 1:x + 2].reshape(-1, 3), axis=0)
                # Nuances mais escuras que a bolinha #262626 não se reproduzem:
                # no A73 o canto escuro da faixa desenhou (55,55,55) e não (25,25,25).
                if float(bgr.max()) < DARKEST_RELIABLE:
                    continue
                tx, ty = to_screen(x, y)
                entries.append(entry(f"spec_p{page}_{tx}_{ty}", "continuous_spectrum",
                                     color_name(bgr[::-1]), page, bgr[::-1], (tx, ty), LONG_PRESS_BASE))

    x0, y0, w, h = bounds_by_page[1]
    sx0, sy0 = to_screen(x0, y0)
    return {
        "schema_version": 3,
        "device": os.path.basename(os.path.normpath(profile_dir)),
        "instagram_package": "com.instagram.android",
        "instagram_version": "451.0.0.0.58",
        "source": "capturas reais do seletor de nuances (toque longo) e da paleta base",
        "sample_count": len(entries),
        "composition_breakdown": {
            "base_palette_samples": sum(1 for e in entries if not e["is_long_press"]),
            "continuous_spectrum_samples": sum(1 for e in entries if e["is_long_press"]),
        },
        "selector_bounds": {"x_min": sx0, "x_max": sx0 + w, "y_top": sy0, "y_center": SWATCH_Y,
                            "y_bottom": sy0 + h},
        "long_press_base": {"x": LONG_PRESS_BASE[0], "y": LONG_PRESS_BASE[1]},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
    }


def main():
    profile_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "profiles", "SM-A736B")
    lut = build(profile_dir)
    out = os.path.join(profile_dir, "color_lut.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(lut, f, ensure_ascii=False, separators=(",", ":"))
    print(f"{lut['sample_count']} entradas ({lut['composition_breakdown']}) -> {out}")
    print("limites da faixa:", lut["selector_bounds"])


if __name__ == "__main__":
    main()
