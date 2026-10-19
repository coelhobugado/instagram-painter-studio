"""
Cores reais de traços e áreas de um desenho, agrupadas para trocar de tinta pouco.

Cada troca de cor no Instagram custa de 0,1 a 1,5 s (página da paleta, toque
longo no seletor). Por isso cores visualmente iguais viram uma só tinta.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

RGB = Tuple[int, int, int]


def to_lab(rgb_or_bgr_img: np.ndarray, is_bgr: bool = True) -> np.ndarray:
    img = rgb_or_bgr_img.astype(np.float32) / 255.0
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB if is_bgr else cv2.COLOR_RGB2LAB)


def lab_to_rgb(lab: Sequence[float]) -> RGB:
    px = np.array([[lab]], np.float32)
    rgb = cv2.cvtColor(px, cv2.COLOR_LAB2RGB)[0, 0]
    return tuple(int(round(float(np.clip(v, 0, 1)) * 255)) for v in rgb)


def uniform_color_mask(lab_img: np.ndarray, window: int = 5, max_spread: float = 10.0) -> np.ndarray:
    """Pixels de miolo: a cor varia pouco na vizinhança (fora das transições antisserrilhadas)."""
    spread = np.zeros(lab_img.shape[:2], np.float32)
    kernel = np.ones((window, window), np.uint8)
    for ch in range(3):
        c = np.ascontiguousarray(lab_img[..., ch], dtype=np.float32)
        spread = np.maximum(spread, cv2.dilate(c, kernel) - cv2.erode(c, kernel))
    return spread < max_spread


def labels_from_nearest_pure(labels: np.ndarray, pure: np.ndarray) -> np.ndarray:
    """Cada pixel de transição herda o rótulo do pixel de miolo mais próximo no espaço.

    Pela cor mais parecida, a borda entre cabelo escuro e fundo claro caía num
    cinza intermediário e virava um halo em volta de tudo; pela vizinhança, a
    borda se divide entre as duas cores que ela separa.
    """
    if pure.all() or not pure.any():
        return labels
    src = (~pure).astype(np.uint8)
    _, idx = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    out = labels.copy()
    out[~pure] = labels[pure][idx[~pure] - 1]  # rótulos de pixel numerados em ordem de leitura
    return out


def ink_mask_from_background(img_bgr: np.ndarray, bg_rgb: RGB, gray_mask: np.ndarray,
                             min_delta_e: float = 22.0) -> np.ndarray:
    """Tinta = pixels escuros (Otsu) OU de cor bem diferente do fundo.

    Só o Otsu em cinza perde linhas claras e coloridas (amarelo, rosa) sobre fundo branco.
    """
    lab = to_lab(img_bgr)
    bg_lab = to_lab(np.uint8([[bg_rgb[::-1]]]))[0, 0]
    de = np.linalg.norm(lab - bg_lab, axis=2)
    colored = (de > min_delta_e).astype(np.uint8) * 255
    mask = cv2.bitwise_or(gray_mask, colored)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))


def stroke_colors(strokes: Sequence[Sequence[Tuple[int, int]]], img_bgr: np.ndarray,
                  ink_mask: np.ndarray, radius: int = 3) -> List[Optional[np.ndarray]]:
    """Cor (Lab) de cada traço: mediana dos pixels de tinta sob o traço.

    Usa só o miolo da tinta (erosão de 1 px) quando possível, para que a borda
    suavizada (mistura com o fundo) não clareie a cor.
    """
    h, w = ink_mask.shape[:2]
    lab = to_lab(img_bgr)
    core = cv2.erode(ink_mask, np.ones((3, 3), np.uint8))
    out: List[Optional[np.ndarray]] = []
    for s in strokes:
        if not s:
            out.append(None)
            continue
        pts = np.asarray(s, np.int32).reshape(-1, 2)
        x0, y0 = np.maximum(pts.min(axis=0) - radius - 1, 0)
        x1, y1 = np.minimum(pts.max(axis=0) + radius + 2, (w, h))
        local = np.zeros((y1 - y0, x1 - x0), np.uint8)
        shifted = (pts - (x0, y0)).reshape(-1, 1, 2)
        if len(pts) >= 2:
            cv2.polylines(local, [shifted], False, 255, 2 * radius + 1)
        else:
            cv2.circle(local, tuple(shifted[0, 0]), radius, 255, -1)
        for source in (core, ink_mask):
            sel = (local > 0) & (source[y0:y1, x0:x1] > 0)
            if np.count_nonzero(sel) >= 3:
                out.append(np.median(lab[y0:y1, x0:x1][sel], axis=0))
                break
        else:
            out.append(None)
    return out


def point_colors(stroke, lab_img: np.ndarray, ink_mask: np.ndarray, bg_lab: np.ndarray,
                 radius: int = 3) -> List[np.ndarray]:
    """Cor (Lab) em cada ponto do traço, só com a tinta mais forte por perto.

    A borda suavizada das linhas mistura tinta e fundo; ficar com a metade dos
    pixels mais distantes do fundo evita linhas finas "clareadas" em vários cinzas.
    Pontos sem tinta por perto herdam a cor do vizinho.
    """
    h, w = ink_mask.shape[:2]
    out: List[Optional[np.ndarray]] = []
    for x, y in stroke:
        x, y = int(round(x)), int(round(y))
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        sel = ink_mask[y0:y1, x0:x1] > 0
        if not sel.any():
            out.append(None)
            continue
        pix = lab_img[y0:y1, x0:x1][sel]
        de = np.linalg.norm(pix - bg_lab, axis=1)
        strong = pix[de >= np.median(de)]
        out.append(np.median(strong, axis=0))
    last = next((c for c in out if c is not None), None)
    if last is None:
        return [bg_lab.copy() for _ in out]
    filled = []
    for c in out:
        last = c if c is not None else last
        filled.append(last)
    return filled


def region_color(mask: np.ndarray, img_bgr: np.ndarray, erode_px: int = 2) -> Optional[np.ndarray]:
    inner = cv2.erode(mask, np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)) if erode_px else mask
    sel = (inner if np.count_nonzero(inner) >= 5 else mask) > 0
    if not sel.any():
        return None
    return np.median(to_lab(img_bgr)[sel], axis=0)


def group_colors(colors_lab: Sequence[Optional[np.ndarray]], weights: Sequence[float],
                 merge_delta_e: float = 12.0, max_colors: int = 10) -> Tuple[List[int], List[np.ndarray]]:
    """Agrupa cores parecidas (ΔE76 < ``merge_delta_e``), priorizando as de maior peso.

    Retorna (índice do grupo de cada item, centro Lab de cada grupo). Itens sem
    cor (None) recebem o grupo mais pesado.
    """
    order = sorted((i for i, c in enumerate(colors_lab) if c is not None), key=lambda i: -weights[i])
    centers: List[np.ndarray] = []
    center_w: List[float] = []
    labels = [-1] * len(colors_lab)
    for i in order:
        c = colors_lab[i]
        if centers:
            d = [float(np.linalg.norm(c - k)) for k in centers]
            j = int(np.argmin(d))
            # Cinzas e pretos (pouca saturação) diferem pouco para o olho: limite maior
            neutral = float(np.hypot(c[1], c[2])) < 12 and float(np.hypot(centers[j][1], centers[j][2])) < 12
            limit = merge_delta_e * (2.0 if neutral else 1.0)
            if d[j] < limit or len(centers) >= max_colors:
                wsum = center_w[j] + weights[i]
                centers[j] = (centers[j] * center_w[j] + c * weights[i]) / max(1e-9, wsum)
                center_w[j] = wsum
                labels[i] = j
                continue
        centers.append(np.array(c, np.float32))
        center_w.append(float(weights[i]))
        labels[i] = len(centers) - 1
    if centers:
        heaviest = int(np.argmax(center_w))
        labels = [heaviest if l < 0 else l for l in labels]
    return labels, centers


def enclosed_regions(ink_mask: np.ndarray, img_bgr: np.ndarray, bg_rgb: RGB,
                     min_area: int = 120, min_delta_e_from_bg: float = 10.0,
                     grow_px: int = 3, background_painted: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Áreas entre as linhas que têm cor própria (o "colorir" do livro de colorir).

    Cada área é um componente sem tinta; a que encosta na borda da imagem com a
    cor do fundo é o fundo e fica de fora. A área cresce ``grow_px`` para baixo
    das linhas, assim não sobra fresta entre o preenchimento e o contorno.
    """
    h, w = ink_mask.shape[:2]
    free = cv2.bitwise_not(cv2.dilate(ink_mask, np.ones((3, 3), np.uint8)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    bg_lab = to_lab(np.uint8([[bg_rgb[::-1]]]))[0, 0]
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_px + 1, 2 * grow_px + 1))
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        x, y, bw, bh = stats[i, :4]
        comp = (labels == i).astype(np.uint8) * 255
        color = region_color(comp, img_bgr)
        if color is None:
            continue
        like_bg = float(np.linalg.norm(color - bg_lab)) < min_delta_e_from_bg
        touches_border = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        # Áreas da cor do fundo só precisam de tinta se o fundo não for pintado
        # (ex.: branco dos olhos sobre a tela escura do celular); o fundo em si nunca.
        if like_bg and (background_painted or touches_border):
            continue
        out.append((cv2.bitwise_and(cv2.dilate(comp, grow), cv2.bitwise_or(comp, ink_mask)), color))
    return out
