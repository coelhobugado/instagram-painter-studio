# -*- coding: utf-8 -*-
"""
Gera as imagens de exemplo de inputs/ por código (sem direitos de terceiros).

- flores.jpg      line art preto e branco: buquê de tulipas com hachuras (modo Traçado colorido)
- personagem.jpg  cartoon original de cores chapadas, contorno preto e olhos brancos grandes (Livro de colorir)
- retrato.jpg     retrato pintado com cara de foto: pele, cabelo, fundo florido desfocado (Ilustração chapada)

Determinístico: rodar de novo produz os mesmos arquivos.
Uso: python scripts/generate_sample_images.py
"""
import os

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUTS = os.path.join(ROOT, "inputs")
BLACK = (0, 0, 0)


def _bezier(p0, p1, p2, p3, n=40):
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3
    return np.round(pts).astype(np.int32)


def _curve(img, pts, thickness=2):
    p = [np.array(v, dtype=np.float64) for v in pts]
    cv2.polylines(img, [_bezier(*p)], False, BLACK, thickness, cv2.LINE_AA)


def _tulip(img, cx, cy, size, tilt, rng):
    """Cabeça de tulipa: três pétalas em curvas e hachuras internas."""
    rot = np.array([[np.cos(tilt), -np.sin(tilt)], [np.sin(tilt), np.cos(tilt)]])

    def P(x, y):
        v = rot @ np.array([x, y]) * size
        return (cx + v[0], cy + v[1])

    # pétalas externas e central
    _curve(img, [P(-0.55, 0.55), P(-0.85, -0.10), P(-0.55, -0.85), P(-0.10, -0.95)])
    _curve(img, [P(0.55, 0.55), P(0.85, -0.10), P(0.55, -0.85), P(0.10, -0.95)])
    _curve(img, [P(-0.55, 0.55), P(-0.30, 0.80), P(0.30, 0.80), P(0.55, 0.55)])
    _curve(img, [P(-0.10, -0.95), P(-0.35, -0.40), P(-0.25, 0.30), P(0.00, 0.62)])
    _curve(img, [P(0.10, -0.95), P(0.35, -0.40), P(0.25, 0.30), P(0.00, 0.62)])
    _curve(img, [P(-0.30, -0.75), P(-0.05, -1.10), P(0.05, -1.10), P(0.30, -0.75)])
    # hachuras curtas na sombra das pétalas
    for k in range(5):
        y = -0.45 + 0.17 * k + rng.uniform(-0.03, 0.03)
        _curve(img, [P(-0.62, y), P(-0.55, y - 0.08), P(-0.45, y - 0.12), P(-0.38, y - 0.16)], 1)
        _curve(img, [P(0.40, y + 0.10), P(0.48, y + 0.05), P(0.56, y), P(0.63, y - 0.05)], 1)
    return P(0.0, 0.70)


def make_flores(path, size=612, seed=7):
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 255, np.uint8)
    base = np.array([size * 0.50, size * 0.96])
    heads = [(0.28, 0.33, 0.085, -0.35), (0.44, 0.18, 0.090, -0.10), (0.62, 0.25, 0.085, 0.20),
             (0.36, 0.48, 0.080, -0.25), (0.70, 0.45, 0.090, 0.35), (0.53, 0.40, 0.080, 0.05)]
    # Caules e folhas saem de pontos espalhados da base: convergindo num ponto
    # só, as linhas formariam uma mancha preta sólida (vira área, não linha).
    for i, (fx, fy, fs, tilt) in enumerate(heads):
        bottom = _tulip(img, fx * size, fy * size, fs * size, tilt, rng)
        # caule duplo até a base do buquê
        top = np.array(bottom)
        foot = base + [(i - 2.5) * 26, -14 * (i % 2)]
        mid = (top + foot) / 2 + np.array([rng.uniform(-25, 25), 0])
        for off in (-3, 3):
            _curve(img, [top + [off, 0], mid + [off, -40], mid + [off * 0.6, 40], foot + [off, 0]])
    # folhas longas com nervura e hachuras: (início x, ponta x, ponta y, meia largura), em fração
    leaves = [(0.34, 0.10, 0.60, 0.045), (0.40, 0.20, 0.47, 0.035),
              (0.66, 0.90, 0.58, 0.045), (0.60, 0.80, 0.46, 0.035)]
    for sx, tx, ty, hw in leaves:
        start = np.array([sx * size, 0.95 * size])
        tip = np.array([tx * size, ty * size])
        axis = tip - start
        n = np.array([-axis[1], axis[0]]) / (np.linalg.norm(axis) + 1e-9) * hw * size
        for side in (-1, 1):
            _curve(img, [start, start + axis * 0.35 + n * side * 1.6, start + axis * 0.75 + n * side, tip])
        _curve(img, [start, start + axis * 0.35 + n * 0.2, start + axis * 0.75 + n * 0.1, tip], 1)
        for k in range(2, 7):
            t = k / 8.0
            p = start + axis * t + n * 0.15
            _curve(img, [p, p + n * 0.3 + axis * 0.02, p + n * 0.6 + axis * 0.04, p + n * 0.8 + axis * 0.05], 1)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])


def make_personagem(path, size=500):
    """Mascote original: capacete vermelho, olhos brancos grandes, traje azul, estrela amarela."""
    img = np.full((size, size, 3), 255, np.uint8)
    red, blue, yellow = (40, 30, 220), (150, 55, 45), (40, 200, 250)
    o = 7  # contorno
    # pernas e botas
    for x in (195, 305):
        cv2.rectangle(img, (x - 38, 360), (x + 38, 455), BLACK, -1)
        cv2.rectangle(img, (x - 38 + o, 360), (x + 38 - o, 410), blue, -1)
        cv2.rectangle(img, (x - 38 + o, 410), (x + 38 - o, 455 - o), red, -1)
    # corpo
    cv2.ellipse(img, (250, 320), (115, 85), 0, 0, 360, BLACK, -1)
    cv2.ellipse(img, (250, 320), (115 - o, 85 - o), 0, 0, 360, blue, -1)
    # braços
    for sx in (-1, 1):
        cv2.ellipse(img, (250 + sx * 140, 300), (45, 70), sx * 30, 0, 360, BLACK, -1)
        cv2.ellipse(img, (250 + sx * 140, 300), (45 - o, 70 - o), sx * 30, 0, 360, red, -1)
    # estrela no peito
    star = []
    for k in range(10):
        r = 38 if k % 2 == 0 else 16
        a = -np.pi / 2 + k * np.pi / 5
        star.append((250 + r * np.cos(a), 330 + r * np.sin(a)))
    star = np.round(star).astype(np.int32)
    cv2.fillPoly(img, [star], yellow)
    cv2.polylines(img, [star], True, BLACK, 4, cv2.LINE_AA)
    # cabeça (capacete) e olhos
    cv2.circle(img, (250, 165), 135, BLACK, -1)
    cv2.circle(img, (250, 165), 135 - o, red, -1)
    for sx in (-1, 1):
        center = (250 + sx * 58, 175)
        cv2.ellipse(img, center, (46, 62), sx * -25, 0, 360, BLACK, -1)
        cv2.ellipse(img, center, (37, 53), sx * -25, 0, 360, (255, 255, 255), -1)
    # antena com bolinha
    cv2.line(img, (250, 32), (250, 5), BLACK, 6, cv2.LINE_AA)
    cv2.circle(img, (250, 10), 12, BLACK, -1)
    cv2.circle(img, (250, 10), 7, yellow, -1)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])


def make_retrato(path, w=476, h=645, seed=11):
    """Retrato pintado com aparência de foto: gradientes, desfoque e grão."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32)
    # fundo bege com luz lateral e flores rosadas desfocadas no topo
    img[:] = (205, 220, 238)
    img += ((xx / w) * 25)[..., None]
    flowers = np.zeros_like(img)
    for _ in range(40):
        x, y = rng.uniform(0, w), rng.uniform(0, h * 0.22)
        r = rng.uniform(10, 26)
        cv2.circle(flowers, (int(x), int(y)), int(r), (150, 40, 190), -1)
    for _ in range(25):
        x, y = rng.uniform(0, w), rng.uniform(0, h * 0.25)
        cv2.ellipse(flowers, (int(x), int(y)), (int(rng.uniform(15, 40)), 4), rng.uniform(0, 180), 0, 360,
                    (60, 110, 70), -1)
    fmask = (flowers.sum(axis=2) > 0).astype(np.float32)
    fmask = cv2.GaussianBlur(fmask, (0, 0), 3)[..., None]
    img = img * (1 - fmask) + cv2.GaussianBlur(flowers, (0, 0), 3) * fmask
    # cabelo escuro atrás do rosto
    hair = np.zeros((h, w), np.uint8)
    cv2.ellipse(hair, (238, 250), (150, 190), 0, 0, 360, 255, -1)
    cv2.rectangle(hair, (95, 250), (385, 470), 255, -1)
    # roupa: paletó preto e gola branca
    body = np.zeros((h, w), np.uint8)
    cv2.ellipse(body, (238, 700), (260, 260), 0, 180, 360, 255, -1)
    collar = np.zeros((h, w), np.uint8)
    cv2.fillPoly(collar, [np.array([(180, 445), (296, 445), (330, 560), (238, 645), (146, 560)], np.int32)], 255)
    # pescoço e rosto (pele)
    skin = np.zeros((h, w), np.uint8)
    cv2.rectangle(skin, (200, 330), (276, 460), 255, -1)
    cv2.ellipse(skin, (238, 265), (95, 125), 0, 0, 360, 255, -1)

    def paint(mask, color, blur=2.5):
        m = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), blur)[..., None]
        return img * (1 - m) + np.array(color, np.float32) * m

    img = paint(hair, (30, 32, 38))
    img = paint(body, (35, 32, 30))
    img = paint(collar, (238, 240, 242))
    # pele com sombreado: mais clara no centro e à esquerda
    shade = 1.0 - 0.18 * np.clip(((xx - 215) ** 2 / 95 ** 2 + (yy - 250) ** 2 / 140 ** 2), 0, 1.5)
    skin_color = np.stack([150 * shade, 178 * shade, 228 * shade], axis=-1)
    m = cv2.GaussianBlur(skin.astype(np.float32) / 255.0, (0, 0), 3)[..., None]
    img = img * (1 - m) + skin_color * m
    # franja e mechas sobre a testa
    fringe = np.zeros((h, w), np.uint8)
    cv2.ellipse(fringe, (238, 175), (110, 70), 0, 180, 360, 255, -1)
    cv2.ellipse(fringe, (150, 280), (30, 120), 10, 0, 360, 255, -1)
    cv2.ellipse(fringe, (326, 280), (30, 120), -10, 0, 360, 255, -1)
    img = paint(fringe, (32, 34, 40))
    # sobrancelhas, olhos, nariz e boca
    feat = img.copy()
    for sx in (-1, 1):
        cx = 238 + sx * 40
        cv2.ellipse(feat, (cx, 228), (24, 6), 0, 180, 360, (45, 50, 60), -1)
        cv2.ellipse(feat, (cx, 252), (20, 9), 0, 0, 360, (235, 235, 238), -1)
        cv2.circle(feat, (cx, 252), 7, (150, 140, 110), -1)
        cv2.circle(feat, (cx, 252), 3, (20, 20, 25), -1)
        cv2.ellipse(feat, (cx, 252), (20, 9), 0, 180, 360, (40, 40, 50), 2)
    cv2.ellipse(feat, (238, 292), (10, 6), 0, 0, 180, (110, 135, 190), 2)
    cv2.ellipse(feat, (238, 330), (28, 10), 0, 0, 360, (120, 120, 200), -1)
    cv2.line(feat, (212, 330), (264, 330), (90, 90, 160), 2)
    img = cv2.GaussianBlur(feat, (0, 0), 1.0)
    # broche vermelho no paletó
    cv2.circle(img, (330, 560), 11, (40, 60, 220), -1)
    # grão de câmera
    img += rng.normal(0.0, 4.0, img.shape).astype(np.float32)
    cv2.imwrite(path, np.clip(img, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    os.makedirs(INPUTS, exist_ok=True)
    make_flores(os.path.join(INPUTS, "flores.jpg"))
    make_personagem(os.path.join(INPUTS, "personagem.jpg"))
    make_retrato(os.path.join(INPUTS, "retrato.jpg"))
    print("Imagens geradas em", INPUTS)


if __name__ == "__main__":
    main()
