"""
Injetor de toques residente no aparelho (``app_process`` + ``InputManager.injectInputEvent``).

Medido no SM-A736B (Android 16, Instagram 451):
- ``input swipe`` custa ~300 ms por aresta e levanta o dedo entre arestas;
- ``input motionevent`` em processos separados perde os MOVEs após o primeiro;
- ``sendevent``/escrita em /dev/input é bloqueada pelo SELinux.

O injetor fica aberto numa única sessão ``adb shell`` e recebe um traço por
linha. Cada traço vira um gesto DOWN → MOVE… → UP com o mesmo ``downTime``.
No modo 2 (WAIT_FOR_FINISH) cada evento espera o app consumi-lo, o que
cadencia os pontos ao ritmo de quadros do Instagram (~16,7 ms) sem perdas.
"""
import hashlib
import logging
import math
import os
import queue
import subprocess
import threading
import time
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

JAR_LOCAL_PATH = os.path.join(os.path.dirname(__file__), "injector", "igp-injector.jar")
JAR_REMOTE_PATH = "/data/local/tmp/igp-injector.jar"
MAIN_CLASS = "igp.TouchInjector"

# O Instagram consome no máximo um ponto por quadro e suaviza a curva entre
# eles. Passos maiores achatam curvas; passos menores só gastam tempo.
DEFAULT_MAX_STEP_PX = 24.0
# Em trechos curvos (arestas curtas após a simplificação) o passo cai para
# este valor: com 24 px por quadro as curvas fechadas saíam facetadas no A73.
CURVE_MAX_STEP_PX = 10.0
CURVE_EDGE_PX = 60.0
# Passos curtos no início e no fim evitam a bolinha de toque parado e o
# "gancho" de ponta que aparecem quando o traço começa ou termina rápido.
RAMP_STEPS_PX = (4.0, 8.0, 14.0)


def resample_stroke(points: Sequence[Tuple[float, float]], max_step_px: float = DEFAULT_MAX_STEP_PX,
                    tolerance_px: float = 1.0) -> List[Tuple[int, int]]:
    """Reduz o traço ao mínimo de pontos que o Instagram reproduz fielmente.

    Simplifica arestas colineares (Douglas-Peucker) e então subdivide cada
    aresta para que nenhum quadro avance mais que ``max_step_px``. As pontas
    recebem rampa de passos curtos.
    """
    pts = []
    for x, y in points:
        p = (float(x), float(y))
        if not pts or p != pts[-1]:
            pts.append(p)
    if len(pts) < 2:
        return [(int(round(x)), int(round(y))) for x, y in pts] * 2

    import cv2
    import numpy as np
    closed = len(pts) > 3 and math.dist(pts[0], pts[-1]) < 1.0
    if closed:
        # Douglas-Peucker "aberto" com início = fim degenera e corta o último
        # trecho: o anel não fecharia. Simplifica como polígono e fecha depois.
        ring = np.asarray(pts[:-1], dtype=np.float32).reshape(-1, 1, 2)
        simplified = cv2.approxPolyDP(ring, tolerance_px, True)
        vertices = [(float(p[0][0]), float(p[0][1])) for p in simplified]
        start = min(range(len(vertices)), key=lambda i: math.dist(vertices[i], pts[0]))
        vertices = vertices[start:] + vertices[:start]
        vertices.append(vertices[0])
    else:
        simplified = cv2.approxPolyDP(np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2), tolerance_px, False)
        vertices = [(float(p[0][0]), float(p[0][1])) for p in simplified]
    if len(vertices) < 2:
        vertices = [pts[0], pts[-1]]

    # Distância acumulada ao longo do polígono simplificado.
    seg_len = [math.dist(a, b) for a, b in zip(vertices, vertices[1:])]
    total = sum(seg_len)
    if total < 1e-6:
        return [(int(round(vertices[0][0])), int(round(vertices[0][1])))] * 2

    def point_at(d):
        for (a, b), length in zip(zip(vertices, vertices[1:]), seg_len):
            if d <= length:
                t = 0.0 if length < 1e-9 else d / length
                return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            d -= length
        return vertices[-1]

    # Os vértices são obrigatórios (preservam cantos); entre eles, passos
    # limitados pela rampa nas pontas e por max_step no miolo.
    vertex_d = [0.0]
    for length in seg_len:
        vertex_d.append(vertex_d[-1] + length)

    def edge_limit(d):
        # Aresta onde o ponto está: arestas curtas indicam curva
        for start, length in zip(vertex_d, seg_len):
            if d < start + length or length == seg_len[-1]:
                return max_step_px if length > CURVE_EDGE_PX else min(max_step_px, CURVE_MAX_STEP_PX)
        return max_step_px

    def step_limit(d):
        from_start = d
        from_end = total - d
        limit = edge_limit(d)
        for i, ramp in enumerate(RAMP_STEPS_PX):
            reach = sum(RAMP_STEPS_PX[:i + 1])
            if from_start < reach or from_end < reach:
                limit = min(limit, ramp)
                break
        return max(1.0, limit)

    def next_step(d):
        # O passo também respeita a rampa no ponto de chegada (fim do traço).
        step = step_limit(d)
        for _ in range(len(RAMP_STEPS_PX) + 1):
            step = min(step, step_limit(min(total, d + step)))
        return step

    out_d = [0.0]
    d = 0.0
    for target in vertex_d[1:]:
        while target - d > next_step(d) + 1e-6:
            d += next_step(d)
            out_d.append(d)
        d = target
        out_d.append(d)

    result = []
    for dist in out_d:
        x, y = point_at(dist)
        p = (int(round(x)), int(round(y)))
        if not result or p != result[-1]:
            result.append(p)
    if len(result) == 1:
        result.append(result[0])
    return result


def _file_md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


class TouchInjectorClient:
    """Sessão persistente com o injetor ``app_process`` no aparelho."""

    def __init__(self, adb_path: str, device_target: Optional[str] = None,
                 inject_mode: int = 2, start_timeout_s: float = 8.0):
        self.adb_path = adb_path
        self.device_target = device_target
        self.inject_mode = inject_mode
        self.start_timeout_s = start_timeout_s
        self._proc: Optional[subprocess.Popen] = None
        self._lines: Optional[queue.Queue] = None
        self._lock = threading.RLock()
        self.last_error = ""

    # ------------------------------------------------------------------ infra
    def _adb(self, *args) -> List[str]:
        cmd = [self.adb_path]
        if self.device_target:
            cmd += ["-s", self.device_target]
        return cmd + list(args)

    def _run(self, *args, timeout=15):
        return subprocess.run(self._adb(*args), capture_output=True, text=True, timeout=timeout,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)

    def deploy(self) -> bool:
        """Envia o .jar ao aparelho apenas quando o conteúdo mudou."""
        if not os.path.exists(JAR_LOCAL_PATH):
            self.last_error = f"Injetor ausente: {JAR_LOCAL_PATH}"
            return False
        local_md5 = _file_md5(JAR_LOCAL_PATH)
        try:
            res = self._run("shell", f"md5sum {JAR_REMOTE_PATH} 2>/dev/null")
            if res.returncode == 0 and res.stdout.split()[:1] == [local_md5]:
                return True
            res = self._run("push", JAR_LOCAL_PATH, JAR_REMOTE_PATH, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            self.last_error = f"Falha ao enviar injetor: {exc}"
            return False
        if res.returncode != 0:
            self.last_error = f"Falha ao enviar injetor: {res.stderr.strip() or res.stdout.strip()}"
            return False
        return True

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> bool:
        with self._lock:
            if self.is_alive():
                return True
            if not self.deploy():
                return False
            try:
                self._proc = subprocess.Popen(
                    self._adb("shell", f"CLASSPATH={JAR_REMOTE_PATH} app_process / {MAIN_CLASS}"),
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            except OSError as exc:
                self.last_error = f"Falha ao iniciar injetor: {exc}"
                self._proc = None
                return False
            self._lines = queue.Queue()

            def pump(proc, lines):
                try:
                    for line in proc.stdout:
                        lines.put(line.strip())
                finally:
                    lines.put(None)

            threading.Thread(target=pump, args=(self._proc, self._lines), daemon=True).start()
            if self._read_reply(self.start_timeout_s) != "READY":
                self.last_error = self.last_error or "Injetor não respondeu READY"
                self.close()
                return False
            if not self._command(f"M {int(self.inject_mode)}", 5.0):
                self.close()
                return False
            return True

    def _read_reply(self, timeout_s: float) -> Optional[str]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                line = self._lines.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty:
                break
            if line is None:
                self.last_error = "Sessão do injetor encerrada"
                return None
            if line:
                return line
        self.last_error = "Tempo esgotado aguardando o injetor"
        return None

    def _command(self, line: str, timeout_s: float) -> bool:
        with self._lock:
            if not self.is_alive():
                self.last_error = "Injetor não está em execução"
                return False
            try:
                self._proc.stdin.write(line + "\n")
                self._proc.stdin.flush()
            except (OSError, ValueError) as exc:
                self.last_error = f"Falha ao escrever no injetor: {exc}"
                self.close()
                return False
            reply = self._read_reply(timeout_s)
            if reply is None:
                # Sem resposta o gesto pode ter sido executado: fecha em vez de repetir.
                self.close()
                return False
            if not reply.startswith("OK"):
                self.last_error = reply
                return False
            return True

    def close(self):
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.write("Q\n")
                proc.stdin.flush()
            proc.wait(timeout=1.0)
        except Exception:
            pass
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass

    # ----------------------------------------------------------------- gestos
    @staticmethod
    def _timeout_for(n_points: int, dt_ms: float) -> float:
        # Modo 2 leva ~1 quadro por evento mesmo com dt_ms=0.
        return 5.0 + n_points * max(dt_ms, 17.0) / 1000.0 * 2.0

    def stroke(self, points: Sequence[Tuple[float, float]], dt_ms: float = 0.0) -> bool:
        """Um traço contínuo exatamente pelos pontos dados."""
        pts = [(int(round(x)), int(round(y))) for x, y in points]
        if not pts:
            return True
        if len(pts) == 1:
            pts.append(pts[0])
        payload = " ".join(f"{x} {y}" for x, y in pts)
        return self._command(f"S {dt_ms:g} {payload}", self._timeout_for(len(pts), dt_ms))

    def tap(self, x: float, y: float) -> bool:
        return self._command(f"T {int(round(x))} {int(round(y))}", 5.0)

    def drag(self, start: Tuple[float, float], end: Tuple[float, float], frames: int = 6,
             hold_ms: float = 0.0, settle_frames: int = 3) -> bool:
        """Arrasto paced por quadros, com espera opcional parado no início.

        Serve ao slider de espessura e ao seletor de nuances, que só abre
        enquanto o dedo continua pressionado sobre a bolinha de cor.
        """
        frame_ms = 16.7
        hold = [start] * max(1, int(round(hold_ms / frame_ms)))
        path = [(start[0] + (end[0] - start[0]) * f / frames, start[1] + (end[1] - start[1]) * f / frames)
                for f in range(1, frames + 1)]
        return self.stroke(hold + path + [end] * settle_frames, frame_ms)
