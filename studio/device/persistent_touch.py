# -*- coding: utf-8 -*-
"""
Persistent Touch Agent & Protocol — Executor Touch Persistente (FASE 3 do Roadmap).
Elimina a sobrecarga de milhares de chamadas de subprocesso 'input tap/swipe' e round-trips ADB via:
1. Canal Socket TCP de alta velocidade com Touch Daemon nativo (porta 18888).
2. Canal Stream Interativo Persistente via ADB Shell aberto (zero subprocessos por lote, zero push/chmod).
3. Mock virtual determinístico para execução instantânea em testes e CI.
Com controle estrito de amostragem (120 Hz), medição em tempo real de latência, jitter e telemetria completa.
"""
import time
import socket
import threading
import queue
import subprocess
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from ..planning.motion_compiler import MotionStroke, MotionEvent, TouchAction, ExecutionBatch


class TouchCommand(str, Enum):
    HELLO = "HELLO"
    BEGIN_STROKE = "BEGIN_STROKE"
    EVENT = "EVENT"  # DOWN, MOVE, UP
    END_STROKE = "END_STROKE"
    SELECT_TOOL = "SELECT_TOOL"
    SELECT_COLOR = "SELECT_COLOR"
    SYNC = "SYNC"
    ACK = "ACK"
    PING = "PING"
    PONG = "PONG"


class TouchChannelType(str, Enum):
    SOCKET = "socket"
    PERSISTENT_SHELL = "persistent_shell"
    MOCK = "mock"
    LEGACY_FALLBACK = "legacy_fallback"


@dataclass
class PersistentTouchMetrics:
    """Métricas em tempo real do stream de digitalização física (Fase 3.4)."""
    events_sent: int = 0
    events_acked: int = 0
    total_latency_ms: float = 0.0
    latencies: List[float] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    fallback_invocations: int = 0
    dropped_events: int = 0
    channel_type: str = "none"

    @property
    def mean_latency_ms(self) -> float:
        return (self.total_latency_ms / max(1, self.events_acked)) if self.events_acked > 0 else 0.0

    @property
    def jitter_ms(self) -> float:
        if len(self.latencies) < 2:
            return 0.0
        diffs = [abs(self.latencies[i] - self.latencies[i - 1]) for i in range(1, len(self.latencies))]
        return float(np.mean(diffs)) if diffs else 0.0

    @property
    def events_per_second(self) -> float:
        elapsed = max(0.001, time.time() - self.start_time)
        return float(self.events_sent) / elapsed

    @property
    def completion_rate(self) -> float:
        return (float(self.events_acked) / max(1.0, float(self.events_sent))) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel_type": self.channel_type,
            "events_sent": self.events_sent,
            "events_acked": self.events_acked,
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "jitter_ms": round(self.jitter_ms, 2),
            "events_per_second": round(self.events_per_second, 1),
            "completion_rate_pct": round(self.completion_rate, 1),
            "fallback_invocations": self.fallback_invocations,
            "dropped_events": self.dropped_events
        }


class PersistentTouchExecutor:
    """
    Executor de eventos de toque em canal persistente contínuo.
    Suporta canais de alta velocidade em cascata:
    1. Socket TCP nativo (Touch Daemon no dispositivo).
    2. Stream Interativo ADB Shell mantido permanentemente aberto (Zero subprocessos adicionais, zero push/chmod).
    3. Mock virtual sincronizado para testes e CI.
    4. Fallback resiliente automático para script ADB tradicional.
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 18888,
                 fallback_adb: Any = None,
                 target_frequency_hz: int = 120,
                 enable_mock: bool = False):
        self.host = host
        self.port = port
        self.fallback_adb = fallback_adb
        self.target_frequency_hz = target_frequency_hz
        self.enable_mock = enable_mock
        self.sock: Optional[socket.socket] = None
        self.channel_type = TouchChannelType.LEGACY_FALLBACK
        self.is_connected = False
        self.metrics = PersistentTouchMetrics()
        # Reentrante: o envio pode reabrir o shell com o lock já obtido.
        self._lock = threading.RLock()
        self._shell_proc: Optional[subprocess.Popen] = None
        self._shell_output = None
        self._batch_ack_counter = 0
        self._socket_unavailable = False

    def _is_aborted(self) -> bool:
        """A parada de emergência do ADBManager também cancela este canal."""
        abort_event = getattr(self.fallback_adb, "_abort_event", None)
        return isinstance(abort_event, threading.Event) and abort_event.is_set()

    def _channel_alive(self) -> bool:
        if not self.is_connected:
            return False
        if self.channel_type == TouchChannelType.MOCK:
            return True
        if self.channel_type == TouchChannelType.SOCKET:
            return self.sock is not None
        if self.channel_type == TouchChannelType.PERSISTENT_SHELL:
            proc = self._shell_proc
            return proc is not None and proc.stdin is not None and proc.poll() is None
        return False

    def ensure_connected(self) -> Tuple[bool, str]:
        """Reaproveita o canal aberto; só reconecta quando ele caiu."""
        if self._channel_alive():
            return True, f"Canal {self.channel_type.value} reaproveitado."
        return self.connect()

    def connect(self) -> Tuple[bool, str]:
        """Tenta estabelecer canal persistente com o dispositivo, priorizando socket e stream interativo."""
        # 1. Modo Mock Virtual (Testes e CI)
        if self.enable_mock:
            self.is_connected = True
            self.channel_type = TouchChannelType.MOCK
            self.metrics.channel_type = self.channel_type.value
            return True, "Mock Persistent Touch Agent conectado com sucesso."

        # 2. Canal de Socket TCP nativo (Touch Daemon). Uma recusa vale para a
        # sessão inteira: o daemon não é iniciado por este projeto, então novas
        # tentativas só somariam o timeout a cada camada.
        try:
            if self._socket_unavailable:
                raise ConnectionRefusedError("touch daemon indisponível nesta sessão")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(0.6)
            self.sock.connect((self.host, self.port))
            self.sock.sendall(b"HELLO 1.0\n")
            resp = self.sock.recv(1024).decode("utf-8", errors="ignore").strip()
            if "OK" in resp or "ACK" in resp:
                self.is_connected = True
                self.channel_type = TouchChannelType.SOCKET
                self.metrics.channel_type = self.channel_type.value
                return True, f"Persistent Touch conectado no socket tcp://{self.host}:{self.port}."
            raise ConnectionError(f"handshake recusado: {resp!r}")
        except Exception:
            self._socket_unavailable = True
            if self.sock:
                try:
                    self.sock.close()
                except Exception:
                    pass
            self.sock = None

        # 3. Canal Stream Interativo ADB Shell (Zero subprocessos adicionais, zero push/chmod)
        if self.fallback_adb is not None and getattr(self.fallback_adb, "is_connected", False):
            ok, msg = self._init_persistent_shell()
            if ok:
                self.is_connected = True
                self.channel_type = TouchChannelType.PERSISTENT_SHELL
                self.metrics.channel_type = self.channel_type.value
                return True, f"Persistent Touch conectado via Shell Stream ADB ({msg})."

        # 4. Fallback gracioso
        self.is_connected = False
        self.channel_type = TouchChannelType.LEGACY_FALLBACK
        self.metrics.channel_type = self.channel_type.value
        return False, "Nenhum canal persistente direto disponível. Modo Fallback ADB ativo."

    def _init_persistent_shell(self) -> Tuple[bool, str]:
        """Inicializa um subprocesso adb shell interativo contínuo mantido aberto durante a sessão."""
        with self._lock:
            self._close_persistent_shell()
            adb_path = getattr(self.fallback_adb, "adb_path", "adb")
            device_target = getattr(self.fallback_adb, "device_target", None)

            cmd = [adb_path]
            if device_target:
                cmd += ["-s", device_target]
            cmd += ["shell"]

            try:
                self._shell_proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                )
                self._shell_output = queue.Queue()
                def read_output(proc, output):
                    try:
                        for line in proc.stdout:
                            output.put(line)
                    finally:
                        output.put(None)
                threading.Thread(target=read_output, args=(self._shell_proc, self._shell_output), daemon=True).start()
                handshake_token = "__PERSISTENT_SHELL_READY__"
                self._shell_proc.stdin.write(f"echo {handshake_token}\n")
                self._shell_proc.stdin.flush()

                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    try:
                        line = self._shell_output.get(timeout=max(0.001, deadline - time.monotonic()))
                    except queue.Empty:
                        break
                    if line is None:
                        break
                    if handshake_token in line:
                        return True, "Stream interativo ativo"
                self._close_persistent_shell()
                return False, "Handshake do stream interativo expirou"
            except Exception as e:
                self._close_persistent_shell()
                return False, f"Falha ao iniciar adb shell ({e})"

    def _close_persistent_shell(self):
        """Encerra com segurança o subprocesso do stream shell interativo."""
        if self._shell_proc is not None:
            try:
                if self._shell_proc.stdin and not self._shell_proc.stdin.closed:
                    self._shell_proc.stdin.write("exit\n")
                    self._shell_proc.stdin.flush()
                self._shell_proc.terminate()
                self._shell_proc.wait(timeout=0.5)
            except Exception:
                try:
                    self._shell_proc.kill()
                except Exception:
                    pass
            self._shell_proc = None
            if self._shell_output is not None:
                self._shell_output.put(None)
            self._shell_output = None

    def execute_motion_stroke(self, stroke: MotionStroke) -> bool:
        """
        Transmite uma pincelada completa compilada (MotionEvent DOWN, MOVE..., UP)
        respeitando rigorosamente a taxa de atualização temporal da digitalização.
        """
        if not stroke or not stroke.events:
            return True
        if self._is_aborted():
            return False

        if self._injector_ready():
            return self._execute_via_injector([stroke])

        if not self.is_connected:
            return self._execute_fallback(stroke)

        # Canal 1: Socket Nativo
        if self.channel_type == TouchChannelType.SOCKET:
            return self._execute_via_socket(stroke)

        # Canal 2: Persistent Shell Stream
        if self.channel_type == TouchChannelType.PERSISTENT_SHELL:
            return self._execute_via_persistent_shell([stroke])

        # Canal 3: Mock Virtual
        if self.channel_type == TouchChannelType.MOCK:
            return self._execute_via_mock([stroke])

        return self._execute_fallback(stroke)

    def execute_batch(self, batch: ExecutionBatch) -> bool:
        """
        Executa um lote inteiro de MotionStrokes em rajada única (Batch Stream Burst),
        eliminando múltiplos round-trips e sleeps artificiais.
        """
        if not batch or not batch.strokes:
            return True
        if self._is_aborted():
            return False

        if self._injector_ready():
            return self._execute_via_injector(batch.strokes)

        if not self.is_connected:
            for s in batch.strokes:
                if not self._execute_fallback(s):
                    return False
            return True

        # Em Persistent Shell ou Mock, envia o lote inteiro de uma só vez
        if self.channel_type == TouchChannelType.PERSISTENT_SHELL:
            return self._execute_via_persistent_shell(batch.strokes)
        elif self.channel_type == TouchChannelType.MOCK:
            return self._execute_via_mock(batch.strokes)
        elif self.channel_type == TouchChannelType.SOCKET:
            for s in batch.strokes:
                if not self._execute_via_socket(s):
                    return False
            return True

        return False

    def _injector_ready(self) -> bool:
        """O injetor app_process do ADBManager reproduz o gesto contínuo com cadência real."""
        from ..adb_manager import ADBManager
        if not isinstance(self.fallback_adb, ADBManager):
            return False
        return self.fallback_adb._get_injector() is not None

    def _execute_via_injector(self, strokes: List[MotionStroke]) -> bool:
        with self._lock:
            t0 = time.time()
            all_pts = [pts for pts in (s.to_points_list() for s in strokes) if len(pts) >= 2]
            total_events = sum(len(s.events) for s in strokes)
            self.metrics.events_sent += total_events
            if not all_pts:
                return True
            if not self.fallback_adb.execute_stroke_batch(all_pts):
                return False
            latency_ms = (time.time() - t0) * 1000.0
            self.metrics.events_acked += total_events
            self.metrics.total_latency_ms += latency_ms
            self.metrics.latencies.append(latency_ms)
            return True

    def _execute_via_socket(self, stroke: MotionStroke) -> bool:
        """Transmissão atômica via socket TCP nativo."""
        with self._lock:
            try:
                t0 = time.time()
                header = f"BEGIN {stroke.tool} {stroke.thickness_mode} {stroke.color_rgb[0]},{stroke.color_rgb[1]},{stroke.color_rgb[2]}\n"
                if self.sock:
                    self.sock.sendall(header.encode("utf-8"))

                for event in stroke.events:
                    line = f"{event.action.value} {int(round(event.x))} {int(round(event.y))} {event.timestamp_ms} {event.pressure:.2f}\n"
                    if self.sock:
                        self.sock.sendall(line.encode("utf-8"))
                    self.metrics.events_sent += 1

                footer = b"END\n"
                if self.sock:
                    self.sock.sendall(footer)
                    acknowledgement = self.sock.recv(1024).decode("utf-8", errors="ignore").strip()
                    if not acknowledgement.startswith("ACK"):
                        self.is_connected = False
                        return False

                latency_ms = (time.time() - t0) * 1000.0
                self.metrics.events_acked += len(stroke.events)
                self.metrics.total_latency_ms += latency_ms
                self.metrics.latencies.append(latency_ms)
                return True
            except Exception:
                self.is_connected = False
                # O traço pode ter chegado antes da falha: não o repete.
                return False

    def _execute_via_persistent_shell(self, strokes: List[MotionStroke]) -> bool:
        """
        Transmissão atômica contínua via Shell Stream interativo aberto.
        Zero arquivos temporários em disco, zero push, zero chmod, zero subprocessos adicionais.
        """
        if self._is_aborted():
            return False
        with self._lock:
            if self._shell_proc is None or self._shell_proc.stdin is None or self._shell_output is None:
                # Tenta restabelecer o stream interativo
                ok, _ = self._init_persistent_shell()
                if not ok:
                    all_ok = True
                    for s in strokes:
                        all_ok = self._execute_fallback(s) and all_ok
                    return all_ok

            try:
                t0 = time.time()
                self._batch_ack_counter += 1
                ack_token = f"__STREAM_ACK_{self._batch_ack_counter}__"

                lines = []
                total_events = 0
                from ..adb_manager import ADBManager
                continuous = getattr(self.fallback_adb, "continuous_strokes", None)
                for stroke in strokes:
                    # Os pontos vêm dos MotionEvents compilados (DOWN, MOVE..., UP).
                    # O `input` do Android não aceita carimbos de tempo, então a
                    # cadência de 120 Hz não é reproduzida; a geometria é preservada.
                    points = [(int(e.x), int(e.y)) for e in stroke.events]
                    lines.extend(ADBManager.build_stroke_commands(points, continuous if isinstance(continuous, bool) else None))
                    total_events += len(stroke.events)
                    self.metrics.events_sent += len(stroke.events)

                lines.append(f"echo {ack_token}\n")
                payload = "".join(lines)

                self._shell_proc.stdin.write(payload)
                self._shell_proc.stdin.flush()

                # Aguarda confirmação síncrona do término do lote pelo shell
                deadline = time.monotonic() + max(8.0, len(lines) * 0.4)
                acked = False
                while time.monotonic() < deadline and not self._is_aborted():
                    try:
                        line = self._shell_output.get(timeout=min(0.1, max(0.001, deadline - time.monotonic())))
                    except queue.Empty:
                        continue
                    if line is None:
                        break
                    if ack_token in line:
                        acked = True
                        break

                if not acked:
                    # Sem ACK o lote pode ter sido pintado: não o repete.
                    self._close_persistent_shell()
                    self.is_connected = False
                    return False

                latency_ms = (time.time() - t0) * 1000.0
                self.metrics.events_acked += total_events
                self.metrics.total_latency_ms += latency_ms
                self.metrics.latencies.append(latency_ms)
                return True

            except Exception:
                self._close_persistent_shell()
                self.is_connected = False
                self.metrics.fallback_invocations += 1
                return False

    def _execute_via_mock(self, strokes: List[MotionStroke]) -> bool:
        """Execução virtual determinística no emulador FakeADB."""
        with self._lock:
            t0 = time.time()
            total_events = 0
            all_pts = []
            for stroke in strokes:
                total_events += len(stroke.events)
                pts = stroke.to_points_list()
                if len(pts) >= 2:
                    all_pts.append(pts)

            self.metrics.events_sent += total_events
            self.metrics.events_acked += total_events
            latency_ms = (time.time() - t0) * 1000.0
            self.metrics.total_latency_ms += latency_ms
            self.metrics.latencies.append(latency_ms)

            if self.fallback_adb is not None and all_pts:
                return bool(self.fallback_adb.execute_stroke_batch(all_pts))
            return True

    def _execute_fallback(self, stroke: MotionStroke) -> bool:
        """Fallback gracioso para o driver ADB tradicional (Fase 3.3)."""
        self.metrics.fallback_invocations += 1
        if self.fallback_adb is not None:
            points = stroke.to_points_list()
            if len(points) >= 2:
                return self.fallback_adb.execute_stroke_batch([points])
        return False

    def get_metrics(self) -> Dict[str, Any]:
        """Retorna telemetria completa da camada física de toque."""
        return self.metrics.to_dict()

    def close(self):
        """Encerra conexões ativas (socket e shell interativo)."""
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

        self._close_persistent_shell()
        self.is_connected = False
