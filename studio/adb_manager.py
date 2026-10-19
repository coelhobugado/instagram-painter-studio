"""
Módulo Resiliente de Gerenciamento e Comunicação ADB (Wi-Fi e USB).
Inclui descoberta automática de dispositivos, tolerância a falhas com retry logic,
verificação de tela/app em primeiro plano, injeção fluída de toques e Kill Switch de hardware.
"""
import os
import shutil
import subprocess
import time
import threading
import queue
import math
import tempfile
import uuid
import logging
import cv2
import numpy as np
from . import config
from .device.profile_loader import DeviceProfileLoader
from .device.safe_executor import ExecutionTokenManager
from .device.ui_timing import UITimingManager
from .device.touch_injector import TouchInjectorClient, resample_stroke
from .device.brush_width import parse_width_level, slider_y_for_width, DEFAULT_SLIDER_CURVE

logger = logging.getLogger(__name__)


class CommandResult(tuple):
    """Resultado de ``run_cmd``: desempacota como ``(ok, stdout, stderr)`` e guarda o código de saída."""
    def __new__(cls, ok, stdout, stderr, returncode=None):
        result = super().__new__(cls, (ok, stdout, stderr))
        result.returncode = returncode
        return result


class ADBManager:
    supports_visual_thickness_check = True
    # Padrões de classe: instâncias criadas sem __init__ (testes) usam o
    # caminho por "input" e nunca abrem uma sessão real do injetor.
    touch_transport = "shell"
    _injector = None
    _injector_failed = False
    _palette_page = None
    def __init__(self, adb_path=None, device_target=None):
        self.adb_path = adb_path or self._detect_adb()
        self.device_target = device_target or config.DEFAULT_DEVICE_IP
        self.is_connected = False
        self.device_info = {"model": "Desconhecido", "resolution": "Desconhecida", "type": "Desconhecido"}
        self.profile_loader = DeviceProfileLoader(config.DEFAULT_DEVICE_PROFILE)
        self.timing = UITimingManager(is_mock=False)
        self.active_token = None
        self.continuous_strokes = config.CONTINUOUS_STROKES
        self.touch_transport = config.TOUCH_TRANSPORT
        self._injector = None
        self._injector_failed = False
        # Página atual da paleta de cores; None = desconhecida (força voltar à 1ª).
        self._palette_page = None
        self._interactive_shell = None
        # Reentrante: get_interactive_shell e o envio fecham o shell com o lock já obtido.
        self._shell_lock = threading.RLock()
        self._shell_counter = 0
        self._shell_output = None
        self._abort_event = threading.Event()

    def _detect_adb(self):
        """Localiza de forma inteligente o executável do adb.exe."""
        if os.path.exists(config.DEFAULT_ADB_PATH):
            return config.DEFAULT_ADB_PATH
        sys_adb = shutil.which("adb")
        if sys_adb:
            return sys_adb
        fallbacks = [
            os.path.expanduser(r"~\AppData\Local\Android\Sdk\platform-tools\adb.exe"),
            r"C:\Android\platform-tools\adb.exe",
            r"C:\platform-tools\adb.exe",
        ]
        for fb in fallbacks:
            if os.path.exists(fb):
                return fb
        return "adb"

    def _ensure_server_running(self):
        """Inicia o daemon do ADB caso não esteja em execução."""
        try:
            subprocess.run(
                [self.adb_path, "start-server"],
                capture_output=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            )
        except Exception:
            pass

    def run_cmd(self, args, timeout=config.ADB_COMMAND_TIMEOUT, ignore_device=False):
        """Executa um comando ADB seguro com timeout e isolamento de processo."""
        cmd = [self.adb_path]
        if not ignore_device and self.device_target:
            cmd += ["-s", self.device_target]
        cmd += args
        
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            )
            stderr = res.stderr.strip()
            if res.returncode != 0 and not stderr:
                stderr = f"adb saiu com código {res.returncode}"
            return CommandResult(res.returncode == 0, res.stdout.strip(), stderr, res.returncode)
        except subprocess.TimeoutExpired:
            return CommandResult(False, "", "Tempo limite esgotado (Timeout no ADB)")
        except Exception as e:
            return CommandResult(False, "", str(e))

    def discover_devices(self):
        """
        Descobre todos os dispositivos conectados via USB ou Wi-Fi.
        Retorna lista de dicionários: [{'id': ..., 'model': ..., 'status': ..., 'is_wifi': ...}]
        """
        ok, out, _ = self.run_cmd(["devices", "-l"], timeout=5, ignore_device=True)
        devices = []
        if not ok or not out:
            return devices
            
        for line in out.splitlines()[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            dev_id = parts[0]
            status = parts[1] if len(parts) > 1 else "unknown"
            
            # Extrai modelo
            model = "Android Device"
            for p in parts[2:]:
                if p.startswith("model:"):
                    model = p.split("model:")[-1]
                elif p.startswith("device:"):
                    if model == "Android Device":
                        model = p.split("device:")[-1]
                        
            is_wifi = ":" in dev_id
            devices.append({
                "id": dev_id,
                "model": model,
                "status": status,
                "is_wifi": is_wifi
            })
        return devices

    def connect(self, target=None):
        """Conecta com detecção inteligente de dispositivos ativos, priorizando dispositivos já disponíveis."""
        if target:
            self.device_target = target.strip()

        # Primeiro verifica se já existe algum dispositivo conectado e autorizado
        devices = self.discover_devices()
        matched = None
        
        # Procura correspondência exata
        for dev in devices:
            if dev["id"] == self.device_target and dev["status"] == "device":
                matched = dev
                break
                
        # Se não encontrou o target específico mas há exatamente 1 dispositivo ativo no ADB
        if not matched and devices:
            active_devs = [d for d in devices if d["status"] == "device"]
            if len(active_devs) == 1:
                matched = active_devs[0]
                self.device_target = matched["id"]

        # Se ainda não encontrou e for endereço de rede (IP:Porta), tenta conectar via Wi-Fi
        if not matched and ":" in self.device_target:
            ok, out, err = self.run_cmd(["connect", self.device_target], timeout=config.ADB_CONNECT_TIMEOUT, ignore_device=True)
            if ok and ("connected to" in out.lower() or "already connected" in out.lower()):
                devices = self.discover_devices()
                for dev in devices:
                    if dev["id"] == self.device_target and dev["status"] == "device":
                        matched = dev
                        break

        if not matched:
            if not devices:
                self.is_connected = False
                return False, f"Nenhum dispositivo encontrado no ADB. Conecte o cabo USB ou ative a Depuração Wi-Fi."
            first = devices[0]
            if first["status"] == "unauthorized":
                self.is_connected = False
                return False, "Dispositivo não autorizado. Aceite o prompt de Depuração USB na tela do celular!"
            self.is_connected = False
            return False, f"Dispositivo '{self.device_target}' não está pronto (Status: {first.get('status')})."

        self.is_connected = True
        # Novo alvo ou reconexão: a sessão anterior do injetor não vale mais.
        self._close_injector()
        self._injector_failed = False
        self._palette_page = None
        self.update_device_info()
        return True, f"Conectado a {self.device_target} ({self.device_info['model']})"

    def auto_reconnect(self):
        """Tenta restabelecer a conexão silenciosamente em caso de queda transitória."""
        for attempt in range(config.ADB_MAX_RECONNECT_RETRIES):
            if self._abort_event.wait(config.ADB_RETRY_BACKOFF_SECONDS * (attempt + 1)):
                return False
            ok, _ = self.connect(self.device_target)
            if ok:
                return True
        return False

    def update_device_info(self):
        """Atualiza metadados do aparelho (modelo, resolução e tipo)."""
        ok, model, _ = self.run_cmd(["shell", "getprop", "ro.product.model"])
        if ok and model:
            self.device_info["model"] = model
            self._select_profile_for_model(model)

        ok, size_out, _ = self.run_cmd(["shell", "wm", "size"])
        if ok and "Physical size:" in size_out:
            res_str = size_out.split("Physical size:")[-1].strip()
            self.device_info["resolution"] = res_str
            try:
                parts = res_str.split("x")
                if len(parts) == 2:
                    w = int(parts[0].strip())
                    h = int(parts[1].strip())
                    self.profile_loader.set_target_resolution(w, h)
            except Exception:
                pass
            
        self.device_info["type"] = "Wi-Fi" if ":" in self.device_target else "USB"

    def _select_profile_for_model(self, model):
        """Usa profiles/<modelo>/ quando existir; caso contrário mantém o perfil atual."""
        model = model.strip()
        if model == self.profile_loader.profile_name:
            return
        if DeviceProfileLoader.profile_exists(model):
            self.profile_loader = DeviceProfileLoader(model)
        else:
            logger.warning("Sem perfil calibrado para %s; usando %s.", model, self.profile_loader.profile_name)

    def ping(self):
        """Mede a latência da conexão."""
        if not self.is_connected:
            return False, "Desconectado"
        t0 = time.time()
        ok, out, err = self.run_cmd(["shell", "echo", "ping"], timeout=3)
        if not ok and self.auto_reconnect():
            t0 = time.time()
            ok, out, err = self.run_cmd(["shell", "echo", "ping"], timeout=3)
            
        elapsed_ms = int((time.time() - t0) * 1000)
        if ok and "ping" in out:
            return True, f"{elapsed_ms} ms"
        return False, f"Falha de resposta: {err or 'Sem resposta'}"

    def is_screen_on(self):
        """Verifica se a tela do celular está ligada (Princípio Fail-Closed: Seção 20 da Auditoria Real)."""
        # O filtro roda no aparelho: só as linhas de estado atravessam o ADB.
        ok, out, _ = self.run_cmd(["shell", "dumpsys power | grep -iE 'mWakefulness=|Display Power: state='"])
        if ok and out:
            out_lower = out.lower()
            # mHoldingDisplaySuspendBlocker continua true durante o desligamento
            # da tela e não prova que ela está acesa; só o estado explícito vale.
            return "display power: state=on" in out_lower or "mwakefulness=awake" in out_lower
        # Fail-closed: se não conseguir confirmar, rejeita a execução por segurança
        return False

    def is_instagram_foreground(self):
        """Verifica se o Instagram está estritamente em primeiro plano e com foco de janela (Fail-Closed)."""
        ok, out, _ = self.run_cmd(["shell", "dumpsys window | grep -E 'mCurrentFocus|mFocusedApp'"])
        if ok and out:
            for line in out.splitlines():
                line_s = line.strip()
                if line_s.startswith("mCurrentFocus") or line_s.startswith("mFocusedApp"):
                    return "com.instagram.android" in line_s
                    
        # Fallback para dumpsys activity
        ok2, out2, _ = self.run_cmd(["shell", "dumpsys activity activities | grep -E 'topResumedActivity|mResumedActivity'"])
        if ok2 and out2:
            for line in out2.splitlines():
                line_s = line.strip()
                if "topResumedActivity" in line_s or "mResumedActivity" in line_s:
                    return "com.instagram.android" in line_s
        return False

    def open_instagram_canvas(self):
        """Navega pelo chat do Instagram para abrir a tela de pintura usando o perfil ativo."""
        plus_btn = self.profile_loader.get_ui_anchor("chat_plus_button", config.UI_CHAT_PLUS_BUTTON)
        draw_btn = self.profile_loader.get_ui_anchor("chat_draw_button", config.UI_CHAT_DRAW_BUTTON)
        self._palette_page = None
        # Clica no '+'
        ok, err = self._tap(*plus_btn)
        if not ok:
            return False, f"Falha ao abrir menu de desenho: {err}"
        self.timing.settle("canvas_open_step")
        # Clica em 'Desenhar'
        ok, err = self._tap(*draw_btn)
        if not ok:
            return False, f"Falha ao abrir tela de desenho: {err}"
        self.timing.settle("canvas_open_step")
        return True, "Tela de desenho acionada com sucesso"

    def select_pen_tool(self, pen="solid"):
        """Seleciona a caneta de desenho a partir das âncoras do perfil."""
        self._require_not_aborted()
        coords = self.profile_loader.get_ui_anchor(f"pen_{pen}", config.UI_PEN_SOLID)
        ok, err = self._tap(*coords)
        if not ok:
            raise RuntimeError(f"Falha ao selecionar caneta: {err}")
        self.timing.settle("tool_select")
        return True

    def set_brush_thickness(self, mode="min"):
        """Ajusta o slider lateral de espessura de traço a partir do perfil do dispositivo."""
        self._require_not_aborted()
        self.last_thickness_note = None
        sl = self.profile_loader.get_slider_thickness()
        sx = sl.get("x", config.UI_SLIDER_X)
        injector = self._get_injector()
        if injector is not None:
            # Medido no A73: tocar no trilho nem sempre move a bolinha, e um
            # arrasto que não começa sobre ela vira traço no canvas. Por isso o
            # arrasto parte da posição da bolinha localizada na tela.
            targets = {"min": sl.get("min_y", config.UI_SLIDER_MIN_Y),
                       "thick": sl.get("thick_y", config.UI_SLIDER_THICK_Y)}
            exact = parse_width_level(mode)
            if exact is not None:
                ty = slider_y_for_width(exact, self.profile_loader.get_width_curve() or DEFAULT_SLIDER_CURVE)
            else:
                ty = int(targets.get(mode, sl.get("medium_y", config.UI_SLIDER_MED_Y)))
            thumb_y = self.detect_slider_thumb_y(self.capture_screenshot(), sx)
            if thumb_y is None:
                raise RuntimeError("Bolinha do slider de espessura não encontrada na tela")
            min_y = sl.get("min_y", config.UI_SLIDER_MIN_Y)
            if ty > min_y + 25:
                return self._set_below_track_thickness(sx, thumb_y, min_y, mode)
            # A bolinha costuma parar até ~20 px além do alvo: confere e corrige.
            for _ in range(2):
                if abs(thumb_y - ty) <= config.UI_SLIDER_TOLERANCE_PX:
                    break
                # Abaixo do trilho visível (posição mais fina) a bolinha só
                # responde se o dedo ficar parado ~300 ms antes de arrastar.
                below_track = thumb_y > sl.get("min_y", config.UI_SLIDER_MIN_Y) + 25
                ok = injector.drag((sx, thumb_y), (sx, ty), frames=16 if below_track else 8,
                                   hold_ms=300 if below_track else 67)
                if not ok:
                    raise RuntimeError(f"Falha ao ajustar espessura: {injector.last_error}")
                self.timing.settle("thickness_slider")
                thumb_y = self.detect_slider_thumb_y(self.capture_screenshot(), sx)
                if thumb_y is None:
                    raise RuntimeError("Bolinha do slider de espessura sumiu após o ajuste")
            if abs(thumb_y - ty) > 2 * config.UI_SLIDER_TOLERANCE_PX:
                raise RuntimeError(f"Espessura '{mode}' não confirmada: bolinha em y={thumb_y}, alvo {ty}")
            return True
        mode = self._nearest_named_level(mode)
        if mode == "min":
            # Calibrado no aparelho: tocar no fim do trilho posiciona a bolinha
            # no mínimo. O swipe anterior saltava para uma espessura grossa.
            ty = sl.get("min_y", config.UI_SLIDER_MIN_Y)
            ok, err = self._tap(sx, ty)
        elif mode == "thick":
            ty = sl.get("thick_y", config.UI_SLIDER_THICK_Y)
            ok, err = self._tap(sx, ty)
        else:
            my = sl.get("medium_y", config.UI_SLIDER_MED_Y)
            ok, err = self._tap(sx, my)
        if not ok:
            raise RuntimeError(f"Falha ao ajustar espessura: {err}")
        self.timing.settle("thickness_slider")
        return True

    @staticmethod
    def _nearest_named_level(mode):
        """Sem o injetor só existem os toques calibrados: usa o nível nomeado mais próximo."""
        exact = parse_width_level(mode)
        if exact is None:
            return mode
        named = {"min": config.BRUSH_WIDTHS["min"], "med": config.BRUSH_WIDTHS["med"],
                 "thick": config.BRUSH_WIDTHS["thick"]}
        return min(named, key=lambda k: abs(named[k] - exact))

    @staticmethod
    def detect_palette_page(screen):
        """Página ativa da paleta (1–3) pelo indicador de três pontos, ou None."""
        if screen is None:
            return None
        h, w = screen.shape[:2]
        values = []
        for x, y in config.UI_PALETTE_PAGE_DOTS:
            if not (6 <= x < w - 6 and 6 <= y < h - 6):
                return None
            roi = screen[y - 6:y + 6, x - 6:x + 6]
            values.append(int(roi.max()))
        ordered = sorted(values)
        # O ponto ativo é branco; os demais, cinza. Sem contraste a leitura é incerta.
        if ordered[-1] < 200 or ordered[-1] - ordered[-2] < 60:
            return None
        return values.index(ordered[-1]) + 1

    def _confirm_palette_page(self, page, reset_swipe, next_swipe):
        for _ in range(3):
            seen = self.detect_palette_page(self.capture_screenshot())
            if seen == page:
                return page
            if seen is None:
                raise RuntimeError("Página da paleta não identificada na tela; cor não selecionada")
            start, end = reset_swipe if seen > page else next_swipe
            for _ in range(abs(seen - page)):
                self._require_not_aborted()
                ok, err = self._swipe(start, end, 350)
                if not ok:
                    raise RuntimeError(f"Falha ao corrigir página da paleta: {err}")
                self.timing.settle("color_palette_swipe")
        raise RuntimeError(f"Paleta não chegou à página {page}; cor não selecionada")

    def _set_below_track_thickness(self, sx, thumb_y, min_y, mode):
        """Leva a caneta sólida à posição mais fina, abaixo do trilho visível (~3 px).

        Medido no SM-A736B (Instagram 451, 2026-10-18): arrastar a bolinha
        nunca passa da base do trilho, mas com a neon no mínimo, ao voltar
        para a sólida a bolinha dela desce sozinha até y≈1572 (4 de 4
        tentativas, partindo de 10 a 40 px). Outras canetas não servem: depois
        da marcador ela vai para ~1328 e depois da seta para ~1515.
        """
        if thumb_y > min_y + 50:
            return True
        self.select_pen_tool("neon")
        self.set_brush_thickness("min")
        self.select_pen_tool("solid")
        self.timing.settle("thickness_slider")
        thumb_y = self.detect_slider_thumb_y(self.capture_screenshot(), sx)
        if thumb_y is None:
            raise RuntimeError(f"Espessura '{mode}' não confirmada: bolinha do slider não encontrada")
        if thumb_y <= min_y + 50:
            # O truque depende de um comportamento do Instagram que pode mudar.
            # Plano B: o mínimo do trilho (~13 px), avisando, em vez de parar a pintura.
            self.set_brush_thickness("min")
            self.last_thickness_note = (
                "Traço fino indisponível nesta versão do Instagram: usando o pincel mínimo do trilho "
                "(linhas mais grossas que na prévia).")
            logger.warning(self.last_thickness_note)
        return True

    @staticmethod
    def detect_slider_thumb_y(screen, x=config.UI_SLIDER_X, y_min=840, y_max=1620):
        """Centro vertical da bolinha branca do slider de espessura, ou None.

        A bolinha é um círculo branco de ~38 px sobre o trilho; o trilho em si
        é uma linha fina e não forma componente desse tamanho.
        """
        if screen is None:
            return None
        h, w = screen.shape[:2]
        x0, x1 = max(0, int(x) - 30), min(w, int(x) + 30)
        roi = screen[max(0, y_min):min(h, y_max), x0:x1]
        if roi.size == 0:
            return None
        white = np.all(roi > 235, axis=2).astype(np.uint8)
        n, _, stats, centroids = cv2.connectedComponentsWithStats(white)
        best = None
        for i in range(1, n):
            cw, ch, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
            if 24 <= cw <= 60 and 24 <= ch <= 60 and area >= 0.5 * cw * ch:
                if best is None or area > best[0]:
                    best = (area, centroids[i][1])
        return None if best is None else int(round(max(0, y_min) + best[1]))

    @staticmethod
    def _slider_changed_at_targets(before, after, x, y_a, y_b):
        """Exige mudança visível nas duas posições do controle lateral."""
        if before is None or after is None or before.shape != after.shape:
            return False
        h, w = before.shape[:2]
        if not (0 <= x < w and 0 <= y_a < h and 0 <= y_b < h):
            return False
        diff = cv2.absdiff(before, after)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        for y in (y_a, y_b):
            roi = gray[max(0, y - 55):min(h, y + 56), max(0, x - 35):min(w, x + 36)]
            if int(np.count_nonzero(roi > 24)) < 20:
                return False
        return True

    def prepare_brush_thickness(self, mode="min"):
        """Força uma transição do slider e verifica que a interface respondeu."""
        if mode not in ("min", "med", "thick") and parse_width_level(mode) is None:
            raise ValueError(f"Espessura inválida: {mode}")
        if self._get_injector() is not None:
            # Com o injetor o ajuste já localiza e confere a bolinha na tela.
            return self.set_brush_thickness(mode)
        mode = self._nearest_named_level(mode)
        sl = self.profile_loader.get_slider_thickness()
        x = int(sl.get("x", config.UI_SLIDER_X))
        positions = {
            "min": int(sl.get("min_y", config.UI_SLIDER_MIN_Y)),
            "med": int(sl.get("medium_y", config.UI_SLIDER_MED_Y)),
            "thick": int(sl.get("thick_y", config.UI_SLIDER_THICK_Y)),
        }
        reference = "thick" if mode != "thick" else "min"
        # A animação do slider pode atrasar um frame da captura. Uma segunda
        # transição confirmada evita reprovar um ajuste que ocorreu de fato.
        for attempt in range(2):
            self.set_brush_thickness(reference)
            before = self.capture_screenshot()
            self.set_brush_thickness(mode)
            after = self.capture_screenshot()
            if self._slider_changed_at_targets(before, after, x, positions[reference], positions[mode]):
                return True
            if attempt == 0:
                self.timing.settle("thickness_slider")
        raise RuntimeError(
            f"Não foi possível confirmar visualmente a espessura '{mode}' após duas tentativas; "
            "desenho bloqueado antes do primeiro traço."
        )

    def select_palette_color(self, touch_info):
        """Aplica a cor desejada via Toque Simples ou Gesto Atômico no espectro com suporte canônico."""
        if not isinstance(touch_info, dict):
            raise TypeError("touch_info deve ser um dicionário")
        self._require_not_aborted()

        page = int(touch_info.get("palette_page", 1))
        if page not in (1, 2, 3):
            raise ValueError(f"Página da paleta inválida: {page}")
        (rsx, rsy), (rex, rey) = self.profile_loader.get_palette_swipe(
            "reset_to_first_page", config.UI_PALETTE_RESET_SWIPE)
        (nsx, nsy), (nex, ney) = self.profile_loader.get_palette_swipe(
            "next_page", config.UI_PALETTE_NEXT_SWIPE)
        # O Instagram conserva a página da paleta entre escolhas. Com a página
        # conhecida, desliza só o necessário; sem ela, volta à primeira com
        # um deslize a mais (sobrar na página 1 é inofensivo).
        current, self._palette_page = self._palette_page, None
        if current is None:
            for _ in range(3):
                self._require_not_aborted()
                ok, err = self._swipe((rsx, rsy), (rex, rey), 350)
                if not ok:
                    raise RuntimeError(f"Falha ao posicionar paleta: {err}")
                self.timing.settle("color_palette_swipe")
            current = 1
        while current > page:
            self._require_not_aborted()
            ok, err = self._swipe((rsx, rsy), (rex, rey), 350)
            if not ok:
                raise RuntimeError(f"Falha ao voltar página da paleta: {err}")
            self.timing.settle("color_palette_swipe")
            current -= 1
        while current < page:
            self._require_not_aborted()
            ok, err = self._swipe((nsx, nsy), (nex, ney), 350)
            if not ok:
                raise RuntimeError(f"Falha ao avançar paleta: {err}")
            self.timing.settle("color_palette_swipe")
            current += 1
        if self._get_injector() is not None:
            # No aparelho real um deslize pode não virar a página (medido no A73):
            # confere o indicador de pontos antes de tocar, ou a cor sai de outra página.
            current = self._confirm_palette_page(page, ((rsx, rsy), (rex, rey)), ((nsx, nsy), (nex, ney)))

        if touch_info.get("is_black") or touch_info.get("name") == "Preto Puro":
            self._require_not_aborted()
            # A amostra mais escura observada nesta versão é #262626.
            # A antiga tentativa de arrastar o branco para cima não funcionou.
            x, y = self.profile_loader.get_palette_base_anchor("dark_swatch", config.UI_PALETTE_DARK_SWATCH)
            if page != 2:
                self._require_not_aborted()
                ok, err = self._swipe((nsx, nsy), (nex, ney), 350)
                if not ok:
                    raise RuntimeError(f"Falha ao buscar cor escura: {err}")
                self.timing.settle("color_palette_swipe")
                current += 1
            ok, err = self._tap(x, y)
            self.timing.settle("color_palette_tap")
        elif touch_info.get("is_long_press"):
            self._require_not_aborted()
            base = touch_info.get("base_touch") or {}
            bx = touch_info.get("base_x", base.get("x"))
            by = touch_info.get("base_y", base.get("y"))
            tx = touch_info.get("target_x", touch_info.get("touch_x", touch_info.get("x")))
            ty = touch_info.get("target_y", touch_info.get("touch_y", touch_info.get("y")))
            if bx is None or by is None or tx is None or ty is None:
                raise KeyError(f"Coordenadas incompletas para toque longo na paleta: {touch_info}")
            bx, by = self.profile_loader.scale_point(bx, by)
            tx, ty = self.profile_loader.scale_point(tx, ty)
            # O seletor de nuances só existe enquanto o dedo segue pressionado
            # sobre a bolinha: DOWN, espera de abertura, arrasto e UP no mesmo gesto.
            injector = self._get_injector()
            if injector is not None:
                ok = injector.drag((bx, by), (tx, ty), frames=8, hold_ms=config.PALETTE_LONG_PRESS_MS)
                err = injector.last_error
            else:
                gesture = (f"input motionevent DOWN {bx} {by}; sleep 0.5; "
                           f"input motionevent MOVE {tx} {ty}; "
                           f"input motionevent UP {tx} {ty}")
                ok, _, err = self.run_cmd(["shell", gesture])
            self.timing.settle("color_palette_swipe")
        else:
            self._require_not_aborted()
            x = touch_info.get("target_x", touch_info.get("touch_x", touch_info.get("x")))
            y = touch_info.get("target_y", touch_info.get("touch_y", touch_info.get("y")))
            if x is None or y is None:
                raise KeyError(f"Coordenadas incompletas para toque simples na paleta: {touch_info}")
            x, y = self.profile_loader.scale_point(x, y)
            ok, err = self._tap(x, y)
            self.timing.settle("color_palette_tap")
        if not ok:
            raise RuntimeError(f"Falha ao selecionar cor no Android: {err}")
        self._palette_page = current
        return True

    @staticmethod
    def _compact_stroke(stroke, tolerance=2.0):
        """Converte um caminho denso em poucos segmentos retos executáveis."""
        points = []
        for x, y in stroke:
            point = (int(round(x)), int(round(y)))
            if not points or point != points[-1]:
                points.append(point)
        if len(points) <= 2:
            return points
        # Cada aresta vira um único swipe Android. Manter pontos intermediários
        # colineares criaria gestos separados e engrossaria a linha.
        simplified = cv2.approxPolyDP(np.asarray(points, dtype=np.int32), tolerance, False)
        vertices = [tuple(int(v) for v in p[0]) for p in simplified]
        return vertices

    @classmethod
    def build_swipe_commands(cls, stroke):
        vertices = cls._compact_stroke(stroke)
        commands = []
        for start, end in zip(vertices, vertices[1:]):
            distance = math.dist(start, end)
            if distance < 2.5:
                continue
            duration = min(180, max(55, round(distance / 2.8)))
            commands.append(f"input swipe {start[0]} {start[1]} {end[0]} {end[1]} {duration}\n")
        return commands

    @classmethod
    def build_motion_commands(cls, stroke):
        """Um único gesto DOWN → MOVE... → UP por traço (Android 11+).

        Diferente de ``build_swipe_commands``, o dedo não levanta entre as
        arestas, então não surgem pontos de junção nem traços engrossados.
        """
        vertices = cls._compact_stroke(stroke)
        if len(vertices) < 2 or all(math.dist(vertices[0], v) < 2.5 for v in vertices[1:]):
            return []
        commands = [f"input motionevent DOWN {vertices[0][0]} {vertices[0][1]}\n"]
        commands += [f"input motionevent MOVE {x} {y}\n" for x, y in vertices[1:]]
        commands.append(f"input motionevent UP {vertices[-1][0]} {vertices[-1][1]}\n")
        return commands

    @classmethod
    def build_stroke_commands(cls, stroke, continuous=None):
        """Gesto contínuo ou um swipe por aresta, conforme ``config.CONTINUOUS_STROKES``."""
        if continuous is None:
            continuous = config.CONTINUOUS_STROKES
        return cls.build_motion_commands(stroke) if continuous else cls.build_swipe_commands(stroke)

    def _execute_remote_script(self, script_content, filename=None):
        """Grava e executa um script shell temporário dentro de /data/local/tmp/ com token de processo."""
        token = ExecutionTokenManager.generate_token()
        self.active_token = token
        if isinstance(script_content, str):
            lines = [l + "\n" if not l.endswith("\n") else l for l in script_content.splitlines()]
        else:
            lines = list(script_content)
        tracked_script = ExecutionTokenManager.build_tracked_script(lines, token)

        # Nome único: execuções simultâneas não sobrescrevem o script uma da outra.
        filename = filename or f"igp_batch_{token}.sh"
        local_path = os.path.join(tempfile.gettempdir(), filename)
        with open(local_path, "w", newline="\n") as f:
            f.write(tracked_script)
            
        remote_path = f"/data/local/tmp/{filename}"
        ok, _, err = self.run_cmd(["push", local_path, remote_path], timeout=6)
        if not ok and not self._abort_event.is_set() and self.auto_reconnect():
            ok, _, err = self.run_cmd(["push", local_path, remote_path], timeout=6)
        if not ok or self._abort_event.is_set():
            self.active_token = None
            return False
        ok, _, err = self.run_cmd(["shell", f"chmod +x {remote_path}"])
        if not ok or self._abort_event.is_set():
            self.active_token = None
            return False
        try:
            ok, out, err = self.run_cmd(["shell", f"sh {remote_path}"], timeout=60)
            return ok and not self._abort_event.is_set()
        finally:
            self.active_token = None
            try:
                os.remove(local_path)
            except OSError:
                pass

    def get_interactive_shell(self):
        """Retorna uma sessão interativa persistente do adb shell para envio de fluxos contínuos sem subprocessos."""
        if self._abort_event.is_set():
            return None
        with self._shell_lock:
            if self._interactive_shell is not None:
                if self._interactive_shell.poll() is None and self._interactive_shell.stdin is not None:
                    return self._interactive_shell
                self.close_interactive_shell()

            cmd = [self.adb_path]
            if self.device_target:
                cmd += ["-s", self.device_target]
            cmd += ["shell"]

            try:
                self._interactive_shell = subprocess.Popen(
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
                threading.Thread(target=read_output, args=(self._interactive_shell, self._shell_output), daemon=True).start()
                handshake_token = "__ADB_SHELL_STREAM_READY__"
                self._interactive_shell.stdin.write(f"echo {handshake_token}\n")
                self._interactive_shell.stdin.flush()
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    try:
                        line = self._shell_output.get(timeout=max(0.001, deadline - time.monotonic()))
                    except queue.Empty:
                        break
                    if line is None:
                        break
                    if handshake_token in line:
                        return self._interactive_shell
                self.close_interactive_shell()
                return None
            except Exception:
                self.close_interactive_shell()
                return None

    def close_interactive_shell(self, lock_timeout=1.0):
        """Encerra a sessão interativa persistente do adb shell de forma limpa.

        Toma ``_shell_lock`` para não trocar o shell sob um envio em curso. Se o
        envio estiver preso (ex.: pipe cheio), mata o processo mesmo sem o lock:
        a escrita bloqueada falha e o remetente libera o lock e limpa o estado.
        """
        locked = self._shell_lock.acquire(timeout=lock_timeout)
        try:
            shell, output = self._interactive_shell, self._shell_output
            if shell is None:
                return
            try:
                if not locked:
                    shell.kill()
                else:
                    if shell.stdin and not shell.stdin.closed:
                        shell.stdin.write("exit\n")
                        shell.stdin.flush()
                    shell.terminate()
                    shell.wait(timeout=0.5)
            except Exception:
                try:
                    shell.kill()
                except Exception:
                    pass
            if output is not None:
                output.put(None)
            if locked:
                self._interactive_shell = None
                self._shell_output = None
        finally:
            if locked:
                self._shell_lock.release()

    def _send_to_interactive_shell(self, lines, timeout_s=10.0) -> bool:
        """Envia comandos diretamente pelo pipe stdin do shell persistente sem subprocessos adicionais."""
        shell = self.get_interactive_shell()
        if shell is None or shell.stdin is None or self._shell_output is None:
            return False

        with self._shell_lock:
            output = self._shell_output
            if output is None or self._interactive_shell is not shell:
                return False
            self._shell_counter += 1
            ack_token = f"__STREAM_BATCH_ACK_{self._shell_counter}__"
            full_payload = "".join(lines) + f"echo {ack_token}\n"
            try:
                if self._abort_event.is_set():
                    return False
                shell.stdin.write(full_payload)
                shell.stdin.flush()

                deadline = time.monotonic() + timeout_s
                while time.monotonic() < deadline and not self._abort_event.is_set():
                    try:
                        line = output.get(timeout=min(0.1, max(0.001, deadline - time.monotonic())))
                    except queue.Empty:
                        continue
                    if line is None:
                        break
                    if ack_token in line:
                        return True
                return False
            except Exception:
                self.close_interactive_shell()
                return False

    def execute_stroke_batch(self, strokes):
        """Executa um micro-lote de traços contínuos com controle de erros."""
        if self._abort_event.is_set():
            return False
        if not strokes:
            return False
            
        injector = self._get_injector()
        if injector is not None:
            return self._execute_strokes_via_injector(injector, strokes)

        continuous = getattr(self, "continuous_strokes", config.CONTINUOUS_STROKES)
        lines = []
        for stroke in strokes:
            if len(stroke) < 2:
                continue
            lines.extend(self.build_stroke_commands(stroke, continuous))

        if not lines:
            return False

        # 1. Tenta envio direto de alta performance pelo Stream Interativo Persistente (Zero subprocessos)
        if self.get_interactive_shell() is not None:
            # Sem ACK o resultado é desconhecido. Repetir o lote poderia
            # engrossar traços já pintados.
            return self._send_to_interactive_shell(lines, timeout_s=max(10.0, len(lines) * 0.4))

        # 2. Fallback resiliente para script remoto caso o stream interativo oscile
        if self._abort_event.is_set():
            return False
        return self._execute_remote_script(lines)

    # ------------------------------------------------------------------
    # Injetor residente (app_process)
    # ------------------------------------------------------------------
    def _get_injector(self):
        """Sessão do injetor pronta, ou None para usar o caminho por ``input``."""
        if (self.touch_transport != "injector" or not self.is_connected
                or self._injector_failed or self._abort_event.is_set()):
            return None
        if self._injector is not None and self._injector.is_alive():
            return self._injector
        injector = TouchInjectorClient(self.adb_path, self.device_target)
        if injector.start():
            self._injector = injector
            return injector
        # Não tenta a cada lote: volta para "input" até a próxima conexão.
        logger.warning("Injetor indisponível, usando input swipe: %s", injector.last_error)
        self._injector_failed = True
        self._injector = None
        return None

    def _close_injector(self):
        injector, self._injector = self._injector, None
        if injector is not None:
            injector.close()

    def _execute_strokes_via_injector(self, injector, strokes):
        """Um gesto contínuo por traço, cadenciado por quadro pelo próprio app."""
        sent = False
        for stroke in strokes:
            if self._abort_event.is_set():
                return False
            if len(stroke) < 2:
                continue
            points = resample_stroke(stroke, config.INJECTOR_MAX_STEP_PX)
            if len(points) < 2 or all(math.dist(points[0], p) < 2.5 for p in points[1:]):
                continue
            if not injector.stroke(points):
                # Sem confirmação o traço pode ter sido pintado: não repete pelo outro caminho.
                logger.warning("Falha no injetor: %s", injector.last_error)
                return False
            sent = True
        return sent

    def _tap(self, x, y):
        """Toque simples pelo injetor quando disponível. Retorna ``(ok, erro)``."""
        self._require_not_aborted()
        injector = self._get_injector()
        if injector is not None:
            return injector.tap(x, y), injector.last_error
        ok, _, err = self.run_cmd(["shell", f"input tap {int(x)} {int(y)}"])
        return ok, err

    def _swipe(self, start, end, duration_ms):
        """Arrasto de interface (paleta, slider). Retorna ``(ok, erro)``."""
        self._require_not_aborted()
        injector = self._get_injector()
        if injector is not None:
            frames = max(2, int(round(duration_ms / 16.7)))
            return injector.drag(start, end, frames=frames), injector.last_error
        ok, _, err = self.run_cmd(["shell", f"input swipe {int(start[0])} {int(start[1])} "
                                            f"{int(end[0])} {int(end[1])} {int(duration_ms)}"])
        return ok, err

    def reset_abort(self):
        """Libera uma nova sessão após uma parada solicitada pelo operador."""
        self._abort_event.clear()

    def _require_not_aborted(self):
        if self._abort_event.is_set():
            raise RuntimeError("Pintura interrompida pelo operador")

    def _capture_screenshot_exec_out(self, timeout=8):
        """Captura com um único subprocesso: o PNG chega pelo stdout, sem arquivo no aparelho."""
        cmd = [self.adb_path]
        if self.device_target:
            cmd += ["-s", self.device_target]
        cmd += ["exec-out", "screencap", "-p"]
        try:
            res = subprocess.run(
                cmd, capture_output=True, timeout=timeout,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            )
        except Exception:
            return None
        if res.returncode != 0 or not res.stdout:
            return None
        image = cv2.imdecode(np.frombuffer(res.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
        return image if image is not None and image.size else None

    def capture_screenshot(self):
        """Captura o frame atual do celular via screencap de forma segura."""
        image = self._capture_screenshot_exec_out()
        if image is not None:
            return image
        # Fallback: grava no aparelho, puxa e remove (três subprocessos).
        image_id = uuid.uuid4().hex
        remote_img = f"/sdcard/ig_painter_live_{image_id}.png"
        local_img = os.path.join(tempfile.gettempdir(), f"ig_painter_live_{image_id}.png")
        
        ok, _, _ = self.run_cmd(["shell", f"screencap -p {remote_img}"], timeout=8)
        if not ok and self.auto_reconnect():
            ok, _, _ = self.run_cmd(["shell", f"screencap -p {remote_img}"], timeout=8)
            
        if not ok:
            return None
            
        try:
            ok, _, _ = self.run_cmd(["pull", remote_img, local_img], timeout=8)
            if ok and os.path.exists(local_img):
                return cv2.imread(local_img)
            return None
        finally:
            self.run_cmd(["shell", f"rm -f {remote_img}"])
            if os.path.exists(local_img):
                os.remove(local_img)

    def emergency_kill(self):
        """Interrompe o envio local e, se houver script remoto, encerra só seu PID."""
        self._abort_event.set()
        self.close_interactive_shell()
        # O injetor termina o traço em curso (o UP chega ao app) e encerra.
        self._close_injector()
        self._palette_page = None
        token = self.active_token
        self.active_token = None
        if token:
            cmd_pid = (f"pid=$(cat /data/local/tmp/{token}.pid 2>/dev/null); "
                       f"if [ -n \"$pid\" ]; then kill -9 \"$pid\" 2>/dev/null; fi; "
                       f"rm -f /data/local/tmp/{token}.pid")
            ok, _, err = self.run_cmd(["shell", cmd_pid], timeout=3)
            return ok, "Script de desenho interrompido." if ok else f"Falha ao encerrar script: {err}"
        return True, "Envio de novos traços interrompido."
