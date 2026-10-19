"""
Interface Gráfica do Usuário (GUI v3 Enterprise) para o Instagram Painter Studio.
Construída com CustomTkinter com Dark Mode, modo AUTO inteligente,
inspetor de paleta e camadas, mapa de calor de erro perceptual, checkpoints em disco e Kill Switch.
"""
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk

from . import config
from .adb_manager import ADBManager
from .painter_engine import PainterEngine
from .execution_worker import ExecutionWorker
from .planning.auto_mode import AutoModeDecisor
from .feedback.error_map import ErrorMapEngine
from .device.safe_executor import PersistentCheckpointManager
from .benchmarks.benchmark_suite import BenchmarkSuite
from .storage.editable_project import save_project, load_project
from .gui_services import GuiProjectState, DeviceService, PlanningService
from .gui.theme import COLORS, apply_global_theme, apply_dark_title_bar, variant_colors, font
from .gui.layout import StudioLayoutMixin, PREVIEW_BUTTON_TEXT
from .gui.widgets import brand_logo

STYLE_NAMES = {"LINHAS": "Traçado colorido", "COLORIR": "Livro de colorir",
               "CAMADAS": "Ilustração chapada", "STENCIL": "Contornos"}

PRESET_COLORS = {
    "🖤 Preto": (38, 38, 38),
    "🖤 Preto disponível": (38, 38, 38),
    "🤍 Branco": (255, 255, 255),
    "🤍 Branco Puro": (255, 255, 255),
    "❤️ Vermelho": (230, 35, 40),
    "💛 Amarelo": (255, 225, 30),
    "💙 Azul": (40, 130, 240),
    "💚 Verde": (40, 180, 70),
    "🌸 Rosa": (245, 120, 170),
    "🧡 Laranja": (245, 130, 30),
    "💜 Roxo": (145, 60, 200),
    "🟫 Marrom": (120, 70, 45)
}

apply_global_theme()


class InstagramPainterStudioGUI(StudioLayoutMixin, ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Painter Studio")
        self.geometry("1440x860")
        self.minsize(1080, 640)
        self.after_idle(self._maximize_window)
        self.bind("<F11>", self._toggle_fullscreen)
        self.bind("<Escape>", self._exit_fullscreen)
        self.bind("<Configure>", self._on_window_resize, add="+")

        # Motores e Serviços Desacoplados (P2.1, P2.2)
        self.adb = ADBManager()
        self.engine = PainterEngine()
        self.gui_state = GuiProjectState()
        self.device_service = DeviceService(self.adb)
        self.planning_service = PlanningService(self.engine)
        self._is_log_collapsed = True
        self.current_plan = None
        self._undo_plans = []
        self._redo_plans = []
        self._selected_region_id = None
        self._selected_region_ids = set()
        self._selection_notice = None
        self._selection_start = None
        self._selection_drag = None
        self._editing_plan = False
        self._generating_preview = False
        self._preview_display_size = None
        self._resize_after_id = None
        self._last_main_size = None
        self._plan_revision = 0
        self._plan_dirty = False
        self._fullscreen = False
        self.worker = None
        self.selected_image_path = None
        self.latest_screencap_raw = None
        self.lbl_canvas_live = None
        self.checkpoint = {"layer": 0, "stroke": 0}
        self._last_progress_ui_ts = 0.0
        self._paint_started_at = None

        # Personalização Manual de Cores
        self.custom_line_ink_rgb = None
        self.custom_line_bg_rgb = None
        self.custom_stencil_ink_rgb = None

        # Estado da Ferramenta de Seleção de Área Interativa (ROI)
        self.custom_canvas_bounds = None
        self.roi_drag_start = None
        self.roi_temp_rect_id = None
        self.roi_active_rect_id = None
        self.roi_handles_ids = []
        self.roi_text_id = None
        self.screencap_scale_info = None
        self.screencap_photo_img = None

        # Layout principal (studio/gui/layout.py)
        self._build_layout()
        self._update_device_controls()
        self._refresh_inspector()
        self._refresh_progress_indicators()
        self._set_window_icon()
        self.after_idle(lambda: apply_dark_title_bar(self))

        # Carrega imagem padrão inicial se disponível
        default_img = os.path.join(os.path.dirname(os.path.dirname(__file__)), "inputs", "retrato.jpg")
        if os.path.exists(default_img):
            self.load_image_file(default_img)

        # Inicia descoberta silenciosa de dispositivos
        self.after(400, self.action_scan_devices)
        # Verifica checkpoint salvo da sessão anterior
        self.after(800, self._check_existing_checkpoint)

    def _set_window_icon(self):
        try:
            self._icon_photo = ImageTk.PhotoImage(brand_logo(64), master=self)
            self.wm_iconphoto(True, self._icon_photo)
        except Exception:
            pass

    def _maximize_window(self):
        """Abre quase maximizada, dentro da área visível da tela."""
        screen_w, screen_h = self.winfo_screenwidth(), self.winfo_screenheight()
        width = min(1600, max(1080, screen_w - 60))
        height = min(940, max(640, screen_h - 90))
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 2 - 18)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _toggle_fullscreen(self, _event=None):
        self._fullscreen = not self._fullscreen
        self.attributes("-fullscreen", self._fullscreen)

    def _exit_fullscreen(self, _event=None):
        if self._fullscreen:
            self._fullscreen = False
            self.attributes("-fullscreen", False)

    def _on_window_resize(self, event):
        if event.widget is not self or not hasattr(self, "connection_pill") or getattr(self, "_closing", False):
            return
        try:
            self._update_responsive_layout(event.width)
        except tk.TclError:
            pass  # janela em fechamento: o cabeçalho já não existe

    def _is_text_mode(self):
        return hasattr(self, "seg_mode") and self.seg_mode.get() == "TEXTO"

    def _get_text_input(self):
        if not hasattr(self, "txt_text_input"):
            return ""
        return self.txt_text_input.get("1.0", "end").strip("\n")

    def _has_source(self):
        """Há o que desenhar: uma imagem, ou um texto digitado no modo Texto."""
        if self._is_text_mode():
            return bool(self._get_text_input().strip())
        return bool(self.selected_image_path)

    def _is_painting(self):
        """Pintura em andamento. O fim é marcado pelo aviso do executor, não pela thread:
        o aviso chega com a thread ainda viva por alguns milissegundos."""
        return bool(self.worker and self.worker.is_alive() and not getattr(self, "_worker_finished", True))

    def safe_after(self, ms, func):
        """Executa callback agendado no Tkinter apenas se a janela estiver ativa."""
        try:
            if not getattr(self, "_closing", False) and self.winfo_exists():
                return self.after(ms, func)
        except Exception:
            pass
        return None

    def destroy(self):
        """Cancela callbacks pendentes antes de destruir a janela Tk."""
        self._closing = True
        try:
            # Cancela só o agendamento. ``after_cancel`` também apagaria o comando
            # Tcl, que pertence ao widget que agendou (ex.: CTkTextbox); ao ser
            # destruído ele tentaria apagá-lo de novo ("can't delete Tcl command")
            # e o fechamento pararia no meio.
            for callback_id in self.tk.splitlist(self.tk.call("after", "info")):
                self.tk.call("after", "cancel", callback_id)
        except tk.TclError:
            pass
        try:
            super().destroy()
        except tk.TclError:
            # Algum widget interno falhou ao se destruir: remove o que sobrou da
            # janela direto no Tcl para o fechamento nunca ficar pela metade.
            try:
                self.tk.call("destroy", self._w)
            except tk.TclError:
                pass
        finally:
            # Se o fechamento falhar no meio, o Tk continuaria apontando para esta
            # janela morta e as imagens da próxima janela seriam criadas nela.
            if getattr(tk, "_default_root", None) is self:
                tk._default_root = None

    # --------------------------------------------------------------------------
    # LOGS E IMAGENS
    # --------------------------------------------------------------------------
    def log(self, message):
        t_str = time.strftime("%H:%M:%S")
        formatted = f"[{t_str}] {message}\n"
        self.log_box.insert("end", formatted)
        self.log_box.see("end")

    def _toggle_log_panel(self):
        """Alterna a visibilidade do log para maximizar a área de trabalho do Canvas (P2.6)."""
        if not self._is_log_collapsed:
            self.log_box.grid_remove()
            self.btn_toggle_log.configure(text="▲ Expandir Log")
            self.main_frame.grid_rowconfigure(1, weight=0, minsize=32)
            self._is_log_collapsed = True
        else:
            self.log_box.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
            self.btn_toggle_log.configure(text="▼ Recolher Log")
            self.main_frame.grid_rowconfigure(1, weight=0, minsize=140)
            self._is_log_collapsed = False
        self._refresh_preview_size()

    def _clear_log(self):
        """Limpa o conteúdo do painel de logs."""
        self.log_box.delete("1.0", "end")

    def _display_cv_image(self, target_label, cv_bgr_img):
        if cv_bgr_img is None:
            return
        rgb_img = cv2.cvtColor(cv_bgr_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        # O tamanho real da etiqueta já desconta barras e controles da aba.
        max_w = max(220, target_label.winfo_width() - 20)
        max_h = max(160, target_label.winfo_height() - 16)
        if target_label.winfo_width() <= 10 or target_label.winfo_height() <= 10:
            max_w = max(220, self.main_frame.winfo_width() - 55)
            max_h = max(160, self.tabview.winfo_height() - 170)
        orig_w, orig_h = pil_img.size
        ratio = min(max_w / orig_w, max_h / orig_h)
        display_w = int(orig_w * ratio)
        display_h = int(orig_h * ratio)

        try:
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(display_w, display_h))
            target_label.configure(image=ctk_img, text="")
            target_label.image = ctk_img
            if target_label is self.lbl_canvas_preview:
                # A imagem é ampliada pelo fator de escala do Windows antes
                # de chegar ao Tk; cliques e winfo_* usam pixels físicos.
                self._preview_display_size = ctk_img._get_scaled_size(
                    target_label._get_widget_scaling()
                )
        except tk.TclError:
            pass

    def _on_main_resize(self, event):
        size = (event.width, event.height)
        if size == self._last_main_size:
            return
        self._last_main_size = size
        if self._resize_after_id is not None:
            self.after_cancel(self._resize_after_id)
        self._resize_after_id = self.after(180, self._refresh_preview_size)

    def _refresh_preview_size(self):
        self._resize_after_id = None
        if getattr(self, "_closing", False):
            return
        if self.current_plan:
            self._render_selected_preview()
            comp = self.current_plan.get("comparison_image")
            if comp is not None:
                self._display_cv_image(self.lbl_canvas_compare, comp)
            heatmap = self.current_plan.get("error_heatmap")
            if heatmap is not None:
                self._display_cv_image(self.lbl_canvas_error, heatmap)
        elif self.selected_image_path:
            path = self.selected_image_path

            def worker():
                image = cv2.imread(path)
                if image is not None:
                    self.safe_after(0, lambda i=image: self._display_cv_image(self.lbl_canvas_preview, i)
                                    if self.selected_image_path == path and not self.current_plan else None)

            threading.Thread(target=worker, daemon=True).start()

    # --------------------------------------------------------------------------
    # CHECKPOINT PERSISTENTE
    # --------------------------------------------------------------------------
    def _check_existing_checkpoint(self):
        cp = PersistentCheckpointManager.load_checkpoint()
        if cp:
            self.saved_checkpoint = cp
            self.log(f"📌 Checkpoint salvo detectado em disco (Plano: {cp.get('plan_hash')}, Camada {cp.get('layer_index', 0)+1}).")

    def _validate_and_offer_resume(self):
        """Validação estrita de Checkpoint contra plano atual e dispositivo (P1.4 / Seção 22 da Auditoria Real)."""
        if not getattr(self, "saved_checkpoint", None) or not self.current_plan:
            return
        cp_hash = self.saved_checkpoint.get("plan_hash")
        curr_hash = self.current_plan.get("plan_hash")
        if not cp_hash or cp_hash != curr_hash:
            self.checkpoint = {"layer": 0, "stroke": 0}
            return

        # Validação do dispositivo vinculado (P1.4)
        cp_dev = self.saved_checkpoint.get("device_id")
        curr_dev = getattr(self.adb, "device_target", None)
        if cp_dev and curr_dev and cp_dev != curr_dev:
            self.log(f"⚠️ Checkpoint salvo pertence ao dispositivo '{cp_dev}', mas o dispositivo conectado atual é '{curr_dev}'. Retomada automática bloqueada.")
            messagebox.showwarning(
                "Dispositivo Divergente",
                f"O progresso salvo em disco pertence a outro dispositivo:\n\n"
                f"• Dispositivo do Checkpoint: {cp_dev}\n"
                f"• Dispositivo Atual: {curr_dev}\n\n"
                f"Para evitar toques fora de posição ou com calibração divergente, a pintura iniciará do início."
            )
            self.checkpoint = {"layer": 0, "stroke": 0}
            return

        l_idx = self.saved_checkpoint.get("layer_index", 0)
        s_idx = self.saved_checkpoint.get("stroke_index", 0)
        if l_idx == 0 and s_idx == 0:
            self.checkpoint = {"layer": 0, "stroke": 0}
            return
        tot_strokes = self.current_plan.get("total_strokes", 0)
        tot_layers = len(self.current_plan.get("layers", []))

        ans = messagebox.askyesno(
            "Checkpoint Válido Detectado",
            f"Este desenho possui progresso salvo em disco:\n\n"
            f"• Dispositivo: {cp_dev or 'Padrão'}\n"
            f"• Camada: {l_idx + 1} de {tot_layers}\n"
            f"• Traço registrado: {s_idx} (Total do plano: {tot_strokes})\n"
            f"• Data: {self.saved_checkpoint.get('timestamp')}\n\n"
            f"Deseja usar este ponto de retomada? Você também poderá escolher começar do zero ao iniciar."
        )
        if ans:
            self.checkpoint = {"layer": l_idx, "stroke": s_idx}
            self.log(f"Checkpoint validado e vinculado: Camada {l_idx + 1}, Traço {s_idx} (Dispositivo: {cp_dev or curr_dev}).")
        else:
            self.checkpoint = {"layer": 0, "stroke": 0}

    # --------------------------------------------------------------------------
    # DESCOBERTA DE DISPOSITIVOS E EVENTOS DE UI
    # --------------------------------------------------------------------------
    def action_scan_devices(self):
        def worker():
            try:
                devs = self.adb.discover_devices()
            except Exception:
                devs = []
            self.safe_after(0, lambda d=devs: self._update_device_dropdown(d))
        threading.Thread(target=worker, daemon=True).start()

    def _update_device_dropdown(self, devices):
        if not devices:
            self.opt_devices.configure(values=["Nenhum dispositivo detectado"])
            self.opt_devices.set("Nenhum dispositivo detectado")
            return

        values = []
        for d in devices:
            mode_tag = "Wi-Fi" if d["is_wifi"] else "USB"
            entry_txt = f"{d['id']} ({d['model']} - {mode_tag})"
            values.append(entry_txt)

        self.opt_devices.configure(values=values)
        self.opt_devices.set(values[0])
        self._on_device_selected(values[0])

    def _on_device_selected(self, chosen):
        if "Nenhum" in chosen or "Procurando" in chosen:
            return
        dev_id = chosen.split()[0]
        self.entry_ip.delete(0, "end")
        self.entry_ip.insert(0, dev_id)

    def _select_mode(self, value):
        self.seg_mode.set(value)
        self._on_mode_change(value)

    def _on_mode_change(self, val):
        if self._is_painting():
            messagebox.showwarning("Pintura em Execução", "Não é permitido alterar o modo de pintura durante a execução física (P1.8).")
            return
        for key in self.mode_buttons:
            self._set_mode_tile_selected(key, key == val)
        if val != "TEXTO":
            self.frame_text_opts.pack_forget()
        if val in ("LINHAS", "COLORIR"):
            self.chk_skip_background.pack(fill="x", pady=(6, 0), before=self._style_options_end)
            self.frame_layered_opts.pack_forget()
            self.frame_line_opts.pack(fill="x", pady=(0, 4), before=self._style_options_end)
            self.lbl_line_bg.pack(anchor="w", pady=(2, 1))
            self.opt_line_bg_color.pack(fill="x", pady=(0, 6))
            self.lbl_auto_desc.configure(text=(
                "Traçado colorido: cada linha com a cor e a espessura dela no original. Para desenhos, "
                "rabiscos e letras." if val == "LINHAS" else
                "Livro de colorir: as linhas com suas cores e as áreas entre elas pintadas com a cor de cada "
                "uma. Para desenhos coloridos, personagens e logos."))
        elif val == "TEXTO":
            self.chk_skip_background.pack_forget()
            self.frame_layered_opts.pack_forget()
            self.frame_text_opts.pack(fill="x", pady=(0, 4), before=self._style_options_end)
            self.frame_line_opts.pack(fill="x", pady=(0, 4), before=self._style_options_end)
            self.lbl_line_bg.pack_forget()
            self.opt_line_bg_color.pack_forget()
            self.lbl_auto_desc.configure(text=(
                "Texto: escreve o que você digitar, no maior tamanho que couber na área de pintura, com quebra "
                "de linha automática. Não usa a imagem da seção 1. Traço fino escreve como caneta (~4 px); "
                "\"Seguir a espessura da fonte\" reproduz o peso da letra."))
        elif val == "CAMADAS":
            self.chk_skip_background.pack_forget()
            self.frame_line_opts.pack_forget()
            self.frame_layered_opts.pack(fill="x", pady=(0, 4), before=self._style_options_end)
            self.lbl_auto_desc.configure(text="Ilustração chapada: a imagem vira poucas cores planas, com os detalhes "
                                              "escuros (olhos, boca) por cima. Para fotos e ilustrações.")
        else:
            self.chk_skip_background.pack(fill="x", pady=(6, 0), before=self._style_options_end)
            self.frame_layered_opts.pack_forget()
            self.frame_line_opts.pack_forget()
            self.lbl_auto_desc.configure(text="Automático: desenho de linhas → Traçado colorido; desenho colorido ou "
                                              "logo → Livro de colorir; foto → Ilustração chapada.")
        self._mark_plan_dirty()

    def _set_input_controls_locked(self, locked: bool):
        """Bloqueia/desbloqueia controles de entrada para impedir alterações divergentes durante a execução física (P1.8)."""
        state = "disabled" if locked else "normal"
        if hasattr(self, "btn_load_file"):
            self.btn_load_file.configure(state=state)
        for b in getattr(self, "mode_buttons", {}).values():
            b.configure(state=state)
        if hasattr(self, "btn_preview"):
            self.btn_preview.configure(state=state)
        # Bloqueia/desbloqueia controles de ajustes auxiliares defensivamente
        for attr in [
            "slider_colors", "slider_spacing", "chk_fidelity", "chk_artistic",
            "chk_sim_opt", "chk_verify_layer", "chk_include_contours", "chk_semantic",
            "opt_line_ink_color", "opt_line_bg_color", "chk_skip_background",
            "txt_text_input", "opt_text_font", "opt_text_align", "opt_text_stroke"
        ]:
            widget = getattr(self, attr, None)
            if widget is not None and hasattr(widget, "configure"):
                try:
                    widget.configure(state=state)
                except Exception:
                    pass

    def _mark_plan_dirty(self):
        """Impede executar, exportar ou salvar um plano desatualizado em relação a imagem, modo ou ajustes (P1.7, P2.1)."""
        self._plan_revision += 1
        if hasattr(self, "gui_state"):
            self.gui_state.mark_dirty()
        if self.current_plan is None:
            # Sem prévia ainda: o botão do rodapé passa a "Gerar prévia" quando há o que desenhar
            if hasattr(self, "btn_start"):
                self._update_start_availability()
            return
        self._plan_dirty = True
        self.btn_start.configure(state="disabled")
        self._update_start_availability()
        self.lbl_plan_summary.configure(text="Ajustes alterados. Gere a prévia novamente antes de pintar ou exportar.")
        self._refresh_edit_controls()
        if hasattr(self, "lbl_preview_mode"):
            self.lbl_preview_mode.configure(text="Prévia anterior • ajustes alterados — gere novamente",
                                            text_color=COLORS["warning"])
        if hasattr(self, "stat_tiles"):
            self._refresh_inspector()
        if hasattr(self, "lbl_error_metrics"):
            self.lbl_error_metrics.configure(text="Métricas Closed-Loop: Ajustes alterados — gere nova prévia.")

    def _can_start_current_plan(self):
        if not self.current_plan or self._plan_dirty or self._generating_preview or self._editing_plan:
            return False
        return (bool(getattr(self.current_plan.get("quality_gate"), "passed", False))
                and bool(getattr(self.adb, "is_connected", False)))

    def _update_start_availability(self):
        if hasattr(self, "sections"):
            self._refresh_progress_indicators()
        if not hasattr(self, "btn_start") or self._is_painting():
            return
        # O botão diz o que falta; a barra de status explica onde resolver.
        if not self.current_plan:
            label = "Gere a prévia primeiro"
            if self._is_text_mode():
                hint = ("Digite o texto (seção 2)." if not self._has_source()
                        else "Gere a prévia do texto (seção 2).")
            else:
                hint = ("Escolha uma imagem (seção 1)." if not self.selected_image_path
                        else "Escolha o estilo e gere a prévia (seção 2).")
        elif self._plan_dirty:
            label = "Atualizar prévia"
            hint = "Os ajustes mudaram: gere a prévia de novo (seção 2)."
        elif not getattr(self.current_plan.get("quality_gate"), "passed", False):
            label = "Prévia bloqueada"
            hint = "A prévia não passou na verificação. Veja a aba Qualidade."
        elif not getattr(self.adb, "is_connected", False):
            label = "Conecte o celular"
            hint = "Prévia pronta. Conecte o celular (seção 3)."
        else:
            label = "Iniciar pintura"
            hint = "Tudo pronto. Abra a tela Desenhar no Instagram e inicie."
        if hasattr(self.btn_start, "set_variant"):
            self.btn_start.set_variant("success" if label == "Iniciar pintura" else "primary")
        # Gerar a prévia e conectar também são ações válidas do botão; só a
        # pintura exige prévia aprovada e celular conectado (_can_start_current_plan).
        actionable = (self._can_start_current_plan()
                      or (label in ("Gere a prévia primeiro", "Atualizar prévia") and self._has_source()
                          and not self._generating_preview and not self._editing_plan)
                      or label == "Conecte o celular")
        self.btn_start.configure(text=label, state="normal" if actionable else "disabled")
        if hasattr(self, "lbl_status") and not self._generating_preview:
            self.lbl_status.configure(text=hint)

    def _update_device_controls(self):
        connected = bool(getattr(self.adb, "is_connected", False))
        for name in ("btn_screencap", "btn_test_center"):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.configure(state="normal" if connected else "disabled")
        has_capture = connected and self.latest_screencap_raw is not None
        for name in ("btn_nudge_up", "btn_nudge_down", "btn_nudge_left", "btn_nudge_right",
                     "btn_scale_plus", "btn_scale_minus", "chk_overlay"):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.configure(state="normal" if has_capture else "disabled")

    def _on_line_ink_color_change(self, chosen):
        if "Customizar" in chosen:
            init_c = "#000000"
            color_tuple, chosen_hex = colorchooser.askcolor(color=init_c, title="Escolha a Cor da Caneta / Traço")
            if chosen_hex and color_tuple:
                r, g, b = int(round(color_tuple[0])), int(round(color_tuple[1])), int(round(color_tuple[2]))
                self.custom_line_ink_rgb = (r, g, b)
                self.opt_line_ink_color.set(f"🎨 Custom ({chosen_hex})")
                self.log(f"Cor de traço personalizada: {chosen_hex} (RGB: {r},{g},{b})")
            else:
                self.opt_line_ink_color.set("🎨 Original (Auto)")
        else:
            self.custom_line_ink_rgb = None
        self._mark_plan_dirty()

    def _on_line_bg_color_change(self, chosen):
        if "Customizar" in chosen:
            init_c = "#FFFFFF"
            color_tuple, chosen_hex = colorchooser.askcolor(color=init_c, title="Escolha a Cor da Camada de Fundo")
            if chosen_hex and color_tuple:
                r, g, b = int(round(color_tuple[0])), int(round(color_tuple[1])), int(round(color_tuple[2]))
                self.custom_line_bg_rgb = (r, g, b)
                self.opt_line_bg_color.set(f"🎨 Custom ({chosen_hex})")
                self.log(f"Cor de fundo personalizada: {chosen_hex} (RGB: {r},{g},{b})")
            else:
                self.opt_line_bg_color.set("🎨 Fundo Original (Auto)")
        else:
            self.custom_line_bg_rgb = None
        self._mark_plan_dirty()

    def _get_line_color_overrides(self):
        ink_override = None
        bg_override = None
        paint_bg = None

        if hasattr(self, "opt_line_ink_color"):
            chosen_ink = self.opt_line_ink_color.get()
            if "Original" not in chosen_ink:
                if self.custom_line_ink_rgb is not None and "Custom" in chosen_ink:
                    ink_override = self.custom_line_ink_rgb
                elif chosen_ink in PRESET_COLORS:
                    ink_override = PRESET_COLORS[chosen_ink]

        if hasattr(self, "opt_line_bg_color"):
            chosen_bg = self.opt_line_bg_color.get()
            if "Não Pintar" in chosen_bg or "🚫" in chosen_bg:
                paint_bg = False
            elif "Original" not in chosen_bg:
                if self.custom_line_bg_rgb is not None and "Custom" in chosen_bg:
                    bg_override = self.custom_line_bg_rgb
                    paint_bg = True
                elif chosen_bg.replace("Fundo ", "") in PRESET_COLORS:
                    bg_override = PRESET_COLORS[chosen_bg.replace("Fundo ", "")]
                    paint_bg = True

        if hasattr(self, "var_skip_background") and self.var_skip_background.get():
            paint_bg = False

        return ink_override, bg_override, paint_bg

    def _on_colors_slider(self, val):
        self.lbl_colors.configure(text=f"Paleta de cores: {int(val)} cores")
        self._mark_plan_dirty()

    def _on_spacing_slider(self, val):
        sp = int(val)
        desc = "Alta Fidelidade" if sp <= 4 else ("Equilibrado" if sp <= 6 else "Rápido")
        self.lbl_spacing.configure(text=f"Espaçamento de Pincel: {sp}px ({desc})")
        self._mark_plan_dirty()

    def _on_inspect_layer_slide(self, val):
        if not self.current_plan:
            return
        cum_imgs = self.current_plan.get("cumulative_layer_images", [])
        if not cum_imgs:
            return
        layer_idx = max(0, min(len(cum_imgs) - 1, int(round(val)) - 1))
        self.lbl_layer_slider.configure(text=f"Inspecionar Camadas: 1 até {layer_idx + 1} de {len(cum_imgs)}")
        self._display_cv_image(self.lbl_canvas_preview, cum_imgs[layer_idx])

    # --------------------------------------------------------------------------
    # CARREGAMENTO DE IMAGEM & DEMOS
    # --------------------------------------------------------------------------
    def action_choose_image(self):
        file_types = [("Imagens", "*.jpg *.jpeg *.png *.webp *.bmp"), ("Todos os arquivos", "*.*")]
        path = filedialog.askopenfilename(title="Selecione a Imagem de Referência", filetypes=file_types)
        if path:
            self.load_image_file(path)

    def load_demo(self, filename):
        demo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "inputs", filename)
        if os.path.exists(demo_path):
            self.load_image_file(demo_path)
        else:
            messagebox.showwarning("Aviso", f"Arquivo não encontrado: {filename}")

    def load_image_file(self, path):
        if self._is_painting():
            messagebox.showwarning("Pintura em Execução", "Não é permitido alterar a imagem de referência enquanto a pintura física estiver em andamento (P1.8).")
            return
        base_name = os.path.basename(path)
        img = cv2.imread(path)
        if img is not None:
            self.selected_image_path = path
            self._mark_plan_dirty()
            h, w = img.shape[:2]
            self.lbl_image_info.configure(text=f"{base_name}\n{w} × {h} px")
            self._set_image_thumbnail(img)
            self.log(f"Imagem carregada: {base_name} ({w}x{h})")
            self._display_cv_image(self.lbl_canvas_preview, img)
            self._update_start_availability()
            self.lbl_plan_summary.configure(text="Nova imagem carregada. Gere a prévia para ver os traços.")
        else:
            messagebox.showerror("Imagem inválida", f"Não foi possível abrir: {base_name}")

    def _set_image_thumbnail(self, img):
        if not hasattr(self, "lbl_image_thumb"):
            return
        h, w = img.shape[:2]
        side = min(h, w)
        crop = img[(h - side) // 2:(h - side) // 2 + side, (w - side) // 2:(w - side) // 2 + side]
        thumb = Image.fromarray(cv2.cvtColor(cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA),
                                             cv2.COLOR_BGR2RGB))
        self._thumb_image = ctk.CTkImage(light_image=thumb, dark_image=thumb, size=(64, 64))
        self.lbl_image_thumb.configure(image=self._thumb_image, text="")

    # --------------------------------------------------------------------------
    # AÇÕES ADB (CONEXÃO, PING, ABERTURA DE TELA)
    # --------------------------------------------------------------------------
    def action_connect_adb(self):
        target = self.entry_ip.get().strip()
        self.log(f"Tentando conectar ao ADB em: {target}...")
        self.btn_connect.configure(state="disabled")

        def worker():
            try:
                ok, msg = self.adb.connect(target)
            except Exception as exc:  # sem isto o botão Conectar ficaria desativado para sempre
                ok, msg = False, f"Falha inesperada ao conectar: {exc}"
            self.safe_after(0, lambda o=ok, m=msg: self._post_connect(o, m))

        threading.Thread(target=worker, daemon=True).start()

    def _post_connect(self, ok, msg):
        self.btn_connect.configure(state="normal")
        if ok:
            from .gui.settings import save_setting
            save_setting("last_device_target", self.adb.device_target)
            self.log(f"Conexão OK: {msg}")
            self.toast(f"Celular conectado · {self.adb.device_info['model']}", "ok")
        else:
            self.log(f"Erro ao conectar: {msg}")
            messagebox.showerror("Erro de Conexão", msg)
        self._update_start_availability()
        self._update_device_controls()
        if ok and self.current_plan and not self._plan_dirty:
            self.lbl_progress_text.configure(text="Prévia pronta • iniciar quando desejar")

    def action_ping_adb(self):
        def worker():
            try:
                ok, res = self.adb.ping()
            except Exception as exc:
                ok, res = False, str(exc)
            self.safe_after(0, lambda o=ok, r=res: self._post_ping(o, r))

        threading.Thread(target=worker, daemon=True).start()

    def _post_ping(self, ok, res):
        if ok:
            self.log(f"Ping OK: {res}")
            self.toast(f"Celular respondeu em {res}", "ok")
        else:
            self.log(f"Falha no ping: {res}")
            messagebox.showwarning("Ping ADB", f"Não responsivo: {res}")

    def action_open_canvas(self):
        msg = (
            "📱 Como abrir a tela de pintura no Instagram:\n\n"
            "1. No celular, abra o Instagram e entre em qualquer conversa do Direct.\n"
            "2. Toque no ícone de Câmera ou '+' -> 'Desenhar' (Doodle).\n"
            "3. Confirme que a tela preta com as canetas no topo e cores embaixo está aberta.\n\n"
            "Deseja que o robô tente enviar a sequência automática de toques para abrir agora?"
        )
        ans = messagebox.askyesno("Abrir Tela de Desenho", msg)
        if not ans:
            return
        self.log("Tentando sequência automática para abrir tela de pintura...")
        def worker():
            try:
                ok, msg = self.adb.open_instagram_canvas()
            except Exception as exc:  # thread de fundo: sem isto a falha some em silêncio
                ok, msg = False, f"Falha ao abrir a tela de desenho: {exc}"
            self.safe_after(0, lambda m=msg: self.log(m))
            if not ok:
                self.safe_after(0, lambda m=msg: self.toast(m, "error", 6000))
                return
            time.sleep(1.0)
            # Widgets só podem ser mexidos pela thread da interface
            self.safe_after(0, self.action_refresh_screencap)
        threading.Thread(target=worker, daemon=True).start()

    def action_emergency_kill(self):
        self.log("🚨 BOTÃO DE EMERGÊNCIA (KILL SWITCH) ACIONADO!")
        if self._is_painting():
            self.worker.stop()
        else:
            def stop_idle():
                try:
                    self.adb.emergency_kill()
                except Exception:
                    pass
                finally:
                    # Nada estava pintando: o freio não pode bloquear as próximas ações
                    if hasattr(self.adb, "reset_abort"):
                        self.adb.reset_abort()
                    self.safe_after(0, lambda: self.btn_start.configure(
                        state="normal" if self._can_start_current_plan() else "disabled"))
            threading.Thread(target=stop_idle, daemon=True).start()
        self.btn_start.configure(state="disabled")
        self.btn_pause.configure(state="disabled")
        self.btn_resume.configure(state="disabled")
        self.lbl_status.configure(text="Interrompendo pintura...")

    def _switch_to_roi_tab(self):
        """Alterna diretamente para a aba de Seleção de Área (ROI)."""
        self.tabview.set("Tela do celular")
        if self.latest_screencap_raw is None and self.adb.is_connected:
            self.action_refresh_screencap()

    def _regenerate_after_area_change(self):
        """A área entra no plano: com uma prévia já gerada, refaz para a área nova."""
        if self.current_plan is not None and self._has_source() and not self._is_painting():
            self.toast("Área alterada · atualizando a prévia", "info")
            self.safe_after(150, self.action_generate_preview)

    def _reset_roi_selection(self):
        """Reseta para a área de pintura padrão central do Instagram."""
        self.custom_canvas_bounds = None
        self._mark_plan_dirty()
        self.lbl_roi_sidebar_status.configure(
            text="Área: Central Padrão (750x950 px)",
            text_color=COLORS["info"]
        )
        self.lbl_roi_coords.configure(
            text="↺ Área de pintura redefinida para o padrão central do Instagram (750x950 px).",
            text_color=COLORS["info"]
        )
        if self.screencap_scale_info and self.latest_screencap_raw is not None:
            self._display_screencap_canvas(self.latest_screencap_raw)
        self.log("Área de pintura redefinida para o padrão central do Instagram.")
        self._regenerate_after_area_change()

    def _test_roi_center_tap(self):
        """Envia um toque de teste no ponto central da área selecionada."""
        if not self.adb.is_connected:
            messagebox.showwarning("ADB Desconectado", "Conecte o celular antes de testar.")
            return
        if self.custom_canvas_bounds:
            cx = self.custom_canvas_bounds["center_x"]
            cy = self.custom_canvas_bounds["center_y"]
        else:
            cx = config.CANVAS_SAFE_CX
            cy = config.CANVAS_SAFE_CY
            if hasattr(self.adb, "profile_loader") and self.adb.profile_loader:
                cx, cy = self.adb.profile_loader.scale_point(cx, cy)

        def worker():
            try:
                ok, _out, err = self.adb.run_cmd(["shell", f"input tap {cx} {cy}"])
            except Exception as exc:
                ok, err = False, str(exc)
            if ok:
                self.safe_after(0, lambda: self._post_test_tap(cx, cy))
            else:
                self.safe_after(0, lambda e=err: self.toast(f"Toque de teste falhou: {e}", "error", 6000))

        threading.Thread(target=worker, daemon=True).start()

    def _post_test_tap(self, cx, cy):
        self.log(f"Toque de teste físico executado no centro: ({cx}, {cy})")
        messagebox.showinfo("Toque de Teste", f"Toque enviado no ponto central da tela: ({cx}, {cy})\nVerifique a reação no seu celular!")

    def _on_toggle_overlay(self):
        if self.latest_screencap_raw is not None:
            self._display_screencap_canvas(self.latest_screencap_raw)

    def action_refresh_screencap(self):
        self.log("Capturando screenshot da tela do celular via ADB...")
        self.btn_screencap.configure(state="disabled", text="Capturando…")
        def worker():
            try:
                img = self.adb.capture_screenshot()
            except Exception:
                img = None
            def update_ui():
                self.btn_screencap.configure(state="normal" if self.adb.is_connected else "disabled",
                                             text="Tirar print do celular")
                if img is not None:
                    self._display_screencap_canvas(img)
                    self._update_device_controls()
                    self.log("Screenshot recebido com sucesso e renderizado no Canvas interativo.")
                else:
                    self.log("Falha ao capturar screenshot do celular via ADB.")
                    messagebox.showwarning("Erro Screencap", "Não foi possível capturar a tela. Verifique se a tela está ligada e o ADB conectado.")
            self.safe_after(0, update_ui)
        threading.Thread(target=worker, daemon=True).start()

    def _display_screencap_canvas(self, cv_bgr_img):
        """Renderiza a imagem do celular no Canvas interativo com cálculo de proporção e limites."""
        if cv_bgr_img is None:
            return
        self.latest_screencap_raw = cv_bgr_img
        rgb = cv2.cvtColor(cv_bgr_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        orig_w, orig_h = pil_img.size

        # Dimensões do Canvas
        canv_w = self.canvas_roi.winfo_width()
        canv_h = self.canvas_roi.winfo_height()
        if canv_w < 100 or canv_h < 100:
            canv_w, canv_h = 840, 560

        max_w = max(50, canv_w - 24)
        max_h = max(50, canv_h - 24)
        ratio = min(max_w / orig_w, max_h / orig_h)
        disp_w = max(1, int(orig_w * ratio))
        disp_h = max(1, int(orig_h * ratio))

        resized_pil = pil_img.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        self.screencap_photo_img = ImageTk.PhotoImage(resized_pil)

        off_x = (canv_w - disp_w) // 2
        off_y = (canv_h - disp_h) // 2

        self.screencap_scale_info = {
            "orig_w": orig_w,
            "orig_h": orig_h,
            "disp_w": disp_w,
            "disp_h": disp_h,
            "off_x": off_x,
            "off_y": off_y,
            "scale_x": orig_w / float(disp_w),
            "scale_y": orig_h / float(disp_h)
        }

        self.canvas_roi.delete("all")
        self.canvas_roi.create_image(off_x, off_y, anchor="nw", image=self.screencap_photo_img)

        # Se houver custom_canvas_bounds salvo, desenha por cima
        if self.custom_canvas_bounds:
            self._draw_roi_rect_on_canvas(self.custom_canvas_bounds)

        # Se a opção de marcadores de calibração estiver ativa, desenha os botões do IG
        if self.var_show_overlay.get():
            self._draw_overlay_markers_on_canvas()

    def _on_roi_mouse_press(self, event):
        self.canvas_roi.focus_set()
        if self._is_painting():
            messagebox.showwarning("Pintura em Execução", "Não é permitido redefinir a área de pintura enquanto o desenho estiver em andamento (P1.8).")
            return
        if not self.screencap_scale_info:
            return
        info = self.screencap_scale_info
        if not (info["off_x"] <= event.x <= info["off_x"] + info["disp_w"] and
                info["off_y"] <= event.y <= info["off_y"] + info["disp_h"]):
            return
        self.roi_drag_start = (event.x, event.y)
        if self.roi_temp_rect_id:
            self.canvas_roi.delete(self.roi_temp_rect_id)
            self.roi_temp_rect_id = None

    def _on_roi_mouse_motion(self, event):
        if not self.roi_drag_start or not self.screencap_scale_info:
            return
        info = self.screencap_scale_info
        sx, sy = self.roi_drag_start
        cx = max(info["off_x"], min(event.x, info["off_x"] + info["disp_w"]))
        cy = max(info["off_y"], min(event.y, info["off_y"] + info["disp_h"]))

        if self.roi_temp_rect_id:
            self.canvas_roi.coords(self.roi_temp_rect_id, sx, sy, cx, cy)
        else:
            self.roi_temp_rect_id = self.canvas_roi.create_rectangle(
                sx, sy, cx, cy, outline="#00F0FF", width=2, dash=(4, 4)
            )

        w_real = int(abs(cx - sx) * info["scale_x"])
        h_real = int(abs(cy - sy) * info["scale_y"])
        self.lbl_roi_coords.configure(
            text=f"📐 Selecionando área: {w_real} × {h_real} px (Solte o clique para confirmar)...",
            text_color=COLORS["warning"]
        )

    def _on_roi_mouse_release(self, event):
        if not self.roi_drag_start or not self.screencap_scale_info:
            return
        info = self.screencap_scale_info
        sx, sy = self.roi_drag_start
        cx = max(info["off_x"], min(event.x, info["off_x"] + info["disp_w"]))
        cy = max(info["off_y"], min(event.y, info["off_y"] + info["disp_h"]))
        self.roi_drag_start = None

        x1_disp = min(sx, cx) - info["off_x"]
        x2_disp = max(sx, cx) - info["off_x"]
        y1_disp = min(sy, cy) - info["off_y"]
        y2_disp = max(sy, cy) - info["off_y"]

        real_left = int(round(x1_disp * info["scale_x"]))
        real_right = int(round(x2_disp * info["scale_x"]))
        real_top = int(round(y1_disp * info["scale_y"]))
        real_bottom = int(round(y2_disp * info["scale_y"]))

        w_real = real_right - real_left
        h_real = real_bottom - real_top

        if w_real < 50 or h_real < 50:
            if self.roi_temp_rect_id:
                self.canvas_roi.delete(self.roi_temp_rect_id)
                self.roi_temp_rect_id = None
            return

        self.custom_canvas_bounds = {
            "left": real_left,
            "top": real_top,
            "right": real_right,
            "bottom": real_bottom,
            "center_x": int((real_left + real_right) / 2),
            "center_y": int((real_top + real_bottom) / 2),
            "max_width": w_real,
            "max_height": h_real
        }
        self._mark_plan_dirty()

        self._draw_roi_rect_on_canvas(self.custom_canvas_bounds)
        msg = f"✅ Área de Pintura Definida: {w_real}x{h_real} px (Centro: {self.custom_canvas_bounds['center_x']}, {self.custom_canvas_bounds['center_y']})"
        self.lbl_roi_coords.configure(text=msg, text_color=COLORS["success"])
        self.lbl_roi_sidebar_status.configure(
            text=f"Área Customizada: {w_real}x{h_real} px",
            text_color=COLORS["success"]
        )
        self.log(f"Área de pintura customizada gravada com sucesso: {w_real}x{h_real} px.")
        self._regenerate_after_area_change()

    def _draw_roi_rect_on_canvas(self, bounds):
        """Desenha a caixa delimitadora da área de pintura no Canvas com cantos destacados."""
        if not self.screencap_scale_info:
            return
        info = self.screencap_scale_info
        x1 = int(bounds["left"] / info["scale_x"]) + info["off_x"]
        x2 = int(bounds["right"] / info["scale_x"]) + info["off_x"]
        y1 = int(bounds["top"] / info["scale_y"]) + info["off_y"]
        y2 = int(bounds["bottom"] / info["scale_y"]) + info["off_y"]

        if self.roi_temp_rect_id:
            self.canvas_roi.delete(self.roi_temp_rect_id)
            self.roi_temp_rect_id = None

        if self.roi_active_rect_id:
            self.canvas_roi.delete(self.roi_active_rect_id)
        for h_id in self.roi_handles_ids:
            self.canvas_roi.delete(h_id)
        self.roi_handles_ids = []
        if self.roi_text_id:
            self.canvas_roi.delete(self.roi_text_id)

        self.roi_active_rect_id = self.canvas_roi.create_rectangle(
            x1, y1, x2, y2, outline="#00F0FF", width=3
        )
        # Cantos
        hs = 6
        for hx, hy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            hid = self.canvas_roi.create_rectangle(hx - hs, hy - hs, hx + hs, hy + hs, fill="#FFFFFF", outline="#00F0FF")
            self.roi_handles_ids.append(hid)

        # Rótulo de dimensões
        w_px = bounds["max_width"]
        h_px = bounds["max_height"]
        self.roi_text_id = self.canvas_roi.create_text(
            x1 + 10, max(info["off_y"] + 15, y1 - 12),
            text=f"ÁREA ATIVA: {w_px} x {h_px} px",
            fill="#00F0FF", anchor="w", font=("Consolas", 11, "bold")
        )

    def _draw_overlay_markers_on_canvas(self):
        """Desenha marcadores visuais das canetas e cores sobre o Canvas interativo."""
        if not self.screencap_scale_info:
            return
        info = self.screencap_scale_info

        # Canetas no Topo
        for pen_name in ["pen_solid", "pen_marker", "pen_neon", "pen_eraser", "button_undo", "button_done"]:
            pt = self.adb.profile_loader.get_ui_anchor(pen_name, (0, 0))
            if pt[0] > 0 and pt[1] > 0:
                cx = int(pt[0] / info["scale_x"]) + info["off_x"]
                cy = int(pt[1] / info["scale_y"]) + info["off_y"]
                r = 10
                self.canvas_roi.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#FBBF24", width=2)

        # Cores da Paleta
        for col in config.PALETTE_BASE_COLORS:
            pt = self.adb.profile_loader.scale_point(col["x"], col["y"])
            cx = int(pt[0] / info["scale_x"]) + info["off_x"]
            cy = int(pt[1] / info["scale_y"]) + info["off_y"]
            r = 11
            hex_c = f"#{col['rgb'][0]:02X}{col['rgb'][1]:02X}{col['rgb'][2]:02X}"
            self.canvas_roi.create_oval(cx - r, cy - r, cx + r, cy + r, fill=hex_c, outline="#FFFFFF", width=1)

    def _nudge_roi(self, dx: int, dy: int):
        """Desloca a ROI no Canvas e na configuração do plano por botões ou setas do teclado (P2.7)."""
        if self._is_painting():
            return
        orig_w = 1080
        orig_h = 2400
        if self.screencap_scale_info:
            orig_w = self.screencap_scale_info.get("orig_w", 1080)
            orig_h = self.screencap_scale_info.get("orig_h", 2400)
        elif self.custom_canvas_bounds:
            orig_w = max(1080, self.custom_canvas_bounds.get("right", 1080))
            orig_h = max(2400, self.custom_canvas_bounds.get("bottom", 2400))

        if self.custom_canvas_bounds:
            self.gui_state.custom_canvas_bounds = dict(self.custom_canvas_bounds)
        new_bounds = self.gui_state.nudge_roi(dx, dy, orig_w, orig_h)
        self.custom_canvas_bounds = new_bounds
        self._mark_plan_dirty()
        if self.screencap_scale_info:
            self._draw_roi_rect_on_canvas(self.custom_canvas_bounds)
        w_real = new_bounds["max_width"]
        h_real = new_bounds["max_height"]
        msg = f"📐 Ajuste de Área: {w_real}x{h_real} px (Centro: {new_bounds['center_x']}, {new_bounds['center_y']})"
        self.lbl_roi_coords.configure(text=msg, text_color=COLORS["info"])
        self.lbl_roi_sidebar_status.configure(
            text=f"Área Customizada: {w_real}x{h_real} px",
            text_color=COLORS["success"]
        )

    def _scale_roi(self, delta: int):
        """Aumenta ou reduz a dimensão da ROI simetricamente por botões ou teclas +/- (P2.7)."""
        if self._is_painting():
            return
        orig_w = 1080
        orig_h = 2400
        if self.screencap_scale_info:
            orig_w = self.screencap_scale_info.get("orig_w", 1080)
            orig_h = self.screencap_scale_info.get("orig_h", 2400)
        elif self.custom_canvas_bounds:
            orig_w = max(1080, self.custom_canvas_bounds.get("right", 1080))
            orig_h = max(2400, self.custom_canvas_bounds.get("bottom", 2400))

        if self.custom_canvas_bounds:
            self.gui_state.custom_canvas_bounds = dict(self.custom_canvas_bounds)
        new_bounds = self.gui_state.scale_roi(delta, min_size=60, max_w=orig_w, max_h=orig_h)
        self.custom_canvas_bounds = new_bounds
        self._mark_plan_dirty()
        if self.screencap_scale_info:
            self._draw_roi_rect_on_canvas(self.custom_canvas_bounds)
        w_real = new_bounds["max_width"]
        h_real = new_bounds["max_height"]
        msg = f"📐 Escala de Área: {w_real}x{h_real} px (Centro: {new_bounds['center_x']}, {new_bounds['center_y']})"
        self.lbl_roi_coords.configure(text=msg, text_color=COLORS["info"])
        self.lbl_roi_sidebar_status.configure(
            text=f"Área Customizada: {w_real}x{h_real} px",
            text_color=COLORS["success"]
        )

    def action_save_simulation(self):
        if not self.current_plan:
            messagebox.showwarning("Aviso", "Gere uma prévia primeiro.")
            return
        if self._plan_dirty:
            messagebox.showwarning("Aviso", "O plano atual está desatualizado em relação aos ajustes. Gere uma nova prévia antes de exportar a simulação (P1.7).")
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")],
            title="Salvar Prévia da Pintura"
        )
        if save_path:
            cv2.imwrite(save_path, self.current_plan["simulation_image"])
            self.log(f"Prévia salva em: {save_path}")
            messagebox.showinfo("Salvo", f"Imagem salva com sucesso em:\n{save_path}")

    def action_run_benchmark(self):
        """Executa a suíte de benchmark de regressão diretamente pela interface."""
        self.log("Executando Suíte de Benchmark Oficial...")
        def worker():
            try:
                rep = BenchmarkSuite().run_full_benchmark()
            except Exception as exc:
                self.safe_after(0, lambda e=str(exc): self.toast(f"Benchmark falhou: {e}", "error", 6000))
                return
            def update_ui():
                self.log(f"Benchmark finalizado em {rep['total_runtime_s']}s!")
                msg = f"Benchmark Concluído com Sucesso ({rep['total_runtime_s']}s)!\n\n"
                for r in rep["results"]:
                    msg += f"• {r['case_name']}: {r['total_strokes']} traços | Mean ΔE: {r['mean_delta_e']} | SSIM: {r['ssim']}\n"
                messagebox.showinfo("Relatório de Benchmark", msg)
            self.safe_after(0, update_ui)
        threading.Thread(target=worker, daemon=True).start()

    # --------------------------------------------------------------------------
    # GERAÇÃO DA PRÉ-VISUALIZAÇÃO (SIMULAÇÃO DETERMINÍSTICA)
    # --------------------------------------------------------------------------
    def action_generate_preview(self):
        if self._is_text_mode():
            if not self._get_text_input().strip():
                messagebox.showwarning("Aviso", "Digite o texto que será escrito.")
                return
        elif not self.selected_image_path:
            messagebox.showwarning("Aviso", "Por favor, selecione uma imagem primeiro.")
            return
        if self._editing_plan or self._generating_preview or self._is_painting():
            return
        self._generating_preview = True
        self.btn_start.configure(state="disabled")
        self._undo_plans.clear()
        self._redo_plans.clear()
        self._selected_region_id = None
        self._selected_region_ids.clear()
        self._refresh_edit_controls()

        self.btn_preview.configure(state="disabled", text="⏳ Processando...")
        self.log("Iniciando sintetizador v3 com determinismo e análise perceptual...")

        mode_val = self.seg_mode.get()
        image_path = self.selected_image_path
        revision = self._plan_revision
        ink_ov, bg_ov, p_bg = self._get_line_color_overrides()
        bounds = self.custom_canvas_bounds.copy() if self.custom_canvas_bounds else None
        settings = {
            "semantic": self.var_semantic_highlights.get(),
            "fidelity": self.var_fidelity_mode.get(),
            "effects": self.var_artistic_effects.get(),
            "simulation": self.var_sim_opt.get(),
            "contours": self.var_include_contours.get(),
            "colors": int(self.slider_colors.get()),
            "spacing": int(self.slider_spacing.get()),
        }
        if mode_val == "TEXTO":
            text = self._get_text_input()
            font_name = self.opt_text_font.get()
            align = {"Esquerda": "left", "Direita": "right"}.get(self.opt_text_align.get(), "center")
            stroke = "font" if "fonte" in self.opt_text_stroke.get() else "fine"

            def text_worker():
                try:
                    plan = self.engine.generate_text_plan(
                        text, font=font_name, custom_bounds=bounds, align=align, stroke=stroke,
                        ink_rgb=ink_ov if ink_ov is not None else (255, 255, 255))
                    self.after(0, lambda p=plan, r=revision: self._post_generate_preview(p, r))
                except Exception as e:
                    err_msg = str(e)
                    self.after(0, lambda msg=err_msg: self._post_preview_error(msg))

            threading.Thread(target=text_worker, daemon=True).start()
            return

        def worker():
            raw_bgr = cv2.imread(image_path)
            if raw_bgr is None:
                self.safe_after(0, lambda: self._post_preview_error("Não foi possível ler a imagem selecionada."))
                return
            try:
                cb = bounds
                sem_opt = settings["semantic"]
                fid_mode = settings["fidelity"]
                art_effects = settings["effects"]
                sim_opt = settings["simulation"]

                style = mode_val
                if mode_val == "AUTO":
                    style = AutoModeDecisor.choose_drawing_style(raw_bgr)
                    self.after(0, lambda s=style: self.log(f"Automático escolheu: {STYLE_NAMES[s]}"))
                if style in ("LINHAS", "COLORIR"):
                    plan = self.engine.generate_line_art_plan(
                        image_path, custom_bounds=cb,
                        override_ink_rgb=ink_ov, override_bg_rgb=bg_ov, paint_background=p_bg,
                        colorize=True, fill_regions=(style == "COLORIR")
                    )
                elif style == "STENCIL":
                    # Projetos e MCP antigos ainda podem pedir o contorno clássico
                    plan = self.engine.generate_stencil_plan(image_path, custom_bounds=cb, override_ink_rgb=ink_ov)
                else:
                    plan = self.engine.generate_flat_illustration_plan(
                        image_path, num_colors=settings["colors"], include_outline=settings["contours"],
                        include_details=sem_opt, custom_bounds=cb
                    )
                self.after(0, lambda p=plan, r=revision: self._post_generate_preview(p, r))
            except Exception as e:
                import traceback
                traceback.print_exc()
                err_msg = str(e)
                self.after(0, lambda msg=err_msg: self._post_preview_error(msg))

        threading.Thread(target=worker, daemon=True).start()

    def _post_generate_preview(self, plan, revision=None):
        self._generating_preview = False
        if hasattr(self, "gui_state"):
            accepted = self.gui_state.set_clean_plan(plan, revision)
            if not accepted and revision is not None and revision != self._plan_revision:
                self.btn_preview.configure(state="normal", text=PREVIEW_BUTTON_TEXT)
                self._mark_plan_dirty()
                self.log("A imagem ou os ajustes mudaram durante o processamento. Gere a prévia novamente.")
                return
        elif revision is not None and revision != self._plan_revision:
            self.btn_preview.configure(state="normal", text=PREVIEW_BUTTON_TEXT)
            self._mark_plan_dirty()
            self.log("A imagem ou os ajustes mudaram durante o processamento. Gere a prévia novamente.")
            return
        previous_hash = self.current_plan.get("plan_hash") if self.current_plan else None
        if previous_hash and previous_hash != plan.get("plan_hash"):
            if self.checkpoint.get("layer", 0) or self.checkpoint.get("stroke", 0):
                self.log("O desenho foi alterado; a retomada anterior foi descartada.")
            self.checkpoint = {"layer": 0, "stroke": 0}
            self.saved_checkpoint = None
            PersistentCheckpointManager.clear_checkpoint()
        self.current_plan = plan
        self._plan_dirty = False
        self._selection_notice = None
        valid_ids = {r["id"] for r in plan.get("editable_regions", [])}
        self._selected_region_ids.intersection_update(valid_ids)
        self._selected_region_id = min(self._selected_region_ids) if self._selected_region_ids else None
        self._refresh_edit_controls()
        self.btn_preview.configure(state="normal", text=PREVIEW_BUTTON_TEXT)
        mode_names = {"line_art": "Traçado fino", "layered_paint": "Pintura em cores", "stencil": "Contornos",
                      "colored_lines": "Traçado colorido", "coloring_book": "Livro de colorir",
                      "flat_illustration": "Ilustração chapada", "text": "Texto"}
        actual_mode = mode_names.get(plan.get("style") or plan.get("mode"), "Modo desconhecido")
        origin = "Automático escolheu: " if self.seg_mode.get() == "AUTO" and revision is not None else "Modo: "
        self.lbl_preview_mode.configure(text=f"{origin}{actual_mode}  •  prévia atualizada", text_color=COLORS["success"])

        # Exibe imagens
        sim_img = plan["simulation_image"]
        comp_img = plan.get("comparison_image", sim_img)
        fit_target = plan.get("target_executable", plan.get("fit_image", sim_img))

        self._render_selected_preview()
        self._display_cv_image(self.lbl_canvas_compare, comp_img)
        self.lbl_compare_heading.configure(text=("Original  |  Tintas possíveis  |  Prévia simulada"
                                                 if plan.get("mode") == "layered_paint"
                                                 else "Original  |  Prévia simulada"))

        # Calcula o Mapa de Calor de Erro Perceptual (Heatmap)
        heatmap = plan.get("error_heatmap")
        metrics = plan.get("error_metrics")
        if heatmap is None or metrics is None:
            heatmap, metrics = ErrorMapEngine.compute_error_map(fit_target, sim_img)
        self._display_cv_image(self.lbl_canvas_error, heatmap)
        self.lbl_error_metrics.configure(
            text=f"Diferença média ΔE: {metrics['mean_delta_e']}   |   Pior faixa ΔE: {metrics['p95_delta_e']}   |   Semelhança SSIM: {metrics['ssim']}"
        )

        # Configura o slider de camadas
        num_layers = len(plan["layers"])
        if num_layers > 1:
            self.slider_inspect_layer.configure(state="normal", from_=1, to=num_layers, number_of_steps=num_layers - 1)
            self.slider_inspect_layer.set(num_layers)
            self.lbl_layer_slider.configure(text=f"Inspecionar Camadas: 1 até {num_layers} de {num_layers}")
        else:
            self.slider_inspect_layer.configure(from_=0, to=1, number_of_steps=1, state="disabled")
            self.slider_inspect_layer.set(1)
            self.lbl_layer_slider.configure(text=f"Inspecionar Camadas: 1 camada única")

        # Popula o Inspetor de Paleta
        self._populate_palette_inspector(plan.get("palette_colors", []))

        # Atualiza resumo
        t_est = plan["estimated_time_s"]
        t_min = int(t_est // 60)
        t_sec = int(t_est % 60)
        
        area_tag = "Central Padrão (750x950 px)"
        if plan.get("custom_bounds"):
            cb = plan["custom_bounds"]
            area_tag = f"Customizada {cb['max_width']}x{cb['max_height']} px (Centro: {cb['center_x']}, {cb['center_y']})"

        qgate = plan.get("quality_gate")
        qgate_str = "Aprovado" if getattr(qgate, "passed", False) else f"Reprovado ({len(getattr(qgate, 'failed_checks', []))} falhas)"
        self.lbl_quality_summary.configure(text=f"Qualidade da prévia: {qgate_str}",
                                           text_color=COLORS["success"] if getattr(qgate, "passed", False) else COLORS["danger"])
        gate_details = "\n".join(getattr(qgate, "failed_checks", [])) if qgate and not qgate.passed else ""
        img_cls = plan.get("image_class", "Geral")

        # Os números ficam nos blocos do inspetor; aqui só o contexto que eles não mostram.
        points = f"{plan['total_points']:,}".replace(",", ".")
        summary = (f"{actual_mode} · imagem do tipo {img_cls.replace('_', ' ').lower()}\n"
                   f"Área: {area_tag}\n"
                   f"{points} pontos vetoriais")
        if gate_details:
            summary += f"\n\nFalhas da verificação:\n{gate_details}"
        self.lbl_plan_summary.configure(text=summary)
        self._refresh_inspector(plan, metrics)
        self.log(f"Plano gerado com sucesso: [{img_cls.upper()} | Gate: {qgate_str}] {plan['total_strokes']} traços, Mean ΔE: {metrics['mean_delta_e']}, ~{t_min}m {t_sec}s.")

        self._update_start_availability()
        self.lbl_progress_text.configure(text=("Prévia pronta • conecte o celular"
                                          if not self.adb.is_connected else "Prévia pronta • iniciar quando desejar"))
        # Verifica se o checkpoint salvo corresponde exatamente a este plano gerado
        self._validate_and_offer_resume()

    def _post_preview_error(self, err):
        self._generating_preview = False
        self.btn_preview.configure(state="normal", text=PREVIEW_BUTTON_TEXT)
        self.log(f"Erro ao sintetizar prévia: {err}")
        self._update_start_availability()
        messagebox.showerror("Erro no Processamento", err)

    def _refresh_edit_controls(self):
        if hasattr(self, "btn_undo_edit"):
            locked = self._editing_plan or self._generating_preview or self._plan_dirty
            self.btn_undo_edit.configure(state="normal" if self._undo_plans and not locked else "disabled")
            self.btn_redo_edit.configure(state="normal" if self._redo_plans and not locked else "disabled")
            can_edit = bool(self._selected_region_ids) and not locked
            self.btn_region_color.configure(state="normal" if can_edit else "disabled")
            self.btn_physical_color.configure(state="normal" if can_edit else "disabled")
            self.btn_clear_selection.configure(state="normal" if self._selected_region_ids else "disabled")
            regions = self.current_plan.get("editable_regions", []) if self.current_plan else []
            has_background = any(r.get("selection_role") == "background" or
                                 (self.current_plan.get("mode") == "line_art" and r.get("name") == "Fundo")
                                 for r in regions) if self.current_plan else False
            self.btn_select_background.configure(state="normal" if has_background and not locked else "disabled")
        if hasattr(self, "lbl_selected_region"):
            regions = self.current_plan.get("editable_regions", []) if self.current_plan else []
            selected = [r for r in regions if r["id"] in self._selected_region_ids]
            if len(selected) == 1:
                region = selected[0]
                palette = self.current_plan.get("palette_colors", [])
                index = region.get("palette_index", -1)
                rgb = region.get("override_rgb")
                color_hex = (f"#{int(rgb[0]):02X}{int(rgb[1]):02X}{int(rgb[2]):02X}" if rgb is not None
                             else palette[index].get("hex", "#FFFFFF") if 0 <= index < len(palette) else "#FFFFFF")
                self.lbl_selected_region.configure(text=f"{region['name']}  •  cor {color_hex}  •  {region['area']} pixels")
                self.selection_swatch.configure(fg_color=color_hex)
            elif selected:
                if self.current_plan.get("mode") in ("line_art", "stencil", "flat_illustration"):
                    message = f"{len(selected)} traços selecionados. Os traços que cruzam a seleção mudam por inteiro."
                else:
                    message = f"{len(selected)} partes selecionadas. Escolha uma cor para alterar todas de uma vez."
                self.lbl_selected_region.configure(text=message)
                self.selection_swatch.configure(fg_color=COLORS["surface_4"])
            else:
                hint = ("Clique em um traço ou arraste sobre vários. Ctrl+clique adiciona à seleção."
                        if self.current_plan and self.current_plan.get("mode") in ("line_art", "stencil", "flat_illustration")
                        else "Clique em uma área ou arraste sobre várias. Ctrl+clique adiciona à seleção.")
                self.lbl_selected_region.configure(text=self._selection_notice or hint)
                self.selection_swatch.configure(fg_color=COLORS["surface_4"])

    def _render_selected_preview(self):
        if not self.current_plan:
            return
        base = self.current_plan.get("simulation_image")
        if base is None:
            return
        if self._show_stroke_view:
            view = np.full_like(base, 22)
            for index, layer in enumerate(self.current_plan.get("layers", [])):
                color = ((47 + index * 89) % 220 + 25,
                         (93 + index * 59) % 220 + 25,
                         (165 + index * 37) % 220 + 25)
                for stroke in layer.get("strokes", []):
                    if len(stroke) >= 2:
                        pts = np.asarray(stroke, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(view, [pts], False, color, 1, cv2.LINE_AA)
        else:
            view = base.copy()
        label_map = self.current_plan.get("region_map")
        if label_map is not None and self._selected_region_ids:
            mask = np.isin(label_map, list(self._selected_region_ids)).astype(np.uint8) * 255
            # Borda externa para a cor nova continuar visível em traços finos.
            outer = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            contours, _ = cv2.findContours(outer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(view, contours, -1, (0, 230, 255), 1, cv2.LINE_AA)
        if self._selection_start and self._selection_drag:
            cv2.rectangle(view, self._selection_start, self._selection_drag, (0, 230, 255), 2)
        self._display_cv_image(self.lbl_canvas_preview, view)

    def _preview_event_to_image(self, event, clamp=False):
        if not self.current_plan or self._editing_plan:
            return None
        label_map = self.current_plan.get("region_map")
        if label_map is None or not self._preview_display_size:
            return None
        display_w, display_h = self._preview_display_size
        if display_w <= 0 or display_h <= 0:
            return None
        left = (self.lbl_canvas_preview.winfo_width() - display_w) / 2
        top = (self.lbl_canvas_preview.winfo_height() - display_h) / 2
        # CTkLabel encaminha eventos de filhos com coordenadas locais distintas.
        click_x = event.x_root - self.lbl_canvas_preview.winfo_rootx()
        click_y = event.y_root - self.lbl_canvas_preview.winfo_rooty()
        if clamp:
            click_x = max(left, min(left + display_w - 1, click_x))
            click_y = max(top, min(top + display_h - 1, click_y))
        elif not (left <= click_x < left + display_w and top <= click_y < top + display_h):
            return None
        h, w = label_map.shape
        x = min(w - 1, int((click_x - left) * w / display_w))
        y = min(h - 1, int((click_y - top) * h / display_h))
        return x, y

    def _select_preview_bounds(self, start, end, additive=False):
        label_map = self.current_plan.get("region_map") if self.current_plan else None
        if label_map is None:
            return
        x0, x1 = sorted((start[0], end[0]))
        y0, y1 = sorted((start[1], end[1]))
        ids = {int(value) for value in np.unique(label_map[y0:y1 + 1, x0:x1 + 1]) if value > 0}
        regions = self.current_plan.get("editable_regions", [])
        valid_ids = {r["id"] for r in regions}
        ids.intersection_update(valid_ids)
        background_ids = {r["id"] for r in regions
                          if r.get("selection_role") == "background"
                          or (self.current_plan.get("mode") == "line_art" and r.get("name") == "Fundo")}
        if ids & background_ids:
            ids.difference_update(background_ids)
            self._selection_notice = ("Para mudar o fundo, use ‘Selecionar fundo’."
                                      if not ids else None)
        else:
            self._selection_notice = None
        if additive:
            ids.update(self._selected_region_ids)
        self._selected_region_ids = ids
        self._selected_region_id = min(ids) if ids else None
        self._refresh_edit_controls()
        self._render_selected_preview()

    def action_select_background(self):
        if not self.current_plan:
            return
        ids = {r["id"] for r in self.current_plan.get("editable_regions", [])
               if r.get("selection_role") == "background"
               or (self.current_plan.get("mode") == "line_art" and r.get("name") == "Fundo")}
        self._selected_region_ids = ids
        self._selected_region_id = min(ids) if ids else None
        self._selection_notice = None
        self._refresh_edit_controls()
        self._render_selected_preview()

    def action_clear_selection(self):
        self._selected_region_ids.clear()
        self._selected_region_id = None
        self._selection_notice = None
        self._refresh_edit_controls()
        self._render_selected_preview()

    def _on_preview_click(self, event):
        point = self._preview_event_to_image(event)
        if point is not None:
            self._select_preview_bounds(point, point, additive=bool(event.state & 0x0004))

    def _on_preview_press(self, event):
        self._selection_start = self._preview_event_to_image(event)
        self._selection_drag = None

    def _on_preview_drag(self, event):
        if self._selection_start is None:
            return
        self._selection_drag = self._preview_event_to_image(event, clamp=True)
        self._render_selected_preview()

    def _on_preview_release(self, event):
        start = self._selection_start
        end = self._preview_event_to_image(event, clamp=True) if start is not None else None
        self._selection_start = None
        self._selection_drag = None
        if start is not None and end is not None:
            self._select_preview_bounds(start, end, additive=bool(event.state & 0x0004))

    def action_toggle_stroke_view(self):
        self._show_stroke_view = not self._show_stroke_view
        self.btn_stroke_view.configure(text="Ver pintura" if self._show_stroke_view else "Ver traços")
        self._render_selected_preview()

    def action_zoom_preview(self):
        if not self.current_plan:
            return
        image = self.current_plan["simulation_image"]
        if self._show_stroke_view:
            image = np.full_like(image, 22)
            for index, layer in enumerate(self.current_plan.get("layers", [])):
                color = ((47 + index * 89) % 220 + 25,
                         (93 + index * 59) % 220 + 25,
                         (165 + index * 37) % 220 + 25)
                for stroke in layer.get("strokes", []):
                    if len(stroke) >= 2:
                        cv2.polylines(image, [np.asarray(stroke, dtype=np.int32).reshape((-1, 1, 2))],
                                      False, color, 1, cv2.LINE_AA)
        window = ctk.CTkToplevel(self)
        window.title("Prévia em tamanho real — clique ou arraste para selecionar")
        window.geometry("900x700")
        frame = tk.Frame(window, bg=COLORS["canvas"])
        frame.pack(fill="both", expand=True)
        canvas = tk.Canvas(frame, bg=COLORS["canvas"], highlightthickness=0)
        yscroll = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        xscroll = tk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
        canvas.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(pil, master=window)
        canvas.photo = photo
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.configure(scrollregion=(0, 0, pil.width, pil.height))

        drag_start = None
        drag_rectangle = None

        def zoom_point(event):
            label_map = self.current_plan.get("region_map") if self.current_plan else None
            if label_map is None:
                return None
            x, y = int(canvas.canvasx(event.x)), int(canvas.canvasy(event.y))
            return max(0, min(label_map.shape[1] - 1, x)), max(0, min(label_map.shape[0] - 1, y))

        def press(event):
            nonlocal drag_start, drag_rectangle
            drag_start = zoom_point(event)
            if drag_rectangle is not None:
                canvas.delete(drag_rectangle)
                drag_rectangle = None

        def drag(event):
            nonlocal drag_rectangle
            end = zoom_point(event)
            if drag_start is None or end is None:
                return
            if drag_rectangle is not None:
                canvas.delete(drag_rectangle)
            drag_rectangle = canvas.create_rectangle(
                drag_start[0], drag_start[1], end[0], end[1],
                outline="#FFE600", width=2,
            )

        def release(event):
            nonlocal drag_start, drag_rectangle
            end = zoom_point(event)
            if drag_rectangle is not None:
                canvas.delete(drag_rectangle)
                drag_rectangle = None
            if drag_start is not None and end is not None:
                self._select_preview_bounds(drag_start, end, additive=bool(event.state & 0x0004))
            drag_start = None

        canvas.bind("<ButtonPress-1>", press)
        canvas.bind("<B1-Motion>", drag)
        canvas.bind("<ButtonRelease-1>", release)

    def action_change_region_color(self):
        region_ids = tuple(sorted(self._selected_region_ids))
        if not region_ids or not self.current_plan:
            return
        region = next((r for r in self.current_plan.get("editable_regions", []) if r["id"] == region_ids[0]), None)
        if region is None:
            return
        palette_index = region["palette_index"]
        palette = self.current_plan.get("palette_colors", [])
        override = region.get("override_rgb")
        initial = (f"#{int(override[0]):02X}{int(override[1]):02X}{int(override[2]):02X}"
                   if override is not None else
                   palette[palette_index]["hex"] if 0 <= palette_index < len(palette) else "#FFFFFF")
        title = f"Cor para {len(region_ids)} partes selecionadas" if len(region_ids) > 1 else f"Cor para {region['name']}"
        color_tuple, _ = colorchooser.askcolor(color=initial, title=title)
        if color_tuple:
            rgb = tuple(int(round(value)) for value in color_tuple)
            self._run_plan_edit(lambda source: self.engine.recolor_regions(source, region_ids, rgb))

    def action_choose_physical_color(self):
        region_ids = tuple(sorted(self._selected_region_ids))
        if not region_ids or self._editing_plan:
            return
        entries = self.engine.color_engine.entries
        window = ctk.CTkToplevel(self)
        window.title("Tintas do Instagram")
        window.geometry("560x570")
        window.transient(self)
        ctk.CTkLabel(window, text="Escolha a página e a tinta física da paleta.",
                     font=ctk.CTkFont(size=15, weight="bold")).pack(pady=10)
        page_var = tk.StringVar(value="Página 1")
        page_menu = ctk.CTkOptionMenu(window, values=["Página 1", "Página 2", "Página 3"], variable=page_var)
        page_menu.pack(pady=4)
        listbox = tk.Listbox(window, height=22, font=("Segoe UI", 11),
                             bg=COLORS["surface_2"], fg=COLORS["text"], selectbackground=COLORS["accent"])
        listbox.pack(fill="both", expand=True, padx=16, pady=10)
        preview = ctk.CTkLabel(window, text="Selecione uma tinta")
        preview.pack(pady=4)
        visible = []

        def fill_list(_value=None):
            visible.clear()
            listbox.delete(0, "end")
            page = int(page_var.get().split()[-1])
            for entry in entries:
                if int(entry.get("palette_page", 1)) == page:
                    visible.append(entry)
                    kind = "nuance" if entry.get("is_long_press") else "base"
                    listbox.insert("end", f"{entry['hex']}  {entry['name']}  ({kind})")
            preview.configure(text=f"{len(visible)} tintas nesta página")

        def show_selection(_event=None):
            chosen = listbox.curselection()
            if chosen:
                entry = visible[chosen[0]]
                gesture = "toque longo" if entry.get("is_long_press") else "toque simples"
                origin = "amostra detalhada medida" if str(entry.get("id", "")).startswith("measured_detail") else "perfil de cores"
                preview.configure(text=f"{entry['hex']} | {entry['name']} | página {entry.get('palette_page', 1)} | {gesture} | {origin}",
                                  text_color=entry["hex"])

        def apply_selection():
            chosen = listbox.curselection()
            if not chosen:
                return
            entry = visible[chosen[0]]
            rgb = tuple(entry["rgb"])
            lut_id = entry["id"]
            window.destroy()
            self._run_plan_edit(lambda source: self.engine.recolor_regions(
                source, region_ids, rgb, preferred_lut_id=lut_id
            ))

        page_menu.configure(command=fill_list)
        listbox.bind("<<ListboxSelect>>", show_selection)
        ctk.CTkButton(window, text="Aplicar às partes selecionadas", command=apply_selection).pack(pady=10)
        fill_list()

    def action_save_project(self):
        if not self.current_plan or self._editing_plan:
            return
        if self._plan_dirty:
            messagebox.showwarning("Aviso", "O plano atual está desatualizado em relação aos ajustes. Gere uma nova prévia antes de salvar o projeto (P1.7).")
            return
        path = filedialog.asksaveasfilename(defaultextension=".npz",
                                            filetypes=[("Projeto Instagram Painter", "*.npz")])
        if path:
            try:
                save_project(path, self.current_plan)
                self.log(f"Projeto editável salvo em: {path}")
            except Exception as exc:
                messagebox.showerror("Falha ao salvar", str(exc))

    def action_open_project(self):
        if self._editing_plan or self._generating_preview or self._is_painting():
            return
        path = filedialog.askopenfilename(filetypes=[("Projeto Instagram Painter", "*.npz")])
        if not path:
            return
        self._editing_plan = True
        self.btn_start.configure(state="disabled")
        self._refresh_edit_controls()

        def run():
            try:
                loaded = load_project(path, self.engine)
                self.safe_after(0, lambda: finish(loaded, None))
            except Exception as exc:
                self.safe_after(0, lambda message=str(exc): finish(None, message))

        def finish(loaded, error):
            self._editing_plan = False
            if error:
                self._refresh_edit_controls()
                qgate = self.current_plan.get("quality_gate") if self.current_plan else None
                self.btn_start.configure(state="normal" if self._can_start_current_plan() else "disabled")
                messagebox.showerror("Falha ao abrir", error)
                return
            self._undo_plans.clear()
            self._redo_plans.clear()
            self._selected_region_id = None
            self._selected_region_ids.clear()
            self._post_generate_preview(loaded)
            self.log(f"Projeto editável aberto: {path}")

        threading.Thread(target=run, daemon=True).start()

    def _run_plan_edit(self, operation):
        if self._editing_plan or self._generating_preview or self._plan_dirty or not self.current_plan or self._is_painting():
            return
        source = self.current_plan
        revision = self._plan_revision
        self._editing_plan = True
        self.btn_start.configure(state="disabled")
        self._refresh_edit_controls()
        self.log("Atualizando plano e prévia...")

        def run():
            try:
                revised = operation(source)
                self.safe_after(0, lambda: self._finish_plan_edit(source, revised, None, revision))
            except Exception as exc:
                self.safe_after(0, lambda message=str(exc): self._finish_plan_edit(source, None, message, revision))

        threading.Thread(target=run, daemon=True).start()

    def _finish_plan_edit(self, source, revised, error, revision=None):
        self._editing_plan = False
        if revision is not None and revision != self._plan_revision:
            self._mark_plan_dirty()
            self.log("Os ajustes mudaram durante a edição. Gere a prévia novamente.")
            return
        if error:
            self._refresh_edit_controls()
            qgate = source.get("quality_gate")
            self.btn_start.configure(state="normal" if self._can_start_current_plan() else "disabled")
            messagebox.showerror("Falha na edição", error)
            return
        self._undo_plans.append(source)
        if len(self._undo_plans) > 5:
            self._undo_plans.pop(0)
        self._redo_plans.clear()
        self._post_generate_preview(revised)
        self.log("Edição aplicada; plano e prévia atualizados.")

    def action_undo_edit(self):
        if self._editing_plan or self._plan_dirty or not self._undo_plans or not self.current_plan:
            return
        self._redo_plans.append(self.current_plan)
        if len(self._redo_plans) > 5:
            self._redo_plans.pop(0)
        self._post_generate_preview(self._undo_plans.pop())

    def action_redo_edit(self):
        if self._editing_plan or self._plan_dirty or not self._redo_plans or not self.current_plan:
            return
        self._undo_plans.append(self.current_plan)
        if len(self._undo_plans) > 5:
            self._undo_plans.pop(0)
        self._post_generate_preview(self._redo_plans.pop())

    def _populate_palette_inspector(self, palette_colors):
        for widget in self.palette_scroll.winfo_children():
            widget.destroy()

        if not palette_colors:
            ctk.CTkLabel(self.palette_scroll, text="Modo monocromático.").pack(pady=20)
            return

        header_frame = ctk.CTkFrame(self.palette_scroll, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=(5, 10))

        ctk.CTkLabel(header_frame, text=f"Cores da pintura · {len(palette_colors)}",
                     font=ctk.CTkFont(size=14, weight="bold"), anchor="w").pack(fill="x")
        ctk.CTkLabel(header_frame,
                     text=f"Perfil do celular: {len(self.engine.color_engine.entries)} tintas. Clique em Alterar para trocar uma cor.",
                     font=ctk.CTkFont(size=11), text_color=COLORS["text_muted"], anchor="w").pack(fill="x")

        for idx, col in enumerate(palette_colors):
            card = ctk.CTkFrame(self.palette_scroll, fg_color=COLORS["surface_2"], corner_radius=10,
                                border_width=1, border_color=COLORS["border"])
            card.pack(fill="x", padx=10, pady=4)
            card.grid_columnconfigure(1, weight=1)

            # Mini-preview da cor de referência
            c_preview = ctk.CTkFrame(card, width=32, height=32, corner_radius=16, fg_color=col["hex"])
            c_preview.grid(row=0, column=0, rowspan=2, padx=(12, 6), pady=8)

            # Mini-preview da cor física mapeada na LUT
            matched_h = col.get("matched_hex", col["hex"])
            c_matched = ctk.CTkFrame(card, width=28, height=28, corner_radius=14, fg_color=matched_h)
            c_matched.grid(row=0, column=2, rowspan=2, padx=(0, 10), pady=8)

            ctk.CTkLabel(card, text=f"{idx + 1}. {col['name']}", anchor="w",
                         font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=1, sticky="ew")
            detail = (f"{col['hex']} → {matched_h}  ·  Página {col.get('palette_page', '?')}  ·  "
                      f"{'Nuance' if col.get('is_long_press') else 'Base'}  ·  ΔE {col.get('delta_e', 0.0)}")
            ctk.CTkLabel(card, text=detail, anchor="w", text_color=COLORS["text_muted"],
                         font=ctk.CTkFont(size=10)).grid(row=1, column=1, sticky="ew")

            # Botão interativo para mudar esta cor
            btn_change = ctk.CTkButton(
                card,
                text="Alterar",
                width=72,
                height=28,
                **variant_colors("secondary"),
                font=font("button_sm"),
                command=lambda i=idx: self.action_change_palette_color(i)
            )
            btn_change.grid(row=0, column=3, rowspan=2, padx=(0, 12), pady=6)

    def action_change_palette_color(self, color_idx: int):
        """Abre o seletor de cores e atualiza o plano em tempo real com re-renderização imediata."""
        if not self.current_plan or not self.current_plan.get("palette_colors"):
            messagebox.showwarning("Aviso", "Gere uma prévia antes de mudar as cores.")
            return

        pal = self.current_plan["palette_colors"]
        if not (0 <= color_idx < len(pal)):
            return

        current_color_entry = pal[color_idx]
        init_hex = current_color_entry.get("hex", "#FFFFFF")

        color_tuple, chosen_hex = colorchooser.askcolor(
            color=init_hex,
            title=f"Escolha uma nova cor para a Cor {color_idx + 1} ({current_color_entry['name']})"
        )

        if chosen_hex and color_tuple:
            r, g, b = int(round(color_tuple[0])), int(round(color_tuple[1])), int(round(color_tuple[2]))
            new_rgb = (r, g, b)

            self._run_plan_edit(lambda source: self.engine.recolor_plan(source, color_idx, new_rgb))

    # --------------------------------------------------------------------------
    # EXECUÇÃO DA PINTURA NO CELULAR VIA ADB
    # --------------------------------------------------------------------------
    def action_primary(self):
        """Botão do rodapé: executa a próxima etapa que falta."""
        if self._can_start_current_plan():
            self.action_start_painting()
        elif self._has_source() and (not self.current_plan or self._plan_dirty):
            self.action_generate_preview()
        elif self.current_plan and not getattr(self.adb, "is_connected", False):
            self._show_step("device")
        else:
            self.action_start_painting()

    def action_start_painting(self):
        if self._editing_plan or self._generating_preview:
            messagebox.showwarning("Aguarde", "O plano ainda está sendo preparado.")
            return
        if not self.current_plan:
            messagebox.showwarning("Aviso", "Gere e aprove a prévia antes de iniciar.")
            return
        if self._plan_dirty:
            messagebox.showwarning("Prévia desatualizada", "O modo, a imagem ou os ajustes mudaram. Gere a prévia novamente antes de pintar.")
            return
        qgate = self.current_plan.get("quality_gate")
        if qgate is None or not getattr(qgate, "passed", False):
            details = ", ".join(getattr(qgate, "failed_checks", [])) if qgate else "verificação ausente"
            messagebox.showerror("Plano bloqueado", f"A prévia ainda não passou nas verificações: {details}")
            return

        need_connect = not self.adb.is_connected
        if need_connect:
            ans = messagebox.askyesno(
                "ADB Não Conectado",
                "O celular Android ainda não foi conectado. Deseja tentar a conexão automática agora?"
            )
            if not ans:
                return

        # Conexão e leitura do estado da tela podem levar segundos: rodam fora da thread da interface
        self.btn_start.configure(state="disabled")
        self.log("Verificando conexão e tela do celular...")

        def preflight():
            try:
                if need_connect:
                    ok, msg = self.adb.connect()
                    if not ok:
                        self.safe_after(0, lambda m=msg: self._abort_start_painting(
                            messagebox.showerror, "Erro", f"Não foi possível conectar: {m}"))
                        return
                screen_on = self.adb.is_screen_on()
            except Exception as exc:
                # Sem isto o botão Iniciar ficaria desativado para sempre
                self.safe_after(0, lambda e=str(exc): self._abort_start_painting(
                    messagebox.showerror, "Erro", f"Falha ao verificar o celular: {e}"))
                return
            self.safe_after(0, lambda s=screen_on: self._continue_start_painting(s))

        threading.Thread(target=preflight, daemon=True).start()

    def _abort_start_painting(self, dialog=None, title="", message=""):
        self.btn_start.configure(state="normal" if self._can_start_current_plan() else "disabled")
        if dialog:
            dialog(title, message)

    def _continue_start_painting(self, screen_on):
        # O plano pode ter mudado enquanto a pré-checagem rodava
        if not self.current_plan or self._plan_dirty or self._editing_plan or self._generating_preview:
            self._abort_start_painting(messagebox.showwarning, "Prévia desatualizada",
                                       "O plano mudou durante a verificação do celular. Gere a prévia novamente.")
            return

        # Validação de tela
        if not screen_on:
            self._abort_start_painting(messagebox.showwarning, "Aviso de Tela",
                                       "A tela do celular está apagada. Por favor, desbloqueie seu aparelho.")
            return

        start_from_zero = False
        if self.checkpoint.get("layer", 0) > 0 or self.checkpoint.get("stroke", 0) > 0:
            resume_choice = messagebox.askyesnocancel(
                "Progresso salvo",
                "Este projeto possui uma pintura interrompida.\n\n"
                "Sim: retomar do ponto salvo.\n"
                "Não: começar do zero e descartar o progresso salvo.\n"
                "Cancelar: não iniciar agora.\n\n"
                "Se escolher começar do zero, limpe antes os traços antigos no celular."
            )
            if resume_choice is None:
                self._abort_start_painting()
                return
            start_from_zero = not resume_choice

        confirm = messagebox.askyesno(
            "Confirmar Início da Pintura",
            f"O robô irá executar {self.current_plan['total_strokes']} traços no Instagram Direct.\n\n"
            f"⚠️ ATENÇÃO: Verifique a tela do seu celular agora!\n\n"
            f"1. A tela de DESENHO (fundo preto com canetas no topo e paleta de cores no rodapé) DEVE estar aberta no celular.\n"
            f"2. Se o celular estiver no feed, no perfil ou na conversa normal, os toques atingirão fotos e mensagens!\n\n"
            f"A tela de DESENHO já está aberta e pronta no seu celular?"
        )
        if not confirm:
            self._abort_start_painting()
            return

        if start_from_zero:
            self.checkpoint = {"layer": 0, "stroke": 0}
            self.saved_checkpoint = None
            PersistentCheckpointManager.clear_checkpoint()
            self.log("Projeto reiniciado do zero; progresso anterior descartado.")

        self.btn_start.configure(state="disabled")
        self.btn_pause.configure(state="normal")
        self.btn_resume.configure(state="disabled")
        self.progress_bar.set(0)

        callbacks = {
            "on_progress": self._on_worker_progress,
            "on_layer_start": self._on_worker_layer,
            "on_log": self._on_worker_log,
            "on_finished": self._on_worker_finished,
            "on_screencap": self._on_worker_screencap,
            "on_checkpoint": self._on_worker_checkpoint,
            "on_error_map": self._on_worker_error_map,
            "on_pause_state": self._on_worker_pause_state,
        }

        enable_cl = self.var_closed_loop.get() if hasattr(self, "var_closed_loop") else False
        verify_lyr = self.var_verify_layer.get() if hasattr(self, "var_verify_layer") else False
        self.worker = ExecutionWorker(
            self.adb,
            self.current_plan,
            callbacks,
            start_layer=self.checkpoint["layer"],
            start_stroke_offset=self.checkpoint["stroke"],
            enable_closed_loop=enable_cl,
            allow_destructive_corrections=False,
            verify_each_layer=verify_lyr
        )
        if hasattr(self.adb, "reset_abort"):
            self.adb.reset_abort()
        self._set_input_controls_locked(True)
        self._paint_started_at = time.monotonic()
        self._worker_finished = False
        self.lbl_status.configure(text="Iniciando pintura…")
        self.lbl_eta.configure(text="")
        self._show_step("paint")
        self.worker.start()
        self._refresh_progress_indicators()

    def _on_worker_log(self, msg):
        self.after(0, lambda m=msg: self.log(m))

    def _on_worker_screencap(self, img):
        if img is None:
            return
        def update():
            self.latest_screencap_raw = img
            try:
                self._display_screencap_canvas(img)
            except Exception:
                pass
            if getattr(self, "lbl_canvas_live", None) is not None:
                try:
                    self._display_cv_image(self.lbl_canvas_live, img)
                except Exception:
                    pass
        self.after(0, update)

    def _on_worker_error_map(self, heatmap, metrics):
        def update():
            if heatmap is not None:
                self._display_cv_image(self.lbl_canvas_error, heatmap)
            if metrics:
                mean_de = metrics.get('mean_delta_e', 'N/A')
                p95_de = metrics.get('p95_delta_e', 'N/A')
                ssim_val = metrics.get('ssim', 'N/A')
                self.lbl_error_metrics.configure(
                    text=f"Métricas Closed-Loop: Mean ΔE: {mean_de} | P95 ΔE: {p95_de} | SSIM: {ssim_val}"
                )
        self.after(0, update)

    def _on_worker_checkpoint(self, layer_idx, stroke_idx):
        self.checkpoint = {"layer": layer_idx, "stroke": stroke_idx}
        if layer_idx == 0 and stroke_idx == 0:
            PersistentCheckpointManager.clear_checkpoint()
            self.saved_checkpoint = None
            return
        PersistentCheckpointManager.save_checkpoint(
            self.current_plan.get("plan_hash", "plan"),
            layer_idx, stroke_idx, self.adb.device_target
        )

    def _on_worker_progress(self, l_curr, l_tot, s_curr, s_tot, percent, msg):
        now = time.monotonic()
        if percent < 100 and now - self._last_progress_ui_ts < 0.1:
            return
        self._last_progress_ui_ts = now
        def update():
            self.progress_bar.set(percent / 100.0)
            self.lbl_progress_text.configure(text=f"{msg} ({percent}%)")
            if getattr(self, "_paint_started_at", None) and 1 <= percent < 100:
                elapsed = time.monotonic() - self._paint_started_at
                remaining = int(elapsed * (100 - percent) / percent)
                self.lbl_eta.configure(text=f"~{remaining // 60}:{remaining % 60:02d} restantes")
        self.after(0, update)

    def _on_worker_pause_state(self, paused):
        def update():
            if self._is_painting():
                self.btn_pause.configure(state="disabled" if paused else "normal")
                self.btn_resume.configure(state="normal" if paused else "disabled")
                self.lbl_status.configure(text="Pausado" if paused else "Pintando")
        self.after(0, update)

    def _on_worker_layer(self, l_curr, layer_name, color_rgb):
        def update():
            self.lbl_status.configure(text=f"Pintando: {layer_name}")
            if hasattr(self, "_layer_rows"):
                self._highlight_layer(l_curr)
        self.after(0, update)

    def _on_worker_finished(self, success, msg):
        def update():
            self._worker_finished = True
            # O executor terminou: libera o freio do ADB acionado pelo Parar, senão
            # "Abrir desenho", "Toque de teste" etc. falhariam depois da parada.
            if hasattr(self.adb, "reset_abort"):
                self.adb.reset_abort()
            self._set_input_controls_locked(False)
            self.btn_start.configure(state="normal" if self._can_start_current_plan() else "disabled")
            self.btn_pause.configure(state="disabled")
            self.btn_resume.configure(state="disabled")
            self._paint_started_at = None
            if hasattr(self, "lbl_eta"):
                self.lbl_eta.configure(text="")
            if hasattr(self, "sections"):
                self._refresh_progress_indicators()
            if success:
                self.checkpoint = {"layer": 0, "stroke": 0}
                PersistentCheckpointManager.clear_checkpoint()
                self.progress_bar.set(1.0)
                self.lbl_progress_text.configure(text="Pintura concluída com sucesso! 100%")
                self.lbl_status.configure(text="Pintura concluída")
                messagebox.showinfo("Pintura Concluída", "A pintura foi finalizada com sucesso no seu Instagram!")
                self.action_refresh_screencap()
            else:
                self.lbl_status.configure(text="Pintura interrompida")
                has_checkpoint = self.checkpoint.get("stroke", 0) > 0 or self.checkpoint.get("layer", 0) > 0
                visually_checked = bool(getattr(self.worker, "verify_each_layer", False))
                suffix = ("\nProgresso visualmente confirmado salvo para retomada."
                          if has_checkpoint and visually_checked else
                          "\nTraços enviados salvos para retomada, sem confirmação visual. Confira o canvas antes de continuar."
                          if has_checkpoint else
                          "\nNenhuma camada foi confirmada. Confira o canvas antes de tentar novamente.")
                messagebox.showwarning("Interrompido", f"{msg}{suffix}")
                self.btn_resume.configure(state="normal" if has_checkpoint else "disabled")
        self.after(0, update)

    def action_pause_painting(self):
        if self.worker:
            self.worker.pause()
            self.btn_pause.configure(state="disabled")
            self.btn_resume.configure(state="normal")

    def action_resume_painting(self):
        if self._is_painting():
            self.worker.resume()
            self.btn_pause.configure(state="normal")
            self.btn_resume.configure(state="disabled")
        else:
            self.action_start_painting()

def launch_app():
    app = InstagramPainterStudioGUI()
    app.mainloop()


def main() -> int:
    """Ponto de entrada do script ``instagram-painter`` (planos e saídas relativos à raiz do projeto)."""
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    launch_app()
    return 0

if __name__ == "__main__":
    launch_app()
