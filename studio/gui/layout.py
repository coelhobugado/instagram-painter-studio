"""
Construção da janela do estúdio (layout, componentes e estados visuais).

Separado da lógica em ``app_gui.py``: este mixin só cria widgets com os nomes
que a lógica usa e mantém os indicadores visuais em dia.

Organização (tudo visível, sem telas escondidas):
┌ cabeçalho: logo · estado do celular (clicável) · parar ───────────────┐
│ coluna de seções numeradas        │ números do plano                   │
│ 1 Imagem   2 Estilo   3 Celular   │ abas: Prévia · Comparar · Cores ·  │
│ 4 Área     5 Pintura              │       Camadas · Qualidade · Tela   │
└ barra: o que falta / progresso · pausar · retomar · iniciar ──────────┘
"""
import tkinter as tk

import customtkinter as ctk

from .. import config
from .theme import (COLORS, RADIUS, font, button, button_style, variant_colors, checkbox_style,
                    slider_style, option_menu_style, entry_style)
from .widgets import ActionButton, StatusPill, StatTile, Toast, section_label, brand_logo
from .settings import load_settings

PREVIEW_BUTTON_TEXT = "Gerar prévia   Ctrl+Enter"
SIDEBAR_WIDTH = 370
SECTIONS = (("image", "Imagem"), ("style", "Estilo"), ("device", "Celular"), ("area", "Área de pintura"),
            ("paint", "Pintura"))


class _Section(ctk.CTkFrame):
    """Seção numerada da coluna: número, título, estado à direita e conteúdo em ``body``."""

    def __init__(self, master, number, title):
        super().__init__(master, fg_color=COLORS["surface"], corner_radius=RADIUS["lg"],
                         border_width=1, border_color=COLORS["border"])
        head = ctk.CTkFrame(self, fg_color="transparent")
        head.pack(fill="x", padx=14, pady=(12, 0))
        self.badge = ctk.CTkLabel(head, text=str(number), width=24, height=24, corner_radius=12,
                                  fg_color=COLORS["surface_4"], text_color=COLORS["text"], font=font("small_bold"))
        self.badge.pack(side="left")
        ctk.CTkLabel(head, text=title, font=font("title"), text_color=COLORS["text"]).pack(side="left", padx=10)
        self.status = ctk.CTkLabel(head, text="", font=font("small_bold"), text_color=COLORS["text_faint"])
        self.status.pack(side="right")
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.pack(fill="x", padx=14, pady=(10, 14))
        self._number = number

    def set_state(self, done, text="", tone=None):
        color = {"ok": COLORS["success"], "warn": COLORS["warning"], "off": COLORS["text_faint"],
                 "info": COLORS["info"]}.get(tone or ("ok" if done else "off"))
        self.badge.configure(text="✓" if done else str(self._number),
                             fg_color=COLORS["success_soft"] if done else COLORS["surface_4"],
                             text_color=COLORS["success"] if done else COLORS["text"])
        self.status.configure(text=text, text_color=color)

    def highlight(self, on=True):
        self.configure(border_color=COLORS["accent"] if on else COLORS["border"])


class StudioLayoutMixin:
    """Cria e atualiza os widgets da janela principal."""

    # ------------------------------------------------------------------ raiz
    def _build_layout(self):
        self.configure(fg_color=COLORS["bg"])
        self.grid_columnconfigure(0, weight=0, minsize=SIDEBAR_WIDTH)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._toast = None
        self._create_header()
        self._create_sidebar()
        self._create_main_preview_area()
        self._create_status_bar()
        self._bind_shortcuts()

    def toast(self, message, tone="info", duration_ms=3500):
        """Aviso rápido que não interrompe o usuário (para confirmações, não para erros)."""
        if self._toast is not None:
            try:
                self._toast.destroy()
            except Exception:
                pass
        try:
            self._toast = Toast(self, message, tone, duration_ms)
        except tk.TclError:
            self._toast = None

    # ------------------------------------------------------------- cabeçalho
    def _create_header(self):
        self.header_frame = ctk.CTkFrame(self, height=60, corner_radius=0, fg_color=COLORS["surface"])
        self.header_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.header_frame.grid_columnconfigure(2, weight=1)

        logo = brand_logo(34)
        self._logo_image = ctk.CTkImage(light_image=logo, dark_image=logo, size=(34, 34))
        ctk.CTkLabel(self.header_frame, image=self._logo_image, text="").grid(row=0, column=0, padx=(18, 10), pady=12)
        ctk.CTkLabel(self.header_frame, text="Painter Studio", font=font("display"),
                     text_color=COLORS["text"]).grid(row=0, column=1, sticky="w")
        self.lbl_header_subtitle = ctk.CTkLabel(self.header_frame, text="Desenha suas imagens no Instagram Direct",
                                                font=font("small"), text_color=COLORS["text_faint"])
        self.lbl_header_subtitle.grid(row=0, column=2, sticky="w", padx=14)

        self.connection_pill = StatusPill(self.header_frame, "Celular desconectado · conectar", "off",
                                          command=lambda: self._show_step("device"))
        self.connection_pill.grid(row=0, column=3, padx=(0, 10))
        # Compatibilidade: a lógica antiga se referia ao selo por este nome.
        self.lbl_connection_badge = self.connection_pill

        self.btn_kill = button(self.header_frame, "■  Parar agora", self.action_emergency_kill,
                               variant="danger", size="md", width=130)
        self.btn_kill.grid(row=0, column=4, padx=(0, 18))
        ctk.CTkFrame(self, height=1, corner_radius=0, fg_color=COLORS["border"]).grid(
            row=0, column=0, columnspan=2, sticky="sew")

    # ----------------------------------------------------- coluna de seções
    def _create_sidebar(self):
        self.sidebar_scroll = ctk.CTkScrollableFrame(self, width=SIDEBAR_WIDTH - 24, corner_radius=0,
                                                     fg_color=COLORS["bg"],
                                                     scrollbar_button_color=COLORS["surface_3"],
                                                     scrollbar_button_hover_color=COLORS["surface_4"])
        self.sidebar_scroll.grid(row=1, column=0, sticky="nsew", padx=(10, 0), pady=10)
        self.sections = {}
        for number, (key, title) in enumerate(SECTIONS, start=1):
            section = _Section(self.sidebar_scroll, number, title)
            section.pack(fill="x", pady=(0, 10))
            self.sections[key] = section
        self._build_image_section(self.sections["image"].body)
        self._build_style_section(self.sections["style"].body)
        self._build_device_section(self.sections["device"].body)
        self._build_area_section(self.sections["area"].body)
        self._build_paint_section(self.sections["paint"].body)
        self._on_mode_change("AUTO")

    def _hint(self, master, text):
        label = ctk.CTkLabel(master, text=text, font=font("small"), text_color=COLORS["text_muted"], anchor="w",
                             justify="left", wraplength=SIDEBAR_WIDTH - 70)
        label.pack(fill="x")
        return label

    def _build_image_section(self, body):
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x")
        self.lbl_image_thumb = ctk.CTkLabel(row, text="", width=60, height=60, corner_radius=RADIUS["md"],
                                            fg_color=COLORS["surface_3"])
        self.lbl_image_thumb.pack(side="left", padx=(0, 12))
        info = ctk.CTkFrame(row, fg_color="transparent")
        info.pack(side="left", fill="both", expand=True)
        self.lbl_image_info = ctk.CTkLabel(info, text="Nenhuma imagem escolhida", font=font("small"),
                                           text_color=COLORS["text_muted"], anchor="w", justify="left")
        self.lbl_image_info.pack(fill="x")
        self.btn_load_file = button(info, "Escolher imagem…", self.action_choose_image, "secondary", "sm")
        self.btn_load_file.pack(fill="x", pady=(6, 0))
        demos = ctk.CTkFrame(body, fg_color="transparent")
        demos.pack(fill="x", pady=(10, 0))
        ctk.CTkLabel(demos, text="ou use um exemplo:", font=font("small"), text_color=COLORS["text_faint"]).pack(side="left")
        for label, filename in (("Flores", "flores.jpg"), ("Retrato", "retrato.jpg")):
            button(demos, label, lambda f=filename: self.load_demo(f), "ghost", "sm", width=72).pack(side="left", padx=(8, 0))

    def _build_style_section(self, body):
        self.style_body = body
        self.seg_mode = ctk.StringVar(value="AUTO")
        self.mode_buttons = {}
        grid = ctk.CTkFrame(body, fg_color="transparent")
        grid.pack(fill="x")
        grid.grid_columnconfigure((0, 1), weight=1, uniform="mode")
        for index, (key, title, hint) in enumerate((
            ("AUTO", "Automático", "escolhe sozinho"),
            ("LINHAS", "Traçado colorido", "desenhos e letras"),
            ("COLORIR", "Livro de colorir", "desenhos coloridos e logos"),
            ("CAMADAS", "Ilustração chapada", "fotos e ilustrações"),
            ("TEXTO", "Texto", "escreva com fontes"),
        )):
            tile = ctk.CTkButton(grid, text=f"{title}\n{hint}", command=lambda v=key: self._select_mode(v),
                                 **button_style("tile", "lg"))
            tile.configure(height=54, font=font("small_bold"))
            tile.grid(row=index // 2, column=index % 2, sticky="ew", padx=3, pady=3)
            self.mode_buttons[key] = tile
        self.lbl_auto_desc = ctk.CTkLabel(body, text="", font=font("small"), text_color=COLORS["text_muted"],
                                          wraplength=SIDEBAR_WIDTH - 70, justify="left", anchor="w")
        self.lbl_auto_desc.pack(fill="x", pady=(8, 2))

        # Opções do modo pintura em cores (recolhidas em "Ajustes avançados")
        self.frame_layered_opts = ctk.CTkFrame(body, fg_color="transparent")
        self._advanced_open = False
        self.btn_advanced = button(self.frame_layered_opts, "Ajustes avançados  ▸", self._toggle_advanced, "ghost", "sm")
        self.btn_advanced.pack(fill="x", pady=(6, 0))
        self.frame_advanced = ctk.CTkFrame(self.frame_layered_opts, fg_color="transparent")
        self.lbl_colors = ctk.CTkLabel(self.frame_advanced, text="Paleta de cores: 10 cores", font=font("small_bold"),
                                       text_color=COLORS["text"], anchor="w")
        self.lbl_colors.pack(fill="x", pady=(8, 0))
        self.slider_colors = ctk.CTkSlider(self.frame_advanced, from_=4, to=24, number_of_steps=10,
                                           command=self._on_colors_slider, **slider_style())
        self.slider_colors.set(10)
        self.slider_colors.pack(fill="x", pady=(4, 6))
        for attr, var_attr, text, default, visible in (
            ("chk_semantic", "var_semantic_highlights", "Realçar detalhes escuros (olhos, boca)", True, True),
            ("chk_include_contours", "var_include_contours", "Contorno escuro por cima", False, True),
            # Opções do motor antigo de camadas: mantidas para projetos e MCP, fora da tela
            ("chk_fidelity", "var_fidelity_mode", "Fidelidade estrita", True, False),
            ("chk_artistic", "var_artistic_effects", "Efeitos artísticos", False, False),
            ("chk_sim_opt", "var_sim_opt", "Otimizar na simulação", True, False),
        ):
            var = ctk.BooleanVar(value=default)
            setattr(self, var_attr, var)
            chk = ctk.CTkCheckBox(self.frame_advanced, text=text, variable=var, command=self._mark_plan_dirty,
                                  **checkbox_style())
            if visible:
                chk.pack(anchor="w", pady=3)
            setattr(self, attr, chk)
        self.slider_spacing = ctk.CTkSlider(self.frame_advanced, from_=2, to=6, number_of_steps=4,
                                            command=self._on_spacing_slider, **slider_style())
        self.slider_spacing.set(3)
        self.lbl_spacing = ctk.CTkLabel(self.frame_advanced, text="")

        # Opções do modo Texto (não usa a imagem da seção 1)
        from ..planning.text_engine import available_fonts, CURATED_FONTS
        self.frame_text_opts = ctk.CTkFrame(body, fg_color="transparent")
        ctk.CTkLabel(self.frame_text_opts, text="Texto", font=font("small_bold"), text_color=COLORS["text"],
                     anchor="w").pack(fill="x", pady=(6, 2))
        self.txt_text_input = ctk.CTkTextbox(self.frame_text_opts, height=84, font=font("body"), wrap="word",
                                             fg_color=COLORS["surface_2"], border_color=COLORS["border_strong"],
                                             border_width=1, corner_radius=RADIUS["md"], text_color=COLORS["text"])
        self.txt_text_input.pack(fill="x", pady=(0, 6))
        self.txt_text_input.bind("<KeyRelease>", lambda _e: self._mark_plan_dirty())
        installed = available_fonts()
        font_names = [n for n in CURATED_FONTS if n in installed] +                      sorted(n for n in installed if n not in CURATED_FONTS)
        ctk.CTkLabel(self.frame_text_opts, text="Fonte", font=font("small_bold"), text_color=COLORS["text"],
                     anchor="w").pack(fill="x", pady=(4, 2))
        self.opt_text_font = ctk.CTkOptionMenu(self.frame_text_opts, values=font_names or ["Arial"],
                                               command=lambda _v: self._mark_plan_dirty(), **option_menu_style())
        self.opt_text_font.set(font_names[0] if font_names else "Arial")
        self.opt_text_font.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(self.frame_text_opts, text="Alinhamento", font=font("small_bold"), text_color=COLORS["text"],
                     anchor="w").pack(fill="x", pady=(4, 2))
        self.opt_text_align = ctk.CTkOptionMenu(self.frame_text_opts, values=["Centro", "Esquerda", "Direita"],
                                                command=lambda _v: self._mark_plan_dirty(), **option_menu_style())
        self.opt_text_align.set("Centro")
        self.opt_text_align.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(self.frame_text_opts, text="Traço", font=font("small_bold"), text_color=COLORS["text"],
                     anchor="w").pack(fill="x", pady=(4, 2))
        self.opt_text_stroke = ctk.CTkOptionMenu(self.frame_text_opts,
                                                 values=["Fino (caneta)", "Seguir a espessura da fonte"],
                                                 command=lambda _v: self._mark_plan_dirty(), **option_menu_style())
        self.opt_text_stroke.set("Fino (caneta)")
        self.opt_text_stroke.pack(fill="x", pady=(0, 6))

        # Opções de cor dos modos traçado fino e contornos
        self.frame_line_opts = ctk.CTkFrame(body, fg_color="transparent")
        self.lbl_line_ink = ctk.CTkLabel(self.frame_line_opts, text="Cor do traço", font=font("small_bold"),
                                         text_color=COLORS["text"], anchor="w")
        self.lbl_line_ink.pack(fill="x", pady=(6, 2))
        self.opt_line_ink_color = ctk.CTkOptionMenu(
            self.frame_line_opts,
            values=["🎨 Original (Auto)", "🖤 Preto disponível", "🤍 Branco Puro", "❤️ Vermelho",
                    "💛 Amarelo", "💙 Azul", "💚 Verde", "🌸 Rosa", "🌈 Customizar..."],
            command=self._on_line_ink_color_change, **option_menu_style())
        self.opt_line_ink_color.set("🎨 Original (Auto)")
        self.opt_line_ink_color.pack(fill="x", pady=(0, 6))
        self.lbl_line_bg = ctk.CTkLabel(self.frame_line_opts, text="Fundo", font=font("small_bold"),
                                        text_color=COLORS["text"], anchor="w")
        self.lbl_line_bg.pack(fill="x", pady=(4, 2))
        self.opt_line_bg_color = ctk.CTkOptionMenu(
            self.frame_line_opts,
            values=["🎨 Fundo Original (Auto)", "🚫 Não Pintar (Tela Escura do Celular)",
                    "🤍 Fundo Branco Puro", "🖤 Fundo Preto disponível", "🌈 Customizar..."],
            command=self._on_line_bg_color_change, **option_menu_style())
        self.opt_line_bg_color.set("🎨 Fundo Original (Auto)")
        self.opt_line_bg_color.pack(fill="x", pady=(0, 6))

        self.var_skip_background = ctk.BooleanVar(value=False)
        self.chk_skip_background = ctk.CTkCheckBox(body, text="Não pintar o fundo (usar a tela do celular)",
                                                   variable=self.var_skip_background, command=self._mark_plan_dirty,
                                                   **checkbox_style())
        self.chk_skip_background.pack(fill="x", pady=(6, 0))
        # Marca o fim das opções: _on_mode_change encaixa os blocos do modo antes dela.
        self._style_options_end = ctk.CTkFrame(body, height=1, fg_color="transparent")
        self._style_options_end.pack(fill="x")

        self.btn_preview = ActionButton(body, PREVIEW_BUTTON_TEXT, self.action_generate_preview, "primary", "lg")
        self.btn_preview.pack(fill="x", pady=(12, 0))

    def _toggle_advanced(self):
        self._advanced_open = not self._advanced_open
        if self._advanced_open:
            self.frame_advanced.pack(fill="x")
            self.btn_advanced.configure(text="Ajustes avançados  ▾")
        else:
            self.frame_advanced.pack_forget()
            self.btn_advanced.configure(text="Ajustes avançados  ▸")

    def _build_device_section(self, body):
        self._hint(body, "Ative a depuração por USB ou Wi-Fi no celular e conecte pelo endereço mostrado nele.")
        section_label(body, "Endereço (IP:porta)").pack(fill="x", pady=(10, 4))
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x")
        self.btn_connect = button(row, "Conectar", self.action_connect_adb, "primary", width=100)
        self.btn_connect.pack(side="right")
        self.entry_ip = ctk.CTkEntry(row, placeholder_text="ex.: 192.168.1.56:46247", width=150, **entry_style())
        # O último endereço que conectou vale mais que o padrão do config
        self.entry_ip.insert(0, load_settings().get("last_device_target") or config.DEFAULT_DEVICE_IP)
        self.entry_ip.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.entry_ip.bind("<Return>", lambda _e: self.action_connect_adb())

        section_label(body, "Ou escolha um aparelho encontrado").pack(fill="x", pady=(10, 4))
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x")
        self.btn_scan = button(row, "Buscar", self.action_scan_devices, "ghost", width=78)
        self.btn_scan.pack(side="right")
        self.opt_devices = ctk.CTkOptionMenu(row, values=["Procurando dispositivos..."], width=150,
                                             command=self._on_device_selected, **option_menu_style())
        self.opt_devices.pack(side="left", fill="x", expand=True, padx=(0, 6))

        tools = ctk.CTkFrame(body, fg_color="transparent")
        tools.pack(fill="x", pady=(12, 0))
        tools.grid_columnconfigure((0, 1), weight=1, uniform="tools")
        self.btn_open_canvas = button(tools, "Abrir tela de desenho", self.action_open_canvas, "secondary", "sm")
        self.btn_open_canvas.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.btn_ping = button(tools, "Testar conexão", self.action_ping_adb, "ghost", "sm")
        self.btn_ping.grid(row=0, column=1, sticky="ew", padx=(3, 0))

    def _build_area_section(self, body):
        self.lbl_roi_sidebar_status = ctk.CTkLabel(body, text="Área: Central Padrão (750x950 px)", font=font("subtitle"),
                                                   text_color=COLORS["text"], anchor="w", justify="left",
                                                   wraplength=SIDEBAR_WIDTH - 70)
        self.lbl_roi_sidebar_status.pack(fill="x")
        self._hint(body, "A área padrão serve na maioria dos casos. Para mudar, tire um print do celular e "
                         "arraste um retângulo sobre o canvas.")
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", pady=(10, 0))
        row.grid_columnconfigure((0, 1), weight=1, uniform="roi")
        self.btn_select_roi_side = button(row, "Ajustar no print", self._switch_to_roi_tab, "secondary", "sm")
        self.btn_select_roi_side.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.btn_reset_roi_side = button(row, "Usar padrão", self._reset_roi_selection, "ghost", "sm")
        self.btn_reset_roi_side.grid(row=0, column=1, sticky="ew", padx=(3, 0))

    def _build_paint_section(self, body):
        self._hint(body, "Deixe a tela Desenhar do Instagram aberta e use o botão no rodapé para começar.")
        self.var_verify_layer = ctk.BooleanVar(value=False)
        self.chk_verify_layer = ctk.CTkCheckBox(body, text="Conferir cada camada com um print",
                                                variable=self.var_verify_layer, **checkbox_style())
        self.chk_verify_layer.pack(anchor="w", pady=(10, 3))
        self.var_closed_loop = ctk.BooleanVar(value=False)
        self.chk_closed_loop = ctk.CTkCheckBox(body, text="Auditoria visual ao final",
                                               variable=self.var_closed_loop, **checkbox_style())
        self.chk_closed_loop.pack(anchor="w", pady=3)
        self.lbl_canvas_live = ctk.CTkLabel(body, text="A tela do celular aparece aqui durante a pintura", height=120,
                                            corner_radius=RADIUS["md"], fg_color=COLORS["canvas"],
                                            text_color=COLORS["text_faint"], font=font("small"))
        self.lbl_canvas_live.pack(fill="x", pady=(10, 0))

    def _show_step(self, key):
        """Rola a coluna até a seção e a destaca por um instante."""
        section = getattr(self, "sections", {}).get(key)
        if section is None:
            return
        try:
            self.update_idletasks()
            canvas = self.sidebar_scroll._parent_canvas
            total = max(1, self.sidebar_scroll.winfo_height())
            canvas.yview_moveto(max(0.0, (section.winfo_y() - 8) / total))
        except (tk.TclError, AttributeError):
            pass
        section.highlight(True)
        self.after(1200, lambda: section.highlight(False))
        if key == "device" and hasattr(self, "entry_ip"):
            self.entry_ip.focus_set()

    # --------------------------------------------------------- área central
    def _create_main_preview_area(self):
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color=COLORS["bg"])
        self.main_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_propagate(False)
        self.main_frame.grid_rowconfigure(0, weight=0)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(2, weight=0, minsize=32)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Números do plano, sempre à vista acima da prévia
        stats = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        stats.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.stat_tiles = {}
        specs = (("time", "Tempo estimado"), ("strokes", "Traços"), ("layers", "Camadas"),
                 ("colors", "Cores"), ("delta_e", "Diferença do original"), ("gate", "Qualidade"))
        for index, (key, label) in enumerate(specs):
            stats.grid_columnconfigure(index, weight=1, uniform="stats")
            tile = StatTile(stats, label)
            tile.grid(row=0, column=index, sticky="ew", padx=(0 if index == 0 else 4, 0))
            self.stat_tiles[key] = tile

        self.tabview = ctk.CTkTabview(
            self.main_frame, fg_color=COLORS["surface"], corner_radius=RADIUS["lg"],
            border_width=1, border_color=COLORS["border"],
            segmented_button_fg_color=COLORS["surface_2"],
            segmented_button_selected_color=COLORS["surface_4"],
            segmented_button_selected_hover_color=COLORS["border_strong"],
            segmented_button_unselected_color=COLORS["surface_2"],
            segmented_button_unselected_hover_color=COLORS["surface_3"],
            text_color=COLORS["text"],
            command=lambda: self.after(30, self._refresh_preview_size),
        )
        self.tabview._segmented_button.configure(font=font("small_bold"), height=30)
        self.tabview.grid(row=1, column=0, sticky="nsew")
        self.main_frame.bind("<Configure>", self._on_main_resize)

        self.tab_preview = self.tabview.add("Prévia")
        self.tab_compare = self.tabview.add("Comparar")
        self.tab_palette = self.tabview.add("Cores")
        self.tab_layers = self.tabview.add("Camadas")
        self.tab_error = self.tabview.add("Qualidade")
        self.tab_live = self.tabview.add("Tela do celular")

        self._build_preview_tab(self.tab_preview)
        self._build_compare_tab(self.tab_compare)
        self._build_palette_tab(self.tab_palette)
        self._build_layers_tab(self.tab_layers)
        self._build_quality_tab(self.tab_error)
        self._build_live_tab(self.tab_live)
        self._build_log_panel()

    def _build_preview_tab(self, tab):
        top = ctk.CTkFrame(tab, fg_color="transparent")
        top.pack(fill="x", padx=8, pady=(4, 2))
        self.btn_redo_edit = button(top, "Refazer", self.action_redo_edit, "ghost", "sm", width=76)
        self.btn_redo_edit.pack(side="right", padx=(4, 0))
        self.btn_undo_edit = button(top, "Desfazer", self.action_undo_edit, "ghost", "sm", width=80)
        self.btn_undo_edit.pack(side="right", padx=(4, 0))
        ctk.CTkFrame(top, width=1, height=22, fg_color=COLORS["border_strong"]).pack(side="right", padx=8)
        self.btn_save_sim = button(top, "Exportar PNG", self.action_save_simulation, "ghost", "sm", width=104)
        self.btn_save_sim.pack(side="right", padx=(4, 0))
        self.btn_save_project = button(top, "Salvar", self.action_save_project, "ghost", "sm", width=70)
        self.btn_save_project.pack(side="right", padx=(4, 0))
        self.btn_open_project = button(top, "Abrir", self.action_open_project, "ghost", "sm", width=64)
        self.btn_open_project.pack(side="right", padx=(4, 0))
        self.lbl_preview_mode = ctk.CTkLabel(top, text="Escolha uma imagem e gere a prévia.", font=font("subtitle"),
                                             text_color=COLORS["text_muted"], anchor="w")
        self.lbl_preview_mode.pack(side="left", fill="x", expand=True)

        layers = ctk.CTkFrame(tab, fg_color="transparent")
        layers.pack(fill="x", padx=8, pady=(4, 4))
        self.lbl_layer_slider = ctk.CTkLabel(layers, text="Camadas: concluída", font=font("small_bold"),
                                             text_color=COLORS["text_muted"], width=190, anchor="w")
        self.lbl_layer_slider.pack(side="left")
        self.slider_inspect_layer = ctk.CTkSlider(layers, from_=0, to=1, number_of_steps=1,
                                                  command=self._on_inspect_layer_slide, state="disabled", **slider_style())
        self.slider_inspect_layer.pack(side="left", fill="x", expand=True, padx=(8, 4))
        self.slider_inspect_layer.set(1)

        stage = ctk.CTkFrame(tab, fg_color=COLORS["canvas"], corner_radius=RADIUS["md"])
        stage.pack(fill="both", expand=True, padx=8, pady=4)
        self.lbl_canvas_preview = ctk.CTkLabel(stage, text="Escolha uma imagem na seção 1",
                                               text_color=COLORS["text_faint"], font=font("body"))
        self.lbl_canvas_preview.pack(fill="both", expand=True, padx=6, pady=6)
        self.lbl_canvas_preview.bind("<ButtonPress-1>", self._on_preview_press)
        self.lbl_canvas_preview.bind("<B1-Motion>", self._on_preview_drag)
        self.lbl_canvas_preview.bind("<ButtonRelease-1>", self._on_preview_release)

        bar = ctk.CTkFrame(tab, fg_color=COLORS["surface_2"], corner_radius=RADIUS["md"])
        bar.pack(fill="x", padx=8, pady=(4, 6))
        info = ctk.CTkFrame(bar, fg_color="transparent")
        info.pack(fill="x", padx=10, pady=(8, 4))
        section_label(info, "Editar a prévia").pack(side="left", padx=(0, 10))
        self.selection_swatch = ctk.CTkFrame(info, width=16, height=16, corner_radius=4, fg_color=COLORS["surface_4"])
        self.selection_swatch.pack(side="left", padx=(0, 8))
        self.selection_swatch.pack_propagate(False)
        self.lbl_selected_region = ctk.CTkLabel(info, text="Clique ou arraste sobre a prévia para selecionar uma parte.",
                                                font=font("small"), text_color=COLORS["text_muted"], anchor="w")
        self.lbl_selected_region.pack(side="left", fill="x", expand=True)
        actions = ctk.CTkFrame(bar, fg_color="transparent")
        actions.pack(fill="x", padx=8, pady=(0, 8))
        actions.grid_columnconfigure((0, 1, 2, 3, 4, 5), weight=1, uniform="sel")
        self.btn_region_color = ActionButton(actions, "Trocar cor", self.action_change_region_color, "secondary", "sm")
        self.btn_region_color.grid(row=0, column=0, sticky="ew", padx=2)
        self.btn_physical_color = ActionButton(actions, "Tintas", self.action_choose_physical_color, "secondary", "sm")
        self.btn_physical_color.grid(row=0, column=1, sticky="ew", padx=2)
        self.btn_select_background = button(actions, "Fundo", self.action_select_background, "ghost", "sm")
        self.btn_select_background.grid(row=0, column=2, sticky="ew", padx=2)
        self.btn_clear_selection = button(actions, "Limpar", self.action_clear_selection, "ghost", "sm")
        self.btn_clear_selection.grid(row=0, column=3, sticky="ew", padx=2)
        self.btn_stroke_view = button(actions, "Ver traços", self.action_toggle_stroke_view, "ghost", "sm")
        self.btn_stroke_view.grid(row=0, column=4, sticky="ew", padx=2)
        self.btn_zoom_preview = button(actions, "Ampliar", self.action_zoom_preview, "ghost", "sm")
        self.btn_zoom_preview.grid(row=0, column=5, sticky="ew", padx=2)
        self._show_stroke_view = False

    def _build_compare_tab(self, tab):
        self.lbl_compare_heading = ctk.CTkLabel(tab, text="Original  |  Prévia simulada", font=font("subtitle"),
                                                text_color=COLORS["text_muted"], anchor="w")
        self.lbl_compare_heading.pack(fill="x", padx=10, pady=(6, 0))
        stage = ctk.CTkFrame(tab, fg_color=COLORS["canvas"], corner_radius=RADIUS["md"])
        stage.pack(fill="both", expand=True, padx=8, pady=8)
        self.lbl_canvas_compare = ctk.CTkLabel(stage, text="Aparece depois de gerar a prévia.", text_color=COLORS["text_faint"])
        self.lbl_canvas_compare.pack(fill="both", expand=True, padx=6, pady=6)

    def _build_palette_tab(self, tab):
        self.palette_scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent",
                                                     scrollbar_button_color=COLORS["surface_3"])
        self.palette_scroll.pack(fill="both", expand=True, padx=6, pady=6)
        self.lbl_palette_placeholder = ctk.CTkLabel(self.palette_scroll, text="As cores aparecem depois da prévia.",
                                                    text_color=COLORS["text_faint"])
        self.lbl_palette_placeholder.pack(pady=40)

    def _build_layers_tab(self, tab):
        self._hint(tab, "Ordem em que o celular vai pintar. Clique numa camada para ver a pintura até ela.").pack_configure(
            padx=12, pady=(8, 4))
        self.layer_list = ctk.CTkScrollableFrame(tab, fg_color="transparent", scrollbar_button_color=COLORS["surface_3"])
        self.layer_list.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self._layer_rows = []
        ctk.CTkLabel(self.layer_list, text="Nenhuma camada ainda.", font=font("small"),
                     text_color=COLORS["text_faint"]).pack(pady=20)

    def _build_quality_tab(self, tab):
        top = ctk.CTkFrame(tab, fg_color="transparent")
        top.pack(fill="x", padx=10, pady=(6, 2))
        self.lbl_quality_summary = ctk.CTkLabel(top, text="Qualidade da prévia: aguardando cálculo",
                                                font=font("subtitle"), text_color=COLORS["text_muted"], anchor="w")
        self.lbl_quality_summary.pack(side="left", fill="x", expand=True)
        self.btn_run_bench = button(top, "Benchmark", self.action_run_benchmark, "ghost", "sm", width=104)
        self.btn_run_bench.pack(side="right")
        self.lbl_plan_summary = ctk.CTkLabel(tab, text="Gere a prévia para habilitar o início da pintura.",
                                             font=font("small"), text_color=COLORS["text_muted"], justify="left",
                                             anchor="w")
        self.lbl_plan_summary.pack(fill="x", padx=10, pady=(2, 0))
        self.lbl_error_metrics = ctk.CTkLabel(tab, text="Erro médio, pior faixa e semelhança aparecem aqui.",
                                              font=font("small"), text_color=COLORS["text_muted"], anchor="w")
        self.lbl_error_metrics.pack(fill="x", padx=10)
        stage = ctk.CTkFrame(tab, fg_color=COLORS["canvas"], corner_radius=RADIUS["md"])
        stage.pack(fill="both", expand=True, padx=8, pady=6)
        self.lbl_canvas_error = ctk.CTkLabel(stage, text="O mapa de diferenças aparece depois da prévia.",
                                             text_color=COLORS["text_faint"])
        self.lbl_canvas_error.pack(fill="both", expand=True, padx=6, pady=6)
        self.lbl_heatmap_legend = ctk.CTkLabel(tab, text="Cores mais quentes indicam maior diferença da imagem original.",
                                               font=font("small"), text_color=COLORS["text_faint"], anchor="w")
        self.lbl_heatmap_legend.pack(fill="x", padx=10, pady=(0, 8))

    def _build_live_tab(self, tab):
        top = ctk.CTkFrame(tab, fg_color="transparent")
        top.pack(fill="x", padx=8, pady=(4, 4))
        self.btn_screencap = button(top, "Tirar print do celular", self.action_refresh_screencap, "primary", "sm", width=170)
        self.btn_screencap.pack(side="left", padx=(0, 6))
        self.btn_reset_roi = button(top, "Área padrão", self._reset_roi_selection, "ghost", "sm", width=100)
        self.btn_reset_roi.pack(side="left", padx=4)
        self.btn_test_center = button(top, "Toque de teste", self._test_roi_center_tap, "secondary", "sm", width=120)
        self.btn_test_center.pack(side="left", padx=4)
        self.var_show_overlay = ctk.BooleanVar(value=False)
        self.chk_overlay = ctk.CTkCheckBox(top, text="Marcadores de calibração", variable=self.var_show_overlay,
                                           command=self._on_toggle_overlay, **checkbox_style())
        self.chk_overlay.pack(side="left", padx=12)

        status = ctk.CTkFrame(tab, fg_color=COLORS["surface_2"], corner_radius=RADIUS["md"])
        status.pack(fill="x", padx=8, pady=2)
        self.lbl_roi_coords = ctk.CTkLabel(
            status, text="Tire um print para marcar a área de pintura. Arraste para selecionar; setas ajustam.",
            font=font("small"), text_color=COLORS["info"], anchor="w")
        self.lbl_roi_coords.pack(side="left", padx=10, pady=6)

        nudge = ctk.CTkFrame(tab, fg_color="transparent")
        nudge.pack(fill="x", padx=8, pady=(4, 4))
        section_label(nudge, "Ajuste fino").pack(side="left", padx=(0, 8))
        for attr, text, cb in (("btn_nudge_up", "▲", lambda: self._nudge_roi(0, -20)),
                               ("btn_nudge_down", "▼", lambda: self._nudge_roi(0, 20)),
                               ("btn_nudge_left", "◀", lambda: self._nudge_roi(-20, 0)),
                               ("btn_nudge_right", "▶", lambda: self._nudge_roi(20, 0))):
            widget = button(nudge, text, cb, "ghost", "sm", width=38)
            widget.pack(side="left", padx=2)
            setattr(self, attr, widget)
        self.btn_scale_plus = button(nudge, "+20 px", lambda: self._scale_roi(20), "ghost", "sm", width=64)
        self.btn_scale_plus.pack(side="left", padx=(12, 2))
        self.btn_scale_minus = button(nudge, "−20 px", lambda: self._scale_roi(-20), "ghost", "sm", width=64)
        self.btn_scale_minus.pack(side="left", padx=2)

        self.canvas_roi = tk.Canvas(tab, bg=COLORS["canvas"], highlightthickness=1,
                                    highlightbackground=COLORS["border"], takefocus=1)
        self.canvas_roi.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        self.canvas_roi.create_text(22, 22, anchor="nw", fill=COLORS["text_faint"],
                                    text="Nenhuma captura. Conecte o celular e tire um print.")
        self.canvas_roi.bind("<ButtonPress-1>", self._on_roi_mouse_press)
        self.canvas_roi.bind("<B1-Motion>", self._on_roi_mouse_motion)
        self.canvas_roi.bind("<ButtonRelease-1>", self._on_roi_mouse_release)
        self.canvas_roi.bind("<Left>", lambda e: self._nudge_roi(-20, 0))
        self.canvas_roi.bind("<Right>", lambda e: self._nudge_roi(20, 0))
        self.canvas_roi.bind("<Up>", lambda e: self._nudge_roi(0, -20))
        self.canvas_roi.bind("<Down>", lambda e: self._nudge_roi(0, 20))
        self.canvas_roi.bind("<plus>", lambda e: self._scale_roi(20))
        self.canvas_roi.bind("<KP_Add>", lambda e: self._scale_roi(20))
        self.canvas_roi.bind("<minus>", lambda e: self._scale_roi(-20))
        self.canvas_roi.bind("<KP_Subtract>", lambda e: self._scale_roi(-20))

    def _build_log_panel(self):
        self.f_log_container = ctk.CTkFrame(self.main_frame, fg_color=COLORS["surface"], corner_radius=RADIUS["md"],
                                            border_width=1, border_color=COLORS["border"])
        self.f_log_container.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        self.f_log_container.grid_rowconfigure(1, weight=1)
        self.f_log_container.grid_columnconfigure(0, weight=1)
        head = ctk.CTkFrame(self.f_log_container, fg_color="transparent", height=28)
        head.grid(row=0, column=0, sticky="ew", padx=6, pady=3)
        section_label(head, "Registro de atividades").pack(side="left", padx=6)
        self.btn_toggle_log = button(head, "▲ Expandir Log", self._toggle_log_panel, "ghost", "sm", width=110, height=24)
        self.btn_toggle_log.pack(side="right", padx=2)
        self.btn_clear_log = button(head, "Limpar", self._clear_log, "ghost", "sm", width=64, height=24)
        self.btn_clear_log.pack(side="right", padx=2)
        self.log_box = ctk.CTkTextbox(self.f_log_container, height=110, font=font("small", mono=True),
                                      fg_color=COLORS["surface"], text_color=COLORS["text_muted"], border_width=0)
        self.log_box.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))
        self.log_box.insert("end", "[Sistema] Painter Studio iniciado.\n")
        self.log_box.grid_remove()

    # ------------------------------------------------------ números do plano
    def _refresh_inspector(self, plan=None, metrics=None):
        plan = plan if plan is not None else self.current_plan
        tiles = getattr(self, "stat_tiles", None)
        if not tiles:
            return
        if not plan:
            for tile in tiles.values():
                tile.set("—", "muted")
            self._populate_layer_list([])
            return
        layers = plan.get("layers", [])
        t_est = plan.get("estimated_time_s", 0) or 0
        tiles["layers"].set(str(len(layers)))
        tiles["strokes"].set(f"{plan.get('total_strokes', 0):,}".replace(",", "."))
        tiles["time"].set(f"{int(t_est // 60)} min {int(t_est % 60):02d} s" if t_est >= 60 else f"{int(t_est)} s")
        tiles["colors"].set(str(len(plan.get("palette_colors", []) or [])) or "—")
        # Prefere a comparação com a imagem original (o que o usuário enxerga)
        metrics = plan.get("fidelity_metrics") or metrics or plan.get("error_metrics") or {}
        mean_de = metrics.get("mean_delta_e")
        if isinstance(mean_de, (int, float)):
            tone = "good" if mean_de < 10 else "warn" if mean_de < 20 else "bad"
            word = "baixa" if mean_de < 10 else "média" if mean_de < 20 else "alta"
            tiles["delta_e"].set(f"{word} · {mean_de:.1f}", tone)
        else:
            tiles["delta_e"].set("—", "muted")
        passed = bool(getattr(plan.get("quality_gate"), "passed", False))
        if self._plan_dirty:
            tiles["gate"].set("Desatualizada", "warn")
        else:
            tiles["gate"].set("Aprovada" if passed else "Bloqueada", "good" if passed else "bad")
        self._populate_layer_list(layers)

    def _populate_layer_list(self, layers):
        if not hasattr(self, "layer_list"):
            return
        for child in self.layer_list.winfo_children():
            child.destroy()
        self._layer_rows = []
        if not layers:
            ctk.CTkLabel(self.layer_list, text="Nenhuma camada ainda.", font=font("small"),
                         text_color=COLORS["text_faint"]).pack(pady=20)
            return
        for index, layer in enumerate(layers):
            rgb = layer.get("render_color_rgb") or layer.get("color_rgb") or (200, 200, 200)
            hex_color = "#%02X%02X%02X" % tuple(int(c) for c in rgb[:3])
            row = ctk.CTkFrame(self.layer_list, fg_color=COLORS["surface_2"], corner_radius=RADIUS["sm"], cursor="hand2")
            row.pack(fill="x", pady=2, padx=4)
            swatch = ctk.CTkFrame(row, width=18, height=18, corner_radius=5, fg_color=hex_color,
                                  border_width=1, border_color=COLORS["border_strong"])
            swatch.pack(side="left", padx=(10, 10), pady=8)
            name = str(layer.get("name", f"Camada {index + 1}")).split(": ", 1)[-1]
            label = ctk.CTkLabel(row, text=f"{index + 1}.  {name}", font=font("body"), text_color=COLORS["text"],
                                 anchor="w")
            label.pack(side="left", fill="x", expand=True)
            count = ctk.CTkLabel(row, text=f"{len(layer.get('strokes', []))} traços", font=font("small"),
                                 text_color=COLORS["text_faint"])
            count.pack(side="right", padx=12)
            for widget in (row, swatch, label, count):
                widget.bind("<Button-1>", lambda _e, i=index: self._inspect_layer(i))
            self._layer_rows.append(row)

    def _inspect_layer(self, index):
        if not self.current_plan or not self.current_plan.get("cumulative_layer_images"):
            return
        self.tabview.set("Prévia")
        if str(self.slider_inspect_layer.cget("state")) != "disabled":
            self.slider_inspect_layer.set(index + 1)
        self._on_inspect_layer_slide(index + 1)
        self._highlight_layer(index)

    def _highlight_layer(self, index):
        for i, row in enumerate(getattr(self, "_layer_rows", [])):
            row.configure(fg_color=COLORS["accent_soft"] if i == index else COLORS["surface_2"])

    # ------------------------------------------------------- barra de ação
    def _create_status_bar(self):
        self.status_bar = ctk.CTkFrame(self, height=76, corner_radius=0, fg_color=COLORS["surface"])
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.status_bar.grid_columnconfigure(1, weight=1)
        ctk.CTkFrame(self.status_bar, height=1, corner_radius=0, fg_color=COLORS["border"]).grid(
            row=0, column=0, columnspan=5, sticky="new")

        self.lbl_status = ctk.CTkLabel(self.status_bar, text="Comece escolhendo uma imagem.", font=font("subtitle"),
                                       text_color=COLORS["text"], width=250, anchor="w", wraplength=245, justify="left")
        self.lbl_status.grid(row=0, column=0, rowspan=2, sticky="w", padx=(20, 12), pady=12)

        progress = ctk.CTkFrame(self.status_bar, fg_color="transparent")
        progress.grid(row=0, column=1, rowspan=2, sticky="ew", padx=8, pady=12)
        progress.grid_columnconfigure(0, weight=1)
        self.lbl_progress_text = ctk.CTkLabel(progress, text="Gere uma prévia para começar", font=font("small"),
                                              text_color=COLORS["text_muted"], anchor="w")
        self.lbl_progress_text.grid(row=0, column=0, sticky="ew")
        self.lbl_eta = ctk.CTkLabel(progress, text="", font=font("small_bold"), text_color=COLORS["text_muted"], anchor="e")
        self.lbl_eta.grid(row=0, column=1, sticky="e")
        self.progress_bar = ctk.CTkProgressBar(progress, height=8, corner_radius=4, fg_color=COLORS["surface_3"],
                                               progress_color=COLORS["accent"])
        self.progress_bar.set(0)
        self.progress_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        self.btn_pause = button(self.status_bar, "Pausar", self.action_pause_painting, "ghost", "lg", width=96,
                                state="disabled")
        self.btn_pause.grid(row=0, column=2, rowspan=2, padx=4, pady=14)
        self.btn_resume = button(self.status_bar, "Retomar", self.action_resume_painting, "ghost", "lg", width=100,
                                 state="disabled")
        self.btn_resume.grid(row=0, column=3, rowspan=2, padx=4, pady=14)
        self.btn_start = ActionButton(self.status_bar, "Gere a prévia primeiro", self.action_primary,
                                      "primary", "lg", width=230, state="disabled")
        self.btn_start.grid(row=0, column=4, rowspan=2, padx=(6, 20), pady=14)

    # ----------------------------------------------------------- indicadores
    def _refresh_progress_indicators(self):
        """Estado de cada seção (✓) e o selo de conexão do cabeçalho."""
        sections = getattr(self, "sections", None)
        if not sections:
            return
        has_image = self._has_source()
        plan_ok = bool(self.current_plan) and not self._plan_dirty
        gate_ok = plan_ok and bool(getattr(self.current_plan.get("quality_gate"), "passed", False))
        connected = bool(getattr(self.adb, "is_connected", False))
        painting = self._is_painting()
        if self._is_text_mode():
            sections["image"].set_state(True, "não usada no modo Texto")
        else:
            sections["image"].set_state(has_image, "pronta" if has_image else "escolha uma")
        if plan_ok:
            sections["style"].set_state(True, "prévia pronta" if gate_ok else "prévia bloqueada",
                                        None if gate_ok else "warn")
        else:
            sections["style"].set_state(False, "gere a prévia" if has_image else "")
        sections["device"].set_state(connected, "conectado" if connected else "desconectado")
        custom = bool(getattr(self, "custom_canvas_bounds", None))
        sections["area"].set_state(True, "personalizada" if custom else "padrão")
        if painting:
            sections["paint"].set_state(False, "pintando…", "info")
        else:
            ready = gate_ok and connected
            sections["paint"].set_state(False, "pronto para pintar" if ready else "", "ok" if ready else None)
        self._refresh_connection_pill()

    def _refresh_connection_pill(self):
        if not hasattr(self, "connection_pill"):
            return
        if self._is_painting():
            self.connection_pill.set("Pintando", "busy")
        elif getattr(self.adb, "is_connected", False):
            model = (getattr(self.adb, "device_info", {}) or {}).get("model", "Celular")
            self.connection_pill.set(f"Conectado · {model}", "ok")
        else:
            self.connection_pill.set("Celular desconectado · conectar", "off")

    def _update_responsive_layout(self, width):
        if width < 1300:
            self.lbl_header_subtitle.grid_remove()
        else:
            self.lbl_header_subtitle.grid()

    # ------------------------------------------------------------- atalhos
    def _bind_shortcuts(self):
        def guarded(action):
            def handler(event):
                # Não rouba atalhos de quem está digitando num campo de texto.
                if isinstance(event.widget, (tk.Entry, tk.Text)):
                    return None
                action()
                return "break"
            return handler

        self.bind("<Control-o>", guarded(self.action_choose_image))
        self.bind("<Control-Return>", guarded(self.action_generate_preview))
        self.bind("<Control-z>", guarded(self.action_undo_edit))
        self.bind("<Control-y>", guarded(self.action_redo_edit))
        self.bind("<Control-l>", guarded(self._toggle_log_panel))

    # ----------------------------------------------------------- utilitário
    def _set_mode_tile_selected(self, key, selected):
        tile = self.mode_buttons.get(key)
        if tile is not None:
            tile.configure(**variant_colors("tile_selected" if selected else "tile"))
