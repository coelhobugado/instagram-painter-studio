"""
Configurações, Coordenadas de Interface, Tolerância a Falhas e Parâmetros Globais do Instagram Painter Studio.
Dispositivo Alvo de Referência: Samsung Galaxy A73 5G (1080 x 2400)
"""
import os

# --- ADB e Dispositivo ---
DEFAULT_ADB_PATH = os.path.expandvars(r"%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe")
# Alvo e perfil podem ser trocados sem editar o código (IGP_DEVICE_TARGET / IGP_DEVICE_PROFILE).
# Se o alvo não estiver ativo, connect() usa o único aparelho autorizado no ADB.
DEFAULT_DEVICE_IP = os.environ.get("IGP_DEVICE_TARGET") or ""
DEFAULT_DEVICE_PROFILE = os.environ.get("IGP_DEVICE_PROFILE") or "SM-A736B"
SCREEN_WIDTH = 1080
SCREEN_HEIGHT = 2400

# Parâmetros de Tolerância a Falhas e Resiliência
ADB_CONNECT_TIMEOUT = 10
ADB_COMMAND_TIMEOUT = 15
ADB_MAX_RECONNECT_RETRIES = 3
ADB_RETRY_BACKOFF_SECONDS = 1.5
BATCH_SIZE_DEFAULT = 8          # Micro-lotes de segurança (5 a 10 traços por lote)

# Traços por confirmação (ACK) no ExecutionWorker. Valores maiores reduzem
# round-trips; a pausa passa a valer ao fim do lote (a parada continua imediata).
WORKER_STROKE_BATCH_SIZE = 4
# Respiro entre lotes para o Instagram renderizar (segundos, escalado pelo timing).
WORKER_INTER_BATCH_DELAY_S = 0.04

# Gesto contínuo por traço via processos "input motionevent" separados.
# Testado no SM-A736B (Android 16): o Android descarta os MOVEs após o primeiro,
# então permanece desligado. O caminho contínuo real é o injetor abaixo.
CONTINUOUS_STROKES = os.environ.get("IGP_CONTINUOUS_STROKES", "0") == "1"

# Transporte de toques: "injector" (app_process residente, gesto contínuo e
# cadenciado por quadro, ~35 ms por reta) ou "shell" (input swipe por aresta).
# Se o injetor não iniciar, o ADBManager volta sozinho para "shell".
TOUCH_TRANSPORT = os.environ.get("IGP_TOUCH_TRANSPORT", "injector")
# Avanço máximo por quadro do injetor. O Instagram usa um ponto por quadro e
# suaviza entre eles; acima disso as curvas achatam.
INJECTOR_MAX_STEP_PX = 24.0
# Tempo parado sobre a bolinha até o seletor de nuances abrir.
PALETTE_LONG_PRESS_MS = 700

# Traço fino (~4 px, caneta sólida abaixo do trilho) nas linhas finas dos
# modos de desenho. IGP_FINE_LINES=0 volta ao pincel mínimo do trilho (~13 px).
FINE_LINES = os.environ.get("IGP_FINE_LINES", "1") != "0"

# --- Área Central Segura do Canvas (Doodles do Instagram Direct) ---
CANVAS_SAFE_CX = 540
CANVAS_SAFE_CY = 1180
CANVAS_SAFE_MAX_W = 800
CANVAS_SAFE_MAX_H = 1200

CANVAS_X_MIN = 80
CANVAS_X_MAX = 1000
CANVAS_Y_MIN = 380
CANVAS_Y_MAX = 1980

# --- Coordenadas da Interface do Instagram (UI Coordinates - Direct Doodles) ---
UI_CHAT_PLUS_BUTTON = (988, 2189)    # Botão '+' de anexos do direct
UI_CHAT_DRAW_BUTTON = (811, 2060)    # Botão 'Desenhar' no menu que sobe

# Ferramentas da barra flutuante superior (Y ≈ 318 no Direct)
UI_PEN_SOLID = (287, 318)            # Canetinha comum / sólida (opaca)
UI_PEN_SOLID_ALT = (287, 318)        # Alternativa direta
UI_PEN_ARROW = (413, 318)            # Caneta de seta
UI_PEN_MARKER = (539, 318)           # Marcador / Marca-texto translúcido
UI_PEN_NEON = (665, 318)             # Caneta neon
UI_PEN_ERASER = (791, 318)           # Borracha

# Controles do rodapé (Y ≈ 2193)
UI_BUTTON_CLOSE = (64, 2193)         # Fechar / Cancelar (X)
UI_BUTTON_UNDO = (199, 2193)         # Desfazer traço (Undo)
UI_BUTTON_EYEDROPPER = (317, 2191)   # Conta-gotas
UI_BUTTON_DONE = (1028, 2194)        # Concluir / Enviar (Aviãozinho)

# Slider lateral de espessura (X ≈ 39)
UI_SLIDER_X = 39
UI_SLIDER_MIN_Y_START = 1450
UI_SLIDER_MIN_Y_END = 1520           # Fim do trilho (igual a profiles/SM-A736B/ui_geometry.json)
UI_SLIDER_MIN_Y = 1496               # Base do trilho = espessura fina (~10 px)
UI_SLIDER_MED_Y = 1200               # Meio do trilho = espessura média (~58 px)
UI_SLIDER_THICK_Y = 900              # Topo do trilho = espessura grossa (~104 px)
# Pontas do trilho, de onde o injetor começa o arrasto até o nível desejado.
UI_SLIDER_TRACK_TOP_Y = 880
UI_SLIDER_TRACK_BOTTOM_Y = 1500
# Distância aceita entre a bolinha e o alvo antes de corrigir o arrasto.
UI_SLIDER_TOLERANCE_PX = 10

# Larguras de traço físicas calibradas em pixels da tela (Seção 11 do Plano de Fidelidade Máxima)
BRUSH_WIDTHS = {
    "min": 10.0,
    "med": 58.0,
    "thick": 104.0,
    "max": 104.0
}

# Paleta inferior e seletor de espectro
UI_PALETTE_BASE_Y = 2184             # Linha dos círculos de cores padrão
UI_COLOR_WHITE = (411, 2184)         # Bolinha da cor branca
# Indicador de página da paleta (três pontos sob as cores; o ativo é branco).
UI_PALETTE_PAGE_DOTS = ((652, 2254), (674, 2254), (696, 2254))
UI_PALETTE_RESET_SWIPE = ((450, 2184), (800, 2184))  # Volta a paleta para a página 1
UI_PALETTE_NEXT_SWIPE = ((850, 2184), (500, 2184))   # Avança uma página da paleta
UI_PALETTE_DARK_SWATCH = (936, 2184)                 # Amostra mais escura (página 2)
UI_SPECTRUM_POPUP_Y = 1980           # Altura média onde o espectro contínuo abre
UI_SPECTRUM_POPUP_Y_TOP = 1920       # Altura para nuances mais claras / iluminadas
UI_SPECTRUM_POPUP_Y_BOTTOM = 2040    # Altura para nuances mais escuras / sombreadas

# Coordenadas das 8 cores visíveis da paleta base do Direct
PALETTE_BASE_COLORS = [
    {"name": "Branco Puro", "x": 411, "y": 2184, "rgb": (255, 255, 255)},
    {"name": "Azul Cobalto", "x": 486, "y": 2184, "rgb": (56, 151, 240)},
    {"name": "Verde Folha", "x": 561, "y": 2184, "rgb": (112, 192, 80)},
    {"name": "Amarelo Ouro", "x": 636, "y": 2184, "rgb": (255, 200, 64)},
    {"name": "Laranja Solar", "x": 711, "y": 2184, "rgb": (253, 141, 50)},
    {"name": "Vermelho Intenso", "x": 786, "y": 2184, "rgb": (237, 73, 86)},
    {"name": "Rosa Magenta", "x": 861, "y": 2184, "rgb": (209, 8, 105)},
    {"name": "Roxo Violeta", "x": 936, "y": 2184, "rgb": (163, 7, 186)},
]

# Configurações dos Perfis de Pintura
PROFILE_STENCIL = "stencil"
PROFILE_BALANCED = "balanced"
PROFILE_HIGH = "high"
PROFILE_ULTRA = "ultra"

QUALITY_PROFILES = {
    PROFILE_STENCIL: {
        "name": "Stencil Rápido (Linhas de Alto Contraste)",
        "num_colors": 1,
        "brush_spacing": 2,
        "min_area": 10,
        "description": "Ideal para logos, flores, retratos estilo mangá e desenhos rápidos em 30s."
    },
    PROFILE_BALANCED: {
        "name": "Pintura Balanceada (8 Cores)",
        "num_colors": 8,
        "brush_spacing": 6,
        "min_area": 35,
        "description": "Retratos com pele, sombras e volume facial. Tempo médio de 2 a 3 minutos."
    },
    PROFILE_HIGH: {
        "name": "Pintura Detalhada (16 Cores)",
        "num_colors": 16,
        "brush_spacing": 5,
        "min_area": 25,
        "description": "Rica em meios-tons, luzes e degradês. Tempo médio de 4 a 6 minutos."
    },
    PROFILE_ULTRA: {
        "name": "Pintura Artística Ultra (24 Cores)",
        "num_colors": 24,
        "brush_spacing": 4,
        "min_area": 15,
        "description": "Fidelidade máxima com micro-pinceladas e texturas. Tempo médio de 7 a 10 minutos."
    }
}
