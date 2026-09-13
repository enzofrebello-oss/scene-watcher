"""Parâmetros padrão do Scene Watcher.

Todo valor ajustável do sistema fica aqui. Estes são apenas os padrões iniciais:
a CLI pode sobrescrevê-los na execução, e na interface gráfica eles podem ser
alterados em tempo real e persistidos entre sessões.
"""

from pathlib import Path

# --- Caminhos ---------------------------------------------------------------
# Todos os caminhos são absolutos e derivados da localização deste arquivo, para
# que o app funcione em qualquer pasta, inclusive com caracteres não-ASCII.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = DATA_DIR / "logs"
CLIPS_DIR = DATA_DIR / "clips"
SESSION_LOG_PATH = LOGS_DIR / "session_log.jsonl"
SETTINGS_PATH = DATA_DIR / "settings.json"
MODELS_DIR = BASE_DIR / "models"

# --- Câmera -----------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
# Tempo sem nenhum frame válido a partir do qual a câmera é considerada
# desconectada. Falhas de leitura isoladas dentro desse intervalo são toleradas.
CAMERA_DISCONNECT_TIMEOUT_SECONDS = 2.0
# Espelha a imagem da câmera na horizontal logo após a captura, antes de qualquer
# detecção, como num espelho: a mão esquerda levantada aparece do lado esquerdo da tela.
CAMERA_MIRROR = True

# --- Detecção de objetos (YOLOv8n) -----------------------------------------
YOLO_MODEL_PATH = MODELS_DIR / "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.5
# Os detectores de objetos (YOLOv8n) e de rosto (YuNet) rodam a cada N frames novos;
# nos intermediários o último resultado é reaproveitado. A detecção de movimento roda
# em todo frame novo.
DETECTION_FRAME_INTERVAL = 5

# --- FPS --------------------------------------------------------------------
# Quantidade de frames recentes na média móvel do FPS (frames novos processados por
# segundo).
FPS_AVERAGE_WINDOW_FRAMES = 30

# --- Detecção de rosto (YuNet, OpenCV) -------------------------------------
YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
# Tempo máximo sem resposta do servidor nos downloads de modelo por URL (YuNet).
MODEL_DOWNLOAD_TIMEOUT_SECONDS = 60
# Limiar próprio do rosto, separado do limiar de confiança dos objetos.
FACE_SCORE_THRESHOLD = 0.8
# Largura para a qual o frame é reduzido antes da detecção de rosto; as caixas voltam
# para as coordenadas do frame original. Menor é mais rápido, mas rostos distantes
# são os primeiros a deixar de ser detectados.
FACE_DETECTION_INPUT_WIDTH = 640
# Valores de referência do YuNet (OpenCV Zoo): limiar de sobreposição para descartar
# caixas duplicadas e número máximo de candidatos considerados.
FACE_NMS_THRESHOLD = 0.3
FACE_TOP_K = 5000

# --- Detecção de movimento --------------------------------------------------
# Frames novos usados para aprender o fundo da cena antes de reportar movimento.
MOTION_WARMUP_FRAMES = 30
# Sensibilidade de 0 a 1, ligada ao limiar de variação do MOG2 (varThreshold) em
# escala logarítmica entre os dois extremos abaixo: 0 → 64, 0.5 → 16 (padrão do
# OpenCV), 1 → 4. Mais sensível significa limiar menor: mudanças menores contam
# como movimento.
MOTION_SENSITIVITY = 0.5
MOTION_VAR_THRESHOLD_AT_MIN_SENSITIVITY = 64.0
MOTION_VAR_THRESHOLD_AT_MAX_SENSITIVITY = 4.0
# Área mínima de uma região de movimento, como fração da área do frame.
MOTION_MIN_AREA_FRACTION = 0.001
# Largura para a qual o frame é reduzido antes da detecção de movimento; as caixas
# voltam para as coordenadas do frame original.
MOTION_PROCESSING_WIDTH = 640
# Quantidade de frames que o MOG2 considera ao modelar o fundo (padrão do OpenCV).
# Com cerca de 16 frames novos por segundo, algo parado por cerca de 30 s passa a
# fazer parte do fundo.
MOTION_HISTORY = 500
# Limpeza da máscara de movimento: a erosão remove ruído isolado e a dilatação junta
# partes do mesmo objeto em movimento.
MOTION_KERNEL_SIZE = 3
MOTION_ERODE_ITERATIONS = 1
MOTION_DILATE_ITERATIONS = 2
# Fração do frame em movimento simultâneo a partir da qual a mudança é tratada
# como variação brusca de iluminação e o modelo de fundo é reiniciado.
LIGHTING_CHANGE_FRACTION = 0.8

# --- Rastreamento -----------------------------------------------------------
# Tolerância sem redetecção antes de remover um objeto rastreado, contada em ciclos de
# detecção, não em frames: com DETECTION_FRAME_INTERVAL = 5 e cerca de 16 frames novos
# por segundo, 10 ciclos são cerca de 3 s.
TRACKER_MAX_MISSED = 10
# IoU mínimo para associar uma detecção a um objeto rastreado da mesma classe.
TRACKER_IOU_THRESHOLD = 0.3

# --- Overlay ----------------------------------------------------------------
# O overlay é desenhado numa cópia do frame (src/overlay_renderer.py). Os tamanhos em
# pixels abaixo valem para um frame com OVERLAY_REFERENCE_HEIGHT de altura e são
# escalados proporcionalmente em outras resoluções.
OVERLAY_REFERENCE_HEIGHT = 720
# Os tamanhos de fonte valem na medida da fonte de interface; a fonte monoespaçada é
# ajustada para ter algarismos da mesma altura.
# Classes rastreadas normalmente, mas sem caixa, rótulo nem linha no painel do overlay.
# Log, eventos e alertas continuam vendo essas classes.
OVERLAY_HIDDEN_CLASSES = frozenset({"person"})
# Cores do design da interface, em hex RGB.
COLOR_BACKGROUND = "#0F1417"
COLOR_TEXT = "#E8ECEE"
COLOR_ACCENT = "#4FD1C5"
# Cores das caixas, escolhidas pelo crc32 do nome da classe. Ficam de fora tons próximos
# do destaque (#4FD1C5), do alerta (#F0A868) e do erro (#E06C75), para uma caixa não
# parecer instrumento ou aviso, e tons próximos entre si, para as cores continuarem
# distinguíveis sobre o vídeo.
OVERLAY_CLASS_PALETTE = [
    "#FFE14A",  # amarelo
    "#CFF25A",  # lima
    "#7ED957",  # verde
    "#34D17A",  # esmeralda
    "#4FB8FF",  # azul-céu
    "#7A9CFF",  # azul
    "#A08CFF",  # lavanda
    "#C77DFF",  # violeta
    "#EE7AE9",  # orquídea
]
# Caixas de objetos e rostos e seus rótulos (classe traduzida e confiança).
OVERLAY_BOX_THICKNESS = 2
OVERLAY_LABEL_FONT_SIZE = 18
OVERLAY_LABEL_PADDING = 4
OVERLAY_LABEL_GAP = 5  # entre o nome da classe e a confiança
# Caixas de movimento: tracejado neutro (COLOR_TEXT) e translúcido, sem rótulo.
OVERLAY_MOTION_THICKNESS = 1
OVERLAY_MOTION_DASH_LENGTH = 10
OVERLAY_MOTION_GAP_LENGTH = 6
OVERLAY_MOTION_OPACITY = 0.6
# Instrumentos: contador técnico (canto superior esquerdo) e painel de objetos (canto
# superior direito), em fonte monoespaçada na cor de destaque. São mais discretos que as
# caixas: fonte menor, texto levemente translúcido e fundo translúcido.
OVERLAY_INSTRUMENT_FONT_SIZE = 15
OVERLAY_INSTRUMENT_PADDING = 8
OVERLAY_INSTRUMENT_LINE_SPACING = 2
OVERLAY_INSTRUMENT_MARGIN = 28  # distância das bordas do frame, por dentro dos cantos de viewfinder
OVERLAY_INSTRUMENT_TEXT_OPACITY = 0.85
OVERLAY_INSTRUMENT_BACKGROUND_OPACITY = 0.55
# Cantos de viewfinder nos quatro cantos do vídeo, desenhados por trás de tudo.
OVERLAY_CORNER_LENGTH_FRACTION = 0.04  # comprimento de cada traço, em fração da menor dimensão do frame
OVERLAY_CORNER_THICKNESS = 2
OVERLAY_CORNER_MARGIN = 16
OVERLAY_CORNER_OPACITY = 0.5

# --- Log de sessão ----------------------------------------------------------
LOG_INTERVAL_SECONDS = 2.0

# --- Tipografia do overlay --------------------------------------------------
# O texto sobre o vídeo é desenhado com cv2.FontFace (OpenCV 5). Nomes de arquivo
# TTF em ordem de preferência; o primeiro encontrado nas pastas de fontes do
# sistema é usado. Se nenhum carregar, usa-se a fonte embutida do OpenCV.
# Atenção: cv2.FontFace não gera erro quando falha (arquivo inexistente ou
# caminho com acento), apenas cai para "sans" com getName() vazio.
FONT_DIRS = [
    Path.home() / "AppData" / "Local" / "Microsoft" / "Windows" / "Fonts",
    Path("C:/Windows/Fonts"),
]
UI_FONT_CANDIDATES = ["Inter-Regular.ttf", "Inter.ttf", "segoeui.ttf"]
MONO_FONT_CANDIDATES = ["IBMPlexMono-Regular.ttf", "consola.ttf"]
BUILTIN_FONT_FALLBACK = "sans"
