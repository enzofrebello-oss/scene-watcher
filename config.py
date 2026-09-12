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

# --- Detecção de objetos (YOLOv8n) -----------------------------------------
YOLO_MODEL_PATH = MODELS_DIR / "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.5
# O detector de objetos roda a cada N frames; nos intermediários o último
# resultado é reaproveitado.
DETECTION_FRAME_INTERVAL = 5

# --- Detecção de rosto (YuNet, OpenCV) -------------------------------------
YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

# --- Detecção de movimento --------------------------------------------------
# Frames usados para aprender o fundo da cena antes de reportar movimento.
MOTION_WARMUP_FRAMES = 30
# Fração do frame em movimento simultâneo a partir da qual a mudança é tratada
# como variação brusca de iluminação e o modelo de fundo é reiniciado.
LIGHTING_CHANGE_FRACTION = 0.8

# --- Rastreamento -----------------------------------------------------------
# Tolerância sem redetecção antes de remover um objeto rastreado.
TRACKER_MAX_MISSED = 10

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
