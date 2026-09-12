"""Detecção de rostos com YuNet (OpenCV), complementar ao YOLOv8n, que não tem classe de rosto."""

import cv2
import numpy as np

import config
from src.object_detector import Detection
from src.utils.model_files import ensure_weights, url_downloader
from src.utils.opencv_logging import opencv_warnings_suppressed

FACE_CLASS_NAME = "face"


class FaceDetector:
    """Detecta rostos num frame BGR e devolve só os que atingem o limiar de score.

    O frame é reduzido para `input_width` pixels de largura antes da detecção (frames
    mais estreitos não são ampliados) e as caixas voltam para as coordenadas do frame
    original. Os pontos faciais que o YuNet fornece são descartados.

    `score_threshold` pode ser alterado a qualquer momento e vale a partir da próxima
    chamada a `detect`, sem recarregar o modelo.
    """

    def __init__(
        self,
        model_path=config.YUNET_MODEL_PATH,
        score_threshold=config.FACE_SCORE_THRESHOLD,
        input_width=config.FACE_DETECTION_INPUT_WIDTH,
    ):
        weights = ensure_weights(model_path, download=url_downloader(config.YUNET_MODEL_URL))
        # Lido pelo Python e entregue como buffer: o OpenCV não abre arquivos ONNX em
        # caminhos com acento.
        model = np.fromfile(str(weights), np.uint8)
        self.input_width = input_width
        self._input_size = (input_width, input_width)
        self._score_threshold = score_threshold
        # O OpenCV 5 avisa na criação que o novo motor de grafos não suporta escolher o
        # dispositivo de execução; é inofensivo, a detecção roda na CPU.
        with opencv_warnings_suppressed():
            self._detector = cv2.FaceDetectorYN.create(
                "onnx",
                model,
                np.array([], np.uint8),
                self._input_size,
                score_threshold,
                config.FACE_NMS_THRESHOLD,
                config.FACE_TOP_K,
            )

    @property
    def score_threshold(self):
        return self._score_threshold

    @score_threshold.setter
    def score_threshold(self, value):
        self._score_threshold = value
        self._detector.setScoreThreshold(value)

    def detect(self, frame):
        height, width = frame.shape[:2]
        if width > self.input_width:
            size = (self.input_width, max(1, round(height * self.input_width / width)))
            image = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        else:
            size = (width, height)
            image = frame
        if size != self._input_size:
            self._detector.setInputSize(size)
            self._input_size = size

        _, faces = self._detector.detect(image)
        if faces is None:
            return []

        scale_x, scale_y = width / size[0], height / size[1]
        threshold = self._score_threshold
        detections = []
        for face in faces:
            score = float(face[14])
            if score < threshold:
                continue
            x, y, w, h = (float(v) for v in face[:4])
            # O YuNet pode devolver caixas que ultrapassam a borda da imagem.
            x1 = max(0, round(x * scale_x))
            y1 = max(0, round(y * scale_y))
            x2 = min(width, round((x + w) * scale_x))
            y2 = min(height, round((y + h) * scale_y))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(Detection(FACE_CLASS_NAME, score, (x1, y1, x2, y2)))
        return detections
