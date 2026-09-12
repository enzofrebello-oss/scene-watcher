"""Detecção de objetos com YOLOv8n pré-treinado nas 80 classes do COCO."""

from pathlib import Path
from typing import NamedTuple

import cv2
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset
from ultralytics.utils.events import events as ultralytics_events

import config
from src.utils.model_files import ensure_weights

# O import do ultralytics chama cv2.setNumThreads(0), deixando o OpenCV com uma única
# thread no processo inteiro, para evitar conflito com o DataLoader do PyTorch no
# treino. Aqui só há inferência, e com uma thread o OpenCV (YuNet, detecção de
# movimento) fica 2 a 3 vezes mais lento. Restaurar o padrão do sistema (-1), em vez
# de um valor lido antes do import, funciona mesmo que outro módulo tenha importado
# o ultralytics primeiro.
cv2.setNumThreads(-1)

# O ultralytics envia estatísticas anônimas de uso ao Google Analytics ao fim de cada
# predict quando a configuração global sync está ativa, que é o padrão. O envio é
# desligado só dentro deste processo, sem alterar as configurações do ultralytics na
# máquina.
ultralytics_events.enabled = False


class Detection(NamedTuple):
    class_name: str  # identificador estável: nome do COCO (ex. "cell phone") ou "face"
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2) em pixels do frame


def _download_from_ultralytics(destination):
    """Baixa um peso oficial do Ultralytics e devolve onde ele ficou.

    O Ultralytics pode devolver uma cópia já existente na pasta de pesos global
    dele em vez de baixar para `destination`.
    """
    return Path(attempt_download_asset(str(destination)))


class ObjectDetector:
    """Detecta objetos num frame BGR e devolve só os que atingem o limiar de confiança.

    `confidence_threshold` pode ser alterado a qualquer momento e vale a partir da
    próxima chamada a `detect`, sem recarregar o modelo.
    """

    def __init__(
        self,
        model_path=config.YOLO_MODEL_PATH,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
    ):
        weights = ensure_weights(model_path, download=_download_from_ultralytics)
        self._model = YOLO(str(weights))
        self.class_names = dict(self._model.names)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame):
        threshold = self.confidence_threshold
        result = self._model.predict(frame, conf=threshold, verbose=False)[0]
        boxes = result.boxes
        detections = []
        for xyxy, confidence, class_id in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
            if confidence < threshold:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
            detections.append(Detection(self.class_names[int(class_id)], confidence, (x1, y1, x2, y2)))
        return detections
