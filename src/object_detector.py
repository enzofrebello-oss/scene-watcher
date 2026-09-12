"""Detecção de objetos com YOLOv8n pré-treinado nas 80 classes do COCO."""

import shutil
import tempfile
from pathlib import Path
from typing import NamedTuple

from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset

import config


class Detection(NamedTuple):
    class_name: str  # nome da classe como o COCO define, ex. "cell phone"
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2) em pixels do frame


class ModelUnavailableError(RuntimeError):
    """O arquivo de pesos não existe e não pôde ser baixado."""


def _download_from_ultralytics(destination):
    """Baixa um peso oficial do Ultralytics e devolve onde ele ficou.

    O Ultralytics pode devolver uma cópia já existente na pasta de pesos global
    dele em vez de baixar para `destination`.
    """
    return Path(attempt_download_asset(str(destination)))


def ensure_weights(path, download=_download_from_ultralytics):
    """Garante que o arquivo de pesos existe em `path`, baixando na primeira execução.

    O download é feito numa pasta temporária e o arquivo só é movido para o destino
    quando completo, para que um download interrompido não deixe um peso corrompido
    em `models/`.
    """
    path = Path(path)
    if path.is_file():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="scene_watcher_model_"))
    try:
        tmp_file = tmp_dir / path.name
        downloaded = download(tmp_file)
        if downloaded is not None and Path(downloaded) != tmp_file:
            shutil.copy2(downloaded, tmp_file)
        if not tmp_file.is_file() or tmp_file.stat().st_size == 0:
            raise RuntimeError("o download terminou sem gerar o arquivo")
        shutil.move(str(tmp_file), str(path))
    except Exception as e:
        raise ModelUnavailableError(
            f"modelo {path.name} não encontrado em {path.parent} e o download automático "
            f"falhou, verifique a conexão com a internet ({e})"
        ) from e
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return path


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
        weights = ensure_weights(model_path)
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
