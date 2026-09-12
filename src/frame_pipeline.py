"""Pipeline de processamento de frames, sem nenhuma dependência de Qt.

É usado diretamente pelo modo terminal e, na interface gráfica, envolvido por
uma QThread em `src/ui/pipeline_worker.py`.
"""

import sys
import time
from collections import deque
from typing import NamedTuple

import cv2
import numpy as np

import config
from src.face_detector import FaceDetector
from src.motion_detector import MotionDetector
from src.object_detector import ObjectDetector
from src.utils.opencv_logging import opencv_warnings_suppressed

# Passo da amostra esparsa de pixels comparada antes do frame inteiro ao procurar
# frames repetidos; a comparação completa só acontece se a amostra for igual.
_REPEAT_CHECK_STRIDE = 16


def _open_capture(index):
    """Abre a câmera pelo backend adequado ao sistema.

    No Windows força o Media Foundation (MSMF), que já é o backend que o OpenCV
    escolheria. Sem forçar, quando a câmera não abre, o OpenCV tenta outros backends
    (FFMPEG, obsensor) e imprime avisos enganosos. MSMF e DirectShow entregam a mesma
    quantidade de frames novos por segundo; o MSMF apenas devolve o último frame de
    novo quando a câmera ainda não entregou um novo, e esses repetidos são descartados
    em `CameraSource.read()`.
    """
    backend = cv2.CAP_MSMF if sys.platform == "win32" else cv2.CAP_ANY
    return cv2.VideoCapture(index, backend)


class CameraOpenError(RuntimeError):
    """A câmera não pôde ser aberta."""


class CameraDisconnectedError(RuntimeError):
    """A câmera parou de responder por mais tempo que a tolerância."""


class CameraSource:
    """Fonte de vídeo a partir de uma câmera local.

    `read()` só devolve frames novos: devolve `None` quando a leitura falha ou quando
    o backend repete o frame anterior. Assim, "frame" significa sempre "frame novo"
    no resto do sistema, independente do backend.

    A câmera só é considerada desconectada quando nenhuma leitura é bem-sucedida por
    `disconnect_timeout` segundos. Frames repetidos contam como leitura bem-sucedida,
    então uma cena totalmente estática (ex. lente tampada, com frames idênticos) não
    é tratada como desconexão.
    """

    def __init__(
        self,
        index=config.CAMERA_INDEX,
        width=config.FRAME_WIDTH,
        height=config.FRAME_HEIGHT,
        disconnect_timeout=config.CAMERA_DISCONNECT_TIMEOUT_SECONDS,
        capture_factory=_open_capture,
        clock=time.monotonic,
    ):
        self.index = index
        self.width = width
        self.height = height
        self.disconnect_timeout = disconnect_timeout
        self._capture_factory = capture_factory
        self._clock = clock
        self._cap = None
        self._last_read_time = None
        self._previous_frame = None

    @property
    def is_open(self):
        return self._cap is not None

    @property
    def frame_size(self):
        """Resolução realmente entregue pela câmera, que pode diferir da pedida."""
        if self._cap is None:
            return None
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def open(self):
        self.release()
        # Quando a câmera não abre, o MSMF avisa que não consegue capturar pelo índice;
        # a CameraOpenError já explica o problema.
        with opencv_warnings_suppressed():
            cap = self._capture_factory(self.index)
            opened = cap.isOpened()
            if not opened:
                cap.release()
        if not opened:
            raise CameraOpenError(
                f"não foi possível abrir a câmera {self.index}, verifique se ela "
                "está conectada e se não está em uso por outro aplicativo"
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap
        self._last_read_time = self._clock()
        self._previous_frame = None

    def read(self):
        """Devolve o próximo frame novo, ou `None` se a leitura falhou ou repetiu o anterior.

        Levanta `CameraDisconnectedError` quando nenhuma leitura é bem-sucedida dentro
        da tolerância.
        """
        if self._cap is None:
            raise RuntimeError("a câmera precisa ser aberta antes da leitura")
        # Com a câmera desconectada, o MSMF emite "can't grab frame" várias vezes por
        # segundo até a tolerância estourar; a CameraDisconnectedError já explica.
        with opencv_warnings_suppressed():
            ok, frame = self._cap.read()
        now = self._clock()
        if not ok or frame is None:
            if now - self._last_read_time >= self.disconnect_timeout:
                raise CameraDisconnectedError("câmera desconectada, verifique a conexão")
            return None
        self._last_read_time = now
        if self._is_repeat(frame):
            return None
        return frame

    def _is_repeat(self, frame):
        """Diz se `frame` repete o último frame novo; se não repetir, guarda uma cópia dele.

        A cópia protege contra quem desenha sobre o frame devolvido: a comparação
        seguinte continua sendo com a imagem original da câmera.
        """
        previous = self._previous_frame
        if previous is not None and previous.shape == frame.shape:
            step = _REPEAT_CHECK_STRIDE
            if np.array_equal(frame[::step, ::step], previous[::step, ::step]) and np.array_equal(frame, previous):
                return True
            np.copyto(previous, frame)
        else:
            self._previous_frame = frame.copy()
        return False

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._previous_frame = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()


class FrameResult(NamedTuple):
    frame: np.ndarray  # frame novo da câmera, sem nada desenhado
    frame_index: int  # posição do frame desde a abertura da câmera, a partir de 0
    motion_boxes: list  # caixas (x1, y1, x2, y2) das regiões em movimento
    detections: list  # objetos (YOLOv8n) e rostos (YuNet), como Detection
    detections_updated: bool  # True se os detectores de objetos e rostos rodaram neste frame
    is_calibrating: bool  # True enquanto a detecção de movimento aprende o fundo
    fps: float | None  # frames novos processados por segundo, com média móvel


class FramePipeline:
    """Câmera, detecção de movimento em todo frame novo e de objetos e rostos a cada N frames.

    O YOLOv8n custa bem mais que a detecção de movimento, então objetos e rostos só são
    detectados a cada `detection_interval` frames; nos intermediários, as últimas
    detecções são reaproveitadas.
    """

    def __init__(
        self,
        source=None,
        object_detector=None,
        face_detector=None,
        motion_detector=None,
        detection_interval=config.DETECTION_FRAME_INTERVAL,
        fps_window=config.FPS_AVERAGE_WINDOW_FRAMES,
        clock=time.perf_counter,
    ):
        if detection_interval < 1:
            raise ValueError(f"o intervalo de detecção deve ser pelo menos 1, recebido {detection_interval}")
        self.source = source if source is not None else CameraSource()
        self.object_detector = object_detector if object_detector is not None else ObjectDetector()
        self.face_detector = face_detector if face_detector is not None else FaceDetector()
        self.motion_detector = motion_detector if motion_detector is not None else MotionDetector()
        self.detection_interval = detection_interval
        self._clock = clock
        self._frame_times = deque(maxlen=fps_window)
        self._frame_index = 0
        self._detections = []

    @property
    def fps(self):
        """Frames novos processados por segundo nos últimos frames, ou `None` com menos de dois."""
        if len(self._frame_times) < 2:
            return None
        elapsed = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / elapsed if elapsed > 0 else None

    def warm_up(self):
        """Roda os detectores uma vez num frame vazio, antes de abrir a câmera.

        A primeira inferência de cada modelo é bem mais lenta que as seguintes; assim o
        primeiro frame real não paga essa inicialização.
        """
        blank = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), np.uint8)
        self.object_detector.detect(blank)
        self.face_detector.detect(blank)

    def open(self):
        self.source.open()
        self._frame_index = 0
        self._detections = []
        self._frame_times.clear()

    def close(self):
        self.source.release()

    def process_next(self):
        """Lê e processa o próximo frame novo; devolve `FrameResult`, ou `None` se ainda não chegou um.

        Repassa `CameraDisconnectedError` da câmera.
        """
        frame = self.source.read()
        if frame is None:
            return None
        motion_boxes = self.motion_detector.detect(frame)
        detections_updated = self._frame_index % self.detection_interval == 0
        if detections_updated:
            self._detections = self.object_detector.detect(frame) + self.face_detector.detect(frame)
        self._frame_times.append(self._clock())
        result = FrameResult(
            frame=frame,
            frame_index=self._frame_index,
            motion_boxes=motion_boxes,
            detections=list(self._detections),
            detections_updated=detections_updated,
            is_calibrating=self.motion_detector.is_calibrating,
            fps=self.fps,
        )
        self._frame_index += 1
        return result

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
