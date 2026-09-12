"""Pipeline de processamento de frames, sem nenhuma dependência de Qt.

É usado diretamente pelo modo terminal e, na interface gráfica, envolvido por
uma QThread em `src/ui/pipeline_worker.py`.
"""

import time
from contextlib import contextmanager

import cv2

import config


@contextmanager
def _opencv_warnings_suppressed():
    """Reduz o log do OpenCV a erros durante o bloco, restaurando o nível anterior depois.

    Usado só na leitura da câmera: quando ela é desconectada, o backend do Windows
    (MSMF) emite "can't grab frame" várias vezes por segundo até a tolerância de
    desconexão estourar, e a mensagem de desconexão já explica o problema. Como o
    nível de log do OpenCV é global, o escopo deve ser o menor possível.
    """
    log = cv2.utils.logging
    previous = log.getLogLevel()
    log.setLogLevel(min(previous, log.LOG_LEVEL_ERROR))
    try:
        yield
    finally:
        log.setLogLevel(previous)


class CameraOpenError(RuntimeError):
    """A câmera não pôde ser aberta."""


class CameraDisconnectedError(RuntimeError):
    """A câmera parou de entregar frames por mais tempo que a tolerância."""


class CameraSource:
    """Fonte de vídeo a partir de uma câmera local.

    Falhas de leitura isoladas são toleradas: `read()` devolve `None` e a câmera
    só é considerada desconectada quando nenhum frame válido chega por
    `disconnect_timeout` segundos.
    """

    def __init__(
        self,
        index=config.CAMERA_INDEX,
        width=config.FRAME_WIDTH,
        height=config.FRAME_HEIGHT,
        disconnect_timeout=config.CAMERA_DISCONNECT_TIMEOUT_SECONDS,
        capture_factory=cv2.VideoCapture,
        clock=time.monotonic,
    ):
        self.index = index
        self.width = width
        self.height = height
        self.disconnect_timeout = disconnect_timeout
        self._capture_factory = capture_factory
        self._clock = clock
        self._cap = None
        self._last_frame_time = None

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
        cap = self._capture_factory(self.index)
        if not cap.isOpened():
            cap.release()
            raise CameraOpenError(
                f"não foi possível abrir a câmera {self.index}, verifique se ela "
                "está conectada e se não está em uso por outro aplicativo"
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap
        self._last_frame_time = self._clock()

    def read(self):
        """Devolve o próximo frame, ou `None` se a leitura falhou dentro da tolerância.

        Levanta `CameraDisconnectedError` quando a tolerância é excedida.
        """
        if self._cap is None:
            raise RuntimeError("a câmera precisa ser aberta antes da leitura")
        with _opencv_warnings_suppressed():
            ok, frame = self._cap.read()
        now = self._clock()
        if ok and frame is not None:
            self._last_frame_time = now
            return frame
        if now - self._last_frame_time >= self.disconnect_timeout:
            raise CameraDisconnectedError("câmera desconectada, verifique a conexão")
        return None

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
