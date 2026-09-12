"""Pipeline de processamento de frames, sem nenhuma dependência de Qt.

É usado diretamente pelo modo terminal e, na interface gráfica, envolvido por
uma QThread em `src/ui/pipeline_worker.py`.
"""

import sys
import time
from contextlib import contextmanager

import cv2
import numpy as np

import config

# Passo da amostra esparsa de pixels comparada antes do frame inteiro ao procurar
# frames repetidos; a comparação completa só acontece se a amostra for igual.
_REPEAT_CHECK_STRIDE = 16


@contextmanager
def _opencv_warnings_suppressed():
    """Reduz o log do OpenCV a erros durante o bloco, restaurando o nível anterior depois.

    Usado na abertura e na leitura da câmera. Quando a câmera não abre, o MSMF avisa
    que não consegue capturar pelo índice; quando ela é desconectada, emite
    "can't grab frame" várias vezes por segundo até a tolerância estourar. Nos dois
    casos a mensagem do app já explica o problema. Como o nível de log do OpenCV é
    global, o escopo deve ser o menor possível.
    """
    log = cv2.utils.logging
    previous = log.getLogLevel()
    log.setLogLevel(min(previous, log.LOG_LEVEL_ERROR))
    try:
        yield
    finally:
        log.setLogLevel(previous)


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
        with _opencv_warnings_suppressed():
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
        with _opencv_warnings_suppressed():
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
