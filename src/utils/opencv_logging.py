"""Controle pontual do nível de log do OpenCV."""

from contextlib import contextmanager

import cv2


@contextmanager
def opencv_warnings_suppressed():
    """Reduz o log do OpenCV a erros durante o bloco, restaurando o nível anterior depois.

    Serve para chamadas específicas cujos avisos são inofensivos ou já explicados por
    uma mensagem do próprio app. Erros continuam aparecendo. Como o nível de log do
    OpenCV é global, o escopo deve ser o menor possível.
    """
    log = cv2.utils.logging
    previous = log.getLogLevel()
    log.setLogLevel(min(previous, log.LOG_LEVEL_ERROR))
    try:
        yield
    finally:
        log.setLogLevel(previous)
