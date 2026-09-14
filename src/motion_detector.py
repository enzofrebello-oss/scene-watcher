"""Detecção de regiões em movimento por subtração de fundo (MOG2)."""

import cv2
import numpy as np

import config

# Valor dos pixels de primeiro plano na máscara do MOG2; sombras recebem 127 e são
# descartadas ao limiarizar logo abaixo deste valor.
_FOREGROUND_VALUE = 255


def sensitivity_to_var_threshold(sensitivity):
    """Converte a sensibilidade (0 a 1) no varThreshold do MOG2, em escala logarítmica."""
    if not 0.0 <= sensitivity <= 1.0:
        raise ValueError(f"a sensibilidade deve estar entre 0 e 1, recebido {sensitivity}")
    least = config.MOTION_VAR_THRESHOLD_AT_MIN_SENSITIVITY
    most = config.MOTION_VAR_THRESHOLD_AT_MAX_SENSITIVITY
    return least * (most / least) ** sensitivity


class MotionDetector:
    """Detecta regiões em movimento num frame BGR.

    O frame é reduzido para `processing_width` pixels de largura (frames mais estreitos
    não são ampliados). Na máscara do MOG2 as sombras são descartadas, a erosão remove
    ruído isolado e a dilatação junta partes do mesmo objeto; cada contorno com área a
    partir de `min_area_fraction` da área do frame vira uma caixa (x1, y1, x2, y2) nas
    coordenadas do frame original.

    Nos primeiros `warmup_frames` frames o modelo só aprende o fundo e `detect` devolve
    lista vazia; `is_calibrating` indica esse período para o próximo frame e
    `last_frame_calibrating`, se o último frame passado a `detect` só serviu para aprender o
    fundo. Se a resolução do frame mudar, o fundo é aprendido de novo.

    `sensitivity` pode ser alterada a qualquer momento, sem reiniciar o fundo.
    """

    def __init__(
        self,
        sensitivity=config.MOTION_SENSITIVITY,
        min_area_fraction=config.MOTION_MIN_AREA_FRACTION,
        processing_width=config.MOTION_PROCESSING_WIDTH,
        warmup_frames=config.MOTION_WARMUP_FRAMES,
        history=config.MOTION_HISTORY,
    ):
        self.min_area_fraction = min_area_fraction
        self.processing_width = processing_width
        self.warmup_frames = warmup_frames
        self.history = history
        self._kernel = np.ones((config.MOTION_KERNEL_SIZE, config.MOTION_KERNEL_SIZE), np.uint8)
        self._subtractor = None
        self._frame_size = None
        self._frames_seen = 0
        self._last_frame_calibrating = True
        self.sensitivity = sensitivity
        self._reset_background()

    @property
    def sensitivity(self):
        return self._sensitivity

    @sensitivity.setter
    def sensitivity(self, value):
        var_threshold = sensitivity_to_var_threshold(value)
        self._sensitivity = value
        if self._subtractor is not None:
            self._subtractor.setVarThreshold(var_threshold)

    @property
    def is_calibrating(self):
        """Se o próximo frame passado a `detect` ainda vai servir só para aprender o fundo."""
        return self._frames_seen < self.warmup_frames

    @property
    def last_frame_calibrating(self):
        """Se o último frame passado a `detect` só serviu para aprender o fundo (True antes do primeiro).

        Difere de `is_calibrating` logo depois do último frame de aquecimento, inclusive do
        aquecimento que recomeça após uma troca de resolução: `is_calibrating` já responde
        pelo frame seguinte.
        """
        return self._last_frame_calibrating

    def _reset_background(self):
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=sensitivity_to_var_threshold(self._sensitivity),
            detectShadows=True,
        )
        self._frames_seen = 0

    def detect(self, frame):
        height, width = frame.shape[:2]
        # Compara a resolução original, não a reduzida: resoluções com a mesma proporção
        # viram o mesmo tamanho reduzido, mas a câmera pode mudar o enquadramento entre elas.
        if (width, height) != self._frame_size:
            if self._frame_size is not None:
                self._reset_background()
            self._frame_size = (width, height)
        if width > self.processing_width:
            size = (self.processing_width, max(1, round(height * self.processing_width / width)))
            image = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        else:
            size = (width, height)
            image = frame

        calibrating = self.is_calibrating
        self._last_frame_calibrating = calibrating
        mask = self._subtractor.apply(image)
        self._frames_seen = min(self._frames_seen + 1, self.warmup_frames)
        if calibrating:
            return []

        _, mask = cv2.threshold(mask, _FOREGROUND_VALUE - 1, 255, cv2.THRESH_BINARY)
        mask = cv2.erode(mask, self._kernel, iterations=config.MOTION_ERODE_ITERATIONS)
        mask = cv2.dilate(mask, self._kernel, iterations=config.MOTION_DILATE_ITERATIONS)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = self.min_area_fraction * size[0] * size[1]
        scale_x, scale_y = width / size[0], height / size[1]
        boxes = []
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((
                round(x * scale_x),
                round(y * scale_y),
                min(width, round((x + w) * scale_x)),
                min(height, round((y + h) * scale_y)),
            ))
        return boxes
