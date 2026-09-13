"""Rastreamento de detecções entre ciclos de detecção, com associação por IoU."""

from typing import NamedTuple

import config
from src.utils.geometry import calculate_iou


class TrackedObject(NamedTuple):
    track_id: int  # identidade do objeto, nunca reaproveitada enquanto o tracker existir
    class_name: str  # mesmo identificador da Detection: nome do COCO ou "face"
    confidence: float  # confiança da última detecção associada
    bbox: tuple  # (x1, y1, x2, y2) da última detecção associada
    missed: int  # ciclos seguidos sem redetecção; 0 se foi detectado no último ciclo


class TrackerUpdate(NamedTuple):
    active: list  # objetos rastreados após o ciclo, em ordem de track_id, inclusive os novos
    new: list  # objetos que passaram a ser rastreados neste ciclo
    removed: list  # objetos removidos neste ciclo por excederem a tolerância, no último estado


class DetectionTracker:
    """Mantém a identidade dos objetos entre ciclos de detecção.

    Cada chamada a `update` é um ciclo de detecção. Uma detecção é associada a um objeto
    rastreado da mesma classe quando o IoU entre as caixas é pelo menos `iou_threshold`.
    Os pares são escolhidos do maior IoU para o menor, e cada objeto e cada detecção
    entram em no máximo um par. Detecções sem par viram objetos novos. Um objeto sem par
    continua ativo, na última posição conhecida, por até `max_missed` ciclos seguidos e
    é removido no ciclo seguinte; assim a caixa não pisca quando o detector falha em
    alguns ciclos.
    """

    def __init__(self, max_missed=config.TRACKER_MAX_MISSED, iou_threshold=config.TRACKER_IOU_THRESHOLD):
        if max_missed < 0:
            raise ValueError(f"a tolerância deve ser pelo menos 0, recebido {max_missed}")
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError(f"o IoU mínimo deve ser maior que 0 e no máximo 1, recebido {iou_threshold}")
        self.max_missed = max_missed
        self.iou_threshold = iou_threshold
        self._tracks = []
        self._next_id = 1

    @property
    def tracks(self):
        """Objetos rastreados no momento, em ordem de `track_id`."""
        return list(self._tracks)

    def reset(self):
        """Esquece os objetos rastreados, sem reaproveitar identidades já usadas."""
        self._tracks = []

    def update(self, detections):
        """Associa as detecções de um ciclo aos objetos rastreados; devolve `TrackerUpdate`."""
        detections = list(detections)
        pairs = []
        for t, track in enumerate(self._tracks):
            for d, detection in enumerate(detections):
                if detection.class_name != track.class_name:
                    continue
                iou = calculate_iou(track.bbox, detection.bbox)
                if iou >= self.iou_threshold:
                    pairs.append((iou, t, d))
        # A ordenação é estável: com IoU empatado, vale a ordem dos objetos e das detecções.
        pairs.sort(key=lambda pair: pair[0], reverse=True)
        detection_for_track = {}
        for _, t, d in pairs:
            if t not in detection_for_track and d not in detection_for_track.values():
                detection_for_track[t] = d

        active, removed = [], []
        for t, track in enumerate(self._tracks):
            if t in detection_for_track:
                detection = detections[detection_for_track[t]]
                active.append(track._replace(confidence=detection.confidence, bbox=tuple(detection.bbox), missed=0))
                continue
            track = track._replace(missed=track.missed + 1)
            (removed if track.missed > self.max_missed else active).append(track)

        matched = set(detection_for_track.values())
        new = []
        for d, detection in enumerate(detections):
            if d in matched:
                continue
            new.append(TrackedObject(self._next_id, detection.class_name, detection.confidence, tuple(detection.bbox), 0))
            self._next_id += 1

        self._tracks = active + new
        return TrackerUpdate(active=list(self._tracks), new=new, removed=removed)
