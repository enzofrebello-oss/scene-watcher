"""Overlay desenhado sobre o frame: caixas de objetos e rostos, movimento e instrumentos.

Só depende do OpenCV. O mesmo desenho aparece na janela do modo terminal e, na fase 2,
na área de câmera da interface, sem lógica de desenho duplicada no Qt.

Ordem de desenho, de trás para frente: cantos de viewfinder, caixas de movimento,
instrumentos (contador técnico e painel de objetos), caixas de objetos e rostos e, por
fim, os rótulos das caixas. As caixas são o elemento visual mais forte; os instrumentos
usam a cor de destaque com fonte menor e translucidez.

As funções puras do começo do módulo (cor por classe, opacidade, ordem e agregação do
painel, textos dos instrumentos, tracejado e posicionamento) não desenham nada e podem
ser testadas isoladamente.
"""

import zlib
from typing import NamedTuple

import cv2
import numpy as np

import config
from src.coco_labels import COCO_LABELS_PT, EXTRA_LABELS_PT, translate

EMPTY_PANEL_TEXT = "nenhum objeto detectado no momento"
CALIBRATING_TEXT = "calibrando fundo da cena..."

# Menor tamanho de fonte usado, para frames muito pequenos continuarem legíveis.
_MIN_FONT_SIZE = 6
# Tamanho em que a altura dos algarismos das duas fontes é medida para igualá-las.
_FONT_CALIBRATION_SIZE = 100


def hex_to_bgr(color):
    """Converte "#RRGGBB" na tupla BGR usada pelo OpenCV."""
    value = color.lstrip("#")
    red, green, blue = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return (blue, green, red)


BACKGROUND_BGR = hex_to_bgr(config.COLOR_BACKGROUND)
TEXT_BGR = hex_to_bgr(config.COLOR_TEXT)
ACCENT_BGR = hex_to_bgr(config.COLOR_ACCENT)


# --- Funções puras ------------------------------------------------------------

def class_color(class_name, palette=config.OVERLAY_CLASS_PALETTE):
    """Cor BGR da classe, sempre a mesma entre execuções.

    Usa o crc32 do nome, e não o hash() do Python, que muda a cada processo.
    """
    return hex_to_bgr(palette[zlib.crc32(class_name.encode("utf-8")) % len(palette)])


def fade_opacity(missed, max_missed=config.TRACKER_MAX_MISSED):
    """Opacidade de um objeto rastreado conforme os ciclos seguidos sem redetecção.

    1 quando detectado no último ciclo; cai em degraus iguais, 1 − ciclos/(tolerância + 1),
    e com a tolerância padrão de 10 chega a cerca de 9% no último ciclo antes de o tracker
    remover o objeto, então a remoção quase não é percebida.
    """
    return min(1.0, max(0.0, 1.0 - missed / (max_missed + 1)))


def visible_tracks(tracks, hidden_classes=config.OVERLAY_HIDDEN_CLASSES):
    """Objetos rastreados que aparecem no overlay (sem as classes ocultas)."""
    return [t for t in tracks if t.class_name not in hidden_classes]


def update_class_order(previous_order, tracks):
    """Classes presentes, na ordem em que apareceram.

    Classes que continuam presentes mantêm a posição, mesmo que a confiança mude; classes
    novas entram no fim, na ordem em que o tracker criou o primeiro objeto de cada uma;
    classes sem nenhum objeto rastreado saem.
    """
    oldest_track = {}
    for t in tracks:
        oldest_track[t.class_name] = min(t.track_id, oldest_track.get(t.class_name, t.track_id))
    kept = [name for name in previous_order if name in oldest_track]
    new = sorted((name for name in oldest_track if name not in kept), key=oldest_track.get)
    return kept + new


class PanelRow(NamedTuple):
    class_name: str
    count: int  # objetos rastreados da classe, inclusive os que estão esmaecendo
    confidence: float  # maior confiança entre eles
    opacity: float  # maior opacidade entre eles: a linha só esmaece quando todos esmaecem


def panel_rows(tracks, class_order, max_missed=config.TRACKER_MAX_MISSED):
    """Uma linha por classe de `class_order` que tem objetos em `tracks`."""
    rows = []
    for name in class_order:
        instances = [t for t in tracks if t.class_name == name]
        if instances:
            rows.append(PanelRow(
                class_name=name,
                count=len(instances),
                confidence=max(t.confidence for t in instances),
                opacity=max(fade_opacity(t.missed, max_missed) for t in instances),
            ))
    return rows


def panel_lines(rows):
    """Textos do painel de objetos, com a confiança alinhada em coluna (fonte monoespaçada)."""
    if not rows:
        return [EMPTY_PANEL_TEXT]
    index_width = len(str(len(rows)))
    names = [
        f"{index:>{index_width}}. {translate(row.class_name)}" + (f" ({row.count})" if row.count > 1 else "")
        for index, row in enumerate(rows, start=1)
    ]
    width = max(len(name) for name in names)
    return [f"{name.ljust(width)} {row.confidence:.2f}" for name, row in zip(names, rows)]


def counter_lines(frame_index, fps, detections, is_calibrating):
    """Textos do contador técnico: rótulos à esquerda e valores alinhados à direita."""
    values = [
        ("frame", f"{frame_index:06d}"),
        ("fps", f"{fps:.1f}" if fps is not None else "--.-"),
        ("detecções", str(detections)),
    ]
    label_width = max(len(label) for label, _ in values)
    value_width = max(len(value) for _, value in values)
    lines = [f"{label.ljust(label_width)}  {value.rjust(value_width)}" for label, value in values]
    if is_calibrating:
        lines.append(CALIBRATING_TEXT)
    return lines


def dashed_segments(bbox, dash, gap):
    """Segmentos ((xa, ya), (xb, yb)) do contorno tracejado da caixa (x1, y1, x2, y2).

    O padrão recomeça em cada canto, então os quatro cantos sempre aparecem.
    """
    x1, y1, x2, y2 = bbox
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    segments = []
    for (ax, ay), (bx, by) in zip(corners, corners[1:]):
        length = abs(bx - ax) + abs(by - ay)
        dx, dy = (bx > ax) - (bx < ax), (by > ay) - (by < ay)
        start = 0
        while start < length:
            end = min(start + dash, length)
            segments.append(((ax + dx * start, ay + dy * start), (ax + dx * end, ay + dy * end)))
            start += dash + gap
    return segments


def intersects(a, b):
    """Se os retângulos (x1, y1, x2, y2) se sobrepõem; retângulos que só se tocam não se sobrepõem."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def place_below_blockers(rect, blockers, frame_height):
    """Desce o retângulo até não sobrepor nenhum bloqueio, sem passar da base do frame.

    Cada deslocamento leva o topo para a base de um bloqueio, sempre para baixo, então
    termina. Se não couber antes da base, fica encostado nela, mesmo sobrepondo.
    """
    x1, _, x2, _ = rect
    height = rect[3] - rect[1]
    blocker = next((b for b in blockers if intersects(rect, b)), None)
    while blocker is not None:
        rect = (x1, blocker[3], x2, blocker[3] + height)
        blocker = next((b for b in blockers if intersects(rect, b)), None)
    if rect[3] > frame_height:
        rect = (x1, frame_height - height, x2, frame_height)
    return rect


# --- Desenho ------------------------------------------------------------------

def _blend(img, rect, opacity, draw):
    """Desenha com a opacidade dada, alterando só a região `rect` (x1, y1, x2, y2) da imagem.

    `draw(layer, x0, y0)` recebe a região a desenhar e o canto superior esquerdo dela na
    imagem, para converter coordenadas.
    """
    height, width = img.shape[:2]
    x1, y1 = max(0, rect[0]), max(0, rect[1])
    x2, y2 = min(width, rect[2]), min(height, rect[3])
    if x2 <= x1 or y2 <= y1 or opacity <= 0.0:
        return
    region = img[y1:y2, x1:x2]
    if opacity >= 1.0:
        draw(region, x1, y1)
        return
    layer = region.copy()
    draw(layer, x1, y1)
    cv2.addWeighted(layer, opacity, region, 1.0 - opacity, 0.0, dst=region)


def _fill(color):
    def draw(layer, x0, y0):
        layer[:] = color
    return draw


def _load_font(candidates):
    """Primeira fonte candidata que o OpenCV consegue abrir; se nenhuma abrir, a fonte embutida.

    cv2.FontFace não gera erro quando falha ao abrir o arquivo: devolve getName() vazio.
    """
    for name in candidates:
        for folder in config.FONT_DIRS:
            path = folder / name
            if path.is_file():
                face = cv2.FontFace(path.as_posix())
                if face.getName():
                    return face
    return cv2.FontFace(config.BUILTIN_FONT_FALLBACK)


def _ink_height(face, size, text):
    """Altura, em pixels, dos traços efetivamente desenhados de `text`."""
    canvas = np.zeros((3 * size, (len(text) + 2) * size, 3), np.uint8)
    cv2.putText(canvas, text, (size, 2 * size), (255, 255, 255), face, size)
    rows = np.flatnonzero(canvas[:, :, 0].max(axis=1) > 127)
    return int(rows[-1] - rows[0] + 1) if rows.size else 0


class _TextStyle:
    """Uma fonte num tamanho, com altura de linha fixa, independente de acentos e descendentes."""

    def __init__(self, face, size):
        self.face = face
        self.size = size
        _, top, _, height = cv2.getTextSize((0, 0), "ÁÇgjpy", (0, 0), face, size)
        self.ascent = -top
        self.height = height

    def width(self, text):
        return cv2.getTextSize((0, 0), text, (0, 0), self.face, self.size)[2]

    def draw(self, img, text, x, top, color):
        cv2.putText(img, text, (x, top + self.ascent), color, self.face, self.size)


class _Metrics(NamedTuple):
    """Tamanhos do config convertidos para a altura de um frame."""

    box_thickness: int
    label_padding: int
    label_gap: int
    label_name: _TextStyle
    label_confidence: _TextStyle
    motion_thickness: int
    dash: int
    gap: int
    instrument: _TextStyle
    instrument_padding: int
    instrument_line_spacing: int
    instrument_margin: int
    corner_thickness: int
    corner_margin: int


class _InstrumentBlock(NamedTuple):
    rect: tuple  # (x1, y1, x2, y2), com o fundo translúcido
    lines: list  # (texto, opacidade relativa)


class _Label(NamedTuple):
    track: object  # TrackedObject
    name: str  # classe traduzida
    confidence: str
    rect: tuple  # (x1, y1, x2, y2)


class _Layout(NamedTuple):
    counter: _InstrumentBlock
    panel: _InstrumentBlock
    labels: list  # _Label, na ordem em que as posições foram escolhidas


# Caracteres que o overlay pode escrever, rasterizados no aquecimento.
_WARM_UP_TEXT = "".join(sorted(set(
    "".join(COCO_LABELS_PT.values()) + "".join(EXTRA_LABELS_PT.values())
    + "0123456789.-() " + EMPTY_PANEL_TEXT + CALIBRATING_TEXT + "frame fps detecções"
)))


class OverlayRenderer:
    """Desenha o overlay de um `FrameResult` numa cópia do frame; o frame recebido não é alterado.

    Guarda apenas a ordem de aparição das classes do painel; `reset()` a esquece, para uma
    nova sessão de câmera.
    """

    def __init__(self, hidden_classes=config.OVERLAY_HIDDEN_CLASSES, max_missed=config.TRACKER_MAX_MISSED):
        self.hidden_classes = frozenset(hidden_classes)
        self.max_missed = max_missed
        self._ui_font = _load_font(config.UI_FONT_CANDIDATES)
        self._mono_font = _load_font(config.MONO_FONT_CANDIDATES)
        # No mesmo tamanho de FontFace, os algarismos da Consolas saem cerca de 30% mais
        # altos que os da Segoe UI. A fonte monoespaçada é reduzida na proporção medida, para
        # que os tamanhos do config valham na medida da fonte de interface.
        ui_height = _ink_height(self._ui_font, _FONT_CALIBRATION_SIZE, "0123456789")
        mono_height = _ink_height(self._mono_font, _FONT_CALIBRATION_SIZE, "0123456789")
        self._mono_scale = ui_height / mono_height if ui_height and mono_height else 1.0
        self._metrics = {}
        self._class_order = []

    @property
    def font_names(self):
        """Fontes em uso (interface, monoespaçada); o nome é o caminho do arquivo ou a fonte embutida."""
        return self._ui_font.getName(), self._mono_font.getName()

    def reset(self):
        self._class_order = []

    def warm_up(self, frame_height=config.FRAME_HEIGHT):
        """Escreve uma vez todos os caracteres usados, em todos os estilos, antes do primeiro frame.

        A primeira escrita com FontFace no processo chega a dezenas de milissegundos.
        """
        metrics = self._metrics_for(frame_height)
        styles = (metrics.label_name, metrics.label_confidence, metrics.instrument)
        canvas = np.zeros((2 * max(s.height for s in styles), 4096, 3), np.uint8)
        for style in styles:
            style.draw(canvas, _WARM_UP_TEXT, 0, 0, TEXT_BGR)

    def render(self, result):
        view = result.frame.copy()
        metrics = self._metrics_for(view.shape[0])
        tracks = visible_tracks(result.tracks, self.hidden_classes)
        self._class_order = update_class_order(self._class_order, tracks)
        rows = panel_rows(tracks, self._class_order, self.max_missed)
        layout = self._layout(view.shape, result, tracks, rows, metrics)

        self._draw_corners(view, metrics)
        for box in result.motion_boxes:
            self._draw_motion_box(view, box, metrics)
        self._draw_instrument(view, layout.counter, metrics)
        self._draw_instrument(view, layout.panel, metrics)
        # Objetos mantidos sem redetecção primeiro; os detectados agora ficam por cima.
        for track in sorted(tracks, key=lambda t: t.missed == 0):
            self._draw_box(view, track, metrics)
        for label in sorted(layout.labels, key=lambda label: label.track.missed == 0):
            self._draw_label(view, label, metrics)
        return view

    def _metrics_for(self, frame_height):
        metrics = self._metrics.get(frame_height)
        if metrics is not None:
            return metrics

        def px(value):
            return max(1, round(value * frame_height / config.OVERLAY_REFERENCE_HEIGHT))

        def ui_font(value):
            return _TextStyle(self._ui_font, max(_MIN_FONT_SIZE, px(value)))

        def mono_font(value):
            return _TextStyle(self._mono_font, max(_MIN_FONT_SIZE, round(px(value) * self._mono_scale)))

        metrics = _Metrics(
            box_thickness=px(config.OVERLAY_BOX_THICKNESS),
            label_padding=px(config.OVERLAY_LABEL_PADDING),
            label_gap=px(config.OVERLAY_LABEL_GAP),
            label_name=ui_font(config.OVERLAY_LABEL_FONT_SIZE),
            label_confidence=mono_font(config.OVERLAY_LABEL_FONT_SIZE),
            motion_thickness=px(config.OVERLAY_MOTION_THICKNESS),
            dash=px(config.OVERLAY_MOTION_DASH_LENGTH),
            gap=px(config.OVERLAY_MOTION_GAP_LENGTH),
            instrument=mono_font(config.OVERLAY_INSTRUMENT_FONT_SIZE),
            instrument_padding=px(config.OVERLAY_INSTRUMENT_PADDING),
            instrument_line_spacing=px(config.OVERLAY_INSTRUMENT_LINE_SPACING),
            instrument_margin=px(config.OVERLAY_INSTRUMENT_MARGIN),
            corner_thickness=px(config.OVERLAY_CORNER_THICKNESS),
            corner_margin=px(config.OVERLAY_CORNER_MARGIN),
        )
        self._metrics[frame_height] = metrics
        return metrics

    def _layout(self, frame_shape, result, tracks, rows, metrics):
        """Posições do contador, do painel e dos rótulos, sem desenhar nada."""
        height, width = frame_shape[:2]
        margin = metrics.instrument_margin
        counter_text = counter_lines(result.frame_index, result.fps, len(tracks), result.is_calibrating)
        counter = self._instrument_block([(text, 1.0) for text in counter_text], margin, margin, metrics, align_right=False)
        panel_text = panel_lines(rows)
        opacities = [row.opacity for row in rows] or [1.0]
        panel = self._instrument_block(list(zip(panel_text, opacities)), width - margin, margin, metrics, align_right=True)

        # O rótulo fica logo acima da caixa ou, sem espaço, dentro dela; objetos detectados
        # agora escolhem a posição primeiro, e um rótulo que cairia sobre um instrumento ou
        # outro rótulo desce até ficar livre.
        blockers = [counter.rect, panel.rect]
        labels = []
        for track in sorted(tracks, key=lambda t: t.missed != 0):
            name = translate(track.class_name)
            confidence = f"{track.confidence:.2f}"
            label_width = (2 * metrics.label_padding + metrics.label_name.width(name)
                           + metrics.label_gap + metrics.label_confidence.width(confidence))
            label_height = 2 * metrics.label_padding + self._label_ascent(metrics) + max(
                style.height - style.ascent for style in (metrics.label_name, metrics.label_confidence)
            )
            box_x1, box_y1 = track.bbox[0], track.bbox[1]
            x = max(0, min(box_x1, width - label_width))
            top = box_y1 - label_height if box_y1 >= label_height else box_y1
            rect = place_below_blockers((x, top, x + label_width, top + label_height), blockers, height)
            blockers.append(rect)
            labels.append(_Label(track, name, confidence, rect))
        return _Layout(counter, panel, labels)

    @staticmethod
    def _instrument_block(lines, x, y, metrics, align_right):
        style = metrics.instrument
        padding, spacing = metrics.instrument_padding, metrics.instrument_line_spacing
        block_width = 2 * padding + max(style.width(text) for text, _ in lines)
        block_height = 2 * padding + len(lines) * style.height + (len(lines) - 1) * spacing
        x1 = x - block_width if align_right else x
        return _InstrumentBlock((x1, y, x1 + block_width, y + block_height), lines)

    def _draw_corners(self, img, metrics):
        height, width = img.shape[:2]
        length = max(1, round(min(height, width) * config.OVERLAY_CORNER_LENGTH_FRACTION))
        thickness, margin = metrics.corner_thickness, metrics.corner_margin
        corners = (
            (margin, margin, 1, 1),
            (width - 1 - margin, margin, -1, 1),
            (margin, height - 1 - margin, 1, -1),
            (width - 1 - margin, height - 1 - margin, -1, -1),
        )
        for cx, cy, dx, dy in corners:
            ex, ey = cx + dx * length, cy + dy * length
            rect = (min(cx, ex) - thickness, min(cy, ey) - thickness, max(cx, ex) + thickness + 1, max(cy, ey) + thickness + 1)

            def draw(layer, x0, y0, cx=cx, cy=cy, ex=ex, ey=ey):
                cv2.line(layer, (cx - x0, cy - y0), (ex - x0, cy - y0), ACCENT_BGR, thickness)
                cv2.line(layer, (cx - x0, cy - y0), (cx - x0, ey - y0), ACCENT_BGR, thickness)

            _blend(img, rect, config.OVERLAY_CORNER_OPACITY, draw)

    @staticmethod
    def _draw_motion_box(img, box, metrics):
        x1, y1, x2, y2 = box
        thickness = metrics.motion_thickness
        segments = dashed_segments(box, metrics.dash, metrics.gap)

        def draw(layer, x0, y0):
            for (ax, ay), (bx, by) in segments:
                cv2.line(layer, (ax - x0, ay - y0), (bx - x0, by - y0), TEXT_BGR, thickness)

        _blend(img, (x1 - thickness, y1 - thickness, x2 + thickness + 1, y2 + thickness + 1), config.OVERLAY_MOTION_OPACITY, draw)

    @staticmethod
    def _draw_instrument(img, block, metrics):
        style = metrics.instrument
        padding, spacing = metrics.instrument_padding, metrics.instrument_line_spacing
        x1, y1, x2, _ = block.rect
        _blend(img, block.rect, config.OVERLAY_INSTRUMENT_BACKGROUND_OPACITY, _fill(BACKGROUND_BGR))
        for index, (text, opacity) in enumerate(block.lines):
            top = y1 + padding + index * (style.height + spacing)

            def draw(layer, x0, y0, text=text, top=top):
                style.draw(layer, text, x1 + padding - x0, top - y0, ACCENT_BGR)

            _blend(img, (x1 + padding, top, x2, top + style.height), config.OVERLAY_INSTRUMENT_TEXT_OPACITY * opacity, draw)

    def _draw_box(self, img, track, metrics):
        x1, y1, x2, y2 = track.bbox
        thickness = metrics.box_thickness
        color = class_color(track.class_name)

        def draw(layer, x0, y0):
            cv2.rectangle(layer, (x1 - x0, y1 - y0), (x2 - x0, y2 - y0), color, thickness)

        _blend(img, (x1 - thickness, y1 - thickness, x2 + thickness + 1, y2 + thickness + 1),
               fade_opacity(track.missed, self.max_missed), draw)

    @staticmethod
    def _label_ascent(metrics):
        return max(metrics.label_name.ascent, metrics.label_confidence.ascent)

    def _draw_label(self, img, label, metrics):
        x1, y1, _, _ = label.rect
        padding = metrics.label_padding
        color = class_color(label.track.class_name)
        confidence_x = x1 + padding + metrics.label_name.width(label.name) + metrics.label_gap
        # Nome e confiança usam fontes de tamanhos diferentes; os dois ficam na mesma linha de base.
        baseline = y1 + padding + self._label_ascent(metrics)
        name_top = baseline - metrics.label_name.ascent
        confidence_top = baseline - metrics.label_confidence.ascent

        def draw(layer, x0, y0):
            layer[:] = BACKGROUND_BGR
            metrics.label_name.draw(layer, label.name, x1 + padding - x0, name_top - y0, color)
            metrics.label_confidence.draw(layer, label.confidence, confidence_x - x0, confidence_top - y0, color)

        _blend(img, label.rect, fade_opacity(label.track.missed, self.max_missed), draw)
