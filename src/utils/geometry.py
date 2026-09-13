"""Funções geométricas puras sobre caixas (x1, y1, x2, y2)."""


def box_area(box):
    """Área de uma caixa com coordenadas contínuas: (x2 - x1) · (y2 - y1).

    Uma caixa invertida (x2 < x1 ou y2 < y1) tem área 0.
    """
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def calculate_iou(box_a, box_b):
    """Interseção sobre união (IoU) de duas caixas (x1, y1, x2, y2), entre 0 e 1.

    As coordenadas são tratadas como contínuas: a caixa (0, 0, 10, 10) tem área 100.
    Caixas que só se tocam na borda não se sobrepõem (IoU 0), e uma caixa de área zero
    tem IoU 0 com qualquer outra, inclusive com ela mesma.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection = box_area((max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)))
    if intersection == 0:
        return 0.0
    # Com interseção positiva, as duas caixas têm área positiva e a união também.
    union = box_area(box_a) + box_area(box_b) - intersection
    return intersection / union
