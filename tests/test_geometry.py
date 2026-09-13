import pytest

from src.utils.geometry import calculate_iou


def test_caixas_identicas_tem_iou_1():
    assert calculate_iou((10, 20, 110, 70), (10, 20, 110, 70)) == 1.0


@pytest.mark.parametrize(
    "box_a, box_b",
    [
        ((0, 0, 10, 10), (20, 20, 30, 30)),  # afastadas nos dois eixos
        ((0, 0, 10, 10), (0, 20, 10, 30)),  # alinhadas na horizontal, afastadas na vertical
        ((0, 0, 10, 10), (10, 0, 20, 10)),  # só se tocam na borda
        ((0, 0, 10, 10), (10, 10, 20, 20)),  # só se tocam no canto
    ],
)
def test_caixas_sem_sobreposicao_tem_iou_0(box_a, box_b):
    assert calculate_iou(box_a, box_b) == 0.0
    assert calculate_iou(box_b, box_a) == 0.0


@pytest.mark.parametrize(
    "box_a, box_b, expected",
    [
        # interseção 5x5 = 25; união 100 + 100 - 25 = 175
        ((0, 0, 10, 10), (5, 5, 15, 15), 25 / 175),
        # interseção 2x4 = 8; união 16 + 16 - 8 = 24
        ((0, 0, 4, 4), (2, 0, 6, 4), 1 / 3),
        # caixa menor inteira dentro da maior: interseção 30; união 100
        ((0, 0, 10, 10), (0, 0, 3, 10), 0.3),
    ],
)
def test_sobreposicao_parcial_com_valor_conhecido(box_a, box_b, expected):
    assert calculate_iou(box_a, box_b) == pytest.approx(expected)
    assert calculate_iou(box_b, box_a) == pytest.approx(expected)


@pytest.mark.parametrize(
    "box_a, box_b",
    [
        ((0, 0, 0, 10), (0, 0, 10, 10)),  # largura zero dentro de uma caixa normal
        ((0, 0, 10, 0), (0, 0, 10, 0)),  # altura zero, idênticas
        ((5, 5, 5, 5), (5, 5, 5, 5)),  # pontos idênticos
    ],
)
def test_caixa_de_area_zero_tem_iou_0(box_a, box_b):
    assert calculate_iou(box_a, box_b) == 0.0
