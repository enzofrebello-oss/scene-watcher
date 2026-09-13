import pytest

from src.detection_tracker import DetectionTracker, TrackedObject
from src.object_detector import Detection
from src.utils.geometry import calculate_iou

CUP_BOX = (100, 100, 200, 200)


def cup(bbox=CUP_BOX, confidence=0.9):
    return Detection("cup", confidence, bbox)


def test_objeto_some_por_3_ciclos_e_reaparece_mantem_identidade():
    tracker = DetectionTracker(max_missed=10, iou_threshold=0.3)
    (first,) = tracker.update([cup()]).new
    for missed in range(1, 4):
        update = tracker.update([])
        assert [(t.track_id, t.missed) for t in update.active] == [(first.track_id, missed)]
        assert update.removed == []

    update = tracker.update([cup((104, 98, 206, 203))])

    assert update.new == []
    assert update.active == [TrackedObject(first.track_id, "cup", 0.9, (104, 98, 206, 203), 0)]


def test_objeto_some_alem_da_tolerancia_e_removido():
    tracker = DetectionTracker(max_missed=10, iou_threshold=0.3)
    (first,) = tracker.update([cup()]).new
    for missed in range(1, 11):
        update = tracker.update([])
        assert [(t.track_id, t.missed) for t in update.active] == [(first.track_id, missed)]
        assert update.removed == []

    update = tracker.update([])

    assert update.active == []
    assert [t.track_id for t in update.removed] == [first.track_id]
    assert tracker.update([]).removed == []  # removido uma única vez


def test_objeto_que_volta_depois_de_removido_ganha_nova_identidade():
    tracker = DetectionTracker(max_missed=2)
    (first,) = tracker.update([cup()]).new
    for _ in range(3):
        tracker.update([])

    (second,) = tracker.update([cup()]).new

    assert second.track_id != first.track_id


def test_tolerancia_zero_remove_no_primeiro_ciclo_sem_redeteccao():
    tracker = DetectionTracker(max_missed=0)
    tracker.update([cup()])

    update = tracker.update([])

    assert update.active == [] and len(update.removed) == 1


def test_objeto_sem_redeteccao_mantem_ultima_caixa_e_confianca():
    tracker = DetectionTracker()
    tracker.update([cup(confidence=0.72)])

    (track,) = tracker.update([]).active

    assert (track.bbox, track.confidence, track.missed) == (CUP_BOX, 0.72, 1)


def test_objeto_novo_so_aparece_em_new_no_ciclo_em_que_surge():
    tracker = DetectionTracker()
    update = tracker.update([cup()])
    assert update.new == update.active and len(update.new) == 1

    assert tracker.update([cup()]).new == []


def test_classes_diferentes_nunca_sao_associadas():
    tracker = DetectionTracker()
    (person,) = tracker.update([Detection("person", 0.9, CUP_BOX)]).new

    update = tracker.update([Detection("face", 0.95, CUP_BOX)])

    assert [t.class_name for t in update.new] == ["face"]
    assert [(t.track_id, t.missed) for t in update.active if t.class_name == "person"] == [(person.track_id, 1)]


@pytest.mark.parametrize(
    "bbox, associates",
    [
        ((0, 0, 3, 10), True),  # IoU exatamente 0,3 com (0, 0, 10, 10): associa
        ((0, 0, 2, 10), False),  # IoU 0,2: vira objeto novo
    ],
)
def test_iou_minimo_e_inclusivo(bbox, associates):
    tracker = DetectionTracker(iou_threshold=0.3)
    (track,) = tracker.update([cup((0, 0, 10, 10))]).new

    update = tracker.update([cup(bbox)])

    assert (update.new == []) is associates
    assert (update.active[0].missed == 0) is associates
    assert update.active[0].track_id == track.track_id


def test_associacao_escolhe_primeiro_os_pares_de_maior_iou():
    tracker = DetectionTracker(iou_threshold=0.3)
    left, right = tracker.update([cup((0, 0, 10, 10)), cup((4, 0, 14, 10))]).new
    near_right, near_left = (5, 0, 14, 10), (-5, 0, 5, 10)
    # near_right está quase sobre o objeto da direita, mas também é a melhor opção do
    # objeto da esquerda. Associar objeto por objeto, na ordem, daria near_right ao da
    # esquerda e deixaria o da direita sem par.
    assert calculate_iou(right.bbox, near_right) == pytest.approx(0.9)
    assert calculate_iou(left.bbox, near_right) > calculate_iou(left.bbox, near_left) >= 0.3

    update = tracker.update([cup(near_right), cup(near_left)])

    assert update.new == []
    assert {t.track_id: t.bbox for t in update.active} == {left.track_id: near_left, right.track_id: near_right}


def test_duas_deteccoes_para_o_mesmo_objeto_a_de_menor_iou_vira_objeto_novo():
    tracker = DetectionTracker(iou_threshold=0.3)
    (track,) = tracker.update([cup((0, 0, 10, 10))]).new

    update = tracker.update([cup((3, 0, 13, 10)), cup((1, 0, 11, 10))])

    assert [(t.track_id, t.bbox) for t in update.active if t.track_id == track.track_id] == [(track.track_id, (1, 0, 11, 10))]
    assert [t.bbox for t in update.new] == [(3, 0, 13, 10)]


def test_reset_esquece_objetos_sem_reaproveitar_identidades():
    tracker = DetectionTracker()
    (first,) = tracker.update([cup()]).new

    tracker.reset()

    assert tracker.tracks == []
    (second,) = tracker.update([cup()]).new
    assert second.track_id > first.track_id


def test_listas_devolvidas_nao_alteram_o_tracker():
    tracker = DetectionTracker()
    tracker.update([cup()]).active.clear()
    tracker.tracks.clear()

    assert len(tracker.update([cup()]).active) == 1


@pytest.mark.parametrize("kwargs", [{"max_missed": -1}, {"iou_threshold": 0.0}, {"iou_threshold": 1.5}])
def test_parametros_invalidos_sao_recusados(kwargs):
    with pytest.raises(ValueError):
        DetectionTracker(**kwargs)
