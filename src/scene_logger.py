"""Log da sessão em JSON Lines: eventos de objetos, movimento e sessão, e registros periódicos.

Não depende de Qt. O `FramePipeline` é dono do logger: cada frame processado passa por
`update`, e os registros gerados voltam em `FrameResult.events`, na mesma ordem em que
foram gravados. Na fase 2, a interface só repassa esses eventos ao feed, aos alertas e à
gravação de clipes.

Cada linha do arquivo é um objeto JSON com `timestamp` (ISO 8601, horário local com fuso
e milissegundos), `type` e os campos do tipo. Chaves e tipos ficam em inglês e as classes
usam o nome do COCO, como identificadores estáveis; os textos em português só são gerados
na exibição (`event_text` e `session_summary_lines`). O formato está documentado no README.

O `session_ended` traz os totais da sessão, a fonte do resumo impresso no terminal e, na
fase 2, do diálogo de resumo.

As funções e a classe `MotionHysteresis` do começo do módulo não gravam nada e podem ser
testadas isoladamente.
"""

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import config
from src.coco_labels import translate

SESSION_STARTED = "session_started"
SESSION_ENDED = "session_ended"
CAMERA_DISCONNECTED = "camera_disconnected"
OBJECT_APPEARED = "object_appeared"
OBJECT_DISAPPEARED = "object_disappeared"
MOTION_STARTED = "motion_started"
MOTION_STOPPED = "motion_stopped"
SUMMARY = "summary"

# Motivos de object_disappeared: todo object_appeared tem o seu object_disappeared.
REASON_NOT_REDETECTED = "not_redetected"  # o tracker removeu o objeto por exceder a tolerância
REASON_SESSION_ENDED = "session_ended"  # o objeto ainda estava rastreado quando a sessão terminou
REASON_CAMERA_REOPENED = "camera_reopened"  # a câmera foi reaberta no meio da sessão e o tracker recomeçou

# Casas decimais de confianças, frações e segundos gravados no log.
DECIMALS = 3

# Largura dos rótulos do resumo da sessão no terminal ("duração", "movimento", "classes").
SUMMARY_LABEL_WIDTH = 12


class SceneEvent(NamedTuple):
    type: str
    timestamp: datetime  # horário local com fuso
    data: dict  # campos do tipo, com os mesmos nomes e a mesma ordem do JSON


# --- Funções puras ------------------------------------------------------------

def local_now():
    """Horário local com fuso."""
    return datetime.now().astimezone()


def format_timestamp(moment):
    """ISO 8601 com milissegundos e fuso, ex. 2026-09-13T14:32:07.412-03:00."""
    return moment.isoformat(timespec="milliseconds")


def to_log_precision(moment):
    """O mesmo instante truncado em milissegundos, como `format_timestamp` grava no log."""
    return moment.replace(microsecond=moment.microsecond // 1000 * 1000)


def json_value(value):
    """Valor pronto para o JSON: datetimes viram ISO 8601, inclusive dentro de dicts e listas."""
    if isinstance(value, datetime):
        return format_timestamp(value)
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def event_record(event):
    """Objeto gravado como uma linha JSON: `timestamp`, `type` e os campos do tipo."""
    record = {"timestamp": format_timestamp(event.timestamp), "type": event.type}
    for key, value in event.data.items():
        record[key] = json_value(value)
    return record


def rounded(value):
    """Número do log: float do Python (não numpy) com `DECIMALS` casas."""
    return round(float(value), DECIMALS)


def summarize_classes(tracks):
    """Por classe, em ordem alfabética: objetos rastreados e a média da última confiança conhecida deles."""
    confidences = {}
    for t in tracks:
        confidences.setdefault(t.class_name, []).append(float(t.confidence))
    return {
        name: {"count": len(values), "avg_confidence": rounded(sum(values) / len(values))}
        for name, values in sorted(confidences.items())
    }


def motion_fields(frames_observed, frames_with_motion):
    """(`motion`, `motion_ratio`) de um registro periódico.

    Só contam frames fora da calibração do fundo; se nenhum frame foi observado (a janela
    inteira em calibração), os dois campos ficam None.
    """
    if frames_observed == 0:
        return None, None
    return frames_with_motion > 0, rounded(frames_with_motion / frames_observed)


def detected_per_class(tracks):
    """Por classe, quantos objetos foram redetectados no último ciclo (`missed == 0`).

    Objetos mantidos sem redetecção não contam: quando o tracker dá uma identidade nova a
    um objeto que se moveu rápido, a antiga continua rastreada por alguns ciclos, e contá-la
    faria um único rosto aparecer como dois.
    """
    counts = {}
    for t in tracks:
        if t.missed == 0:
            counts[t.class_name] = counts.get(t.class_name, 0) + 1
    return counts


def union_seconds(intervals):
    """Duração da união de intervalos `(início, fim)` de datetimes, em segundos; sobreposições contam uma vez.

    Um intervalo com o fim antes do início (o relógio do sistema foi atrasado no meio dele) conta
    como duração zero, sem descontar tempo dos outros.
    """
    total = 0.0
    start = end = None
    for interval_start, interval_end in sorted(intervals):
        interval_end = max(interval_start, interval_end)
        if end is not None and interval_start <= end:
            end = max(end, interval_end)
            continue
        if end is not None:
            total += (end - start).total_seconds()
        start, end = interval_start, interval_end
    if end is not None:
        total += (end - start).total_seconds()
    return total


def session_totals(duration_seconds, frames, frames_observed, frames_with_motion, classes):
    """Campos do `session_ended`, na ordem do JSON.

    `classes` é {classe: {"first_seen", "appearances", "max_simultaneous", "intervals"}}, com
    `intervals` = [(aparecimento, última detecção)] de cada objeto da classe já encerrado.
    """
    _, motion_ratio = motion_fields(frames_observed, frames_with_motion)
    return {
        "duration_seconds": rounded(duration_seconds),
        "frames": frames,
        "frames_observed": frames_observed,
        "frames_with_motion": frames_with_motion,
        "motion_ratio": motion_ratio,
        "classes_seen": {
            name: {
                "first_seen": totals["first_seen"],
                "visible_seconds": rounded(union_seconds(totals["intervals"])),
                "max_simultaneous": totals["max_simultaneous"],
                "appearances": totals["appearances"],
            }
            for name, totals in sorted(classes.items())
        },
    }


class MotionHysteresis:
    """Decide quando o movimento começou e parou, sem rajadas com movimento fragmentado.

    Começa depois de `start_frames` frames seguidos com movimento e para depois de
    `stop_seconds` sem nenhum movimento. Frames em calibração não trazem informação e não
    mudam o estado.
    """

    def __init__(self, start_frames=config.MOTION_START_FRAMES, stop_seconds=config.MOTION_STOP_SECONDS):
        if start_frames < 1:
            raise ValueError(f"são necessários pelo menos 1 frame para o movimento começar, recebido {start_frames}")
        if stop_seconds < 0:
            raise ValueError(f"o tempo para o movimento parar não pode ser negativo, recebido {stop_seconds}")
        self.start_frames = start_frames
        self.stop_seconds = stop_seconds
        self.active = False
        self._consecutive = 0
        self._last_motion = None

    def interrupt(self):
        """A sequência de frames foi interrompida (câmera reaberta).

        A contagem de frames seguidos com movimento recomeça; o estado ativo e o horário do
        último movimento são mantidos.
        """
        self._consecutive = 0

    def update(self, has_motion, is_calibrating, now):
        """Devolve MOTION_STARTED, MOTION_STOPPED ou None para um frame; `now` em segundos monotônicos."""
        if is_calibrating:
            return None
        if has_motion:
            self._last_motion = now
            if self.active:
                return None
            self._consecutive += 1
            if self._consecutive >= self.start_frames:
                self.active = True
                self._consecutive = 0
                return MOTION_STARTED
            return None
        self._consecutive = 0
        if self.active and now - self._last_motion >= self.stop_seconds:
            self.active = False
            return MOTION_STOPPED
        return None


def event_level(event):
    return "ERRO" if event.type == CAMERA_DISCONNECTED else "INFO"


def event_text(event):
    """Texto do evento em português, para o terminal e, na fase 2, o feed e o rodapé.

    Registros periódicos não têm texto (None).
    """
    data = event.data
    if event.type == OBJECT_APPEARED:
        return f"{translate(data['class'])} apareceu ({data['confidence']:.2f})"
    if event.type == OBJECT_DISAPPEARED:
        detail = {REASON_SESSION_ENDED: ", sessão encerrada", REASON_CAMERA_REOPENED: ", câmera reaberta"}.get(data["reason"], "")
        return f"{translate(data['class'])} sumiu ({data['confidence']:.2f}{detail})"
    if event.type == MOTION_STARTED:
        return "movimento começou"
    if event.type == MOTION_STOPPED:
        return "movimento parou"
    if event.type == SESSION_STARTED:
        resolution = data["resolution"]
        size = f" em {resolution[0]}x{resolution[1]}" if resolution else ""
        return f"sessão iniciada, câmera {data['camera']}{size}"
    if event.type == SESSION_ENDED:
        return "sessão encerrada"
    if event.type == CAMERA_DISCONNECTED:
        return data["message"]
    return None


def terminal_line(event):
    """Linha "HH:MM:SS [INFO] texto" do evento, ou None para registros periódicos."""
    text = event_text(event)
    if text is None:
        return None
    return f"{event.timestamp:%H:%M:%S} [{event_level(event)}] {text}"


def format_duration(seconds):
    """HH:MM:SS, com a fração de segundo descartada."""
    total = int(seconds)
    return f"{total // 3600:02d}:{total // 60 % 60:02d}:{total % 60:02d}"


def format_visible_time(seconds):
    """Tempo visível de uma classe: HH:MM:SS, ou "<1 s" abaixo de um segundo (inclusive zero)."""
    return "<1 s" if seconds < 1 else format_duration(seconds)


def format_share(part, whole):
    """Percentual inteiro de `part` em `whole`, arredondado; nunca mostra 0% nem 100% quando não é exato."""
    percent = math.floor(100 * part / whole + 0.5)
    if percent == 0 and part > 0:
        return "<1%"
    if percent == 100 and part < whole:
        return ">99%"
    return f"{percent}%"


def session_summary_lines(event):
    """Linhas do resumo da sessão em português, a partir do `session_ended`; lista vazia para outros tipos.

    As classes saem traduzidas, em ordem de primeira aparição (empate pelo nome do COCO), com
    o máximo de objetos redetectados no mesmo ciclo e o tempo visível.
    """
    if event.type != SESSION_ENDED:
        return []
    data = event.data
    lines = ["resumo da sessão", _summary_row("duração", format_duration(data["duration_seconds"]))]
    if data["frames"] == 0:
        lines.append("  nenhuma imagem recebida da câmera")
        return lines

    observed = data["frames_observed"]
    if observed == 0:
        motion = "sem dados, fundo ainda em calibração"
    else:
        share = format_share(data["frames_with_motion"], observed)
        motion = f"{share} de 1 frame analisado" if observed == 1 else f"{share} dos {observed} frames analisados"
    lines.append(_summary_row("movimento", motion))

    classes = data["classes_seen"]
    if not classes:
        lines.append(_summary_row("classes", "nenhum objeto detectado na sessão"))
        return lines
    lines.append(_summary_row("classes", str(len(classes))))
    ordered = sorted(classes.items(), key=lambda item: (item[1]["first_seen"], item[0]))
    names = [translate(name) for name, _ in ordered]
    name_width = max(len(name) for name in names)
    count_width = max(len(str(totals["max_simultaneous"])) for _, totals in ordered)
    for name, (_, totals) in zip(names, ordered):
        lines.append(
            f"    {name:<{name_width}}  máx. {totals['max_simultaneous']:>{count_width}}   "
            f"visível {format_visible_time(totals['visible_seconds'])}"
        )
    return lines


def _summary_row(label, value):
    return f"  {label:<{SUMMARY_LABEL_WIDTH}} {value}"


def _warn_on_stderr(message):
    print(message, file=sys.stderr)


# --- Logger -------------------------------------------------------------------

class SceneLogger:
    """Gera os eventos da sessão a partir de cada frame processado e grava tudo em JSON Lines.

    O arquivo é sobrescrito no início de cada sessão, e cada linha é gravada com flush, então
    o que já foi registrado sobrevive a um travamento. Se o arquivo não puder ser gravado, o
    logger avisa uma vez por sessão e continua devolvendo os eventos, sem log em disco.
    """

    def __init__(
        self,
        path=config.SESSION_LOG_PATH,
        summary_interval=config.LOG_INTERVAL_SECONDS,
        motion_start_frames=config.MOTION_START_FRAMES,
        motion_stop_seconds=config.MOTION_STOP_SECONDS,
        clock=time.monotonic,
        wall_clock=local_now,
        warn=_warn_on_stderr,
    ):
        if summary_interval <= 0:
            raise ValueError(f"o intervalo dos registros periódicos deve ser positivo, recebido {summary_interval}")
        self.path = Path(path)
        self.summary_interval = summary_interval
        self._motion_start_frames = motion_start_frames
        self._motion_stop_seconds = motion_stop_seconds
        self._clock = clock
        self._wall_clock = wall_clock
        self._warn = warn
        self._file = None
        self._warned = False
        self._session_active = False
        # track_id -> {"class", "confidence", "appeared", "last_seen"} dos objetos com object_appeared em aberto
        self._objects = {}
        self._tracks = []
        self._motion = MotionHysteresis(motion_start_frames, motion_stop_seconds)
        self._next_summary = None
        self._frames_in_window = 0
        self._frames_observed = 0
        self._frames_with_motion = 0
        self._reset_session_totals(None)

    @property
    def session_active(self):
        return self._session_active

    def start_session(self, camera_index, resolution):
        """Sobrescreve o arquivo e registra `session_started`; devolve os eventos gerados."""
        if self._session_active:
            raise RuntimeError("a sessão do log já está ativa")
        self._warned = False
        self._open_file()
        self._session_active = True
        self._objects = {}
        self._tracks = []
        self._motion = MotionHysteresis(self._motion_start_frames, self._motion_stop_seconds)
        now = self._clock()
        self._next_summary = now + self.summary_interval
        self._reset_window()
        self._reset_session_totals(now)
        return [self._emit(SESSION_STARTED, {
            "format_version": config.LOG_FORMAT_VERSION,
            "camera": camera_index,
            "resolution": list(resolution) if resolution is not None else None,
        })]

    def update(self, result):
        """Eventos de um frame processado (`FrameResult`), já gravados, na ordem do arquivo."""
        self._require_session()
        now, moment = self._clock(), self._wall_clock()
        events = []
        if result.detections_updated:
            # Removidos antes dos novos: se um objeto sai e outro da mesma classe entra no
            # mesmo ciclo, o novo aparece com class_count 1, como primeiro da classe.
            for track in result.removed_tracks:
                events.append(self._object_disappeared(track.track_id, track.class_name, track.confidence, moment,
                                                       REASON_NOT_REDETECTED))
            for track in result.new_tracks:
                self._objects[track.track_id] = {
                    "class": track.class_name, "confidence": track.confidence, "appeared": moment, "last_seen": moment,
                }
                totals = self._classes.setdefault(
                    track.class_name, {"first_seen": moment, "appearances": 0, "max_simultaneous": 0, "intervals": []}
                )
                totals["appearances"] += 1
                events.append(self._emit(OBJECT_APPEARED, {
                    "class": track.class_name,
                    "track_id": track.track_id,
                    "confidence": rounded(track.confidence),
                    "class_count": self._class_count(track.class_name),
                }, moment))
            for track in result.tracks:
                known = self._objects.get(track.track_id)
                if known is not None:
                    known["confidence"] = track.confidence
                    if track.missed == 0:
                        known["last_seen"] = moment
            for class_name, detected in detected_per_class(result.tracks).items():
                totals = self._classes.get(class_name)
                if totals is not None:
                    totals["max_simultaneous"] = max(totals["max_simultaneous"], detected)
        self._tracks = list(result.tracks)

        self._frames_in_window += 1
        self._session_frames += 1
        if not result.is_calibrating:
            self._frames_observed += 1
            self._session_frames_observed += 1
            if result.motion_boxes:
                self._frames_with_motion += 1
                self._session_frames_with_motion += 1
        transition = self._motion.update(bool(result.motion_boxes), result.is_calibrating, now)
        if transition is not None:
            events.append(self._emit(transition, {}, moment))

        if now >= self._next_summary:
            events.append(self._summary(moment))
            # Agenda fixa a partir do início da sessão, sem acumular atraso; intervalos
            # inteiros sem frames (câmera travada) não geram registros atrasados.
            while self._next_summary <= now:
                self._next_summary += self.summary_interval
        return events

    def camera_disconnected(self, message):
        """Registra a desconexão da câmera; devolve os eventos gerados."""
        self._require_session()
        return [self._emit(CAMERA_DISCONNECTED, {"message": message})]

    def camera_reopened(self):
        """A câmera foi reaberta no meio da sessão e o tracker recomeça do zero.

        Os objetos ainda em aberto viram object_disappeared com motivo `camera_reopened`, e a
        contagem de frames seguidos com movimento recomeça; devolve os eventos gerados.
        """
        self._require_session()
        events = self._forget_objects(self._wall_clock(), REASON_CAMERA_REOPENED)
        self._tracks = []
        self._motion.interrupt()
        return events

    def end_session(self):
        """Encerra a sessão; devolve os eventos gerados, na ordem do arquivo.

        Grava o registro periódico da janela incompleta (se algum frame chegou nela), um
        object_disappeared com motivo `session_ended` para cada objeto ainda em aberto e
        `session_ended` com os totais da sessão, e fecha o arquivo.
        """
        self._require_session()
        now, moment = self._clock(), self._wall_clock()
        events = []
        if self._frames_in_window:
            events.append(self._summary(moment))
        events += self._forget_objects(moment, REASON_SESSION_ENDED)
        events.append(self._emit(SESSION_ENDED, session_totals(
            now - self._session_start,
            self._session_frames,
            self._session_frames_observed,
            self._session_frames_with_motion,
            self._classes,
        ), moment))
        self._session_active = False
        self._close_file()
        return events

    def _require_session(self):
        if not self._session_active:
            raise RuntimeError("a sessão do log não foi iniciada")

    def _class_count(self, class_name):
        return sum(1 for known in self._objects.values() if known["class"] == class_name)

    def _object_disappeared(self, track_id, class_name, confidence, moment, reason):
        known = self._objects.pop(track_id, None)
        if known is not None and class_name in self._classes:
            # Na precisão do log, para o tempo visível poder ser refeito exatamente a partir do arquivo.
            self._classes[class_name]["intervals"].append((to_log_precision(known["appeared"]), to_log_precision(known["last_seen"])))
        return self._emit(OBJECT_DISAPPEARED, {
            "class": class_name,
            "track_id": track_id,
            "confidence": rounded(confidence),
            "class_count": self._class_count(class_name),
            "last_seen": known["last_seen"] if known is not None else None,
            "reason": reason,
        }, moment)

    def _forget_objects(self, moment, reason):
        events = []
        for track_id in sorted(self._objects):
            known = self._objects[track_id]
            events.append(self._object_disappeared(track_id, known["class"], known["confidence"], moment, reason))
        return events

    def _summary(self, moment):
        motion, motion_ratio = motion_fields(self._frames_observed, self._frames_with_motion)
        event = self._emit(SUMMARY, {
            "classes": summarize_classes(self._tracks),
            "motion": motion,
            "motion_ratio": motion_ratio,
            "frames_observed": self._frames_observed,
            "frames_with_motion": self._frames_with_motion,
        }, moment)
        self._reset_window()
        return event

    def _reset_window(self):
        self._frames_in_window = 0
        self._frames_observed = 0
        self._frames_with_motion = 0

    def _reset_session_totals(self, start):
        self._session_start = start
        self._session_frames = 0
        self._session_frames_observed = 0
        self._session_frames_with_motion = 0
        # classe -> {"first_seen", "appearances", "max_simultaneous", "intervals"}
        self._classes = {}

    def _emit(self, event_type, data, moment=None):
        event = SceneEvent(event_type, moment if moment is not None else self._wall_clock(), data)
        self._write(event)
        return event

    def _open_file(self):
        self._close_file()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, "w", encoding="utf-8", newline="\n")
        except OSError as error:
            self._file = None
            self._warn_once(error)

    def _write(self, event):
        if self._file is None:
            return
        line = json.dumps(event_record(event), ensure_ascii=False)
        try:
            self._file.write(line + "\n")
            self._file.flush()
        except OSError as error:
            self._close_file()
            self._warn_once(error)

    def _close_file(self):
        if self._file is None:
            return
        try:
            self._file.close()
        except OSError:
            pass
        self._file = None

    def _warn_once(self, error):
        if self._warned:
            return
        self._warned = True
        self._warn(f"aviso: não foi possível gravar o log da sessão em {self.path} ({error}); a sessão continua sem log em disco")
