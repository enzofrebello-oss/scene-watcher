"""Ponto de entrada do Scene Watcher."""

import sys

import cv2

from src.frame_pipeline import CameraDisconnectedError, CameraOpenError, FramePipeline
from src.scene_logger import SESSION_ENDED, SESSION_STARTED, event_level, terminal_line
from src.utils.model_files import ModelUnavailableError

WINDOW_TITLE = "Scene Watcher"
EXIT_KEYS = {ord("q"), ord("Q"), 27}  # 27 = ESC

# Abertura da câmera e fim da sessão já têm mensagens próprias no terminal; esses eventos
# ficam só no arquivo de log.
TERMINAL_SKIPPED_EVENTS = {SESSION_STARTED, SESSION_ENDED}


def print_events(events):
    """Imprime os eventos da sessão no terminal, sem os registros periódicos; erros vão para o stderr."""
    for event in events:
        if event.type in TERMINAL_SKIPPED_EVENTS:
            continue
        line = terminal_line(event)
        if line is not None:
            print(line, file=sys.stderr if event_level(event) == "ERRO" else sys.stdout)


def run_terminal():
    """Modo terminal: janela simples do OpenCV com o overlay. Devolve o código de saída do processo."""
    print("carregando modelos...")
    try:
        pipeline = FramePipeline()
        pipeline.warm_up()
    except ModelUnavailableError as e:
        print(f"erro: {e}", file=sys.stderr)
        return 1

    try:
        opening_events = pipeline.open()
    except CameraOpenError as e:
        print(f"erro: {e}", file=sys.stderr)
        return 1

    width, height = pipeline.source.frame_size
    print(
        f"câmera {pipeline.source.index} aberta em {width}x{height}. "
        "pressione q ou ESC na janela para sair, ou Ctrl+C no terminal"
    )
    print_events(opening_events)
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    try:
        while True:
            result = pipeline.process_next()
            if result is not None:
                print_events(result.events)
                cv2.imshow(WINDOW_TITLE, result.annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in EXIT_KEYS:
                break
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
    except CameraDisconnectedError:
        # A mensagem sai como evento [ERRO], impresso junto com o encerramento da sessão abaixo.
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        print_events(pipeline.close())
        cv2.destroyAllWindows()

    print("sessão encerrada, câmera liberada")
    return 0


def main():
    print("interface gráfica ainda não implementada, rodando em modo terminal")
    return run_terminal()


if __name__ == "__main__":
    sys.exit(main())
