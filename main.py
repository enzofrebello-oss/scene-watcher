"""Ponto de entrada do Scene Watcher."""

import sys

import cv2

from src.frame_pipeline import CameraDisconnectedError, CameraOpenError, FramePipeline
from src.utils.model_files import ModelUnavailableError

WINDOW_TITLE = "Scene Watcher"
EXIT_KEYS = {ord("q"), ord("Q"), 27}  # 27 = ESC


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
        pipeline.open()
    except CameraOpenError as e:
        print(f"erro: {e}", file=sys.stderr)
        return 1

    width, height = pipeline.source.frame_size
    print(
        f"câmera {pipeline.source.index} aberta em {width}x{height}. "
        "pressione q ou ESC na janela para sair, ou Ctrl+C no terminal"
    )
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    try:
        while True:
            result = pipeline.process_next()
            if result is not None:
                cv2.imshow(WINDOW_TITLE, result.annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in EXIT_KEYS:
                break
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
    except CameraDisconnectedError as e:
        print(f"erro: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.close()
        cv2.destroyAllWindows()

    print("sessão encerrada, câmera liberada")
    return 0


def main():
    print("interface gráfica ainda não implementada, rodando em modo terminal")
    return run_terminal()


if __name__ == "__main__":
    sys.exit(main())
