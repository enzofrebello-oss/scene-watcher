"""Pesos dos modelos: garante que existem localmente, baixando na primeira execução."""

import shutil
import tempfile
import urllib.request
from pathlib import Path

import config


class ModelUnavailableError(RuntimeError):
    """O arquivo de pesos não existe e não pôde ser baixado."""


def url_downloader(url, timeout=config.MODEL_DOWNLOAD_TIMEOUT_SECONDS):
    """Cria uma função de download para `ensure_weights` que baixa `url` por HTTP."""

    def download(destination):
        with urllib.request.urlopen(url, timeout=timeout) as response, open(destination, "wb") as out:
            shutil.copyfileobj(response, out)
        return destination

    return download


def ensure_weights(path, download):
    """Garante que o arquivo de pesos existe em `path`, baixando na primeira execução.

    `download(destino)` deve gravar o arquivo em `destino` e devolver onde ele ficou;
    se devolver outro caminho (ex. um cache global), o arquivo é copiado de lá.

    O download é feito numa pasta temporária e o arquivo só é movido para o destino
    quando completo, para que um download interrompido não deixe um peso corrompido
    em `models/`.
    """
    path = Path(path)
    if path.is_file():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="scene_watcher_model_"))
    try:
        tmp_file = tmp_dir / path.name
        downloaded = download(tmp_file)
        if downloaded is not None and Path(downloaded) != tmp_file:
            shutil.copy2(downloaded, tmp_file)
        if not tmp_file.is_file() or tmp_file.stat().st_size == 0:
            raise RuntimeError("o download terminou sem gerar o arquivo")
        shutil.move(str(tmp_file), str(path))
    except Exception as e:
        raise ModelUnavailableError(
            f"modelo {path.name} não encontrado em {path.parent} e o download automático "
            f"falhou, verifique a conexão com a internet ({e})"
        ) from e
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return path
