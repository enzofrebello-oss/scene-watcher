# Scene Watcher

Aplicativo de desktop que usa a câmera para, em tempo real, identificar o que aparece no campo de visão (objetos, pessoas, animais e rostos) e detectar regiões em movimento na cena.

> Em desenvolvimento. As seções sobre throttling, limiares, formato de eventos, classes do COCO e atalhos são adicionadas conforme cada etapa é implementada.

## Stack

| Dependência | Uso |
|---|---|
| Python 3.11+ (desenvolvido em 3.14) | linguagem |
| `opencv-python` (5.x) | captura de vídeo, detecção de movimento (MOG2), detecção de rosto (YuNet) e texto do overlay com fonte TTF via `cv2.FontFace`, que exibe acentos corretamente ("avião", "ônibus") |
| `ultralytics` (YOLOv8n) | detecção de objetos pré-treinada nas 80 classes do COCO |
| **YuNet** (`face_detection_yunet_2023mar.onnx`, via `cv2.FaceDetectorYN`) | detector de rosto dedicado e leve, rodando em paralelo ao YOLO (o COCO não tem classe "rosto") |
| `numpy` | manipulação de frames |
| `pytest` | testes |
| `PyQt6`, `pyqtgraph` | interface gráfica (fase 2) |

## Instalação

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Execução

```powershell
.\.venv\Scripts\python.exe main.py
```

Enquanto a interface gráfica não existe, o app roda em modo terminal: abre uma janela simples do OpenCV com a imagem da câmera. Para sair, pressione `q` ou `ESC` na janela, feche a janela, ou use `Ctrl+C` no terminal. A câmera é sempre liberada ao sair.

Se a câmera não abrir, ou parar de entregar frames por mais de `CAMERA_DISCONNECT_TIMEOUT_SECONDS` (padrão 2 s), o app encerra com uma mensagem explicando o problema e código de saída 1.

## Modelos

Os pesos ficam em `models/` e não são versionados:

- `models/yolov8n.pt`: YOLOv8n do Ultralytics.
- `models/face_detection_yunet_2023mar.onnx`: YuNet, do [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).

Na primeira execução, pesos ausentes são baixados automaticamente (é preciso internet uma única vez). O download é feito numa pasta temporária e o arquivo só vai para `models/` quando completo, então uma queda de conexão não deixa um peso corrompido. Se o download falhar, o app encerra explicando qual modelo faltou.

## Classes detectadas

O YOLOv8n detecta as 80 classes do COCO. Internamente (detecções, logs) cada classe é identificada pelo nome original do COCO; na interface e no overlay aparece a tradução abaixo, definida em `src/coco_labels.py`.

| COCO | Interface | COCO | Interface |
|---|---|---|---|
| person | pessoa | elephant | elefante |
| bicycle | bicicleta | bear | urso |
| car | carro | zebra | zebra |
| motorcycle | moto | giraffe | girafa |
| airplane | avião | backpack | mochila |
| bus | ônibus | umbrella | guarda-chuva |
| train | trem | handbag | bolsa |
| truck | caminhão | tie | gravata |
| boat | barco | suitcase | mala |
| traffic light | semáforo | frisbee | frisbee |
| fire hydrant | hidrante | skis | esquis |
| stop sign | placa de pare | snowboard | snowboard |
| parking meter | parquímetro | sports ball | bola |
| bench | banco | kite | pipa |
| bird | pássaro | baseball bat | taco de beisebol |
| cat | gato | baseball glove | luva de beisebol |
| dog | cachorro | skateboard | skate |
| horse | cavalo | surfboard | prancha de surfe |
| sheep | ovelha | tennis racket | raquete de tênis |
| cow | vaca | bottle | garrafa |
| wine glass | taça | dining table | mesa |
| cup | copo | toilet | vaso sanitário |
| fork | garfo | tv | televisão |
| knife | faca | laptop | notebook |
| spoon | colher | mouse | mouse |
| bowl | tigela | remote | controle remoto |
| banana | banana | keyboard | teclado |
| apple | maçã | cell phone | celular |
| sandwich | sanduíche | microwave | micro-ondas |
| orange | laranja | oven | forno |
| broccoli | brócolis | toaster | torradeira |
| carrot | cenoura | sink | pia |
| hot dog | cachorro-quente | refrigerator | geladeira |
| pizza | pizza | book | livro |
| donut | rosquinha | clock | relógio |
| cake | bolo | vase | vaso |
| chair | cadeira | scissors | tesoura |
| couch | sofá | teddy bear | ursinho de pelúcia |
| potted plant | planta | hair drier | secador de cabelo |
| bed | cama | toothbrush | escova de dentes |

## Caminhos com caracteres não-ASCII

No Windows, algumas funções do OpenCV que abrem arquivos diretamente falham em pastas com acentos: `cv2.imwrite` retorna `False` sem gravar nada, e o carregamento de modelos ONNX por caminho (YuNet) não encontra o arquivo. O projeto contorna isso para funcionar em qualquer pasta:

- imagens são codificadas com `cv2.imencode` e gravadas pelo Python;
- modelos ONNX são lidos pelo Python e entregues ao OpenCV como buffer em memória;
- vídeos são gravados numa pasta temporária e movidos para `data/clips/` só depois de finalizados. O `cv2.VideoWriter` do OpenCV 5 já aceita caminhos com acento, mas gravar fora do destino garante que um clipe incompleto nunca apareça em `data/clips/` enquanto ainda está sendo escrito;
- fontes TTF do overlay são lidas das pastas de fontes do sistema. O `cv2.FontFace` não abre fontes em pastas com acento e, nesse caso, cai silenciosamente para a fonte embutida do OpenCV, que também exibe acentos.

## Estrutura de pastas

```
scene-watcher/
├── requirements.txt
├── README.md
├── config.py
├── main.py
├── models/
│   ├── yolov8n.pt
│   └── face_detection_yunet_2023mar.onnx
├── src/
│   ├── object_detector.py
│   ├── coco_labels.py           # tradução das 80 classes do COCO
│   ├── face_detector.py
│   ├── motion_detector.py
│   ├── detection_tracker.py
│   ├── frame_pipeline.py        # processamento sem dependência de Qt
│   ├── overlay_renderer.py
│   ├── scene_logger.py
│   ├── alert_manager.py
│   ├── clip_recorder.py
│   ├── settings_store.py
│   ├── utils/geometry.py
│   └── ui/
│       ├── main_window.py
│       ├── pipeline_worker.py   # QThread fina em volta do frame_pipeline
│       ├── camera_view.py
│       ├── info_panel.py
│       ├── event_feed.py
│       ├── controls_bar.py
│       ├── roi_selector.py
│       ├── history_chart.py
│       ├── toast.py
│       ├── shortcuts.py
│       └── theme.py
├── data/
│   ├── logs/session_log.jsonl
│   ├── clips/
│   └── settings.json
└── tests/
    ├── test_geometry.py
    ├── test_detection_tracker.py
    ├── test_alert_manager.py
    └── test_roi_selector.py
```
