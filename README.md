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

Enquanto a interface gráfica não existe, o app roda em modo terminal: abre uma janela simples do OpenCV com a imagem da câmera e o [overlay](#overlay). Para sair, pressione `q` ou `ESC` na janela, feche a janela, ou use `Ctrl+C` no terminal. A câmera é sempre liberada ao sair.

Se a câmera não abrir, ou parar de entregar frames por mais de `CAMERA_DISCONNECT_TIMEOUT_SECONDS` (padrão 2 s), o app encerra com uma mensagem explicando o problema e código de saída 1.

Por padrão, a imagem da câmera é espelhada na horizontal (`CAMERA_MIRROR`), como num espelho: a mão esquerda levantada aparece do lado esquerdo da tela. O espelhamento acontece logo após a captura, antes de qualquer detecção, então as caixas de objetos, rostos e movimento já saem na posição certa sobre a imagem espelhada.

## Testes

```powershell
.\.venv\Scripts\python.exe -m pytest
```

O `pytest.ini` na raiz coloca a pasta do projeto no caminho de importação, para que os testes em `tests/` importem `config` e `src`.

## Modelos

Os pesos ficam em `models/` e não são versionados:

- `models/yolov8n.pt`: YOLOv8n do Ultralytics.
- `models/face_detection_yunet_2023mar.onnx`: YuNet, do [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).

Na primeira execução, pesos ausentes são baixados automaticamente (é preciso internet uma única vez): o YOLOv8n pelo Ultralytics e o YuNet direto do OpenCV Zoo. O download é feito numa pasta temporária e o arquivo só vai para `models/` quando completo, então uma queda de conexão não deixa um peso corrompido. Se o download falhar, o app encerra explicando qual modelo faltou.

## Telemetria do Ultralytics

Por padrão, o ultralytics envia estatísticas anônimas de uso ao Google Analytics ao fim de cada detecção: versões das bibliotecas, modelo do processador, sistema operacional, nome do arquivo do modelo e tempos de inferência. Pelo código da biblioteca, nenhuma imagem, detecção ou caminho de arquivo é enviado.

O Scene Watcher desliga esse envio dentro do próprio processo (`src/object_detector.py`), sem alterar a configuração global do ultralytics na máquina: outros usos do ultralytics continuam seguindo o que estiver em `yolo settings`.

## Detecção de rosto

O COCO não tem uma classe de rosto, então os rostos vêm de um detector dedicado, o YuNet, rodando em paralelo ao YOLOv8n:

- cada rosto é uma detecção no mesmo formato das de objetos, com a classe `face`, exibida como "rosto";
- o limiar de score é próprio (`FACE_SCORE_THRESHOLD`, padrão 0.8), separado do limiar de confiança dos objetos;
- o frame é reduzido para 640 px de largura (`FACE_DETECTION_INPUT_WIDTH`) antes da detecção, e as caixas voltam para as coordenadas do frame original. Na CPU, com frames 1280×720, isso custa cerca de 8 ms por frame, contra cerca de 30 ms na resolução cheia. Quanto menor essa largura, mais cedo rostos distantes deixam de ser detectados;
- os pontos faciais (olhos, nariz, boca) que o YuNet também fornece são descartados.

## Detecção de movimento

Regiões em movimento são detectadas por subtração de fundo (MOG2), de forma independente do detector de objetos:

- o frame é reduzido para 640 px de largura (`MOTION_PROCESSING_WIDTH`) e as caixas voltam para as coordenadas do frame original;
- nos primeiros 30 frames (`MOTION_WARMUP_FRAMES`) o modelo só aprende o fundo e nenhum movimento é reportado. Esse estado é exibido como "calibrando fundo da cena..." e, com a câmera integrada a cerca de 16 frames novos por segundo, dura cerca de 2 s. Se a resolução do frame mudar, o fundo é aprendido de novo;
- a sensibilidade (`MOTION_SENSITIVITY`, de 0 a 1, padrão 0.5) controla o limiar de variação do MOG2 em escala logarítmica: 0 → 64, 0.5 → 16 (padrão do OpenCV), 1 → 4. Quanto mais sensível, menores as mudanças que contam como movimento;
- sombras são detectadas pelo MOG2 e não contam como movimento;
- a máscara passa por erosão (1×, remove ruído isolado) e dilatação (2×, junta partes do mesmo objeto) com núcleo 3×3;
- regiões menores que 0,1% da área do frame (`MOTION_MIN_AREA_FRACTION`) são descartadas;
- o fundo é modelado com os últimos 500 frames (`MOTION_HISTORY`, padrão do OpenCV): algo que fica parado por cerca de 30 s passa a fazer parte do fundo.

## Pipeline de processamento e FPS

A cada frame novo da câmera, o `FramePipeline` (`src/frame_pipeline.py`):

- roda a detecção de movimento, que é leve;
- a cada `DETECTION_FRAME_INTERVAL` frames (padrão 5), roda o YOLOv8n e o YuNet e atualiza o `DetectionTracker` com as detecções; nos frames intermediários, as últimas detecções e os objetos rastreados são reaproveitados;
- calcula o FPS como frames novos processados por segundo, com média móvel dos últimos `FPS_AVERAGE_WINDOW_FRAMES` frames (padrão 30);
- desenha o overlay numa cópia do frame. O `FrameResult` traz o frame limpo (`frame`) e o desenhado (`annotated_frame`).

Antes de abrir a câmera, os detectores rodam uma vez num frame vazio, para que o primeiro frame real não pague a inicialização dos modelos.

### Trade-off do throttling

Medido na CPU (sem GPU), com frames 1280×720 e a detecção de movimento em todo frame:

| Estratégia | FPS com a câmera integrada | FPS sem limite de câmera | Processamento médio por frame, sem limite de câmera |
|---|---|---|---|
| YOLOv8n e YuNet a cada 5 frames (padrão) | 16,1 | 85,9 | 11,6 ms |
| YOLOv8n a cada 5 frames, YuNet em todo frame | 16,5 | 51,7 | 19,4 ms |
| YOLOv8n e YuNet em todo frame | 16,7 | 20,1 | 49,6 ms |

- Com a câmera integrada na luz do ambiente de desenvolvimento, a própria câmera entrega cerca de 16 frames novos por segundo, e as três estratégias chegam ao mesmo FPS: o processamento ainda cabe no intervalo entre frames.
- A diferença aparece quando a câmera entrega mais frames (boa iluminação, câmeras de 30 fps ou mais). O teto medido com frames em memória, sem esperar a câmera, mostra que processar tudo em todo frame fica em cerca de 20 FPS, enquanto o padrão passa de 80.
- O custo do throttling é que objetos e rostos só são atualizados a cada 5 frames (cerca de 0,3 s a 16 FPS); entre uma detecção e outra, as caixas ficam na última posição conhecida.
- Os frames em que os detectores rodam são mais lentos que os demais: cerca de 50 a 55 ms com frames em memória e, com a câmera aberta, chegaram a 100 a 160 ms nas medições. Isso pode aparecer como uma pequena irregularidade a cada 5 frames.

## Rastreamento de detecções

O `DetectionTracker` (`src/detection_tracker.py`) mantém a identidade de cada objeto e rosto entre ciclos de detecção, para que as caixas não pisquem quando o detector falha em alguns ciclos:

- a cada ciclo de detecção (a cada `DETECTION_FRAME_INTERVAL` frames), as detecções são associadas aos objetos já rastreados da mesma classe pela sobreposição das caixas (IoU, função `calculate_iou` em `src/utils/geometry.py`). Só há associação com IoU de pelo menos `TRACKER_IOU_THRESHOLD` (padrão 0,3), e os pares de maior IoU são escolhidos primeiro;
- detecções sem par viram objetos novos, com uma identidade (`track_id`) que nunca é reaproveitada durante a execução;
- um objeto que não é redetectado continua na última posição conhecida por até `TRACKER_MAX_MISSED` ciclos seguidos (padrão 10) e é removido no ciclo seguinte. A tolerância é contada em ciclos de detecção, não em frames: com o intervalo padrão de 5 frames e cerca de 16 frames novos por segundo, 10 ciclos são cerca de 3 s. Por isso, a caixa de um objeto que sai de cena continua visível por até cerca de 3 s;
- rostos passam pelo tracker com a classe `face`. Como a associação é só entre caixas da mesma classe, um rosto nunca é confundido com a pessoa que o contém;
- o IoU usa coordenadas contínuas: a caixa (0, 0, 10, 10) tem área 100, caixas que só se tocam na borda têm IoU 0, e uma caixa de área zero tem IoU 0 com qualquer outra.

Limitação: a associação depende de sobreposição. Se um objeto se desloca tanto entre duas detecções que a caixa nova e a anterior ficam com IoU abaixo de 0,3 (por exemplo, atravessando o quadro rapidamente ou ficando alguns ciclos sem ser detectado enquanto se move), ele ganha uma identidade nova.

## Overlay

O `OverlayRenderer` (`src/overlay_renderer.py`) desenha sobre uma cópia de cada frame tudo o que aparece na área de câmera: na janela do modo terminal e, na fase 2, na interface. O frame original continua sem desenho, para recortes como a miniatura do objeto em foco. De trás para frente:

- **cantos de viewfinder**: quatro ângulos finos na cor de destaque (`#4FD1C5`), a 50% de opacidade;
- **caixas de movimento**: tracejado neutro (`#E8ECEE`) a 60% de opacidade, sem rótulo;
- **contador técnico**, no canto superior esquerdo: frame desde a abertura da câmera, FPS com média móvel e quantidade de objetos rastreados visíveis. Enquanto o fundo da cena é aprendido, aparece a linha "calibrando fundo da cena...";
- **painel de objetos**, no canto superior direito: uma linha por classe, na ordem em que as classes apareceram, com a contagem entre parênteses quando há mais de um objeto e a maior confiança entre eles, por exemplo "1. celular (2) 0.89". Sem objetos, mostra "nenhum objeto detectado no momento";
- **caixas de objetos e rostos**, na cor da classe, com um rótulo com a classe traduzida e a confiança.

O contador e o painel usam fonte monoespaçada na cor de destaque, menor que a dos rótulos, com o texto a 85% e fundo `#0F1417` a 55% de opacidade, para que as caixas continuem sendo o elemento visual mais forte.

### Cores por classe

Cada classe usa uma das 9 cores de `OVERLAY_CLASS_PALETTE`, escolhida pelo crc32 do nome da classe, então a cor é a mesma em qualquer execução. A paleta exclui tons próximos das cores de destaque, alerta e erro da interface. Com 81 classes e 9 cores, classes diferentes podem ter a mesma cor.

### Objetos mantidos sem redetecção

A caixa e o rótulo de um objeto que o detector deixou de encontrar esmaecem em degraus, a cada ciclo de detecção sem redetecção: a opacidade é 1 − ciclos/11, ou seja, 100% quando o objeto foi detectado no último ciclo e cerca de 9% no 10º ciclo, logo antes de o tracker removê-lo. Esses objetos continuam contando no contador e no painel até serem removidos. Uma linha do painel só esmaece quando todos os objetos daquela classe estão esmaecendo.

### Classes ocultas

`OVERLAY_HIDDEN_CLASSES` (padrão: `person`) tira do overlay a caixa, o rótulo e a linha do painel dessas classes: a caixa do corpo inteiro é grande e disputa atenção com rostos e objetos. Essas classes continuam sendo detectadas e rastreadas.

### Rótulos, fontes e tamanhos

- O rótulo fica logo acima da caixa ou, sem espaço, dentro dela. Se cair sobre outro rótulo, o contador ou o painel, desce até ficar livre. Objetos detectados no último ciclo escolhem a posição primeiro e são desenhados por cima.
- O nome da classe usa a fonte de interface e a confiança, a fonte monoespaçada, com os algarismos na mesma altura das letras. Com Inter e IBM Plex Mono instaladas, o overlay usa essas fontes; sem elas, Segoe UI e Consolas. As fontes são carregadas e aquecidas antes de abrir a câmera.
- Os tamanhos em `config.py` valem para frames de 720 px de altura e são escalados proporcionalmente em outras resoluções.

### Custo

Na CPU, sem outra carga, o overlay de uma cena com 5 objetos e uma região de movimento leva cerca de 1,2 ms por frame em 1280×720, dos quais cerca de 0,6 ms são a cópia do frame. Com a câmera e os detectores rodando, a média medida ficou entre 3 e 4 ms, e o FPS continuou limitado pela câmera (entre 14,6 e 16,2 nas medições, conforme a iluminação).

### Limitações

- O tracejado de movimento, claro e translúcido, fica pouco visível em cenas muito claras.
- Cores vizinhas da paleta, como amarelo e lima, ou verde e esmeralda, podem ser confundidas em caixas finas.

## Classes detectadas

O YOLOv8n detecta as 80 classes do COCO. Internamente (detecções, logs) cada classe é identificada pelo nome original do COCO; na interface e no overlay aparece a tradução abaixo, definida em `src/coco_labels.py`. O rosto, que vem do YuNet, usa o identificador `face` e aparece como "rosto".

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

Atenção: no Windows, assim que é importado, o ultralytics substitui `cv2.imread`, `cv2.imwrite` e `cv2.imshow` no processo inteiro por versões próprias que aceitam caminhos com acento. O projeto não depende disso. Os contornos acima continuam necessários, porque o YuNet e o `cv2.FontFace`, por exemplo, não passam por essas funções. Além disso, um `cv2.imwrite` que funciona dentro do app pode falhar num script ou teste que não importa o ultralytics.

## Estrutura de pastas

```
scene-watcher/
├── requirements.txt
├── pytest.ini                   # põe a raiz do projeto no caminho de importação dos testes
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
│   ├── utils/
│   │   ├── geometry.py
│   │   ├── model_files.py       # garante os pesos, com download na primeira execução
│   │   └── opencv_logging.py    # supressão pontual de avisos do OpenCV
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
