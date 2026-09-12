"""Texto exibido na interface para cada classe detectada.

`COCO_LABELS_PT` traduz as 80 classes do COCO detectadas pelo YOLOv8n e
`EXTRA_LABELS_PT` traduz classes que vêm de outros detectores, como o rosto do
YuNet. As chaves são os identificadores usados internamente; os valores são o texto
exibido, sempre em minúsculas.
"""

COCO_LABELS_PT = {
    "person": "pessoa",
    "bicycle": "bicicleta",
    "car": "carro",
    "motorcycle": "moto",
    "airplane": "avião",
    "bus": "ônibus",
    "train": "trem",
    "truck": "caminhão",
    "boat": "barco",
    "traffic light": "semáforo",
    "fire hydrant": "hidrante",
    "stop sign": "placa de pare",
    "parking meter": "parquímetro",
    "bench": "banco",
    "bird": "pássaro",
    "cat": "gato",
    "dog": "cachorro",
    "horse": "cavalo",
    "sheep": "ovelha",
    "cow": "vaca",
    "elephant": "elefante",
    "bear": "urso",
    "zebra": "zebra",
    "giraffe": "girafa",
    "backpack": "mochila",
    "umbrella": "guarda-chuva",
    "handbag": "bolsa",
    "tie": "gravata",
    "suitcase": "mala",
    "frisbee": "frisbee",
    "skis": "esquis",
    "snowboard": "snowboard",
    "sports ball": "bola",
    "kite": "pipa",
    "baseball bat": "taco de beisebol",
    "baseball glove": "luva de beisebol",
    "skateboard": "skate",
    "surfboard": "prancha de surfe",
    "tennis racket": "raquete de tênis",
    "bottle": "garrafa",
    "wine glass": "taça",
    "cup": "copo",
    "fork": "garfo",
    "knife": "faca",
    "spoon": "colher",
    "bowl": "tigela",
    "banana": "banana",
    "apple": "maçã",
    "sandwich": "sanduíche",
    "orange": "laranja",
    "broccoli": "brócolis",
    "carrot": "cenoura",
    "hot dog": "cachorro-quente",
    "pizza": "pizza",
    "donut": "rosquinha",
    "cake": "bolo",
    "chair": "cadeira",
    "couch": "sofá",
    "potted plant": "planta",
    "bed": "cama",
    "dining table": "mesa",
    "toilet": "vaso sanitário",
    "tv": "televisão",
    "laptop": "notebook",
    "mouse": "mouse",
    "remote": "controle remoto",
    "keyboard": "teclado",
    "cell phone": "celular",
    "microwave": "micro-ondas",
    "oven": "forno",
    "toaster": "torradeira",
    "sink": "pia",
    "refrigerator": "geladeira",
    "book": "livro",
    "clock": "relógio",
    "vase": "vaso",
    "scissors": "tesoura",
    "teddy bear": "ursinho de pelúcia",
    "hair drier": "secador de cabelo",
    "toothbrush": "escova de dentes",
}

# Classes que não são do COCO, vindas de outros detectores.
EXTRA_LABELS_PT = {
    "face": "rosto",
}


def translate(class_name):
    """Texto exibido para uma classe; sem tradução conhecida, devolve o nome original."""
    if class_name in COCO_LABELS_PT:
        return COCO_LABELS_PT[class_name]
    return EXTRA_LABELS_PT.get(class_name, class_name)
