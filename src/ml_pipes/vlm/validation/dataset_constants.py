"""Shared benchmark definitions for dataset preparation and loading."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetInfo:
    task: str
    splits: tuple[str, ...]


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

REFCOCO_SPLITS = {
    "refcocog": ("train", "val", "test"),
    "refcoco": ("train", "val", "testA", "testB"),
    "refcoco+": ("train", "val", "testA", "testB"),
}

DATASETS = {
    "cifar10": DatasetInfo("classify", ("train", "val", "test")),
    "coco128": DatasetInfo("detect", ("train", "val")),
    "refcocog": DatasetInfo("ground", REFCOCO_SPLITS["refcocog"]),
    "ovad": DatasetInfo("attributes", ("test", "val")),
    "vqav2": DatasetInfo("answer", ("train", "val")),
    "coco_captions2017": DatasetInfo("describe", ("train", "val")),
}

DEFAULT_DATASETS = {
    "classify": "cifar10", "detect": "coco128", "ground": "refcocog",
    "attributes": "ovad", "answer": "vqav2", "describe": "coco_captions2017",
}

CIFAR_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck",
)

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"

CIFAR_CHECKSUM = "md5:c32a1d4ab5d03f1284b67883e8d87530"

COCO_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)

COCO_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"

COCO_CHECKSUM = "sha256:61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e"

MDETR_URL = "https://zenodo.org/records/4729015/files/mdetr_annotations.tar.gz?download=1"

COCO_IMAGE_URL = "https://s3.amazonaws.com/images.cocodataset.org/train2014/"

OVAD_GROUPS = (
    "cleanliness", "clothes pattern", "color quantity", "cooked", "face expression",
    "gender", "group", "hair color", "hair length", "hair tone", "hair type", "length",
    "material", "maturity", "optical property", "order", "pattern", "position", "size",
    "state", "texture", "tone",
)

OVAD_EXCLUDED = {"color", "clothes color"}

OVAD_PROTOCOL = "OVAD categorical object attributes (22 groups; excludes color and clothes color); not official mAP"

OVAD_URL = "https://lmb.informatik.uni-freiburg.de/resources/datasets/ovad.zip"

OVAD_CHECKSUM = "sha256:1716fc5216dd040c3803ca7d91444140d02bc6a98f40a60b85f16c97174dc1ec"

OVAD_IMAGE_URL = "https://s3.amazonaws.com/images.cocodataset.org/val2017/"

VQA_URL = "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_{kind}_{split}_mscoco.zip"

VQA_IMAGE_URL = "https://s3.amazonaws.com/images.cocodataset.org/{split}2014/"

CAPTIONS_URL = "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"

CAPTIONS_IMAGE_URL = "https://s3.amazonaws.com/images.cocodataset.org/{split}2017/"

PREPARATION_VERSION = 2
