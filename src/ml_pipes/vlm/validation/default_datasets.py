"""Default validation datasets: source definitions, annotation joins, and HF loading.

HF builder classes are defined inside loaders so importing validation does not
require the optional dataset loading dependencies.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ._caption import validate_captions
from ._vqa import validate_targets
from .data import _limit

if TYPE_CHECKING:
    from datasets import Dataset


@dataclass(frozen=True)
class DatasetInfo:
    task: str
    splits: tuple[str, ...]


DATASETS = {
    "cifar10": DatasetInfo("classify", ("train", "val", "test")),
    "coco128": DatasetInfo("detect", ("train", "val")),
    "refcocog": DatasetInfo("ground", ("train", "val", "test")),
    "ovad": DatasetInfo("attributes", ("test", "val")),
    "vqav2": DatasetInfo("answer", ("train", "val")),
    "coco_captions2017": DatasetInfo("describe", ("train", "val")),
}

DEFAULT_DATASETS = {
    "classify": "cifar10", "detect": "coco128", "ground": "refcocog",
    "attributes": "ovad", "answer": "vqav2", "describe": "coco_captions2017",
}


# Shared loading options

def _utils():
    try:
        from . import _dataset_utils
    except ImportError as exc:
        raise ImportError("Dataset loaders require: pip install 'ml-pipes-vlm[validation]'") from exc
    return _dataset_utils


def _split(split, allowed=None):
    if (not isinstance(split, str) or not split or Path(split).name != split or split in {".", ".."}):
        raise ValueError("split must be a nonempty split name")
    if allowed is not None and split not in allowed:
        raise ValueError(f"Unsupported split {split!r}; expected {allowed}")


def _local_files(annotations, images):
    if annotations is None:
        if images is not None:
            raise ValueError("images requires local annotations")
        return None, None
    if images is None:
        raise ValueError("Local annotations require an images directory")
    return str(Path(annotations).expanduser().resolve()), str(Path(images).expanduser().resolve())


# Classification

CIFAR_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck",
)

CIFAR_REPO = "uoft-cs/cifar10"

CIFAR_REVISION = "0b2714987fa478483af9968de7c934580d0bb9a2"


def load_cifar10(*, split: str = "test", classes=None, max_samples: int | None = None, cache_dir=None) -> Dataset:
    """Load the pinned HF CIFAR-10 dataset; val aliases the held-out test split."""
    _limit(max_samples)
    _split(split, DATASETS["cifar10"].splits)
    if classes is not None and (not classes or any(name not in CIFAR_NAMES for name in classes)):
        raise ValueError("classes must contain supported CIFAR-10 class names")
    utils = _utils()
    hf = utils.hf
    dataset = hf.load_dataset(CIFAR_REPO, revision=CIFAR_REVISION,
                              split="test" if split == "val" else split, cache_dir=cache_dir)
    dataset = dataset.rename_column("img", "image")
    if classes is not None:
        ids = {dataset.features["label"].str2int(name) for name in classes}
        dataset = dataset.filter(lambda label: label in ids, input_columns=["label"])
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def load_classification(root: str | Path, *, split: str = "val", class_ids=None,
                        max_samples: int | None = None, cache_dir=None) -> Dataset:
    """Load local class folders with HF imagefolder and a complete vocabulary."""
    _limit(max_samples)
    _split(split)
    utils = _utils()
    hf = utils.hf
    directory = Path(root).expanduser().resolve() / split
    if not directory.is_dir():
        raise FileNotFoundError(f"Classification split not found: {directory}")
    names = sorted(path.name for path in directory.iterdir() if path.is_dir())
    if not names:
        raise ValueError(f"No classification images found in {directory}")
    selected = None
    if class_ids is not None:
        try:
            ids = tuple(class_ids)
        except TypeError as exc:
            raise ValueError("class_ids must contain known integer class IDs") from exc
        if (isinstance(class_ids, (str, bytes)) or not ids
                or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                       or not 0 <= value < len(names) for value in ids)):
            raise ValueError("class_ids must contain known integer class IDs")
        selected = set(ids)
    try:
        dataset = hf.load_dataset("imagefolder", data_files={split: str(directory / "**" / "*")},
                                  split=split, cache_dir=cache_dir,
                                  features=hf.Features({"image": hf.Image(), "label": hf.ClassLabel(names=names)}),
                                  drop_labels=False)
    except hf.data_files.EmptyDatasetError as exc:
        raise ValueError(f"No classification images found in {directory}") from exc
    dataset = dataset.cast_column("image", hf.Image(decode=False)).map(_image_paths, batched=True)
    if selected is not None:
        dataset = dataset.filter(lambda label: label in selected, input_columns=["label"])
    if not len(dataset):
        raise ValueError("No samples belong to the selected classification classes")
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset.cast_column("image", hf.Image())


def _image_paths(batch):
    return {"im_file": [image["path"] for image in batch["image"]]}


# Detection: COCO128 and local YOLO

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

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


def load_coco128(*, split: str = "val", max_samples: int | None = None, cache_dir=None) -> Dataset:
    """Load the original COCO128 archive through an HF builder."""
    _limit(max_samples)
    _split(split, DATASETS["coco128"].splits)
    utils = _utils()
    hf = utils.hf

    class COCO128Builder(utils._RowsBuilder):
        """COCO128: train and val share the same 128 COCO training images."""

        def _split_generators(self, manager):
            archive = manager.download(COCO_URL)
            utils.check_checksum(archive, COCO_CHECKSUM)
            directory = Path(manager.extract(archive)) / "coco128" / "images" / "train2017"
            names = dict(enumerate(COCO_NAMES))
            rows = _detection_rows(_image_files(directory), names, self.config.max_samples)
            return self._split(rows, _detection_features(names.values(), hf))

    return utils._build(COCO128Builder, name="coco128", split=split, max_samples=max_samples, cache_dir=cache_dir)


def load_yolo(data: str | Path, *, split: str = "val", max_samples: int | None = None, cache_dir=None) -> Dataset:
    """Load local YOLO YAML/TXT into canonical image/objects rows."""
    _limit(max_samples)
    _split(split)
    utils = _utils()
    hf = utils.hf

    class YOLOBuilder(utils._RowsBuilder):
        """Local YOLO YAML/TXT annotations with original-image pixel xyxy boxes."""

        def _split_generators(self, manager):
            import yaml

            path = Path(self.config.data_files["annotations"][0])
            config = yaml.safe_load(path.read_text())
            if not isinstance(config, dict):
                raise ValueError(f"Expected a dataset YAML mapping: {path}")
            names = _names(config.get("names"))
            root_value = config.get("path", ".")
            if not isinstance(root_value, str) or not root_value:
                raise ValueError("Dataset path must be a nonempty string")
            root = utils._resolve(root_value, path.parent)
            entries = config.get(self.config.source_split)
            if isinstance(entries, str):
                entries = [entries]
            if (not isinstance(entries, list) or not entries
                    or any(not isinstance(entry, str) or not entry for entry in entries)):
                raise ValueError(f"Dataset YAML must define a nonempty {self.config.source_split!r} split")
            images = sorted({image for entry in entries for image in _detection_images(entry, root)})
            rows = _detection_rows(images, names, self.config.max_samples)
            return self._split(rows, _detection_features(names.values(), hf))

    return utils._build(YOLOBuilder, name="yolo", split=split, max_samples=max_samples, cache_dir=cache_dir,
                        files={"annotations": str(Path(data).expanduser().resolve())})


def _names(value: object) -> dict[int, str]:
    if isinstance(value, list):
        names = dict(enumerate(value))
    elif isinstance(value, Mapping):
        names = dict(value)
    else:
        raise ValueError("Detection YAML must define names as a list or integer-keyed mapping")
    if (
        not names
        or any(type(key) is not int for key in names)
        or set(names) != set(range(len(names)))
        or any(not isinstance(name, str) or not name.strip() for name in names.values())
        or len(set(names.values())) != len(names)
    ):
        raise ValueError("Dataset names must be unique nonempty strings with consecutive class IDs from zero")
    return dict(sorted(names.items()))


def _detection_images(entry: str, root: Path) -> list[Path]:
    utils = _utils()
    path = utils._resolve(entry, root)
    if path.is_dir():
        return _image_files(path)
    if not path.is_file():
        raise FileNotFoundError(f"Detection images or image list not found: {path}")
    if path.suffix.lower() == ".txt":
        images = [
            utils._resolve(line.strip(), path.parent)
            for line in path.read_text().splitlines() if line.strip()
        ]
    elif path.suffix.lower() in IMAGE_SUFFIXES:
        images = [path]
    else:
        raise ValueError(f"Expected an image directory, image, or TXT image list: {path}")
    for image in images:
        if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Invalid image entry in {path}: {image}")
    return images


def _image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    return sorted(
        path.resolve() for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _detection_rows(images, names, max_samples):
    from PIL import Image

    rows = []
    for path in images[:max_samples]:
        categories, boxes = _read_labels(_label_path(path), names)
        with Image.open(path) as image:
            width, height = image.size
        rows.append({
            "image": str(path), "im_file": str(path),
            "objects": {"bbox": (boxes * [width, height, width, height]).tolist(),
                        "category": categories.tolist()},
        })
    return rows


def _read_labels(path: Path, names: dict[int, str]) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return np.empty(0, dtype=np.int64), np.empty((0, 4))
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = np.asarray([float(part) for part in line.split()], dtype=np.float64)
            if values.shape != (5,) or not np.isfinite(values).all():
                raise ValueError("expected five finite values: class x_center y_center width height")
            class_id, x, y, w, h = values
            if class_id != int(class_id) or int(class_id) not in names:
                raise ValueError("unknown or noninteger class ID")
            box = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
            # YOLO TXT's rounded centers and sizes can put an edge a fraction
            # of a pixel outside the image. Preserve the annotation values.
            if w <= 0 or h <= 0 or (box < -1e-6).any() or (box > 1 + 1e-6).any():
                raise ValueError("expected positive normalized boxes within the image")
        except ValueError as exc:
            raise ValueError(f"Invalid detection annotation {path}:{line_number}: {exc}") from exc
        rows.append((int(class_id), box))
    return (
        np.array([row[0] for row in rows], dtype=np.int64),
        np.array([row[1] for row in rows], dtype=np.float64).reshape(-1, 4),
    )


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    # Match the final 'images' directory, as in YOLO's image-to-label convention.
    indices = [i for i, part in enumerate(parts[:-1]) if part == "images"]
    if indices:
        parts[indices[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def _detection_features(names, hf):
    return {"objects": {
        "bbox": hf.List(hf.List(hf.Value("float64"), length=4)),
        "category": hf.List(hf.ClassLabel(names=list(names))),
    }}


# Grounding: RefCOCO

MDETR_URL = "https://zenodo.org/records/4729015/files/mdetr_annotations.tar.gz?download=1"

COCO_IMAGE_URL = "https://s3.amazonaws.com/images.cocodataset.org/train2014/"


def load_refcocog(*, split: str = "val", max_samples: int | None = None, cache_dir=None) -> Dataset:
    """Load original RefCOCOg UMD annotations and selected COCO images."""
    _limit(max_samples)
    _split(split, DATASETS["refcocog"].splits)
    return _load_refcoco(dataset="refcocog", split=split, max_samples=max_samples, cache_dir=cache_dir)


def load_refcoco(annotations: str | Path, images: str | Path, *, dataset: str = "refcocog",
                 split: str = "val", max_samples: int | None = None, cache_dir=None) -> Dataset:
    """Load local MDETR RefCOCO annotations, one row per expression."""
    _limit(max_samples)
    if not isinstance(dataset, str) or dataset not in ("refcocog", "refcoco", "refcoco+"):
        raise ValueError("dataset must be refcocog, refcoco, or refcoco+")
    _split(split)
    path = Path(annotations).expanduser().resolve() / f"finetune_{dataset}_{split}.json"
    return _load_refcoco(dataset=dataset, split=split, max_samples=max_samples, cache_dir=cache_dir,
                         files={"annotations": str(path)}, images=str(Path(images).expanduser().resolve()))


def _load_refcoco(*, dataset, split, max_samples, cache_dir, files=None, images=None):
    utils = _utils()
    hf = utils.hf

    class RefCOCOBuilder(utils._RowsBuilder):
        """MDETR RefCOCO annotations; RefCOCOg uses UMD splits, RefCOCO/+ use UNC."""

        def _split_generators(self, manager):
            filename = f"finetune_{self.config.name}_{self.config.source_split}.json"
            if self.config.data_files:
                data = json.loads(Path(self.config.data_files["annotations"][0]).read_text())
            else:
                data = utils._json_member(manager, manager.download(MDETR_URL), filename)
            rows, names = _refcoco_selection(data)
            rows = utils._images(rows[:self.config.max_samples], directory=self.config.data_dir,
                                 manager=manager, prefix=COCO_IMAGE_URL)
            return self._split(rows, {
                "text": hf.Value("string"), "bbox": hf.List(hf.Value("float64"), length=4),
                "label": hf.ClassLabel(names=names), "sample_id": hf.Value("int64"),
            })

    return utils._build(RefCOCOBuilder, name=dataset, split=split, max_samples=max_samples,
                        cache_dir=cache_dir, files=files, images=images)


def _refcoco_selection(data):
    """Validate MDETR joins and emit one canonical row per expression."""
    if not isinstance(data, dict):
        raise ValueError("Expected a grounding annotation mapping")
    queries = _indexed_records(data, "images")
    categories = _indexed_records(data, "categories")
    targets = _indexed_records(data, "annotations")
    names = [item.get("name") for item in categories.values()]
    if (not names or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("Grounding category names must be nonempty and unique")
    category_ids = {key: index for index, key in enumerate(categories)}
    query_targets = {}
    for annotation in targets.values():
        query_id = annotation.get("image_id")
        if type(query_id) is not int or query_id not in queries:
            raise ValueError(f"Orphan grounding annotation: {annotation['id']}")
        if query_id in query_targets:
            raise ValueError(f"Multiple target annotations for grounding query {query_id}")
        query_targets[query_id] = annotation
    rows = []
    for query_id, query in queries.items():
        annotation = query_targets.get(query_id)
        if annotation is None:
            raise ValueError(f"Missing target annotation for grounding query {query_id}")
        text = query.get("caption")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Invalid expression for grounding query {query_id}")
        size = (query.get("width"), query.get("height"))
        if any(type(value) is not int or value <= 0 for value in size):
            raise ValueError(f"Invalid image dimensions for grounding query {query_id}")
        filename = query.get("file_name")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(f"Invalid image filename for grounding query {query_id}")
        category_id = annotation.get("category_id")
        if type(category_id) is not int or category_id not in category_ids:
            raise ValueError(f"Unknown category for grounding query {query_id}")
        bbox = annotation.get("bbox")
        if (not isinstance(bbox, list) or len(bbox) != 4
                or any(type(value) not in (int, float) for value in bbox)):
            raise ValueError(f"Invalid box for grounding query {query_id}: expected numeric xywh")
        x, y, width, height = bbox
        box = [x, y, x + width, y + height]
        if (not np.isfinite(box).all() or width <= 0 or height <= 0
                or box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError(f"Invalid box for grounding query {query_id}: expected finite positive dimensions")
        rows.append({
            "file_name": filename, "image_size": size, "text": text,
            "bbox": box, "label": category_ids[category_id], "sample_id": query_id,
        })
    if not rows:
        raise ValueError("No grounding expressions found")
    return rows, names


def _indexed_records(data: dict, key: str) -> dict[int, dict]:
    records = data.get(key)
    if not isinstance(records, list):
        raise ValueError(f"Grounding annotations must define a {key} list")
    indexed = {}
    for record in records:
        if not isinstance(record, dict) or type(record.get("id")) is not int:
            raise ValueError(f"Grounding {key} records require integer IDs")
        if record["id"] in indexed:
            raise ValueError(f"Duplicate grounding {key} ID: {record['id']}")
        indexed[record["id"]] = record
    return indexed


# Attributes: OVAD

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


def load_ovad(annotations=None, images=None, *, split: str = "val", groups=None,
              max_samples: int | None = None, cache_dir=None) -> Dataset:
    """Load categorical OVAD targets with cached subject crops."""
    _limit(max_samples)
    _split(split, DATASETS["ovad"].splits)
    annotations, images = _local_files(annotations, images)
    utils = _utils()
    hf = utils.hf

    @dataclass
    class _OVADConfig(utils._SourceConfig):
        groups: tuple[str, ...] | None = None

    class OVADBuilder(utils._RowsBuilder):
        """Categorical OVAD targets and cached subject crops."""

        BUILDER_CONFIG_CLASS = _OVADConfig

        def _info(self):
            return hf.DatasetInfo(description=OVAD_PROTOCOL)

        def _split_generators(self, manager):
            if self.config.data_files:
                data = json.loads(Path(self.config.data_files["annotations"][0]).read_text())
            else:
                archive = manager.download(OVAD_URL)
                utils.check_checksum(archive, OVAD_CHECKSUM)
                data = utils._json_member(manager, archive, "ovad2000.json")
            attributes, rows = _ovad_selection(data, self.config.groups, self.config.max_samples)
            if not rows:
                raise ValueError("No eligible OVAD objects for the selected groups")
            rows = utils._images(rows, directory=self.config.data_dir, manager=manager, prefix=OVAD_IMAGE_URL)
            return self._split(rows, {
                "attributes": {group: hf.ClassLabel(names=list(choices)) for group, choices in attributes.items()},
                "sample_id": hf.Value("int64"), "image_id": hf.Value("int64"),
            })

    groups = _ovad_groups(groups)
    # Explicitly selecting all groups shares the same HF cache as the default.
    if groups == OVAD_GROUPS:
        groups = None
    return utils._build(OVADBuilder, name="ovad", split=split, max_samples=max_samples, cache_dir=cache_dir,
                        files=None if annotations is None else {"annotations": annotations}, images=images, groups=groups)


def _ovad_selection(data: dict, groups=None, max_samples=None):
    """Validate source annotations and select objects before image I/O."""
    if not isinstance(data, dict):
        raise ValueError("Expected an OVAD annotation mapping")
    groups = _ovad_groups(groups)
    values = _ovad_index(data, "attributes")
    if set(values) != set(range(len(values))):
        raise ValueError("OVAD attribute IDs must be consecutive from zero for att_vec")
    schema, indices = {}, {}
    for value_id, value in sorted(values.items()):
        group, name = value.get("type"), value.get("name")
        if not isinstance(group, str) or group not in {*OVAD_GROUPS, *OVAD_EXCLUDED}:
            raise ValueError(f"Unsupported OVAD attribute type: {group!r}")
        if not isinstance(name, str) or not name.startswith(group + ":"):
            raise ValueError(f"Invalid OVAD attribute name: {name!r}")
        description = name.split(":", 1)[1]
        synonyms = description.split("/")
        if any(not synonym.strip() or synonym != synonym.strip() for synonym in synonyms):
            raise ValueError(f"Invalid OVAD attribute synonyms: {name!r}")
        if group in OVAD_EXCLUDED:
            continue
        label = synonyms[0]
        choices = schema.setdefault(group, {})
        if label in choices:
            raise ValueError(f"Duplicate OVAD canonical label in {group}: {label}")
        choices[label] = description
        indices.setdefault(group, []).append((value_id, label))
    if not schema:
        raise ValueError("No supported OVAD attribute groups")
    if groups is not None and set(groups) - schema.keys():
        raise ValueError(f"Missing OVAD groups: {sorted(set(groups) - schema.keys())}")
    selected_schema = {group: choices for group, choices in schema.items() if groups is None or group in groups}
    images = _ovad_index(data, "images")
    categories = _ovad_index(data, "categories")
    annotations = _ovad_index(data, "annotations")
    for image in images.values():
        if any(type(image.get(key)) is not int or image[key] <= 0 for key in ("width", "height")):
            raise ValueError(f"Invalid OVAD image dimensions: {image['id']}")
        filename = image.get("file_name")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(f"Invalid OVAD image filename: {image['id']}")
    selected = []
    for annotation in annotations.values():
        sample_id = annotation["id"]
        image_id, category_id = annotation.get("image_id"), annotation.get("category_id")
        if type(image_id) is not int or image_id not in images:
            raise ValueError(f"Orphan OVAD image join for annotation {sample_id}")
        if type(category_id) is not int or category_id not in categories:
            raise ValueError(f"Unknown OVAD category for annotation {sample_id}")
        vector = annotation.get("att_vec")
        if not isinstance(vector, list) or len(vector) != len(values) or any(
            type(value) is not int or value not in (-1, 0, 1) for value in vector
        ):
            raise ValueError(f"Invalid OVAD att_vec for annotation {sample_id}")
        targets = {}
        for group, entries in indices.items():
            positives = [choice_id for choice_id, (index, _) in enumerate(entries) if vector[index] == 1]
            if len(positives) > 1:
                raise ValueError(f"Multiple positive OVAD values for {group} in annotation {sample_id}")
            if positives and group in selected_schema:
                targets[group] = positives[0]
        area, bbox = annotation.get("area"), annotation.get("bbox")
        if type(area) not in (int, float) or not np.isfinite(area) or area < 0:
            raise ValueError(f"Invalid OVAD area for annotation {sample_id}")
        if not isinstance(bbox, list) or len(bbox) != 4 or any(type(v) not in (int, float) for v in bbox):
            raise ValueError(f"Invalid OVAD box for annotation {sample_id}: expected numeric xywh")
        x, y, width, height = bbox
        edges = np.array([x, y, x + width, y + height], dtype=np.float64)
        if not np.isfinite(edges).all() or width <= 0 or height <= 0:
            raise ValueError(f"Invalid OVAD box for annotation {sample_id}: expected finite positive dimensions")
        image = images[image_id]
        size = (image["width"], image["height"])
        clipped = np.clip(edges, 0, [*size, *size])
        crop_box = (*np.floor(clipped[:2]).astype(int), *np.ceil(clipped[2:]).astype(int))
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            raise ValueError(f"Empty clipped OVAD box for annotation {sample_id}")
        if area >= 50 and targets and (max_samples is None or len(selected) < max_samples):
            selected.append({
                "sample_id": sample_id, "image_id": image_id,
                "file_name": image["file_name"], "image_size": size,
                "crop_box": [int(v) for v in crop_box],
                "attributes": {group: targets.get(group) for group in selected_schema},
            })
    return selected_schema, selected


def _ovad_groups(groups) -> tuple[str, ...] | None:
    if groups is None:
        return None
    if isinstance(groups, (str, bytes)):
        raise ValueError("OVAD groups must be a nonempty collection of supported group names")
    groups = tuple(groups)
    if not groups or any(not isinstance(group, str) or group not in OVAD_GROUPS for group in groups):
        raise ValueError(f"Unsupported OVAD groups; expected a subset of {OVAD_GROUPS}")
    return tuple(sorted(set(groups)))


def _ovad_index(data: dict, key: str) -> dict[int, dict]:
    records = data.get(key)
    if not isinstance(records, list) or not records:
        raise ValueError(f"OVAD annotations require a nonempty {key} list")
    indexed = {}
    for item in records:
        if not isinstance(item, dict) or type(item.get("id")) is not int or item["id"] < 0:
            raise ValueError(f"OVAD {key} require nonnegative integer IDs")
        if item["id"] in indexed:
            raise ValueError(f"Duplicate OVAD {key} ID: {item['id']}")
        indexed[item["id"]] = item
    return indexed


# Visual question answering: VQA v2

VQA_URL = "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_{kind}_{split}_mscoco.zip"

VQA_IMAGE_URL = "https://s3.amazonaws.com/images.cocodataset.org/{split}2014/"


def load_vqav2(annotations=None, images=None, *, split: str = "val",
               max_samples: int | None = None, cache_dir=None) -> Dataset:
    """Load VQA v2 questions with all ten reference answers."""
    _limit(max_samples)
    _split(split, DATASETS["vqav2"].splits)
    annotations, images = _local_files(annotations, images)
    utils = _utils()
    hf = utils.hf

    class VQAv2Builder(utils._RowsBuilder):
        """VQA v2 labeled questions, preserving all ten annotator answer votes."""

        def _split_generators(self, manager):
            originals = {}
            for kind, key in (("Questions", "questions"), ("Annotations", "annotations")):
                filename = (f"v2_OpenEnded_mscoco_{self.config.source_split}2014_questions.json" if key == "questions"
                            else f"v2_mscoco_{self.config.source_split}2014_annotations.json")
                if self.config.data_files:
                    originals[key] = json.loads(Path(self.config.data_files[key][0]).read_text())
                else:
                    url = VQA_URL.format(kind=kind, split=self.config.source_split.title())
                    originals[key] = utils._json_member(manager, manager.download(url), filename)
            rows = _vqav2_selection(originals["questions"], originals["annotations"], self.config.max_samples)
            for row in rows:
                row["file_name"] = f"COCO_{self.config.source_split}2014_{row['image_id']:012d}.jpg"
            rows = utils._images(rows, directory=self.config.data_dir, manager=manager,
                                 prefix=VQA_IMAGE_URL.format(split=self.config.source_split))
            return self._split(rows, {
                "question": hf.Value("string"), "answers": hf.List(hf.Value("string")),
                "answer_type": hf.Value("string"), "question_type": hf.Value("string"),
                "sample_id": hf.Value("int64"), "image_id": hf.Value("int64"),
            })

    files = None if annotations is None else {
        "questions": str(Path(annotations) / f"v2_OpenEnded_mscoco_{split}2014_questions.json"),
        "annotations": str(Path(annotations) / f"v2_mscoco_{split}2014_annotations.json"),
    }
    return utils._build(VQAv2Builder, name="vqav2", split=split, max_samples=max_samples, cache_dir=cache_dir,
                        files=files, images=images)


def _vqav2_selection(questions, annotations, max_samples=None):
    """Validate ID joins and targets before selecting in source question order."""
    indexed = []
    for data, key in ((questions, "questions"), (annotations, "annotations")):
        records = data.get(key) if isinstance(data, dict) else None
        if not isinstance(records, list) or not records:
            raise ValueError(f"VQA v2 requires a nonempty {key} list")
        index = {}
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"Invalid VQA v2 {key} record")
            question_id, image_id = record.get("question_id"), record.get("image_id")
            if any(type(value) is not int or value < 0 for value in (question_id, image_id)):
                raise ValueError(f"VQA v2 {key} require nonnegative integer question/image IDs")
            if question_id in index:
                raise ValueError(f"Duplicate VQA v2 {key} question ID: {question_id}")
            index[question_id] = record
        indexed.append(index)
    queries, targets = indexed
    if queries.keys() != targets.keys():
        raise ValueError("VQA v2 questions and annotations must have matching question IDs")
    selected = []
    for question_id, query in queries.items():
        target = targets[question_id]
        if target["image_id"] != query["image_id"]:
            raise ValueError(f"VQA v2 image IDs disagree for question {question_id}")
        question = query.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Invalid VQA v2 question: {question_id}")
        votes = target.get("answers")
        if not isinstance(votes, list) or any(not isinstance(vote, dict) for vote in votes):
            raise ValueError("VQA v2 answers must be a list of answer annotations")
        answers = [vote.get("answer") for vote in votes]
        answer_type, question_type = target.get("answer_type"), target.get("question_type")
        validate_targets(answers, answer_type, question_type)
        if max_samples is None or len(selected) < max_samples:
            selected.append({
                "sample_id": question_id, "image_id": query["image_id"],
                "question": question, "answers": answers,
                "answer_type": answer_type, "question_type": question_type,
            })
    return selected


# Captioning: COCO Captions 2017

CAPTIONS_URL = "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"

CAPTIONS_IMAGE_URL = "https://s3.amazonaws.com/images.cocodataset.org/{split}2017/"


def load_coco_captions2017(annotations=None, images=None, *, split: str = "val",
                          max_samples: int | None = None, cache_dir=None) -> Dataset:
    """Load COCO 2017 images with every reference caption."""
    _limit(max_samples)
    _split(split, DATASETS["coco_captions2017"].splits)
    annotations, images = _local_files(annotations, images)
    utils = _utils()
    hf = utils.hf

    class COCOCaptionsBuilder(utils._RowsBuilder):
        """COCO Captions 2017, one row per image with every reference caption."""

        def _split_generators(self, manager):
            if self.config.data_files:
                data = json.loads(Path(self.config.data_files["annotations"][0]).read_text())
            else:
                data = utils._json_member(manager, manager.download(CAPTIONS_URL),
                                          f"captions_{self.config.source_split}2017.json")
            rows = _coco_captions_selection(data, self.config.max_samples)
            rows = utils._images(rows, directory=self.config.data_dir, manager=manager,
                                 prefix=CAPTIONS_IMAGE_URL.format(split=self.config.source_split))
            return self._split(rows, {
                "captions": hf.List(hf.Value("string")),
                "sample_id": hf.Value("int64"), "image_id": hf.Value("int64"),
            })

    return utils._build(COCOCaptionsBuilder, name="coco_captions2017", split=split,
                        max_samples=max_samples, cache_dir=cache_dir,
                        files=None if annotations is None else {"annotations": annotations}, images=images)


def _coco_captions_selection(data, max_samples=None):
    """Join all captions to image IDs before selecting source-order images."""
    if not isinstance(data, dict):
        raise ValueError("Expected a COCO captions annotation mapping")
    indexed = {}
    for key in ("images", "annotations"):
        records = data.get(key)
        if not isinstance(records, list) or not records:
            raise ValueError(f"COCO captions require a nonempty {key} list")
        index = {}
        for item in records:
            if not isinstance(item, dict) or type(item.get("id")) is not int or item["id"] < 0:
                raise ValueError(f"COCO captions {key} require nonnegative integer IDs")
            if item["id"] in index:
                raise ValueError(f"Duplicate COCO captions {key} ID: {item['id']}")
            index[item["id"]] = item
        indexed[key] = index
    images = indexed["images"]
    captions = {image_id: [] for image_id in images}
    filenames = set()
    for image in images.values():
        size = (image.get("width"), image.get("height"))
        if any(type(value) is not int or value <= 0 for value in size):
            raise ValueError(f"Invalid COCO caption image dimensions: {image['id']}")
        filename = image.get("file_name")
        if (not isinstance(filename, str) or not filename.strip() or "\\" in filename
                or Path(filename).is_absolute() or ".." in Path(filename).parts
                or filename in {".", ".."}):
            raise ValueError(f"Invalid COCO caption image filename: {filename!r}")
        if filename in filenames:
            raise ValueError(f"Duplicate COCO caption image filename: {filename}")
        filenames.add(filename)
    for annotation in indexed["annotations"].values():
        image_id = annotation.get("image_id")
        if type(image_id) is not int or image_id not in images:
            raise ValueError(f"Unknown COCO caption image ID: {image_id!r}")
        caption = annotation.get("caption")
        validate_captions([caption])
        captions[image_id].append(caption)
    for refs in captions.values():
        validate_captions(refs)
    selected = [
        {"sample_id": image_id, "image_id": image_id, "file_name": image["file_name"],
         "image_size": (image["width"], image["height"]), "captions": captions[image_id]}
        for image_id, image in list(images.items())[:max_samples]
    ]
    return selected
