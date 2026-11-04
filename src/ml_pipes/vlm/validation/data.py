"""Local classification, detection, referring expression, and OVAD readers."""
from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path

import cv2
import numpy as np
from ml_pipes.vision import Decode, ImagePayload, LoadFile

_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    return sorted(
        path.resolve() for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )


def _load_image(path: str):
    try:
        return Decode()(LoadFile()(path))
    except (OSError, ValueError, cv2.error) as exc:
        raise ValueError(f"Could not read image {path}: {exc}") from exc


def _split_name(split: str) -> None:
    if (
        not isinstance(split, str) or not split
        or Path(split).name != split or split in {".", ".."}
    ):
        raise ValueError("split must be a nonempty split name")


class ClassificationDataset:
    """Read ``root/split/class/image`` without resizing or cropping.

    ``names`` maps integer IDs to class names. Indexed samples contain
    ``im_file``, ``img`` (an ImagePayload), and ``cls`` (one integer ID).
    """

    def __init__(self, root: str | Path, split: str = "val") -> None:
        _split_name(split)
        directory = Path(root).expanduser().resolve() / split
        if not directory.is_dir():
            raise FileNotFoundError(f"Classification split not found: {directory}")
        classes = sorted(path for path in directory.iterdir() if path.is_dir())
        self.names = {i: path.name for i, path in enumerate(classes)}
        self.samples = [
            {"im_file": str(image), "cls": np.array([class_id], dtype=np.int64)}
            for class_id, path in enumerate(classes) for image in _image_files(path)
        ]
        if not self.samples:
            raise ValueError(f"No classification images found in {directory}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        return {**sample, "img": _load_image(sample["im_file"])}


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


def _resolve(path: str, parent: Path) -> Path:
    result = Path(path).expanduser()
    return (result if result.is_absolute() else parent / result).resolve()


def _detection_images(entry: str, root: Path) -> list[Path]:
    path = _resolve(entry, root)
    if path.is_dir():
        return _image_files(path)
    if not path.is_file():
        raise FileNotFoundError(f"Detection images or image list not found: {path}")
    if path.suffix.lower() == ".txt":
        images = [
            _resolve(line.strip(), path.parent)
            for line in path.read_text().splitlines() if line.strip()
        ]
    elif path.suffix.lower() in _IMAGE_SUFFIXES:
        images = [path]
    else:
        raise ValueError(f"Expected an image directory, image, or TXT image list: {path}")
    for image in images:
        if not image.is_file() or image.suffix.lower() not in _IMAGE_SUFFIXES:
            raise ValueError(f"Invalid image entry in {path}: {image}")
    return images


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    # Match the final 'images' directory, as in YOLO's image-to-label convention.
    indices = [i for i, part in enumerate(parts[:-1]) if part == "images"]
    if indices:
        parts[indices[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def _read_labels(path: Path, names: dict[int, str]) -> tuple[np.ndarray, np.ndarray, str]:
    if not path.exists():
        return np.empty(0, dtype=np.int64), np.empty((0, 4)), "missing_labels"
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
        "labeled_images" if rows else "empty_labels",
    )


class YOLODataset:
    """Read local YOLO detection YAML and TXT labels.

    Samples contain ``im_file``, ``img``, ``cls`` (integer IDs), and ``bboxes``
    (Nx4 xyxy boxes in original-image pixels). Annotation metadata is checked
    at construction; original images are decoded only when indexed.
    """

    def __init__(self, data: str | Path, split: str = "val") -> None:
        _split_name(split)
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("Reading detection YAML requires: pip install 'ml-pipes-vlm[validation]'") from exc
        path = Path(data).expanduser().resolve()
        with path.open() as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, dict):
            raise ValueError(f"Expected a dataset YAML mapping: {path}")
        self.names = _names(config.get("names"))
        root_value = config.get("path", ".")
        if not isinstance(root_value, str) or not root_value:
            raise ValueError("Dataset path must be a nonempty string")
        root = _resolve(root_value, path.parent)
        entries = config.get(split)
        if isinstance(entries, str):
            entries = [entries]
        if not isinstance(entries, list) or not entries or any(
            not isinstance(entry, str) or not entry for entry in entries
        ):
            raise ValueError(f"Dataset YAML must define a nonempty {split!r} split")
        images = sorted({image for entry in entries for image in _detection_images(entry, root)})
        if not images:
            raise ValueError(f"No detection images found for split {split!r} in {path}")
        self.counts = {"labeled_images": 0, "missing_labels": 0, "empty_labels": 0}
        self.samples = []
        for image in images:
            cls, boxes, status = _read_labels(_label_path(image), self.names)
            self.counts[status] += 1
            self.samples.append({"im_file": str(image), "cls": cls, "bboxes": boxes})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = _load_image(sample["im_file"])
        scale = np.array([image.width, image.height, image.width, image.height])
        return {**sample, "img": image, "bboxes": sample["bboxes"] * scale}


_REFCOCO_SPLITS = {
    "refcocog": ("train", "val", "test"),
    "refcoco": ("train", "val", "testA", "testB"),
    "refcoco+": ("train", "val", "testA", "testB"),
}


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


class RefCOCODataset:
    """Read MDETR's local ``finetune_{dataset}_{split}.json`` annotations.

    Each converted image record represents one expression, even when several
    records share a physical image. Samples contain ``im_file``, ``img``, one
    ``cls`` ID, one pixel-xyxy ``bboxes`` row, ``target``, and ``sample_id``.
    Category IDs and expression order are preserved. Images are decoded lazily.
    """

    def __init__(
        self, annotations: str | Path, images: str | Path, *,
        dataset: str = "refcocog", split: str = "val",
    ) -> None:
        if not isinstance(dataset, str) or dataset not in _REFCOCO_SPLITS:
            raise ValueError("dataset must be refcocog, refcoco, or refcoco+")
        _split_name(split)
        if split not in _REFCOCO_SPLITS[dataset]:
            raise ValueError(f"Unsupported {dataset} split: {split!r}")
        path = Path(annotations).expanduser().resolve() / f"finetune_{dataset}_{split}.json"
        image_root = Path(images).expanduser().resolve()
        if not image_root.is_dir():
            raise FileNotFoundError(f"Grounding image directory not found: {image_root}")
        with path.open() as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a grounding annotation mapping: {path}")
        queries = _indexed_records(data, "images")
        categories = _indexed_records(data, "categories")
        targets = _indexed_records(data, "annotations")
        self.names = {key: item.get("name") for key, item in categories.items()}
        if (
            not self.names
            or any(not isinstance(name, str) or not name.strip() for name in self.names.values())
            or len(set(self.names.values())) != len(self.names)
        ):
            raise ValueError("Grounding category names must be nonempty and unique")
        query_targets = {}
        for annotation in targets.values():
            query_id = annotation.get("image_id")
            if type(query_id) is not int or query_id not in queries:
                raise ValueError(f"Orphan grounding annotation: {annotation['id']}")
            if query_id in query_targets:
                raise ValueError(f"Multiple target annotations for grounding query {query_id}")
            query_targets[query_id] = annotation

        self.samples = []
        for query_id, query in queries.items():
            annotation = query_targets.get(query_id)
            if annotation is None:
                raise ValueError(f"Missing target annotation for grounding query {query_id}")
            target = query.get("caption")
            if not isinstance(target, str) or not target.strip():
                raise ValueError(f"Invalid expression for grounding query {query_id}")
            size = (query.get("width"), query.get("height"))
            if any(type(value) is not int or value <= 0 for value in size):
                raise ValueError(f"Invalid image dimensions for grounding query {query_id}")
            filename = query.get("file_name")
            if not isinstance(filename, str) or not filename.strip():
                raise ValueError(f"Invalid image filename for grounding query {query_id}")
            image = _resolve(filename, image_root)
            if not image.is_file():
                raise FileNotFoundError(f"Grounding image not found: {image}")
            category_id = annotation.get("category_id")
            if type(category_id) is not int or category_id not in self.names:
                raise ValueError(f"Unknown category for grounding query {query_id}")
            bbox = annotation.get("bbox")
            if (
                not isinstance(bbox, list) or len(bbox) != 4
                or any(type(value) not in (int, float) for value in bbox)
            ):
                raise ValueError(f"Invalid box for grounding query {query_id}: expected numeric xywh")
            x, y, width, height = bbox
            box = np.array([[x, y, x + width, y + height]], dtype=np.float64)
            if (
                not np.isfinite(box).all() or width <= 0 or height <= 0
                or box[0, 2] <= box[0, 0] or box[0, 3] <= box[0, 1]
            ):
                raise ValueError(f"Invalid box for grounding query {query_id}: expected finite positive dimensions")
            self.samples.append({
                "im_file": str(image), "image_size": size,
                "cls": np.array([category_id], dtype=np.int64), "bboxes": box,
                "target": target, "sample_id": query_id,
            })
        if not self.samples:
            raise ValueError(f"No grounding expressions found in {path}")
        self.counts = {
            "expressions": len(self.samples),
            "unique_images": len({sample["im_file"] for sample in self.samples}),
        }
        self.metadata = {
            "format": "mdetr", "dataset": dataset, "split": split,
            "expected_split_by": "umd" if dataset == "refcocog" else "unc",
            "annotations": str(path), "images": str(image_root),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = _load_image(sample["im_file"])
        if image.size != sample["image_size"]:
            raise ValueError(
                f"Image dimensions differ from grounding annotation for query {sample['sample_id']}: "
                f"expected {sample['image_size']}, got {image.size}"
            )
        return {**sample, "img": image, "cls": sample["cls"].copy(), "bboxes": sample["bboxes"].copy()}


OVAD_GROUPS = (
    "cleanliness", "clothes pattern", "color quantity", "cooked", "face expression",
    "gender", "group", "hair color", "hair length", "hair tone", "hair type", "length",
    "material", "maturity", "optical property", "order", "pattern", "position", "size",
    "state", "texture", "tone",
)
_OVAD_EXCLUDED = {"color", "clothes color"}
OVAD_PROTOCOL = "OVAD categorical object attributes (22 groups; excludes color and clothes color); not official mAP"


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
        if not isinstance(group, str) or group not in {*OVAD_GROUPS, *_OVAD_EXCLUDED}:
            raise ValueError(f"Unsupported OVAD attribute type: {group!r}")
        if not isinstance(name, str) or not name.startswith(group + ":"):
            raise ValueError(f"Invalid OVAD attribute name: {name!r}")
        description = name.split(":", 1)[1]
        synonyms = description.split("/")
        if any(not synonym.strip() or synonym != synonym.strip() for synonym in synonyms):
            raise ValueError(f"Invalid OVAD attribute synonyms: {name!r}")
        if group in _OVAD_EXCLUDED:
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
    small = unscored = eligible = 0
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
            positives = [label for index, label in entries if vector[index] == 1]
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
        if area < 50:
            small += 1
        elif not targets:
            unscored += 1
        else:
            eligible += 1
            if max_samples is None or len(selected) < max_samples:
                selected.append({
                    "sample_id": sample_id, "image_id": image_id, "category_id": category_id,
                    "file_name": image["file_name"], "image_size": size,
                    "bbox": list(bbox), "crop_box": tuple(int(v) for v in crop_box), "attributes": targets,
                })
    counts = {
        "source_objects": len(annotations), "source_images": len(images),
        "excluded_small_objects": small, "excluded_unscored_objects": unscored,
        "excluded_objects": small + unscored, "eligible_objects": eligible,
        "prepared_objects": len(selected), "prepared_images": len({s["image_id"] for s in selected}),
    }
    return selected_schema, selected, counts


class OVADDataset:
    """Read OVAD categorical targets and return clipped object crops.

    ``test`` and ``val`` name the same published benchmark. ``groups`` selects
    complete categorical vocabularies; absent positives are unscored. Filtering
    (area >=50 and at least one selected target) precedes ``max_samples``.
    ``attributes`` contains constructor-compatible canonical choices/descriptions.
    """

    def __init__(
        self, annotations: str | Path, images: str | Path, *, split: str = "val",
        groups=None, max_samples: int | None = None,
    ) -> None:
        if split not in ("test", "val"):
            raise ValueError("OVAD supports test and its val alias; training splits are unsupported")
        if max_samples is not None and (type(max_samples) is not int or max_samples < 1):
            raise ValueError("max_samples must be a positive integer or None")
        path = Path(annotations).expanduser().resolve()
        image_root = Path(images).expanduser().resolve()
        with path.open() as stream:
            self.attributes, selected, self.counts = _ovad_selection(json.load(stream), groups, max_samples)
        if not image_root.is_dir():
            raise FileNotFoundError(f"OVAD image directory not found: {image_root}")
        self.samples = []
        for sample in selected:
            image = _resolve(sample["file_name"], image_root)
            if not image.is_relative_to(image_root):
                raise ValueError(f"OVAD image path escapes image directory: {image}")
            if not image.is_file():
                raise FileNotFoundError(f"OVAD image not found: {image}")
            self.samples.append({**sample, "im_file": str(image)})
        if not self.samples:
            raise ValueError("No eligible OVAD objects for the selected groups")
        self.metadata = {
            "dataset": "ovad", "protocol": OVAD_PROTOCOL, "split": split, "source_split": "test",
            "split_note": "val is an alias for the published OVAD test benchmark",
            "selected_groups": list(self.attributes), "annotations": str(path), "images": str(image_root),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = _load_image(sample["im_file"])
        if image.size != sample["image_size"]:
            raise ValueError(f"Image dimensions differ from OVAD annotation {sample['sample_id']}")
        x1, y1, x2, y2 = sample["crop_box"]
        crop = ImagePayload(np.ascontiguousarray(image.array[y1:y2, x1:x2]), image.color_space, image.layout)
        return {**sample, "img": crop, "bbox": list(sample["bbox"]), "attributes": dict(sample["attributes"])}


class DataLoader:
    """Iterate an indexed dataset synchronously, one original image at a time."""

    def __init__(self, dataset, indices=None) -> None:
        self.dataset = dataset
        self.indices = range(len(dataset)) if indices is None else tuple(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[dict]:
        for index in self.indices:
            yield self.dataset[index]


class ClassificationDataLoader(DataLoader):
    """Select classification samples by class ID before decoding images.

    The dataset exposes annotation metadata through ``.samples``. Omit
    ``class_ids`` to include every class.
    """

    def __init__(self, dataset, *, class_ids=None) -> None:
        indices = None
        if class_ids is not None:
            selected = set(class_ids)
            indices = [
                i for i, sample in enumerate(dataset.samples)
                if int(sample["cls"][0]) in selected
            ]
        super().__init__(dataset, indices=indices)
