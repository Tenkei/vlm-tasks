"""Minimal local classification and YOLO detection dataset readers."""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path

import cv2
import numpy as np
from ml_pipes.vision import Decode, LoadFile

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
