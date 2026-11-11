"""Load default benchmarks and local annotations as Hugging Face datasets."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ._caption import validate_captions
from ._vqa import validate_targets
from .data import _limit
from . import dataset_constants as constants
from . import prepare_data

if TYPE_CHECKING:
    from datasets import Dataset


def _image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    return sorted(
        path.resolve() for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in constants.IMAGE_SUFFIXES
    )


def _split_name(split: str) -> None:
    if (
        not isinstance(split, str) or not split
        or Path(split).name != split or split in {".", ".."}
    ):
        raise ValueError("split must be a nonempty split name")


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
    elif path.suffix.lower() in constants.IMAGE_SUFFIXES:
        images = [path]
    else:
        raise ValueError(f"Expected an image directory, image, or TXT image list: {path}")
    for image in images:
        if not image.is_file() or image.suffix.lower() not in constants.IMAGE_SUFFIXES:
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
    return selected, {
        "source_questions": len(queries),
        "source_images": len({query["image_id"] for query in queries.values()}),
        "selected_questions": len(selected),
        "selected_images": len({sample["image_id"] for sample in selected}),
    }


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
    return selected, {
        "source_images": len(images), "source_captions": len(indexed["annotations"]),
        "selected_images": len(selected),
        "selected_captions": sum(len(sample["captions"]) for sample in selected),
    }


def _ovad_groups(groups) -> tuple[str, ...] | None:
    if groups is None:
        return None
    if isinstance(groups, (str, bytes)):
        raise ValueError("OVAD groups must be a nonempty collection of supported group names")
    groups = tuple(groups)
    if not groups or any(not isinstance(group, str) or group not in constants.OVAD_GROUPS for group in groups):
        raise ValueError(f"Unsupported OVAD groups; expected a subset of {constants.OVAD_GROUPS}")
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
        if not isinstance(group, str) or group not in {*constants.OVAD_GROUPS, *constants.OVAD_EXCLUDED}:
            raise ValueError(f"Unsupported OVAD attribute type: {group!r}")
        if not isinstance(name, str) or not name.startswith(group + ":"):
            raise ValueError(f"Invalid OVAD attribute name: {name!r}")
        description = name.split(":", 1)[1]
        synonyms = description.split("/")
        if any(not synonym.strip() or synonym != synonym.strip() for synonym in synonyms):
            raise ValueError(f"Invalid OVAD attribute synonyms: {name!r}")
        if group in constants.OVAD_EXCLUDED:
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
        if area < 50:
            small += 1
        elif not targets:
            unscored += 1
        else:
            eligible += 1
            if max_samples is None or len(selected) < max_samples:
                selected.append({
                    "sample_id": sample_id, "image_id": image_id,
                    "file_name": image["file_name"], "image_size": size,
                    "crop_box": [int(v) for v in crop_box],
                    "attributes": {group: targets.get(group) for group in selected_schema},
                })
    counts = {
        "source_objects": len(annotations), "source_images": len(images),
        "excluded_small_objects": small, "excluded_unscored_objects": unscored,
        "excluded_objects": small + unscored, "eligible_objects": eligible,
        "prepared_objects": len(selected), "prepared_images": len({s["image_id"] for s in selected}),
    }
    return selected_schema, selected, counts


def _datasets():
    try:
        import datasets
    except ImportError as exc:
        raise ImportError("Dataset loaders require: pip install 'ml-pipes-vlm[validation]'") from exc
    return datasets


def _format_images(batch):
    """Supply the fixed RGB array schema through HF's dataset transform API."""
    images, sizes = [], []
    for index, image in enumerate(batch["image"]):
        size = tuple(batch["image_size"][index]) if "image_size" in batch else image.size
        if image.size != size:
            raise ValueError("Image dimensions differ from dataset annotation")
        if "crop_box" in batch:
            box = batch["crop_box"][index]
            crop_size = (box[2] - box[0], box[3] - box[1])
            # map() may materialize crops while retaining the HF transform.
            if size != crop_size:
                image = image.crop(box)
        sizes.append(list(image.size))
        images.append(np.asarray(image.convert("RGB")))
    result = {**batch, "image": images}
    if "image_size" in batch:
        result["image_size"] = sizes
    return result


def _with_preparation(dataset, prepared):
    if prepared is not None:
        dataset.info.dataset_name = prepared.metadata["dataset"]
        dataset.metadata.update(data=str(prepared.data), preparation=prepared.metadata)
        dataset.counts.update(prepared.metadata.get("counts", {}))
        dataset.counts.update({key: value for key, value in prepared.metadata.items()
                              if key.startswith(("source_", "selected_")) and type(value) is int})
    return dataset


def _check_local_paths(annotations, images):
    if annotations is None and images is not None:
        raise ValueError("images requires local annotations")
    if annotations is not None and images is None:
        raise ValueError("Local annotations require an images directory")


def _build_dataset(rows, features, *, counts=None, metadata=None):
    """Build an HF dataset with lazy image decoding and RGB array formatting."""
    hf = _datasets()
    dataset = hf.Dataset.from_list(rows, features=hf.Features({
        "image": hf.Image(), "im_file": hf.Value("string"), **features,
    })).with_transform(_format_images)
    dataset.counts = dict(counts or {})
    dataset.metadata = dict(metadata or {})
    return dataset


def load_classification(root: str | Path, *, split: str = "val", class_ids=None, max_samples: int | None = None) -> Dataset:
    """Load local class directories as image/label rows with a complete vocabulary."""
    _limit(max_samples)
    hf = _datasets()
    _split_name(split)
    directory = Path(root).expanduser().resolve() / split
    if not directory.is_dir():
        raise FileNotFoundError(f"Classification split not found: {directory}")
    classes = dict(enumerate(sorted(path for path in directory.iterdir() if path.is_dir())))
    selected = None
    if class_ids is not None:
        try:
            ids = tuple(class_ids)
        except TypeError as exc:
            raise ValueError("class_ids must be a nonempty collection of known integer class IDs") from exc
        if (isinstance(class_ids, (str, bytes)) or not ids
                or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                       or value not in classes for value in ids)):
            raise ValueError("class_ids must be a nonempty collection of known integer class IDs")
        selected = set(ids)
    rows = [
        {"image": str(image), "im_file": str(image), "label": class_id}
        for class_id, path in classes.items() if selected is None or class_id in selected
        for image in _image_files(path)
    ]
    if not rows:
        if selected is not None:
            raise ValueError("No samples belong to the selected classification classes")
        raise ValueError(f"No classification images found in {Path(root).expanduser().resolve() / split}")
    return _build_dataset(rows[:max_samples], {"label": hf.ClassLabel(names=[path.name for path in classes.values()])})


def load_yolo(data: str | Path, *, split: str = "val", max_samples: int | None = None) -> Dataset:
    """Load local YOLO YAML/TXT as image/objects rows with pixel xyxy boxes."""
    _limit(max_samples)
    _split_name(split)
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Reading detection YAML requires: pip install 'ml-pipes-vlm[validation]'") from exc
    hf = _datasets()
    from PIL import Image

    path = Path(data).expanduser().resolve()
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Expected a dataset YAML mapping: {path}")
    names = _names(config.get("names"))
    root_value = config.get("path", ".")
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("Dataset path must be a nonempty string")
    root = _resolve(root_value, path.parent)
    entries = config.get(split)
    if isinstance(entries, str):
        entries = [entries]
    if (not isinstance(entries, list) or not entries
            or any(not isinstance(entry, str) or not entry for entry in entries)):
        raise ValueError(f"Dataset YAML must define a nonempty {split!r} split")
    images = sorted({image for entry in entries for image in _detection_images(entry, root)})
    if not images:
        raise ValueError(f"No detection images found for split {split!r} in {path}")
    counts = {"labeled_images": 0, "missing_labels": 0, "empty_labels": 0}
    rows = []
    for index, path in enumerate(images):
        categories, boxes, status = _read_labels(_label_path(path), names)
        counts[status] += 1
        if max_samples is not None and index >= max_samples:
            continue
        # Only read image headers to convert YOLO's normalized geometry.
        try:
            with Image.open(path) as image:
                width, height = image.size
        except OSError as exc:
            raise ValueError(f"Could not read image {path}: {exc}") from exc
        scale = np.array([width, height, width, height])
        rows.append({
            "image": str(path), "im_file": str(path),
            "objects": {"bbox": (boxes * scale).tolist(), "category": categories.tolist()},
        })
    return _build_dataset(rows, {"objects": {
        "bbox": hf.List(hf.List(hf.Value("float64"), length=4)),
        "category": hf.List(hf.ClassLabel(names=list(names.values()))),
    }}, counts=counts)


def load_refcoco(annotations: str | Path, images: str | Path, *, dataset: str = "refcocog", split: str = "val", max_samples: int | None = None) -> Dataset:
    """Load local MDETR RefCOCO annotations, one row per referring expression."""
    _limit(max_samples)
    if not isinstance(dataset, str) or dataset not in constants.REFCOCO_SPLITS:
        raise ValueError("dataset must be refcocog, refcoco, or refcoco+")
    _split_name(split)
    if split not in constants.REFCOCO_SPLITS[dataset]:
        raise ValueError(f"Unsupported {dataset} split: {split!r}")
    hf = _datasets()
    path = Path(annotations).expanduser().resolve() / f"finetune_{dataset}_{split}.json"
    image_root = Path(images).expanduser().resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"Grounding image directory not found: {image_root}")
    rows, names = _refcoco_selection(json.loads(path.read_text()))
    for row in rows:
        image = _resolve(row.pop("file_name"), image_root)
        if not image.is_file():
            raise FileNotFoundError(f"Grounding image not found: {image}")
        row.update(image=str(image), im_file=str(image))
    counts = {"expressions": len(rows), "unique_images": len({row["im_file"] for row in rows})}
    return _build_dataset(rows[:max_samples], {
        "text": hf.Value("string"), "bbox": hf.List(hf.Value("float64"), length=4),
        "label": hf.ClassLabel(names=names), "sample_id": hf.Value("int64"),
        "image_size": hf.List(hf.Value("int64"), length=2),
    }, counts=counts, metadata={
        "format": "mdetr", "dataset": dataset, "split": split,
        "expected_split_by": "umd" if dataset == "refcocog" else "unc",
        "annotations": str(path), "images": str(image_root),
    })


def load_cifar10(*, split: str = "test", classes=None, max_samples: int | None = None, cache_dir=None) -> Dataset:
    """Load CIFAR-10; defaults to the held-out test set."""
    prepared = prepare_data.prepare_cifar10(split=split, classes=classes, max_samples=max_samples, cache_dir=cache_dir)
    return _with_preparation(load_classification(prepared.data, split=split), prepared)


def load_coco128(*, split: str = "val", max_samples: int | None = None, cache_dir=None) -> Dataset:
    """Load the Ultralytics COCO128 detection subset."""
    prepared = prepare_data.prepare_coco128(split=split, max_samples=max_samples, cache_dir=cache_dir)
    return _with_preparation(load_yolo(prepared.data, split=split), prepared)


def load_refcocog(*, split: str = "val", max_samples: int | None = None, cache_dir=None) -> Dataset:
    """Load RefCOCOg UMD, one row per expression; defaults to validation."""
    prepared = prepare_data.prepare_refcocog(split=split, max_samples=max_samples, cache_dir=cache_dir)
    return _with_preparation(load_refcoco(prepared.data, prepared.images, split=split), prepared)


def load_ovad(annotations=None, images=None, *, split: str = "val", groups=None, max_samples: int | None = None, cache_dir=None) -> Dataset:
    """Load OVAD categorical attribute rows; images are lazily cropped subjects."""
    if split not in constants.DATASETS["ovad"].splits:
        raise ValueError("OVAD supports test and its val alias; training splits are unsupported")
    _limit(max_samples)
    _check_local_paths(annotations, images)
    prepared = None
    if annotations is None:
        prepared = prepare_data.prepare_ovad(split=split, max_samples=max_samples, cache_dir=cache_dir, groups=groups)
        annotations, images = prepared.data, prepared.images
    hf = _datasets()
    path = Path(annotations).expanduser().resolve()
    image_root = Path(images).expanduser().resolve()
    attributes, rows, counts = _ovad_selection(json.loads(path.read_text()), groups, max_samples)
    if not image_root.is_dir():
        raise FileNotFoundError(f"OVAD image directory not found: {image_root}")
    for row in rows:
        image = _resolve(row.pop("file_name"), image_root)
        if not image.is_relative_to(image_root):
            raise ValueError(f"OVAD image path escapes image directory: {image}")
        if not image.is_file():
            raise FileNotFoundError(f"OVAD image not found: {image}")
        row.update(image=str(image), im_file=str(image))
    if not rows:
        raise ValueError("No eligible OVAD objects for the selected groups")
    dataset = _build_dataset(rows, {
        "attributes": {group: hf.ClassLabel(names=list(choices)) for group, choices in attributes.items()},
        "crop_box": hf.List(hf.Value("int64"), length=4),
        "image_size": hf.List(hf.Value("int64"), length=2),
        "sample_id": hf.Value("int64"), "image_id": hf.Value("int64"),
    }, counts=counts, metadata={
        "dataset": "ovad", "protocol": constants.OVAD_PROTOCOL, "split": split, "source_split": "test",
        "split_note": "val is an alias for the published OVAD test benchmark",
        "selected_groups": list(attributes), "annotations": str(path), "images": str(image_root),
        "attribute_descriptions": attributes,
    })
    return _with_preparation(dataset, prepared)


def load_vqav2(annotations=None, images=None, *, split: str = "val", max_samples: int | None = None, cache_dir=None) -> Dataset:
    """Load VQA v2 questions with all ten reference answers; defaults to validation."""
    if split not in constants.DATASETS["vqav2"].splits:
        raise ValueError("VQA v2 supports labeled train and val splits only")
    _limit(max_samples)
    _check_local_paths(annotations, images)
    prepared = None
    if annotations is None:
        prepared = prepare_data.prepare_vqav2(split=split, max_samples=max_samples, cache_dir=cache_dir)
        annotations, images = prepared.data, prepared.images
    hf = _datasets()
    root = Path(annotations).expanduser().resolve()
    image_root = Path(images).expanduser().resolve()
    questions_path = root / f"v2_OpenEnded_mscoco_{split}2014_questions.json"
    annotations_path = root / f"v2_mscoco_{split}2014_annotations.json"
    rows, counts = _vqav2_selection(json.loads(questions_path.read_text()), json.loads(annotations_path.read_text()), max_samples)
    if not image_root.is_dir():
        raise FileNotFoundError(f"VQA v2 image directory not found: {image_root}")
    for row in rows:
        image = image_root / f"COCO_{split}2014_{row['image_id']:012d}.jpg"
        if not image.is_file():
            raise FileNotFoundError(f"VQA v2 image not found: {image}")
        row.update(image=str(image), im_file=str(image))
    dataset = _build_dataset(rows, {
        "question": hf.Value("string"), "answers": hf.List(hf.Value("string")),
        "question_type": hf.Value("string"), "answer_type": hf.Value("string"),
        "sample_id": hf.Value("int64"), "image_id": hf.Value("int64"),
    }, counts=counts, metadata={
        "dataset": "vqav2", "version": "2", "split": split,
        "questions": str(questions_path), "annotations": str(annotations_path),
        "images": str(image_root), "selection": "Source JSON question order", "max_samples": max_samples,
    })
    return _with_preparation(dataset, prepared)


def load_coco_captions2017(annotations=None, images=None, *, split: str = "val", max_samples: int | None = None, cache_dir=None) -> Dataset:
    """Load COCO 2017 images with all reference captions; defaults to validation."""
    if split not in constants.DATASETS["coco_captions2017"].splits:
        raise ValueError("COCO captions supports labeled train and val splits only")
    _limit(max_samples)
    _check_local_paths(annotations, images)
    prepared = None
    if annotations is None:
        prepared = prepare_data.prepare_coco_captions2017(split=split, max_samples=max_samples, cache_dir=cache_dir)
        annotations, images = prepared.data, prepared.images
    hf = _datasets()
    path = Path(annotations).expanduser().resolve()
    image_root = Path(images).expanduser().resolve()
    rows, counts = _coco_captions_selection(json.loads(path.read_text()), max_samples)
    if not image_root.is_dir():
        raise FileNotFoundError(f"COCO captions image directory not found: {image_root}")
    for row in rows:
        image = (image_root / row.pop("file_name")).resolve()
        if not image.is_relative_to(image_root):
            raise ValueError(f"COCO caption image path escapes image directory: {image}")
        if not image.is_file():
            raise FileNotFoundError(f"COCO caption image not found: {image}")
        row.update(image=str(image), im_file=str(image))
    dataset = _build_dataset(rows, {
        "captions": hf.List(hf.Value("string")), "sample_id": hf.Value("int64"),
        "image_id": hf.Value("int64"), "image_size": hf.List(hf.Value("int64"), length=2),
    }, counts=counts, metadata={
        "dataset": "coco_captions2017", "split": split,
        "annotations": str(path), "images": str(image_root),
        "selection": "Source JSON image order; all reference captions retained", "max_samples": max_samples,
    })
    return _with_preparation(dataset, prepared)
