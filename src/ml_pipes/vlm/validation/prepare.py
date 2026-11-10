"""Prepare supported benchmark datasets as local inputs for validation.

Only registered sources are fetched. Loaders themselves remain local readers.
Archives are read member-by-member; their paths and download scripts are never
executed or extracted. Cached preparations include provenance and file hashes.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile

import cv2
import numpy as np


@dataclass(frozen=True)
class DatasetInfo:
    task: str
    splits: tuple[str, ...]
    names: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreparedDataset:
    dataset: str
    split: str
    data: Path
    images: Path | None
    names: tuple[str, ...]
    metadata: dict
    attributes: dict[str, dict[str, str]] | None = None


_CIFAR_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck",
)
_COCO_NAMES = (
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
DATASETS = {
    "cifar10": DatasetInfo("classify", ("train", "val", "test"), _CIFAR_NAMES),
    "coco128": DatasetInfo("detect", ("train", "val"), _COCO_NAMES),
    "refcocog": DatasetInfo("ground", ("train", "val", "test")),
    "ovad": DatasetInfo("attributes", ("test", "val")),
    "vqav2": DatasetInfo("answer", ("train", "val")),
    "coco_captions2017": DatasetInfo("describe", ("train", "val")),
}
DEFAULT_DATASETS = {
    "classify": "cifar10", "detect": "coco128", "ground": "refcocog",
    "attributes": "ovad", "answer": "vqav2", "describe": "coco_captions2017",
}

_CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
_CIFAR_CHECKSUM = "md5:c32a1d4ab5d03f1284b67883e8d87530"
_COCO_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
_COCO_CHECKSUM = "sha256:61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e"
_MDETR_URL = "https://zenodo.org/records/4729015/files/mdetr_annotations.tar.gz?download=1"
_COCO_IMAGE_URL = "https://s3.amazonaws.com/images.cocodataset.org/train2014/"
_OVAD_URL = "https://lmb.informatik.uni-freiburg.de/resources/datasets/ovad.zip"
_OVAD_CHECKSUM = "sha256:1716fc5216dd040c3803ca7d91444140d02bc6a98f40a60b85f16c97174dc1ec"
_OVAD_IMAGE_URL = "https://s3.amazonaws.com/images.cocodataset.org/val2017/"
_VQA_URL = "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_{kind}_{split}_mscoco.zip"
_VQA_IMAGE_URL = "https://s3.amazonaws.com/images.cocodataset.org/{split}2014/"
_CAPTIONS_URL = "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"
_CAPTIONS_IMAGE_URL = "https://s3.amazonaws.com/images.cocodataset.org/{split}2017/"
_FORMAT_VERSION = 1


def _cache_root(cache_dir=None) -> Path:
    location = cache_dir or os.environ.get("VLM_TASKS_DATASETS_DIR") or Path.home() / ".cache" / "vlm-tasks" / "datasets"
    return Path(location).expanduser().resolve()


def _hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            pending = Path(stream.name)
            stream.write(content)
        pending.replace(path)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def _write_json(path: Path, content) -> None:
    _atomic_write(path, (json.dumps(content, indent=2) + "\n").encode())


def _open_url(url: str):
    return urlopen(Request(url, headers={"User-Agent": "ml-pipes-vlm dataset preparation"}), timeout=60)


def _download(url: str, path: Path, *, checksum: str | None = None, force: bool = False) -> Path:
    algorithm, expected = checksum.split(":", 1) if checksum else ("sha256", None)
    if not force and path.is_file() and path.stat().st_size > 0:
        if expected is None or _hash(path, algorithm) == expected:
            return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        pending = None
        try:
            digest = hashlib.new(algorithm)
            size = 0
            with _open_url(url) as response, tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                pending = Path(stream.name)
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if not size:
                raise ValueError(f"Empty dataset download: {url}")
            if expected is not None and digest.hexdigest() != expected:
                raise ValueError(f"Dataset checksum mismatch: {url}")
            pending.replace(path)
            return path
        except (URLError, OSError) as exc:
            if attempt == 2:
                raise RuntimeError(f"Could not download {url}: {exc}") from exc
            time.sleep(0.5 * (attempt + 1))
        finally:
            if pending is not None:
                pending.unlink(missing_ok=True)
    raise AssertionError("Unreachable download state")


def _cached(path: Path, cache: Path) -> tuple[PreparedDataset | None, set[Path]]:
    if not path.is_file():
        return None, set()
    try:
        record = json.loads(path.read_text())
        if record["format_version"] != _FORMAT_VERSION:
            return None, set()
        files = {cache / name: checksum for name, checksum in record["files"].items()}
        if not files or any(not item.resolve().is_relative_to(cache) for item in files):
            return None, set()
        bad = {item for item, checksum in files.items() if not item.is_file() or _hash(item) != checksum}
        if bad:
            return None, bad
        data = cache / record["data"]
        images = None if record["images"] is None else cache / record["images"]
        if not data.resolve().is_relative_to(cache) or (
            images is not None and not images.resolve().is_relative_to(cache)
        ):
            return None, set()
        if record["dataset"] == "cifar10" and any(
            not (data / record["split"] / name).is_dir() for name in _CIFAR_NAMES
        ):
            return None, set()
        return PreparedDataset(
            record["dataset"], record["split"], data, images,
            tuple(record["names"]), record["metadata"],
            record.get("attributes"),
        ), set()
    except (OSError, ValueError, KeyError, TypeError):
        return None, set()


def _prepare_cifar(root: Path, output: Path, split: str, limit, classes, bad):
    archive = _download(_CIFAR_URL, root / "sources" / "cifar10-binary.tar.gz", checksum=_CIFAR_CHECKSUM)
    source_split = "test" if split == "val" else split
    batches = ["test_batch.bin"] if source_split == "test" else [f"data_batch_{i}.bin" for i in range(1, 6)]
    samples = []
    total = 0
    with tarfile.open(archive, "r:gz") as source:
        for batch_index, filename in enumerate(batches):
            name = "cifar-10-batches-bin/" + filename
            members = [member for member in source.getmembers() if member.name == name]
            if len(members) != 1 or not members[0].isfile():
                raise ValueError(f"Missing or duplicate CIFAR-10 batch: {name}")
            stream = source.extractfile(members[0])
            assert stream is not None
            content = stream.read()
            if not content or len(content) % 3073:
                raise ValueError(f"Invalid CIFAR-10 binary batch: {name}")
            records = np.frombuffer(content, dtype=np.uint8).reshape(-1, 3073)
            if (records[:, 0] >= len(_CIFAR_NAMES)).any():
                raise ValueError(f"Invalid CIFAR-10 label in {name}")
            total += len(records)
            for index, record in enumerate(records):
                label = _CIFAR_NAMES[int(record[0])]
                if classes is None or label in classes:
                    samples.append((label, f"{batch_index}_{index:05d}.png", record[1:]))
    samples.sort(key=lambda sample: (sample[0], sample[1]))
    selected = samples if limit is None else samples[:limit]
    data = output / "data"
    for label in _CIFAR_NAMES:
        (data / split / label).mkdir(parents=True, exist_ok=True)
    files = []
    for label, filename, pixels in selected:
        rgb = pixels.reshape(3, 32, 32).transpose(1, 2, 0)
        success, encoded = cv2.imencode(".png", rgb[:, :, ::-1])
        if not success:
            raise ValueError("Could not encode CIFAR-10 image")
        path = data / split / label / filename
        _atomic_write(path, encoded.tobytes())
        files.append(path)
    return data, None, _CIFAR_NAMES, files, {
        "source_url": _CIFAR_URL, "archive_checksum": _CIFAR_CHECKSUM,
        "source_split": source_split, "source_samples": total,
        "selected_samples": len(selected), "selection": "Class name and source record order",
        "split_note": "val is an explicit alias for the official CIFAR-10 test split",
    }


def _prepare_coco(root: Path, output: Path, split: str, limit, classes, bad):
    archive = _download(_COCO_URL, root / "sources" / "coco128.zip", checksum=_COCO_CHECKSUM)
    files = []
    with ZipFile(archive) as source:
        names = source.namelist()
        images = sorted(name for name in names if re.fullmatch(r"coco128/images/train2017/[0-9]+\.jpg", name))
        if not images or len(images) != len(set(images)):
            raise ValueError("COCO128 archive has missing or duplicate images")
        selected = images if limit is None else images[:limit]
        for name in selected:
            label = name.replace("/images/", "/labels/").removesuffix(".jpg") + ".txt"
            for member in (name, label):
                if member not in names:
                    continue  # Missing labels remain visible to the detection reader.
                if names.count(member) != 1:
                    raise ValueError(f"Duplicate COCO128 member: {member}")
                path = output / member.removeprefix("coco128/")
                _atomic_write(path, source.read(member))
                files.append(path)
    data = output / "data.yaml"
    # JSON is valid YAML and avoids a YAML dependency in the downloader.
    _write_json(data, {"path": ".", split: "images/train2017", "names": list(_COCO_NAMES)})
    files.append(data)
    return data, None, _COCO_NAMES, files, {
        "source_url": _COCO_URL, "archive_checksum": _COCO_CHECKSUM,
        "source_split": "train2017", "source_samples": len(images),
        "selected_samples": len(selected), "selection": "Image filename order",
        "split_note": "COCO128 train and val reuse the same 128 COCO training images",
    }


def _ground_annotation(path: Path) -> dict:
    if not path.is_file():
        for attempt in range(3):
            try:
                with _open_url(_MDETR_URL) as response, tarfile.open(fileobj=response, mode="r|gz") as source:
                    for member in source:
                        if Path(member.name).name == path.name:
                            if not member.isfile():
                                raise ValueError("Grounding annotation must be a regular file")
                            stream = source.extractfile(member)
                            assert stream is not None
                            content = stream.read()
                            json.loads(content)
                            _atomic_write(path, content)
                            break
                    else:
                        raise FileNotFoundError(f"MDETR archive does not contain {path.name}")
                break
            except (URLError, OSError) as exc:
                if isinstance(exc, FileNotFoundError) or attempt == 2:
                    raise RuntimeError(f"Could not prepare {path.name} from {_MDETR_URL}: {exc}") from exc
                time.sleep(0.5 * (attempt + 1))
    content = json.loads(path.read_text())
    if not isinstance(content, dict) or any(not isinstance(content.get(key), list) for key in ("images", "annotations", "categories")):
        raise ValueError("Invalid MDETR annotation structure")
    return content


def _prepare_ground(root: Path, output: Path, split: str, limit, classes, bad):
    source_path = root / "sources" / f"finetune_refcocog_{split}.json"
    original = _ground_annotation(source_path)
    queries = original["images"] if limit is None else original["images"][:limit]
    ids = [query.get("id") for query in queries]
    if any(type(value) is not int for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Invalid or duplicate grounding query IDs")
    selected_ids = set(ids)
    subset = {**original, "images": queries, "annotations": [item for item in original["annotations"] if item.get("image_id") in selected_ids]}
    images = root / "images" / "train2014"
    files = []
    dimensions = {}
    for query in queries:
        filename = query.get("file_name")
        if not isinstance(filename, str) or not re.fullmatch(r"COCO_train2014_[0-9]{12}\.jpg", filename):
            raise ValueError(f"Invalid COCO train2014 filename: {filename!r}")
        size = (query.get("width"), query.get("height"))
        if any(type(value) is not int or value <= 0 for value in size):
            raise ValueError(f"Invalid grounding image dimensions: {filename}")
        if filename in dimensions and dimensions[filename] != size:
            raise ValueError(f"Inconsistent grounding image dimensions: {filename}")
        dimensions[filename] = size
    for filename, size in dimensions.items():
        path = images / filename
        _download(_COCO_IMAGE_URL + filename, path, force=path in bad)
        decoded = cv2.imread(str(path))
        if decoded is None or (decoded.shape[1], decoded.shape[0]) != size:
            _download(_COCO_IMAGE_URL + filename, path, force=True)
            decoded = cv2.imread(str(path))
            if decoded is None or (decoded.shape[1], decoded.shape[0]) != size:
                raise ValueError(f"Downloaded image dimensions differ from annotation: {filename}")
        files.append(path)
    data = output / "annotations"
    annotation = data / source_path.name
    _write_json(annotation, subset)
    # Reuse the local loader's annotation contract before publishing a cache hit.
    from .data import RefCOCODataset

    RefCOCODataset(data, images, split=split)
    files.append(annotation)
    return data, images, tuple(item["name"] for item in original["categories"]), files, {
        "source_url": _MDETR_URL, "source_version": "MDETR v1.0.1",
        "source_annotation_sha256": _hash(source_path), "expected_split_by": "umd",
        "image_url_prefix": _COCO_IMAGE_URL,
        "source_expressions": len(original["images"]),
        "source_unique_images": len({query["file_name"] for query in original["images"]}),
        "selected_expressions": len(queries), "selected_unique_images": len(dimensions),
        "selection": "Source JSON expression order", "query_ids": ids,
    }


def _prepare_ovad(root: Path, output: Path, split: str, limit, groups, bad):
    from .data import OVADDataset, OVAD_PROTOCOL, _ovad_selection

    archive = _download(_OVAD_URL, root / "sources" / "ovad.zip", checksum=_OVAD_CHECKSUM)
    with ZipFile(archive) as source:
        if source.namelist().count("ovad2000.json") != 1:
            raise ValueError("OVAD archive must contain exactly one ovad2000.json")
        content = source.read("ovad2000.json")
    original = json.loads(content)
    attributes, samples, counts = _ovad_selection(original, groups, limit)
    if not samples:
        raise ValueError("No eligible OVAD objects for the selected groups")
    selected_ids = {sample["sample_id"] for sample in samples}
    image_ids = {sample["image_id"] for sample in samples}
    subset = {
        **original,
        "images": [image for image in original["images"] if image["id"] in image_ids],
        "annotations": [item for item in original["annotations"] if item["id"] in selected_ids],
    }
    dimensions = {}
    for image in subset["images"]:
        filename = image["file_name"]
        if not re.fullmatch(r"[0-9]{12}\.jpg", filename):
            raise ValueError(f"Invalid COCO val2017 filename: {filename!r}")
        size = (image["width"], image["height"])
        if filename in dimensions and dimensions[filename] != size:
            raise ValueError(f"Inconsistent OVAD image dimensions: {filename}")
        dimensions[filename] = size
    images = root / "images" / "val2017"
    files = []
    for filename, size in dimensions.items():
        path = images / filename
        _download(_OVAD_IMAGE_URL + filename, path, force=path in bad)
        decoded = cv2.imread(str(path))
        if decoded is None or (decoded.shape[1], decoded.shape[0]) != size:
            _download(_OVAD_IMAGE_URL + filename, path, force=True)
            decoded = cv2.imread(str(path))
            if decoded is None or (decoded.shape[1], decoded.shape[0]) != size:
                raise ValueError(f"Downloaded image dimensions differ from OVAD annotation: {filename}")
        files.append(path)
    data = output / "ovad2000.json"
    _write_json(data, subset)
    OVADDataset(data, images, split=split, groups=groups)
    files.append(data)
    return data, images, (), files, {
        "source_url": _OVAD_URL, "archive_checksum": _OVAD_CHECKSUM,
        "source_annotation_sha256": hashlib.sha256(content).hexdigest(),
        "image_url_prefix": _OVAD_IMAGE_URL, "source_split": "test",
        "split_note": "val is an alias for the published OVAD test benchmark",
        "protocol": OVAD_PROTOCOL, "selected_groups": list(attributes),
        "selection": "Source JSON object annotation order, after area and categorical target filtering",
        "sample_ids": [sample["sample_id"] for sample in samples], "counts": counts,
    }, attributes


def _prepare_vqav2(root: Path, output: Path, split: str, limit, classes, bad):
    from .data import VQAv2Dataset, _vqav2_selection

    filenames = {
        "Questions": f"v2_OpenEnded_mscoco_{split}2014_questions.json",
        "Annotations": f"v2_mscoco_{split}2014_annotations.json",
    }
    originals, files, urls, hashes = {}, [], {}, {}
    for kind, filename in filenames.items():
        url = _VQA_URL.format(kind=kind, split=split.title())
        archive = root / "sources" / url.rsplit("/", 1)[-1]
        _download(url, archive, force=archive in bad)
        with ZipFile(archive) as source:
            if source.namelist().count(filename) != 1:
                raise ValueError(f"VQA v2 archive must contain exactly one {filename}")
            originals[kind] = json.loads(source.read(filename))
        files.append(archive)
        urls[kind] = url
        hashes[kind] = _hash(archive)
    selected, counts = _vqav2_selection(originals["Questions"], originals["Annotations"], limit)
    ids = {sample["sample_id"] for sample in selected}
    data = output / "annotations"
    for kind, filename in filenames.items():
        key = kind.lower()
        subset = {
            **originals[kind],
            key: [item for item in originals[kind][key] if item["question_id"] in ids],
        }
        path = data / filename
        _write_json(path, subset)
        files.append(path)
    images = root / "images" / f"{split}2014"
    image_url = _VQA_IMAGE_URL.format(split=split)
    for image_id in dict.fromkeys(sample["image_id"] for sample in selected):
        filename = f"COCO_{split}2014_{image_id:012d}.jpg"
        path = images / filename
        _download(image_url + filename, path, force=path in bad)
        if cv2.imread(str(path)) is None:
            _download(image_url + filename, path, force=True)
            if cv2.imread(str(path)) is None:
                raise ValueError(f"Could not decode downloaded VQA v2 image: {filename}")
        files.append(path)
    VQAv2Dataset(data, images, split=split)
    return data, images, (), files, {
        "source_urls": urls, "source_archive_sha256": hashes, "source_version": "VQA v2",
        "image_url_prefix": image_url, "source_split": f"{split}2014",
        "selection": "Source JSON question order", **counts,
        "question_ids": [sample["sample_id"] for sample in selected],
    }


def _prepare_captions(root: Path, output: Path, split: str, limit, classes, bad):
    from .data import COCOCaptionsDataset, _coco_captions_selection

    archive = root / "sources" / "annotations_trainval2017.zip"
    _download(_CAPTIONS_URL, archive, force=archive in bad)
    member = f"annotations/captions_{split}2017.json"
    with ZipFile(archive) as source:
        if source.namelist().count(member) != 1:
            raise ValueError(f"COCO archive must contain exactly one {member}")
        content = source.read(member)
    original = json.loads(content)
    selected, counts = _coco_captions_selection(original, limit)
    ids = {sample["image_id"] for sample in selected}
    subset = {
        **original,
        "images": [image for image in original["images"] if image["id"] in ids],
        "annotations": [item for item in original["annotations"] if item["image_id"] in ids],
    }
    images = root / "images" / f"{split}2017"
    files = [archive]
    image_url = _CAPTIONS_IMAGE_URL.format(split=split)
    for sample in selected:
        filename = sample["file_name"]
        if not re.fullmatch(r"[0-9]{12}\.jpg", filename):
            raise ValueError(f"Invalid COCO {split}2017 filename: {filename!r}")
        path = images / filename
        _download(image_url + filename, path, force=path in bad)
        decoded = cv2.imread(str(path))
        if decoded is None or (decoded.shape[1], decoded.shape[0]) != sample["image_size"]:
            _download(image_url + filename, path, force=True)
            decoded = cv2.imread(str(path))
            if decoded is None or (decoded.shape[1], decoded.shape[0]) != sample["image_size"]:
                raise ValueError(f"Downloaded image dimensions differ from COCO caption annotation: {filename}")
        files.append(path)
    data = output / f"captions_{split}2017.json"
    _write_json(data, subset)
    COCOCaptionsDataset(data, images, split=split)
    files.append(data)
    return data, images, (), files, {
        "source_url": _CAPTIONS_URL, "source_version": "COCO Captions 2017",
        "source_archive_sha256": _hash(archive),
        "source_annotation_sha256": hashlib.sha256(content).hexdigest(),
        "image_url_prefix": image_url, "source_split": f"{split}2017",
        "selection": "Source JSON image order; all reference captions retained",
        "image_ids": [sample["image_id"] for sample in selected], **counts,
    }


def prepare_dataset(
    dataset: str, *, split: str = "val", max_samples: int | None = None,
    cache_dir: str | Path | None = None, classes: tuple[str, ...] | None = None,
    groups: tuple[str, ...] | None = None,
) -> PreparedDataset:
    """Download/cache a supported split and return paths for existing loaders.

    ``max_samples`` selects the same deterministic prefix as validation.
    ``classes`` selects CIFAR-10 classes before the limit, matching class-filtered
    validation. CIFAR-10/COCO128 require their shared source archives; RefCOCOg
    streams a shared annotation archive and fetches only referenced images.
    OVAD ``groups`` selects categorical groups (all 22 by default) before
    filtering/limiting objects and fetching distinct COCO val2017 images.
    VQA v2 limits source-order questions, retaining all ten answers and fetching
    only distinct referenced COCO 2014 images.
    COCO captions limits images in source JSON image order, retaining every
    reference caption and downloading only selected COCO 2017 images.
    """
    if not isinstance(dataset, str) or dataset not in DATASETS:
        raise ValueError(f"Supported downloadable datasets: {', '.join(DATASETS)}")
    info = DATASETS[dataset]
    if split not in info.splits:
        raise ValueError(f"Unsupported split {split!r} for {dataset}; expected {info.splits}")
    if max_samples is not None and (type(max_samples) is not int or max_samples < 1):
        raise ValueError("max_samples must be a positive integer or None")
    if classes is not None:
        if dataset != "cifar10" or not classes or any(name not in info.names for name in classes):
            raise ValueError("classes must contain supported CIFAR-10 class names")
        classes = tuple(sorted(set(classes)))
    if groups is not None:
        from .data import _ovad_groups

        if dataset != "ovad":
            raise ValueError("groups applies only to OVAD preparation")
        groups = _ovad_groups(groups)
    cache = _cache_root(cache_dir)
    class_key = "all" if classes is None else hashlib.sha256(json.dumps(classes).encode()).hexdigest()[:16]
    if groups is not None:
        from .data import OVAD_GROUPS

        class_key = "all" if groups == OVAD_GROUPS else hashlib.sha256(json.dumps(groups).encode()).hexdigest()[:16]
    root = cache / dataset
    output = root / "prepared" / split / ("all" if max_samples is None else str(max_samples)) / class_key
    manifest = output / "prepared.json"
    cached, bad = _cached(manifest, cache)
    if cached is not None:
        return cached
    attributes = None
    if dataset == "ovad":
        data, images, names, files, metadata, attributes = _prepare_ovad(root, output, split, max_samples, groups, bad)
    else:
        preparer = {
            "cifar10": _prepare_cifar, "coco128": _prepare_coco,
            "refcocog": _prepare_ground, "vqav2": _prepare_vqav2,
            "coco_captions2017": _prepare_captions,
        }[dataset]
        data, images, names, files, metadata = preparer(root, output, split, max_samples, classes, bad)
    if not files:
        raise ValueError(f"No samples available in {dataset} split {split!r}")
    metadata = {
        "dataset": dataset, "split": split, "max_samples": max_samples,
        "classes": None if classes is None else list(classes), "preparation_version": _FORMAT_VERSION, **metadata,
    }
    _write_json(manifest, {
        "format_version": _FORMAT_VERSION, "dataset": dataset, "split": split,
        "data": str(data.relative_to(cache)),
        "images": None if images is None else str(images.relative_to(cache)),
        "names": names, "metadata": metadata, "attributes": attributes,
        "files": {str(path.relative_to(cache)): _hash(path) for path in files},
    })
    return PreparedDataset(dataset, split, data, images, names, metadata, attributes)
