"""Prepare supported benchmark datasets as local inputs for validation.

Only registered sources are fetched. Loaders turn prepared files into HF datasets.
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

from . import dataset_constants as constants
from .data import _limit


@dataclass(frozen=True)
class PreparedFiles:
    """Local source paths and preparation provenance for a dataset loader."""

    data: Path
    images: Path | None
    metadata: dict


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


def _download_image(url: str, path: Path, *, size=None, force=False) -> Path:
    """Check decoded pixels and retry a corrupt or incorrectly sized download."""
    for attempt in range(2):
        _download(url, path, force=force or attempt > 0)
        decoded = cv2.imread(str(path))
        if decoded is not None and (size is None or (decoded.shape[1], decoded.shape[0]) == size):
            return path
    if decoded is None:
        raise ValueError(f"Could not decode downloaded image: {path.name}")
    raise ValueError(f"Downloaded image dimensions differ from annotation: {path.name}")


def _cached(path: Path, cache: Path) -> tuple[PreparedFiles | None, set[Path]]:
    if not path.is_file():
        return None, set()
    try:
        record = json.loads(path.read_text())
        if record["format_version"] != constants.PREPARATION_VERSION:
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
        return PreparedFiles(data, images, record["metadata"]), set()
    except (OSError, ValueError, KeyError, TypeError):
        return None, set()


def _preparation_cache(dataset, split, max_samples, cache_dir, *, selection=None):
    """Validate a split and locate its cached preparation and damaged files."""
    info = constants.DATASETS[dataset]
    if split not in info.splits:
        raise ValueError(f"Unsupported split {split!r} for {dataset}; expected {info.splits}")
    _limit(max_samples)
    cache = _cache_root(cache_dir)
    selection_key = "all" if selection is None else hashlib.sha256(json.dumps(selection).encode()).hexdigest()[:16]
    root = cache / dataset
    output = root / "prepared" / split / ("all" if max_samples is None else str(max_samples)) / selection_key
    cached, bad = _cached(output / "prepared.json", cache)
    return root, output, cached, bad


def _save_preparation(root, output, split, *, data, images=None, metadata, files) -> PreparedFiles:
    """Record prepared files and provenance after successful preparation."""
    if not files:
        raise ValueError(f"No samples available in {root.name} split {split!r}")
    cache = root.parent
    metadata = {
        "dataset": root.name, "split": split,
        "preparation_version": constants.PREPARATION_VERSION, **metadata,
    }
    _write_json(output / "prepared.json", {
        "format_version": constants.PREPARATION_VERSION,
        "data": str(data.relative_to(cache)),
        "images": None if images is None else str(images.relative_to(cache)),
        "metadata": metadata,
        "files": {str(path.relative_to(cache)): _hash(path) for path in files},
    })
    return PreparedFiles(data, images, metadata)


def prepare_cifar10(
    *, split: str = "test", max_samples: int | None = None,
    classes=None, cache_dir: str | Path | None = None,
) -> PreparedFiles:
    """Cache CIFAR-10 images, selecting classes before the sample limit."""
    if classes is not None:
        if not classes or any(name not in constants.CIFAR_NAMES for name in classes):
            raise ValueError("classes must contain supported CIFAR-10 class names")
        classes = tuple(sorted(set(classes)))
    root, output, cached, _ = _preparation_cache("cifar10", split, max_samples, cache_dir, selection=classes)
    if cached is not None and all((cached.data / split / name).is_dir() for name in constants.CIFAR_NAMES):
        return cached
    archive = _download(constants.CIFAR_URL, root / "sources" / "cifar10-binary.tar.gz", checksum=constants.CIFAR_CHECKSUM)
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
            if (records[:, 0] >= len(constants.CIFAR_NAMES)).any():
                raise ValueError(f"Invalid CIFAR-10 label in {name}")
            total += len(records)
            for index, record in enumerate(records):
                label = constants.CIFAR_NAMES[int(record[0])]
                if classes is None or label in classes:
                    samples.append((label, f"{batch_index}_{index:05d}.png", record[1:]))
    samples.sort(key=lambda sample: (sample[0], sample[1]))
    selected = samples if max_samples is None else samples[:max_samples]
    data = output / "data"
    for label in constants.CIFAR_NAMES:
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
    return _save_preparation(root, output, split, data=data, files=files, metadata={
        "max_samples": max_samples,
        "classes": None if classes is None else list(classes),
        "source_url": constants.CIFAR_URL, "archive_checksum": constants.CIFAR_CHECKSUM,
        "source_split": source_split, "source_samples": total,
        "selected_samples": len(selected), "selection": "Class name and source record order",
        "split_note": "val is an explicit alias for the official CIFAR-10 test split",
    })


def prepare_coco128(
    *, split: str = "val", max_samples: int | None = None,
    cache_dir: str | Path | None = None,
) -> PreparedFiles:
    """Cache a deterministic image subset from the COCO128 archive."""
    root, output, cached, _ = _preparation_cache("coco128", split, max_samples, cache_dir)
    if cached is not None:
        return cached
    archive = _download(constants.COCO_URL, root / "sources" / "coco128.zip", checksum=constants.COCO_CHECKSUM)
    files = []
    with ZipFile(archive) as source:
        names = source.namelist()
        images = sorted(name for name in names if re.fullmatch(r"coco128/images/train2017/[0-9]+\.jpg", name))
        if not images or len(images) != len(set(images)):
            raise ValueError("COCO128 archive has missing or duplicate images")
        selected = images if max_samples is None else images[:max_samples]
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
    _write_json(data, {"path": ".", split: "images/train2017", "names": list(constants.COCO_NAMES)})
    files.append(data)
    return _save_preparation(root, output, split, data=data, files=files, metadata={
        "max_samples": max_samples,
        "source_url": constants.COCO_URL, "archive_checksum": constants.COCO_CHECKSUM,
        "source_split": "train2017", "source_samples": len(images),
        "selected_samples": len(selected), "selection": "Image filename order",
        "split_note": "COCO128 train and val reuse the same 128 COCO training images",
    })


def _ground_annotation(path: Path) -> dict:
    if not path.is_file():
        for attempt in range(3):
            try:
                with _open_url(constants.MDETR_URL) as response, tarfile.open(fileobj=response, mode="r|gz") as source:
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
                    raise RuntimeError(f"Could not prepare {path.name} from {constants.MDETR_URL}: {exc}") from exc
                time.sleep(0.5 * (attempt + 1))
    content = json.loads(path.read_text())
    if not isinstance(content, dict) or any(not isinstance(content.get(key), list) for key in ("images", "annotations", "categories")):
        raise ValueError("Invalid MDETR annotation structure")
    return content


def prepare_refcocog(
    *, split: str = "val", max_samples: int | None = None,
    cache_dir: str | Path | None = None,
) -> PreparedFiles:
    """Cache RefCOCOg annotations and images referenced by selected expressions."""
    root, output, cached, bad = _preparation_cache("refcocog", split, max_samples, cache_dir)
    if cached is not None:
        return cached
    from .default_datasets import _refcoco_selection

    source_path = root / "sources" / f"finetune_refcocog_{split}.json"
    original = _ground_annotation(source_path)
    queries = original["images"] if max_samples is None else original["images"][:max_samples]
    ids = [query.get("id") for query in queries]
    if any(type(value) is not int for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Invalid or duplicate grounding query IDs")
    selected_ids = set(ids)
    subset = {**original, "images": queries, "annotations": [item for item in original["annotations"] if item.get("image_id") in selected_ids]}
    _refcoco_selection(subset)
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
        files.append(_download_image(constants.COCO_IMAGE_URL + filename, path, size=size, force=path in bad))
    data = output / "annotations"
    annotation = data / source_path.name
    _write_json(annotation, subset)
    files.append(annotation)
    return _save_preparation(root, output, split, data=data, images=images, files=files, metadata={
        "max_samples": max_samples,
        "source_url": constants.MDETR_URL, "source_version": "MDETR v1.0.1",
        "source_annotation_sha256": _hash(source_path), "expected_split_by": "umd",
        "image_url_prefix": constants.COCO_IMAGE_URL,
        "source_expressions": len(original["images"]),
        "source_unique_images": len({query["file_name"] for query in original["images"]}),
        "selected_expressions": len(queries), "selected_unique_images": len(dimensions),
        "selection": "Source JSON expression order", "query_ids": ids,
    })


def prepare_ovad(
    *, split: str = "val", max_samples: int | None = None,
    groups=None, cache_dir: str | Path | None = None,
) -> PreparedFiles:
    """Cache OVAD annotations and images after categorical object filtering."""
    from .default_datasets import _ovad_groups

    groups = _ovad_groups(groups)
    selection = None if groups == constants.OVAD_GROUPS else groups
    root, output, cached, bad = _preparation_cache("ovad", split, max_samples, cache_dir, selection=selection)
    if cached is not None:
        return cached
    from .default_datasets import _ovad_selection

    archive = _download(constants.OVAD_URL, root / "sources" / "ovad.zip", checksum=constants.OVAD_CHECKSUM)
    with ZipFile(archive) as source:
        if source.namelist().count("ovad2000.json") != 1:
            raise ValueError("OVAD archive must contain exactly one ovad2000.json")
        content = source.read("ovad2000.json")
    original = json.loads(content)
    attributes, samples, counts = _ovad_selection(original, groups, max_samples)
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
        files.append(_download_image(constants.OVAD_IMAGE_URL + filename, path, size=size, force=path in bad))
    data = output / "ovad2000.json"
    _write_json(data, subset)
    files.append(data)
    return _save_preparation(root, output, split, data=data, images=images, files=files, metadata={
        "max_samples": max_samples,
        "source_url": constants.OVAD_URL, "archive_checksum": constants.OVAD_CHECKSUM,
        "source_annotation_sha256": hashlib.sha256(content).hexdigest(),
        "image_url_prefix": constants.OVAD_IMAGE_URL, "source_split": "test",
        "split_note": "val is an alias for the published OVAD test benchmark",
        "protocol": constants.OVAD_PROTOCOL, "selected_groups": list(attributes),
        "selection": "Source JSON object annotation order, after area and categorical target filtering",
        "sample_ids": [sample["sample_id"] for sample in samples], "counts": counts,
    })


def prepare_vqav2(
    *, split: str = "val", max_samples: int | None = None,
    cache_dir: str | Path | None = None,
) -> PreparedFiles:
    """Cache VQA v2 annotations and images referenced by selected questions."""
    root, output, cached, bad = _preparation_cache("vqav2", split, max_samples, cache_dir)
    if cached is not None:
        return cached
    from .default_datasets import _vqav2_selection

    filenames = {
        "Questions": f"v2_OpenEnded_mscoco_{split}2014_questions.json",
        "Annotations": f"v2_mscoco_{split}2014_annotations.json",
    }
    originals, files, urls, hashes = {}, [], {}, {}
    for kind, filename in filenames.items():
        url = constants.VQA_URL.format(kind=kind, split=split.title())
        archive = root / "sources" / url.rsplit("/", 1)[-1]
        _download(url, archive, force=archive in bad)
        with ZipFile(archive) as source:
            if source.namelist().count(filename) != 1:
                raise ValueError(f"VQA v2 archive must contain exactly one {filename}")
            originals[kind] = json.loads(source.read(filename))
        files.append(archive)
        urls[kind] = url
        hashes[kind] = _hash(archive)
    selected, counts = _vqav2_selection(originals["Questions"], originals["Annotations"], max_samples)
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
    image_url = constants.VQA_IMAGE_URL.format(split=split)
    for image_id in dict.fromkeys(sample["image_id"] for sample in selected):
        filename = f"COCO_{split}2014_{image_id:012d}.jpg"
        path = images / filename
        files.append(_download_image(image_url + filename, path, force=path in bad))
    return _save_preparation(root, output, split, data=data, images=images, files=files, metadata={
        "max_samples": max_samples,
        "source_urls": urls, "source_archive_sha256": hashes, "source_version": "VQA v2",
        "image_url_prefix": image_url, "source_split": f"{split}2014",
        "selection": "Source JSON question order", **counts,
        "question_ids": [sample["sample_id"] for sample in selected],
    })


def prepare_coco_captions2017(
    *, split: str = "val", max_samples: int | None = None,
    cache_dir: str | Path | None = None,
) -> PreparedFiles:
    """Cache COCO captions and selected images, retaining all references."""
    root, output, cached, bad = _preparation_cache("coco_captions2017", split, max_samples, cache_dir)
    if cached is not None:
        return cached
    from .default_datasets import _coco_captions_selection

    archive = root / "sources" / "annotations_trainval2017.zip"
    _download(constants.CAPTIONS_URL, archive, force=archive in bad)
    member = f"annotations/captions_{split}2017.json"
    with ZipFile(archive) as source:
        if source.namelist().count(member) != 1:
            raise ValueError(f"COCO archive must contain exactly one {member}")
        content = source.read(member)
    original = json.loads(content)
    selected, counts = _coco_captions_selection(original, max_samples)
    ids = {sample["image_id"] for sample in selected}
    subset = {
        **original,
        "images": [image for image in original["images"] if image["id"] in ids],
        "annotations": [item for item in original["annotations"] if item["image_id"] in ids],
    }
    images = root / "images" / f"{split}2017"
    files = [archive]
    image_url = constants.CAPTIONS_IMAGE_URL.format(split=split)
    for sample in selected:
        filename = sample["file_name"]
        if not re.fullmatch(r"[0-9]{12}\.jpg", filename):
            raise ValueError(f"Invalid COCO {split}2017 filename: {filename!r}")
        path = images / filename
        files.append(_download_image(image_url + filename, path, size=sample["image_size"], force=path in bad))
    data = output / f"captions_{split}2017.json"
    _write_json(data, subset)
    files.append(data)
    return _save_preparation(root, output, split, data=data, images=images, files=files, metadata={
        "max_samples": max_samples,
        "source_url": constants.CAPTIONS_URL, "source_version": "COCO Captions 2017",
        "source_archive_sha256": _hash(archive),
        "source_annotation_sha256": hashlib.sha256(content).hexdigest(),
        "image_url_prefix": image_url, "source_split": f"{split}2017",
        "selection": "Source JSON image order; all reference captions retained",
        "image_ids": [sample["image_id"] for sample in selected], **counts,
    })
