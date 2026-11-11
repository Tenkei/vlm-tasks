"""Convenience functions that return loaded HF datasets for validation."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from . import _dataset_utils as utils

if TYPE_CHECKING:
    from datasets import Dataset

__all__ = [
    "load_cifar10", "load_classification", "load_coco128", "load_yolo",
    "load_refcocog", "load_refcoco", "load_ovad", "load_vqav2", "load_coco_captions2017",
]


def _build(builder):
    import datasets as hf

    # Local annotations may be edited between calls. Rebuild rows using HF's
    # source cache; remote benchmark configurations reuse the completed dataset.
    mode = (hf.DownloadMode.REUSE_CACHE_IF_EXISTS if builder.config.data_files
            else hf.DownloadMode.REUSE_DATASET_IF_EXISTS)
    builder.download_and_prepare(download_mode=mode)
    return builder.as_dataset(split=builder.config.source_split)


def load_cifar10(*, split: str = "test", classes=None, max_samples: int | None = None, cache_dir=None) -> Dataset:
    """Load the pinned HF CIFAR-10 dataset; val aliases the held-out test split."""
    utils._limit(max_samples)
    from . import default_datasets as defaults

    hf = defaults.hf
    utils._split(split, defaults.DATASETS["cifar10"].splits)
    if classes is not None and (not classes or any(name not in defaults.CIFAR_NAMES for name in classes)):
        raise ValueError("classes must contain supported CIFAR-10 class names")
    dataset = hf.load_dataset(defaults.CIFAR_REPO, revision=defaults.CIFAR_REVISION,
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
    utils._limit(max_samples)
    from . import default_datasets as defaults

    hf = defaults.hf
    utils._split(split)
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


def load_coco128(*, split: str = "val", max_samples: int | None = None, cache_dir=None) -> Dataset:
    """Load the original COCO128 archive through an HF builder."""
    from . import default_datasets as defaults

    utils._split(split, defaults.DATASETS["coco128"].splits)
    builder = defaults.COCO128Builder(source_split=split, max_samples=max_samples, cache_dir=cache_dir)
    return _build(builder)


def load_yolo(data: str | Path, *, split: str = "val", max_samples: int | None = None, cache_dir=None) -> Dataset:
    """Load local YOLO YAML/TXT into canonical image/objects rows."""
    from . import default_datasets as defaults

    utils._split(split)
    builder = defaults.YOLOBuilder(
        source_split=split, max_samples=max_samples, cache_dir=cache_dir,
        data_files={"annotations": str(Path(data).expanduser().resolve())},
    )
    return _build(builder)


def load_refcocog(*, split: str = "val", max_samples: int | None = None, cache_dir=None) -> Dataset:
    """Load original RefCOCOg UMD annotations and selected COCO images."""
    from . import default_datasets as defaults

    utils._split(split, defaults.DATASETS["refcocog"].splits)
    builder = defaults.RefCOCOBuilder(source_split=split, max_samples=max_samples, cache_dir=cache_dir)
    return _build(builder)


def load_refcoco(annotations: str | Path, images: str | Path, *, dataset: str = "refcocog",
                 split: str = "val", max_samples: int | None = None, cache_dir=None) -> Dataset:
    """Load local MDETR RefCOCO annotations, one row per expression."""
    from . import default_datasets as defaults

    if not isinstance(dataset, str) or dataset not in ("refcocog", "refcoco", "refcoco+"):
        raise ValueError("dataset must be refcocog, refcoco, or refcoco+")
    utils._split(split)
    path = Path(annotations).expanduser().resolve() / f"finetune_{dataset}_{split}.json"
    builder = defaults.RefCOCOBuilder(
        config_name=dataset, source_split=split, max_samples=max_samples, cache_dir=cache_dir,
        data_files={"annotations": str(path)}, data_dir=str(Path(images).expanduser().resolve()),
    )
    return _build(builder)


def load_ovad(annotations=None, images=None, *, split: str = "val", groups=None,
              max_samples: int | None = None, cache_dir=None) -> Dataset:
    """Load categorical OVAD targets with cached subject crops."""
    from . import default_datasets as defaults

    utils._split(split, defaults.DATASETS["ovad"].splits)
    annotations, images = _local_files(annotations, images)

    builder = defaults.OVADBuilder(
        source_split=split, max_samples=max_samples, cache_dir=cache_dir, groups=groups,
        data_files=None if annotations is None else {"annotations": annotations}, data_dir=images,
    )
    return _build(builder)


def load_vqav2(annotations=None, images=None, *, split: str = "val",
               max_samples: int | None = None, cache_dir=None) -> Dataset:
    """Load VQA v2 questions with all ten reference answers."""
    from . import default_datasets as defaults

    utils._split(split, defaults.DATASETS["vqav2"].splits)
    annotations, images = _local_files(annotations, images)

    files = None if annotations is None else {
        "questions": str(Path(annotations) / f"v2_OpenEnded_mscoco_{split}2014_questions.json"),
        "annotations": str(Path(annotations) / f"v2_mscoco_{split}2014_annotations.json"),
    }
    builder = defaults.VQAv2Builder(
        source_split=split, max_samples=max_samples, cache_dir=cache_dir, data_files=files, data_dir=images,
    )
    return _build(builder)


def load_coco_captions2017(annotations=None, images=None, *, split: str = "val",
                          max_samples: int | None = None, cache_dir=None) -> Dataset:
    """Load COCO 2017 images with every reference caption."""
    from . import default_datasets as defaults

    utils._split(split, defaults.DATASETS["coco_captions2017"].splits)
    annotations, images = _local_files(annotations, images)

    builder = defaults.COCOCaptionsBuilder(
        source_split=split, max_samples=max_samples, cache_dir=cache_dir,
        data_files=None if annotations is None else {"annotations": annotations}, data_dir=images,
    )
    return _build(builder)


def _image_paths(batch):
    return {"im_file": [image["path"] for image in batch["image"]]}


def _local_files(annotations, images):
    if annotations is None:
        if images is not None:
            raise ValueError("images requires local annotations")
        return None, None
    if images is None:
        raise ValueError("Local annotations require an images directory")
    return str(Path(annotations).expanduser().resolve()), str(Path(images).expanduser().resolve())
