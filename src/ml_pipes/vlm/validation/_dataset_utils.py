"""Shared HF builders, image resolution, and verified downloads."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import datasets as hf
from PIL import Image

from .data import _limit


# HF dataset construction

@dataclass
class _SourceConfig(hf.BuilderConfig):
    source_split: str = "val"
    max_samples: int | None = None


class _RowsBuilder(hf.GeneratorBasedBuilder):
    """Share HF row writing; each source implements its own acquisition and joins."""

    BUILDER_CONFIG_CLASS = _SourceConfig
    VERSION = hf.Version("1.0.0")

    def _info(self):
        return hf.DatasetInfo(description=self.__doc__ or "")

    def _split(self, rows, features):
        if not rows:
            raise ValueError("No samples available for the selected dataset options")
        self.info.features = hf.Features({
            "image": hf.Image(), "im_file": hf.Value("string"), **features,
        })
        return [hf.SplitGenerator(name=self.config.source_split, gen_kwargs={"rows": rows})]

    def _generate_examples(self, rows):
        for index, original in enumerate(rows):
            row = dict(original)
            size = row.pop("image_size", None)
            crop = row.pop("crop_box", None)
            if size is not None:
                with Image.open(row["image"]) as image:
                    if image.size != tuple(size):
                        raise ValueError("Image dimensions differ from dataset annotation")
                    if crop is not None:
                        row["image"] = image.convert("RGB").crop(crop)
            yield index, row


def _build(builder_type, *, name, split, max_samples, cache_dir, files=None, images=None, **options):
    _limit(max_samples)
    builder = builder_type(
        dataset_name=name, config_name=name, source_split=split, max_samples=max_samples,
        cache_dir=cache_dir, data_files=files, data_dir=images, **options,
    )
    # Local annotations may be edited between calls. Rebuild rows using HF's
    # source cache; remote benchmark configurations reuse the completed dataset.
    mode = hf.DownloadMode.REUSE_CACHE_IF_EXISTS if files else hf.DownloadMode.REUSE_DATASET_IF_EXISTS
    builder.download_and_prepare(download_mode=mode)
    return builder.as_dataset(split=split)


# Source files and images

def _json_member(manager, archive, filename):
    """Read one original annotation without writing an intermediate JSON file."""
    found = None
    for name, stream in manager.iter_archive(archive):
        if Path(name).name == filename:
            if found is not None:
                raise ValueError(f"Duplicate annotation file: {filename}")
            found = json.load(stream)
    if found is None:
        raise FileNotFoundError(f"Archive does not contain {filename}")
    return found


def _images(rows, *, directory=None, manager=None, prefix=None):
    """Resolve only the images referenced by the selected rows."""
    filenames = dict.fromkeys(row["file_name"] for row in rows)
    if directory is not None:
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Image directory not found: {root}")
        paths = {name: _resolve(name, root) for name in filenames}
        for path in paths.values():
            if not path.is_relative_to(root):
                raise ValueError(f"Image path escapes image directory: {path}")
            if not path.is_file():
                raise FileNotFoundError(f"Dataset image not found: {path}")
    else:
        for name in filenames:
            if Path(name).name != name or "\\" in name:
                raise ValueError(f"Invalid image filename: {name!r}")
        paths = manager.download({name: prefix + name for name in filenames})
    for row in rows:
        name = row.pop("file_name")
        row.update(image=str(paths[name]), im_file=str(paths[name]))
    return rows


def _resolve(path: str, parent: Path) -> Path:
    result = Path(path).expanduser()
    return (result if result.is_absolute() else parent / result).resolve()


# Verified downloads

def download_artifact(url, *, checksum, cache_dir=None):
    manager = hf.DownloadManager(download_config=hf.DownloadConfig(cache_dir=cache_dir, max_retries=2))
    path = Path(manager.download(url))
    # Tokenizers execute separately from dataset generation and need their pinned hash.
    try:
        check_checksum(path, checksum)
    except ValueError:
        manager.download_config.force_download = True
        path = Path(manager.download(url))
        check_checksum(path, checksum)
    return path


def check_checksum(path, checksum):
    algorithm, expected = checksum.split(":", 1)
    with Path(path).open("rb") as stream:
        actual = hashlib.file_digest(stream, algorithm).hexdigest()
    if actual != expected:
        raise ValueError(f"Dataset checksum mismatch: {path}")
