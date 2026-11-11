"""Shared image resolution, archive reading, and verified downloads."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _limit(max_samples: int | None) -> None:
    if max_samples is not None and (type(max_samples) is not int or max_samples < 1):
        raise ValueError("max_samples must be a positive integer or None")


def _split(split, allowed=None):
    if (not isinstance(split, str) or not split or Path(split).name != split or split in {".", ".."}):
        raise ValueError("split must be a nonempty split name")
    if allowed is not None and split not in allowed:
        raise ValueError(f"Unsupported split {split!r}; expected {allowed}")


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


def download_artifact(url, *, checksum, cache_dir=None):
    import datasets as hf

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
