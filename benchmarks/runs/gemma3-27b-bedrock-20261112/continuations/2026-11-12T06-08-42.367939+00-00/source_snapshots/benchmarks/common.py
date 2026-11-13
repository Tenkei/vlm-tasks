"""Shared artifact helpers; no inference or dataset downloads at import time."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "classify": "cifar10", "detect": "coco128", "ground_box": "refcocog",
    "ground_point": "refcocog", "attributes": "ovad", "answer": "vqav2",
    "describe": "coco_captions2017",
}


def serialize(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def encoded(value):
    return json.dumps(value, sort_keys=True, default=serialize, ensure_ascii=False)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, default=serialize, ensure_ascii=False) + "\n")
    pending.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def provenance():
    paths = sorted((ROOT / "src/ml_pipes/vlm").rglob("*.py"))
    paths += sorted((ROOT / "benchmarks").glob("*.py"))
    return {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in paths},
    }


def sample_record(sample, index):
    image = sample["image"].convert("RGB")
    target = {key: value for key, value in sample.items() if key not in {"image", "im_file"}}
    return {
        "index": index, "sample_id": sample.get("sample_id", sample.get("image_id", index)),
        "image_size": list(image.size), "image_rgb_sha256": digest(image.tobytes()),
        "targets": target,
    }
