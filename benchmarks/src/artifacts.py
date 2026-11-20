"""Shared artifact helpers; no inference or dataset downloads at import time."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


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


def model_config_revision(config):
    revision = config.get("revision")
    if type(revision) is not int or revision < 1:
        raise ValueError("Model config revision must be a positive integer")
    return revision


def revision_fields(task, config):
    from .task_registry import TASKS

    return {"task_spec_revision": TASKS[task].revision,
            "model_config_revision": model_config_revision(config)}


def save_run_setup(output, config):
    """Save the selected definitions once, before inference starts."""
    from .task_registry import TASKS

    output = Path(output)
    model_config_revision(config)
    specs = {task: asdict(TASKS[task]) for task in config["tasks"]}
    paths = [output / "config.json", *(output / "task-specs" / f"{task}.json" for task in specs)]
    if any(path.exists() for path in paths):
        raise FileExistsError(f"Run setup already exists: {output}")
    write_json(output / "config.json", config)
    for task, spec in specs.items():
        write_json(output / "task-specs" / f"{task}.json", spec)


def check_run_setup(output, config):
    """Reject changed config or task definitions without rewriting saved setup."""
    from .task_registry import TASKS

    output = Path(output)
    model_config_revision(config)
    if json.loads((output / "config.json").read_text()) != config:
        raise ValueError("Run configuration differs from the saved run")
    for task in config["tasks"]:
        path = output / "task-specs" / f"{task}.json"
        if not path.is_file():
            raise ValueError(f"Run has no saved task spec: {task}; use the recorded source version for historical runs")
        if encoded(json.loads(path.read_text())) != encoded(asdict(TASKS[task])):
            raise ValueError(f"Run task spec differs from the saved run: {task}")


def save_svg(figure, path):
    """Save stable SVG metadata and remove whitespace emitted by Matplotlib."""
    path = Path(path)
    figure.savefig(path, metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def provenance():
    paths = sorted((ROOT / "src/ml_pipes/vlm").rglob("*.py"))
    paths += sorted((ROOT / "benchmarks").glob("*.py"))
    paths += sorted((ROOT / "benchmarks/src").rglob("*.py"))
    paths += sorted((ROOT / "benchmarks/setup/task-specs").glob("*.json"))
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
