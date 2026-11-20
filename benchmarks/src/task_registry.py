"""Ordered task definitions shared by preparation, execution and reporting."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .artifacts import ROOT, digest

TASK_SPECS_DIRECTORY = ROOT / "benchmarks/setup/task-specs"


@dataclass(frozen=True)
class TaskSpec:
    order: int
    dataset: str
    operator: str
    title: str
    metrics: tuple[tuple[str, str], ...]
    primary_metric: str
    chart_metric_label: str
    geometry: str | None = None
    context: str | None = None

    @property
    def variation(self):
        return self.geometry.title() if self.geometry else None

    @property
    def chart_title(self):
        return f"{self.title} ({self.variation})" if self.variation else self.title


def load_task_specs(directory=TASK_SPECS_DIRECTORY):
    """Load one task per JSON file, preserving the explicit benchmark order."""
    tasks, orders = {}, set()
    for path in sorted(Path(directory).glob("*.json")):
        try:
            data = json.loads(path.read_text())
            data["metrics"] = tuple(tuple(metric) for metric in data["metrics"])
            spec = TaskSpec(**data)
            if type(spec.order) is not int or spec.order < 1 or spec.order in orders:
                raise ValueError("order must be a unique positive integer")
            if any(len(metric) != 2 for metric in spec.metrics) or spec.primary_metric not in dict(spec.metrics):
                raise ValueError("metrics must contain named pairs including primary_metric")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid task spec {path}: {exc}") from exc
        tasks[path.stem] = spec
        orders.add(spec.order)
    if not tasks:
        raise ValueError(f"No task specs found in {directory}")
    return dict(sorted(tasks.items(), key=lambda item: item[1].order))


def task_spec_hashes():
    """Include task-definition data in run and reporting provenance."""
    return {path.relative_to(ROOT).as_posix(): digest(path.read_bytes())
            for path in sorted(TASK_SPECS_DIRECTORY.glob("*.json"))}


# Primary chart metrics have a natural [0, 1] range, independent of observed models.
TASKS = load_task_specs()

DATASETS = {name: spec.dataset for name, spec in TASKS.items()}
METRICS = {name: spec.metrics for name, spec in TASKS.items()}
AXES = [(name, spec.primary_metric, spec.chart_title, spec.chart_metric_label)
        for name, spec in TASKS.items()]
TASK_SECTIONS = [(title, [(name, spec.variation) for name, spec in TASKS.items() if spec.title == title])
                 for title in dict.fromkeys(spec.title for spec in TASKS.values())]
