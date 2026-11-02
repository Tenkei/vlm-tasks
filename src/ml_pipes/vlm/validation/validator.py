"""Validation orchestration and task-specific evaluation."""
from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from itertools import islice
from time import perf_counter

import numpy as np

from ..errors import InvalidModelResponseError, ModelExecutionError
from ..types import ClassificationResult, DetectionResult
from .data import ClassificationDataset, DataLoader, YOLODataset
from .metrics import ClassificationMetrics, DetectionMetrics


class BaseValidator:
    """Set up a dataset, call the configured task, and accumulate metrics.

    Override dataset construction and metric methods for another task or data
    source. A supplied dataloader must expose its dataset through ``.dataset``;
    the dataset's ``names`` maps class IDs to names. Loaders yield the sample
    dictionaries documented by the concrete dataset classes.
    """

    def __init__(
        self, data=None, *, dataloader=None, split: str = "val",
        class_map: Mapping[str, str] | None = None,
        max_samples: int | None = None, verbose: bool = True,
    ) -> None:
        if (data is None) == (dataloader is None):
            raise ValueError("Supply exactly one of data or dataloader")
        if max_samples is not None and (type(max_samples) is not int or max_samples < 1):
            raise ValueError("max_samples must be a positive integer or None")
        if class_map is not None and (
            not isinstance(class_map, Mapping) or not class_map
            or any(not isinstance(key, str) or not key.strip() for key in class_map)
            or any(not isinstance(value, str) or not value.strip() for value in class_map.values())
        ):
            raise ValueError("class_map must be a nonempty dataset-name to task-label mapping")
        self.data = data
        self.split = split
        self.max_samples = max_samples
        self.verbose = verbose
        self._provided_dataloader = dataloader
        self._requested_class_map = None if class_map is None else dict(class_map)

    def __call__(self, task):
        started = perf_counter()
        self.task = task
        self.dataset = (
            self.build_dataset() if self._provided_dataloader is None
            else self._provided_dataloader.dataset
        )
        self._align_classes()
        self.dataloader = (
            self.get_dataloader(self.dataset) if self._provided_dataloader is None
            else self._provided_dataloader
        )
        self.init_metrics()
        self.metrics.dataset_counts = dict(getattr(self.dataset, "counts", {}))
        self._timings = {"load": 0.0, "inference": 0.0, "metrics": 0.0}
        self._started = started
        last = perf_counter()
        for sample in islice(self._iter_samples(), self.max_samples):
            self._timings["load"] += perf_counter() - last
            inference_started = perf_counter()
            try:
                prediction = task(sample["img"])
            except (ModelExecutionError, InvalidModelResponseError) as exc:
                prediction = None
                self.metrics.failures.append({
                    "im_file": str(sample["im_file"]),
                    "type": type(exc).__name__, "message": str(exc),
                })
            self._timings["inference"] += perf_counter() - inference_started
            metric_started = perf_counter()
            self.update_metrics(prediction, sample)
            self._timings["metrics"] += perf_counter() - metric_started
            if self.verbose:
                print(f"\rValidated {self.metrics.seen} samples", end="", file=sys.stderr, flush=True)
            last = perf_counter()
        if not self.metrics.seen:
            raise ValueError("No samples belong to the evaluated classes")
        self.finalize_metrics()
        if self.verbose:
            print("\n" + json.dumps(self.metrics.results_dict, indent=2), file=sys.stderr)
        return self.metrics

    def _align_classes(self) -> None:
        dataset_names = set(self.dataset.names.values())
        task_names = set(self.task.labels)
        mapping = self._requested_class_map
        if mapping is None:
            mapping = {name: name for name in self.task.labels}
        missing = set(mapping) - dataset_names
        unknown = set(mapping.values()) - task_names
        if missing or unknown:
            raise ValueError(
                f"Invalid class alignment: missing dataset names={sorted(missing)}, "
                f"unknown task labels={sorted(unknown)}. Supply class_map for aliases or a subset."
            )
        self.class_map = {
            class_id: mapping[name] for class_id, name in self.dataset.names.items() if name in mapping
        }
        self.names = tuple(name for name in self.task.labels if name in mapping.values())

    def _iter_samples(self):
        return iter(self.dataloader)

    def build_dataset(self):
        raise NotImplementedError

    def get_dataloader(self, dataset):
        return DataLoader(dataset)

    def init_metrics(self) -> None:
        raise NotImplementedError

    def update_metrics(self, prediction, sample: dict) -> None:
        raise NotImplementedError

    def finalize_metrics(self) -> None:
        self.metrics.speed = {
            name: elapsed * 1000 / self.metrics.seen for name, elapsed in self._timings.items()
        }
        self.metrics.elapsed_seconds = perf_counter() - self._started


class ClassificationValidator(BaseValidator):
    """Validate a single-label Classify task against class directories."""

    def build_dataset(self):
        return ClassificationDataset(self.data, split=self.split)

    def get_dataloader(self, dataset):
        indices = [
            i for i, sample in enumerate(dataset.samples)
            if int(sample["cls"][0]) in self.class_map
        ]
        return DataLoader(dataset, indices=indices)

    def _iter_samples(self):
        # Also apply selection to an externally supplied dataloader.
        return (sample for sample in self.dataloader if int(sample["cls"][0]) in self.class_map)

    def init_metrics(self) -> None:
        self.metrics = ClassificationMetrics(self.names, prediction_names=tuple(self.task.labels))

    def update_metrics(self, prediction, sample: dict) -> None:
        if prediction is not None and (
            not isinstance(prediction, ClassificationResult) or len(prediction.labels) != 1
        ):
            raise TypeError("ClassificationValidator requires one-label ClassificationResult outputs")
        self.metrics.update(
            self.class_map[int(sample["cls"][0])],
            None if prediction is None else prediction.labels[0],
        )


class DetectionValidator(BaseValidator):
    """Validate a Detect task against YOLO detection labels."""

    def __init__(self, *args, match_iou: float = 0.5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Validate before dataset setup or any task execution.
        if (
            isinstance(match_iou, bool) or not isinstance(match_iou, (int, float))
            or not np.isfinite(match_iou) or not 0 < match_iou <= 1
        ):
            raise ValueError("match_iou must be finite and in (0, 1]")
        self.match_iou = float(match_iou)

    def build_dataset(self):
        return YOLODataset(self.data, split=self.split)

    def init_metrics(self) -> None:
        self.metrics = DetectionMetrics(self.names, match_iou=self.match_iou)

    def update_metrics(self, prediction, sample: dict) -> None:
        if prediction is not None and not isinstance(prediction, DetectionResult):
            raise TypeError("DetectionValidator requires DetectionResult outputs")
        indices = [i for i, class_id in enumerate(sample["cls"]) if int(class_id) in self.class_map]
        target_labels = [self.class_map[int(sample["cls"][i])] for i in indices]
        detections = [] if prediction is None else [
            item for item in prediction.detections if item.label in self.names
        ]
        boxes = np.array([
            [item.box.x1, item.box.y1, item.box.x2, item.box.y2] for item in detections
        ]).reshape(-1, 4)
        self.metrics.update(
            target_labels, sample["bboxes"][indices],
            [item.label for item in detections], boxes,
        )
