"""Validation orchestration and task-specific evaluation."""
from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from itertools import islice
from time import perf_counter

import numpy as np

from ..errors import InvalidModelResponseError, ModelExecutionError
from ..types import AttributePredictionResult, ClassificationResult, DetectionResult, GroundingResult
from ._vqa import SCORER_METADATA, validate_targets
from .data import (
    ClassificationDataset, DataLoader, OVADDataset, RefCOCODataset, VQAv2Dataset, YOLODataset,
    _classification_classes,
)
from .metrics import AnswerMetrics, AttributesMetrics, ClassificationMetrics, DetectionMetrics, GroundingMetrics
from .prepare import DATASETS, prepare_dataset


class BaseValidator:
    """Bind a task, set up its dataset, and accumulate validation metrics.

    Override dataset construction and metric methods for another task or data
    source. A supplied dataloader must expose its dataset through ``.dataset``;
    the dataset's ``names`` maps class IDs to names (or ``attributes`` defines
    categorical choices). Answer datasets need neither class metadata nor a
    vocabulary. Loaders yield the sample dictionaries documented by the concrete
    dataset classes.
    """

    default_dataset: str | None = None

    def __init__(
        self, task, data=None, *, dataloader=None, split: str = "val",
        class_map: Mapping[str, str] | None = None,
        max_samples: int | None = None, verbose: bool = True, cache_dir=None,
    ) -> None:
        if data is None and dataloader is None:
            data = self.default_dataset
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
        self.task = task
        self.data = data
        self.split = split
        self.max_samples = max_samples
        self.verbose = verbose
        self.cache_dir = cache_dir
        self._provided_dataloader = dataloader
        self._requested_class_map = None if class_map is None else dict(class_map)

    def __call__(self):
        started = perf_counter()
        self.prepared_dataset = None
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
        if self.prepared_dataset is not None:
            self.metrics.dataset_metadata = {
                **getattr(self.metrics, "dataset_metadata", {}),
                "data": str(self.prepared_dataset.data),
                "preparation": self.prepared_dataset.metadata,
            }
        self._timings = {"load": 0.0, "inference": 0.0, "metrics": 0.0}
        self._started = started
        last = perf_counter()
        for sample in islice(self.dataloader, self.max_samples):
            self._timings["load"] += perf_counter() - last
            inference_started = perf_counter()
            try:
                prediction = self.predict_sample(sample)
            except (ModelExecutionError, InvalidModelResponseError) as exc:
                prediction = None
                self.metrics.failures.append({
                    "im_file": str(sample["im_file"]),
                    "type": type(exc).__name__, "message": str(exc),
                    **{key: sample[key] for key in ("sample_id", "target") if key in sample},
                })
            self._timings["inference"] += perf_counter() - inference_started
            metric_started = perf_counter()
            self.update_metrics(prediction, sample)
            self._timings["metrics"] += perf_counter() - metric_started
            if self.verbose:
                print(f"\rValidated {self.metrics.seen} samples", end="", file=sys.stderr, flush=True)
            last = perf_counter()
        if not self.metrics.seen:
            raise ValueError("No samples to evaluate")
        self.finalize_metrics()
        if self.verbose:
            print("\n" + json.dumps(self.metrics.results_dict, indent=2), file=sys.stderr)
        return self.metrics

    def _align_classes(self) -> None:
        self.class_map = self._resolve_class_map(self.dataset.names)
        self.names = tuple(name for name in self.task.labels if name in self.class_map.values())

    def _resolve_class_map(self, names: Mapping[int, str]) -> dict[int, str]:
        dataset_names = set(names.values())
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
        return {
            class_id: mapping[name] for class_id, name in names.items() if name in mapping
        }

    def build_dataset(self):
        raise NotImplementedError

    def prepare_named_dataset(self, task: str):
        if not isinstance(self.data, str) or self.data not in DATASETS:
            return None
        if DATASETS[self.data].task != task:
            raise ValueError(f"Dataset {self.data!r} does not support {task} validation")
        classes = None
        if task == "classify":
            classes = tuple(self._requested_class_map or self.task.labels)
        self.prepared_dataset = prepare_dataset(
            self.data, split=self.split, max_samples=self.max_samples,
            cache_dir=self.cache_dir, classes=classes,
            **({"groups": tuple(self.task.attributes)} if task == "attributes" and self.task.attributes is not None else {}),
        )
        return self.prepared_dataset

    def get_dataloader(self, dataset):
        return DataLoader(dataset)

    def init_metrics(self) -> None:
        raise NotImplementedError

    def predict_sample(self, sample: dict):
        return self.task(sample["img"])

    def update_metrics(self, prediction, sample: dict) -> None:
        raise NotImplementedError

    def finalize_metrics(self) -> None:
        self.metrics.speed = {
            name: elapsed * 1000 / self.metrics.seen for name, elapsed in self._timings.items()
        }
        self.metrics.elapsed_seconds = perf_counter() - self._started


class AnswerValidator(BaseValidator):
    """Evaluate one Answer result per question using VQA v2 consensus."""

    default_dataset = "vqav2"

    def __init__(self, *args, images=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self._requested_class_map is not None:
            raise ValueError("class_map is not supported for answer validation")
        if self.split not in ("train", "val"):
            raise ValueError("VQA v2 supports labeled train and val splits only")
        automatic = isinstance(self.data, str) and self.data in DATASETS
        if automatic and images is not None:
            raise ValueError("Automatic VQA v2 preparation supplies images; use local data with images")
        if self._provided_dataloader is None and images is None and not automatic:
            raise ValueError("Answer validation requires images or a supplied dataloader")
        self.images = images

    def build_dataset(self):
        prepared = self.prepare_named_dataset("answer")
        return VQAv2Dataset(
            self.data if prepared is None else prepared.data,
            self.images if prepared is None else prepared.images,
            split=self.split, max_samples=self.max_samples,
        )

    def _align_classes(self) -> None:
        # Answers have no fixed vocabulary; answer types are reporting groups.
        pass

    def init_metrics(self) -> None:
        self.metrics = AnswerMetrics()
        self.metrics.dataset_metadata = {
            **getattr(self.dataset, "metadata", {}), **SCORER_METADATA,
        }

    def predict_sample(self, sample: dict):
        # Validate custom-loader targets before making a model call, too.
        validate_targets(sample["answers"], sample["answer_type"], sample["question_type"])
        return self.task((sample["img"], sample["question"]))

    def update_metrics(self, prediction, sample: dict) -> None:
        self.metrics.update(
            sample["answers"], prediction, answer_type=sample["answer_type"],
            question_type=sample["question_type"], im_file=sample["im_file"],
        )

    def finalize_metrics(self) -> None:
        super().finalize_metrics()
        if self.prepared_dataset is not None:
            self.metrics.dataset_counts.update({
                key: self.prepared_dataset.metadata[key]
                for key in ("source_questions", "source_images", "selected_questions", "selected_images")
            })
        self.metrics.dataset_counts.update(
            evaluated_questions=self.metrics.seen,
            evaluated_images=self.metrics.unique_images,
        )


class AttributesValidator(BaseValidator):
    """Evaluate a fixed Attributes schema against categorical object targets.

    Built-in OVAD data owns filtering/cropping. Custom loaders must expose
    ``dataset.attributes`` and yield already selected crops with known targets.
    """

    default_dataset = "ovad"

    def __init__(self, *args, images=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self._requested_class_map is not None:
            raise ValueError("class_map is not supported for attributes validation")
        automatic = isinstance(self.data, str) and self.data in DATASETS
        if automatic and images is not None:
            raise ValueError("Automatic OVAD preparation supplies images; use local data with images")
        if self._provided_dataloader is None and images is None and not automatic:
            raise ValueError("Attributes validation requires images or a supplied dataloader")
        self.images = images

    def build_dataset(self):
        prepared = self.prepare_named_dataset("attributes")
        return OVADDataset(
            self.data if prepared is None else prepared.data,
            self.images if prepared is None else prepared.images,
            split=self.split,
            groups=None if self.task.attributes is None else tuple(self.task.attributes),
            max_samples=self.max_samples,
        )

    def _align_classes(self) -> None:
        from ..ops import _attribute_choices, _choices

        schema = getattr(self.dataset, "attributes", None)
        if not isinstance(schema, Mapping) or not schema:
            raise ValueError("Attributes datasets must expose a nonempty attributes schema")
        self.attributes = _attribute_choices(schema if self.task.attributes is None else self.task.attributes)
        missing = self.attributes.keys() - schema.keys()
        if missing:
            raise ValueError(f"Unsupported dataset attribute groups: {sorted(missing)}")
        for group, choices in self.attributes.items():
            vocabulary = _choices(schema[group])
            if set(choices) != set(vocabulary):
                raise ValueError(f"Attribute choices for {group} must match the complete dataset vocabulary")

    def init_metrics(self) -> None:
        self.metrics = AttributesMetrics(self.attributes)
        self.metrics.dataset_metadata = dict(getattr(self.dataset, "metadata", {}))

    def predict_sample(self, sample: dict):
        return self.task((sample["img"], self.attributes))

    def update_metrics(self, prediction, sample: dict) -> None:
        if prediction is not None and not isinstance(prediction, AttributePredictionResult):
            raise TypeError("AttributesValidator requires AttributePredictionResult outputs")
        self.metrics.update(
            sample["attributes"], None if prediction is None else prediction.attributes,
            sample.get("image_id", str(sample["im_file"])),
        )

    def finalize_metrics(self) -> None:
        super().finalize_metrics()
        if self.prepared_dataset is not None:
            self.metrics.dataset_counts = dict(self.prepared_dataset.metadata["counts"])
        self.metrics.dataset_counts.update(
            evaluated_objects=self.metrics.seen,
            evaluated_attribute_decisions=self.metrics.attribute_decisions,
            evaluated_images=self.metrics.unique_images,
        )


class ClassificationValidator(BaseValidator):
    """Validate a single-label Classify task against class directories."""

    default_dataset = "cifar10"

    def build_dataset(self):
        prepared = self.prepare_named_dataset("classify")
        root = self.data if prepared is None else prepared.data
        names = {class_id: path.name for class_id, path in _classification_classes(root, self.split).items()}
        class_map = self._resolve_class_map(names)
        return ClassificationDataset(root, split=self.split, class_ids=class_map)

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

    default_dataset = "coco128"

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
        prepared = self.prepare_named_dataset("detect")
        return YOLODataset(self.data if prepared is None else prepared.data, split=self.split)

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


class GroundingValidator(BaseValidator):
    """Evaluate one Ground prediction per RefCOCO referring expression."""

    default_dataset = "refcocog"

    def __init__(self, *args, images=None, dataset: str = "refcocog", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self._requested_class_map is not None:
            raise ValueError("class_map is not supported for grounding validation")
        automatic = isinstance(self.data, str) and self.data in DATASETS
        if automatic and (images is not None or dataset != "refcocog"):
            raise ValueError("Automatic grounding preparation supports refcocog; use local data and images for other sources")
        if self._provided_dataloader is None and images is None and not automatic:
            raise ValueError("Grounding validation requires images or a supplied dataloader")
        self.images = images
        self.dataset_name = dataset

    def build_dataset(self):
        prepared = self.prepare_named_dataset("ground")
        return RefCOCODataset(
            self.data if prepared is None else prepared.data,
            self.images if prepared is None else prepared.images,
            dataset=self.dataset_name, split=self.split,
        )

    def _align_classes(self) -> None:
        # Ground has no task vocabulary: categories are reporting groups only.
        self.class_map = dict(self.dataset.names)
        self.names = tuple(self.class_map.values())

    def init_metrics(self) -> None:
        self.metrics = GroundingMetrics(self.names, geometry=self.task.geometry)
        self.metrics.dataset_metadata = dict(getattr(self.dataset, "metadata", {}))

    def predict_sample(self, sample: dict):
        if np.asarray(sample["cls"]).shape != (1,) or np.asarray(sample["bboxes"]).shape != (1, 4):
            raise ValueError("Grounding validation requires exactly one category and target box")
        if sample["cls"][0] not in self.class_map:
            raise ValueError("Unknown grounding category ID")
        return self.task((sample["img"], sample["target"]))

    def update_metrics(self, prediction, sample: dict) -> None:
        if prediction is not None:
            if not isinstance(prediction, GroundingResult):
                raise TypeError("GroundingValidator requires GroundingResult outputs")
            if prediction.image_size != sample["img"].size:
                raise ValueError("Grounding prediction dimensions differ from the input image")
        self.metrics.update(
            self.class_map[int(sample["cls"][0])], sample["bboxes"][0],
            prediction, str(sample["im_file"]),
        )
