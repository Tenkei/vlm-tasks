"""Evaluate loaded datasets using task-specific sample schemas."""
from __future__ import annotations

from collections.abc import Mapping
from itertools import islice
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from ..errors import InvalidModelResponseError, ModelExecutionError
from ..types import AttributePredictionResult, ClassificationResult, DetectionResult, GroundingResult
from ._vqa import validate_targets
from ._caption_tokenizer import DEFAULT_TOKENIZER_VERSION
from .data import ValidationDataset, _limit
from .metrics import AnswerMetrics, AttributesMetrics, ClassificationMetrics, DescribeMetrics, DetectionMetrics, GroundingMetrics


def _names(feature, column):
    names = getattr(feature, "names", None)
    if (not isinstance(names, (list, tuple)) or not names
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)):
        raise ValueError(f"{column} requires a ClassLabel feature with a complete, unique vocabulary")
    return tuple(names)


def _check_label(value, names):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value < len(names):
        raise ValueError(f"Unknown or noninteger dataset label ID: {value!r}")


def _check_boxes(boxes):
    if not isinstance(boxes, list):
        raise ValueError("Boxes must be a list of pixel xyxy boxes")
    for box in boxes:
        if (not isinstance(box, list) or len(box) != 4
                or any(type(value) not in (int, float) for value in box)
                or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError("Boxes must contain four finite pixel xyxy coordinates with positive area")


class BaseValidator:
    """Consume a loaded dataset, run predictions, and accumulate fresh metrics.

    Datasets yield sample mappings and expose HF-style ``features``. Validators
    perform no downloading, path discovery, split selection, or construction.
    Override schema checks, prediction, and metric hooks for another task.
    """

    required_columns = ("image",)

    def __init__(self, task, dataset: ValidationDataset, *, max_samples=None, verbose=True, cache_dir=None):
        _limit(max_samples)
        if isinstance(dataset, (str, bytes, Path, Mapping)) or not hasattr(dataset, "__iter__"):
            raise TypeError("dataset must be a loaded dataset yielding sample mappings")
        self.task = task
        self.dataset = dataset
        self.max_samples = max_samples
        self.verbose = verbose
        self.cache_dir = cache_dir
        self.sample_index = 0
        self.features = getattr(dataset, "features", None)
        if self.features is not None:
            if not isinstance(self.features, Mapping):
                raise TypeError("dataset.features must be a feature mapping or None")
            missing = set(self.required_columns) - self.features.keys()
            if missing:
                raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    def __call__(self):
        started = perf_counter()
        self.setup()
        self.init_metrics()
        self.metrics.dataset_counts = dict(getattr(self.dataset, "counts", {}))
        self.metrics.dataset_metadata = {
            **getattr(self.dataset, "metadata", {}), **getattr(self.metrics, "dataset_metadata", {}),
        }
        self._timings = {"load": 0.0, "inference": 0.0, "metrics": 0.0}
        self._started = started
        last = perf_counter()
        for index, sample in enumerate(islice(self.dataset, self.max_samples)):
            self.sample_index = index
            self.validate_sample(sample)
            self._timings["load"] += perf_counter() - last
            inference_started = perf_counter()
            try:
                prediction = self.predict_sample(sample)
            except (ModelExecutionError, InvalidModelResponseError) as exc:
                prediction = None
                self.metrics.failures.append({
                    "im_file": self.sample_path(sample), "type": type(exc).__name__, "message": str(exc),
                    **self.failure_context(sample),
                })
            self._timings["inference"] += perf_counter() - inference_started
            metric_started = perf_counter()
            self.update_metrics(prediction, sample)
            self._timings["metrics"] += perf_counter() - metric_started
            if self.verbose:
                print(f'\rValidated {self.metrics.seen} samples', end="", file=sys.stderr, flush=True)
            last = perf_counter()
        if not self.metrics.seen:
            raise ValueError("No samples to evaluate")
        self.finalize_metrics()
        if self.verbose:
            print("\n" + json.dumps(self.metrics.results_dict, indent=2), file=sys.stderr)
        return self.metrics

    def setup(self):
        """Preflight task-specific runtime requirements before inference."""

    def validate_sample(self, sample):
        """Require the documented schema; source adaptation belongs in datasets."""
        if not isinstance(sample, Mapping):
            raise TypeError("Dataset samples must be mappings")
        missing = set(self.required_columns) - sample.keys()
        if missing:
            raise ValueError(f"Dataset sample is missing required columns: {sorted(missing)}")
        image = sample["image"]
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3 or not image.shape[0] or not image.shape[1]):
            raise ValueError("Dataset image must be a nonempty HWC uint8 RGB array; format it in the dataset")

    def sample_path(self, sample):
        """Return the explicit image path or a stable reporting identifier."""
        identifier = sample.get("image_id", sample.get("sample_id", self.sample_index))
        return str(sample.get("im_file") or f"sample:{identifier}")

    def failure_context(self, sample):
        identifier = sample.get("sample_id", sample.get("image_id", self.sample_index))
        return {"sample_id": identifier}

    def init_metrics(self):
        raise NotImplementedError

    def predict_sample(self, sample):
        return self.task(sample["image"])

    def update_metrics(self, prediction, sample):
        raise NotImplementedError

    def finalize_metrics(self):
        self.metrics.speed = {name: elapsed * 1000 / self.metrics.seen for name, elapsed in self._timings.items()}
        self.metrics.elapsed_seconds = perf_counter() - self._started


class DescribeValidator(BaseValidator):
    """Score image/captions rows, preserving all reference captions."""

    required_columns = ("image", "captions")

    def __init__(self, *args, tokenizer_version=DEFAULT_TOKENIZER_VERSION, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer_version = tokenizer_version

    def init_metrics(self):
        self.metrics = DescribeMetrics(tokenizer_version=self.tokenizer_version, cache_dir=self.cache_dir).prepare()

    def validate_sample(self, sample):
        super().validate_sample(sample)
        self.metrics._validate_sample(
            sample["captions"], sample.get("image_id", self.sample_index), self.sample_path(sample),
        )

    def update_metrics(self, prediction, sample):
        self.metrics.update(
            sample["captions"], prediction,
            image_id=sample.get("image_id", self.sample_index), im_file=self.sample_path(sample),
        )

    def finalize_metrics(self):
        started = perf_counter()
        self.metrics.finalize()
        self._timings["metrics"] += perf_counter() - started
        self.metrics.dataset_counts.update(evaluated_samples=self.metrics.seen, evaluated_images=self.metrics.unique_images,
                                           evaluated_captions=self.metrics.reference_captions)
        super().finalize_metrics()


class AnswerValidator(BaseValidator):
    """Score image/question/answers rows using VQA v2 consensus."""

    required_columns = ("image", "question", "answers", "answer_type", "question_type")

    def init_metrics(self):
        self.metrics = AnswerMetrics()

    def validate_sample(self, sample):
        super().validate_sample(sample)
        if not isinstance(sample["question"], str) or not sample["question"].strip():
            raise ValueError("Dataset question must be a nonempty string")
        if not isinstance(sample["answers"], list):
            raise ValueError("Dataset answers must be a list of ten strings")
        validate_targets(sample["answers"], sample["answer_type"], sample["question_type"])

    def predict_sample(self, sample):
        return self.task((sample["image"], sample["question"]))

    def update_metrics(self, prediction, sample):
        self.metrics.update(
            sample["answers"], prediction, answer_type=sample["answer_type"],
            question_type=sample["question_type"], im_file=self.sample_path(sample),
        )

    def finalize_metrics(self):
        self.metrics.dataset_counts.update(evaluated_questions=self.metrics.seen, evaluated_images=self.metrics.unique_images)
        super().finalize_metrics()


class AttributesValidator(BaseValidator):
    """Score cropped image/attributes rows with per-group ClassLabel features."""

    required_columns = ("image", "attributes")

    def init_metrics(self):
        from ..ops import _attribute_choices
        feature = (self.features or {}).get("attributes")
        if not isinstance(feature, Mapping) or not feature:
            raise ValueError("Attributes datasets require per-group ClassLabel features")
        self.vocabularies = {group: _names(value, f'attributes.{group}') for group, value in feature.items()}
        descriptions = getattr(self.dataset, "metadata", {}).get("attribute_descriptions", {})
        schema = {group: {name: descriptions.get(group, {}).get(name, name) for name in names}
                  for group, names in self.vocabularies.items()}
        self.attributes = _attribute_choices(schema if self.task.attributes is None else self.task.attributes)
        missing = self.attributes.keys() - schema.keys()
        if missing:
            raise ValueError(f'Unsupported dataset attribute groups: {sorted(missing)}')
        for group, choices in self.attributes.items():
            if set(choices) != set(schema[group]):
                raise ValueError(f'Attribute choices for {group} must match the complete dataset vocabulary')
        self.metrics = AttributesMetrics(self.attributes)

    def validate_sample(self, sample):
        super().validate_sample(sample)
        values = sample["attributes"]
        if not isinstance(values, Mapping) or values.keys() - self.vocabularies.keys():
            raise ValueError("Dataset attributes must map known groups to label IDs or null")
        known = False
        for group, value in values.items():
            if value is not None:
                _check_label(value, self.vocabularies[group])
                known |= group in self.attributes
        if not known:
            raise ValueError("Attribute samples require at least one known target in the selected groups")

    def predict_sample(self, sample):
        return self.task((sample["image"], self.attributes))

    def update_metrics(self, prediction, sample):
        if prediction is not None and not isinstance(prediction, AttributePredictionResult):
            raise TypeError("AttributesValidator requires AttributePredictionResult outputs")
        targets = {
            group: self.vocabularies[group][value]
            for group, value in sample["attributes"].items()
            if value is not None and group in self.attributes
        }
        self.metrics.update(
            targets, None if prediction is None else prediction.attributes,
            sample.get("image_id", self.sample_path(sample)),
        )

    def finalize_metrics(self):
        self.metrics.dataset_counts.update(evaluated_objects=self.metrics.seen,
            evaluated_attribute_decisions=self.metrics.attribute_decisions, evaluated_images=self.metrics.unique_images)
        super().finalize_metrics()


class ClassificationValidator(BaseValidator):
    """Score image/label rows using the label feature's complete vocabulary."""

    required_columns = ("image", "label")

    def init_metrics(self):
        self.names = _names((self.features or {}).get("label"), "label")
        self.metrics = ClassificationMetrics(self.names)

    def validate_sample(self, sample):
        super().validate_sample(sample)
        _check_label(sample["label"], self.names)

    def predict_sample(self, sample):
        return self.task((sample["image"], self.names))

    def update_metrics(self, prediction, sample):
        if prediction is not None and (not isinstance(prediction, ClassificationResult) or len(prediction.labels) != 1):
            raise TypeError("ClassificationValidator requires one-label ClassificationResult outputs")
        self.metrics.update(self.names[sample["label"]], None if prediction is None else prediction.labels[0])


class DetectionValidator(BaseValidator):
    """Score image/objects rows with pixel xyxy boxes and category ClassLabels."""

    required_columns = ("image", "objects")

    def __init__(self, *args, match_iou=0.5, **kwargs):
        if (isinstance(match_iou, bool) or not isinstance(match_iou, (int, float))
                or not np.isfinite(match_iou) or not 0 < match_iou <= 1):
            raise ValueError("match_iou must be finite and in (0, 1]")
        super().__init__(*args, **kwargs)
        self.match_iou = float(match_iou)

    def init_metrics(self):
        feature = (self.features or {}).get("objects")
        category = feature.get("category") if isinstance(feature, Mapping) else None
        self.names = _names(getattr(category, "feature", None), "objects.category")
        self.metrics = DetectionMetrics(self.names, match_iou=self.match_iou)

    def validate_sample(self, sample):
        super().validate_sample(sample)
        objects = sample["objects"]
        if not isinstance(objects, Mapping) or not {"bbox", "category"} <= objects.keys():
            raise ValueError("Dataset objects must be a mapping with bbox and category lists")
        if not isinstance(objects["category"], list):
            raise ValueError("Dataset objects.category must be a list of label IDs")
        _check_boxes(objects["bbox"])
        if len(objects["category"]) != len(objects["bbox"]):
            raise ValueError("Dataset objects must have matching box and category counts")
        for category in objects["category"]:
            _check_label(category, self.names)

    def predict_sample(self, sample):
        return self.task((sample["image"], self.names))

    def update_metrics(self, prediction, sample):
        if prediction is not None and not isinstance(prediction, DetectionResult):
            raise TypeError("DetectionValidator requires DetectionResult outputs")
        target_labels = [self.names[category] for category in sample["objects"]["category"]]
        detections = [] if prediction is None else prediction.detections
        boxes = [[item.box.x1, item.box.y1, item.box.x2, item.box.y2] for item in detections]
        self.metrics.update(target_labels, sample["objects"]["bbox"], [item.label for item in detections], boxes)


class GroundingValidator(BaseValidator):
    """Score image/text/bbox/label rows, one referring expression per sample."""

    required_columns = ("image", "text", "bbox", "label")

    def init_metrics(self):
        self.names = _names((self.features or {}).get("label"), "label")
        self.metrics = GroundingMetrics(self.names, geometry=self.task.geometry)

    def validate_sample(self, sample):
        super().validate_sample(sample)
        if not isinstance(sample["text"], str) or not sample["text"].strip():
            raise ValueError("Dataset referring text must be a nonempty string")
        _check_label(sample["label"], self.names)
        _check_boxes([sample["bbox"]])

    def failure_context(self, sample):
        return {**super().failure_context(sample), "target": sample["text"]}

    def predict_sample(self, sample):
        prediction = self.task((sample["image"], sample["text"]))
        if isinstance(prediction, GroundingResult) and prediction.image_size != (sample["image"].shape[1], sample["image"].shape[0]):
            raise ValueError("Grounding prediction dimensions differ from the input image")
        return prediction

    def update_metrics(self, prediction, sample):
        if prediction is not None and not isinstance(prediction, GroundingResult):
            raise TypeError("GroundingValidator requires GroundingResult outputs")
        self.metrics.update(
            self.names[sample["label"]], sample["bbox"],
            prediction, self.sample_path(sample),
        )
