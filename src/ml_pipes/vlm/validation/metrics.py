"""Metrics for task outputs without confidence scores."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np

from ..types import AnswerResult, DescriptionResult, GroundingResult
from ._caption import SCORER_METADATA as CAPTION_SCORER_METADATA, score_captions, validate_captions
from ._caption_tokenizer import CaptionTokenizer, DEFAULT_TOKENIZER_VERSION
from ._vqa import ANSWER_TYPES, SCORER_METADATA, score_answer, validate_targets


def _ratios(tp: int, fp: int, fn: int) -> dict[str, float]:
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


class _RunMetrics:
    def __init__(self, names: Sequence[str] | None = None, *, unlabeled: bool = False) -> None:
        self.names = () if names is None else tuple(names)
        if (
            (not self.names and not (unlabeled and names is None))
            or any(not isinstance(name, str) or not name.strip() for name in self.names)
            or len(set(self.names)) != len(self.names)
        ):
            raise ValueError("Metric names must be nonempty and unique")
        self.seen = 0
        self.failures: list[dict[str, object]] = []
        self.dataset_counts: dict[str, int] = {}
        self.speed: dict[str, float] = {}
        self.elapsed_seconds = 0.0

    @property
    def failure_rate(self) -> float:
        return len(self.failures) / self.seen if self.seen else 0.0

    def _run_results(self) -> dict[str, int | float]:
        return {
            "samples": self.seen,
            "failures": len(self.failures),
            "failure_rate": self.failure_rate,
        }


class DescribeMetrics(_RunMetrics):
    """COCO corpus caption scores; tokenization and scoring run in finalize().

    Native scales are retained (CIDEr has a factor of ten). The IDF corpus
    includes every evaluated image's references, including failed calls.
    """

    def __init__(self, *, tokenizer_version=DEFAULT_TOKENIZER_VERSION, cache_dir=None):
        super().__init__(unlabeled=True)
        self._tokenizer = CaptionTokenizer(tokenizer_version, cache_dir=cache_dir)
        self._candidates: list[str] = []
        self._references: list[tuple[str, ...]] = []
        self._images: set[int] = set()
        self._image_files: set[Path] = set()
        self._scores = None
        self.dataset_metadata = dict(CAPTION_SCORER_METADATA)

    def prepare(self):
        """Check Java and the pinned tokenizer artifact before model execution."""
        self._tokenizer.prepare()
        self.dataset_metadata.update(self._tokenizer.metadata)
        return self

    def _validate_sample(self, captions, image_id, im_file):
        captions = validate_captions(captions)
        if type(image_id) is not int or image_id < 0:
            raise ValueError("Caption image_id must be a nonnegative integer")
        if image_id in self._images:
            raise ValueError(f"Duplicate caption image_id: {image_id}")
        if not isinstance(im_file, (str, Path)) or not str(im_file).strip():
            raise ValueError("Caption im_file must be a nonempty image path")
        image_file = Path(im_file).resolve()
        if image_file in self._image_files:
            raise ValueError(f"Duplicate caption image path: {im_file}")
        return captions, image_file

    def update(self, captions, prediction: DescriptionResult | None, *, image_id: int, im_file: str):
        captions, image_file = self._validate_sample(captions, image_id, im_file)
        if prediction is not None:
            if not isinstance(prediction, DescriptionResult):
                raise TypeError("DescribeMetrics requires DescriptionResult outputs")
            if not isinstance(prediction.description, str) or not prediction.description.strip():
                raise ValueError("Description predictions require a nonempty string")
        self._candidates.append("" if prediction is None else prediction.description)
        self._references.append(captions)
        self._images.add(image_id)
        self._image_files.add(image_file)
        self.seen += 1
        self._scores = None

    def finalize(self):
        """Tokenize the selected corpus once and cache all six metric scores."""
        if self._scores is not None:
            return self
        if not self.seen:
            raise ValueError("No samples to evaluate")
        self.prepare()
        texts = self._tokenizer.tokenize(self._candidates + [text for refs in self._references for text in refs])
        offset = self.seen
        references = []
        for refs in self._references:
            references.append(texts[offset:offset + len(refs)])
            offset += len(refs)
        self._scores = score_captions(texts[:self.seen], references)
        self.dataset_metadata.update(
            idf_reference_images=self.unique_images,
            degenerate_idf=self.unique_images == 1,
            idf_diagnostic="One reference image gives zero IDF and CIDEr=0" if self.unique_images == 1 else None,
        )
        return self

    @property
    def unique_images(self):
        return len(self._images)

    @property
    def reference_captions(self):
        return sum(len(refs) for refs in self._references)

    def _score(self, name):
        self.finalize()
        return self._scores[name]

    cider = property(lambda self: self._score("cider"))
    bleu_1 = property(lambda self: self._score("bleu_1"))
    bleu_2 = property(lambda self: self._score("bleu_2"))
    bleu_3 = property(lambda self: self._score("bleu_3"))
    bleu_4 = property(lambda self: self._score("bleu_4"))
    rouge_l = property(lambda self: self._score("rouge_l"))

    @property
    def results_dict(self):
        self.finalize()
        return {
            **self._run_results(), "unique_images": self.unique_images,
            "reference_captions": self.reference_captions, **self._scores,
        }

    def summary(self):
        self.finalize()
        return [{"support": self.seen, "unique_images": self.unique_images,
                 "reference_captions": self.reference_captions, **self._scores}]


class AnswerMetrics(_RunMetrics):
    """VQA v2 consensus accuracy over questions, including failures/abstentions.

    Scores are fractions, not percentages. ``summary()`` reports answer types
    (including unsupported types with ``None`` scores), then observed question
    types in sorted order. References and predictions are not retained.
    """

    def __init__(self) -> None:
        super().__init__(ANSWER_TYPES)
        self._score_sum = 0.0
        self._groups = {
            "answer_type": {name: [0, 0.0] for name in ANSWER_TYPES},
            "question_type": {},
        }
        self._images: set[Path] = set()
        self.status_counts = dict.fromkeys(("answered", "unanswerable"), 0)
        self.dataset_metadata: dict[str, object] = dict(SCORER_METADATA)

    def update(
        self, answers: Sequence[str], prediction: AnswerResult | None, *,
        answer_type: str, question_type: str, im_file: str,
    ) -> None:
        validate_targets(answers, answer_type, question_type)
        score = 0.0
        if prediction is not None:
            if not isinstance(prediction, AnswerResult):
                raise TypeError("AnswerMetrics requires AnswerResult outputs")
            if prediction.status == "answered":
                if not isinstance(prediction.answer, str) or not prediction.answer.strip():
                    raise ValueError("Answered predictions require a nonempty string answer")
                score = score_answer(prediction.answer, answers)
            elif prediction.status != "unanswerable" or prediction.answer is not None:
                raise ValueError("Invalid answer status/answer combination")
            self.status_counts[prediction.status] += 1
        self.seen += 1
        self._score_sum += score
        self._images.add(Path(im_file).resolve())
        for group_by, group in (("answer_type", answer_type), ("question_type", question_type)):
            totals = self._groups[group_by].setdefault(group, [0, 0.0])
            totals[0] += 1
            totals[1] += score

    @property
    def vqa_accuracy(self) -> float:
        return self._score_sum / self.seen if self.seen else 0.0

    @property
    def unique_images(self) -> int:
        return len(self._images)

    @property
    def results_dict(self) -> dict[str, int | float]:
        return {
            **self._run_results(), "vqa_accuracy": self.vqa_accuracy,
            "unique_images": self.unique_images,
        }

    def summary(self) -> list[dict]:
        rows = []
        for group_by, groups in self._groups.items():
            names = ANSWER_TYPES if group_by == "answer_type" else sorted(groups)
            for name in names:
                support, score_sum = groups[name]
                rows.append({
                    "group_by": group_by, "group": name, "support": support,
                    "vqa_accuracy": score_sum / support if support else None,
                })
        return rows


class AttributesMetrics(_RunMetrics):
    """Categorical scoring masked to known targets, with one matrix per group.

    Matrix rows and columns follow ``attributes[group]``; the final column is
    reserved for failed calls. Predictions on unscored fields add no support.
    ``summary()`` reports all choices, including those absent from the run.
    """

    def __init__(self, attributes: Mapping[str, Sequence[str] | Mapping[str, str]]) -> None:
        if not isinstance(attributes, Mapping):
            raise ValueError("Metric attributes must be a nonempty mapping")
        super().__init__(tuple(attributes))
        self.attributes = {}
        for group, choices in attributes.items():
            if isinstance(choices, (str, bytes)) or not isinstance(choices, (Mapping, Sequence)):
                raise ValueError("Attribute choices must be a sequence or mapping")
            self.attributes[group] = _RunMetrics(tuple(choices)).names
        self._ids = {group: {label: i for i, label in enumerate(choices)} for group, choices in self.attributes.items()}
        self.confusion_matrices = {
            group: np.zeros((len(choices), len(choices) + 1), dtype=np.int64)
            for group, choices in self.attributes.items()
        }
        self._images = set()

    def update(self, targets: Mapping[str, str], prediction: Mapping[str, str] | None, image_id) -> None:
        if not isinstance(targets, Mapping) or not targets or set(targets) - self.attributes.keys():
            raise ValueError("Attribute targets must be a nonempty subset of the configured groups")
        for group, target in targets.items():
            if not isinstance(target, str) or target not in self._ids[group]:
                raise ValueError(f"Unknown attribute target for {group}: {target!r}")
        if prediction is not None:
            if not isinstance(prediction, Mapping) or set(prediction) != self.attributes.keys():
                raise ValueError("Attribute predictions must contain every configured group exactly once")
            for group, value in prediction.items():
                if not isinstance(value, str) or value not in self._ids[group]:
                    raise ValueError(f"Unsupported attribute prediction for {group}: {value!r}")
        for group, target in targets.items():
            column = len(self._ids[group]) if prediction is None else self._ids[group][prediction[group]]
            self.confusion_matrices[group][self._ids[group][target], column] += 1
        self.seen += 1
        self._images.add(image_id)

    @property
    def attribute_decisions(self) -> int:
        return sum(int(matrix.sum()) for matrix in self.confusion_matrices.values())

    @property
    def unique_images(self) -> int:
        return len(self._images)

    @property
    def accuracy(self) -> float | None:
        correct = sum(int(np.trace(matrix[:, :-1])) for matrix in self.confusion_matrices.values())
        return correct / self.attribute_decisions if self.attribute_decisions else None

    def summary(self) -> list[dict]:
        result = []
        for group, choices in self.attributes.items():
            matrix = self.confusion_matrices[group]
            support = int(matrix.sum())
            values, active_f1 = [], []
            for index, label in enumerate(choices):
                tp = int(matrix[index, index])
                target_support = int(matrix[index].sum())
                prediction_support = int(matrix[:, index].sum())
                scores = _ratios(tp, prediction_support - tp, target_support - tp)
                if target_support or prediction_support:
                    active_f1.append(scores["f1"])
                values.append({
                    "value": label, "support": target_support, "prediction_support": prediction_support,
                    **(scores if support else dict.fromkeys(scores)),
                })
            result.append({
                "attribute": group, "support": support,
                "accuracy": int(np.trace(matrix[:, :-1])) / support if support else None,
                "macro_f1": sum(active_f1) / len(active_f1) if active_f1 else None,
                "values": values,
            })
        return result

    @property
    def mean_attribute_accuracy(self) -> float | None:
        scores = [group["accuracy"] for group in self.summary() if group["support"]]
        return sum(scores) / len(scores) if scores else None

    @property
    def macro_f1(self) -> float | None:
        scores = [group["macro_f1"] for group in self.summary() if group["support"]]
        return sum(scores) / len(scores) if scores else None

    @property
    def results_dict(self) -> dict[str, int | float | None]:
        return {
            **self._run_results(), "attribute_decisions": self.attribute_decisions,
            "unique_images": self.unique_images, "accuracy": self.accuracy,
            "mean_attribute_accuracy": self.mean_attribute_accuracy, "macro_f1": self.macro_f1,
        }


class ClassificationMetrics(_RunMetrics):
    """Top-1 and per-class metrics, including failed task executions.

    Confusion matrix rows and columns follow the complete dataset vocabulary
    in ``names``, with one additional column for failed executions.
    """

    def __init__(self, names: Sequence[str]) -> None:
        super().__init__(names)
        self._ids = {name: i for i, name in enumerate(self.names)}
        self.confusion_matrix = np.zeros((len(self.names), len(self.names) + 1), dtype=np.int64)

    def update(self, target: str, prediction: str | None) -> None:
        row = self._ids[target]
        column = len(self.names) if prediction is None else self._ids[prediction]
        self.confusion_matrix[row, column] += 1
        self.seen += 1

    @property
    def top1(self) -> float:
        correct = int(np.trace(self.confusion_matrix))
        return correct / self.seen if self.seen else 0.0

    @property
    def results_dict(self) -> dict[str, int | float]:
        return {**self._run_results(), "accuracy_top1": self.top1}

    def summary(self) -> list[dict[str, str | int | float]]:
        result = []
        for i, name in enumerate(self.names):
            tp = int(self.confusion_matrix[i, i])
            support = int(self.confusion_matrix[i].sum())
            fp = int(self.confusion_matrix[:, i].sum()) - tp
            result.append({"class": name, "support": support, **_ratios(tp, fp, support - tp)})
        return result


def _box_iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    intersection = np.maximum(
        np.minimum(first[:, None, 2:], second[None, :, 2:])
        - np.maximum(first[:, None, :2], second[None, :, :2]),
        0,
    ).prod(axis=2)
    first_area = (first[:, 2:] - first[:, :2]).prod(axis=1)
    second_area = (second[:, 2:] - second[:, :2]).prod(axis=1)
    union = first_area[:, None] + second_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


class DetectionMetrics(_RunMetrics):
    """Class-aware one-to-one box matching at one IoU threshold.

    ``tp``, ``fp``, and ``fn`` follow ``names``. Aggregate ratios pool these
    counts across classes. No confidence ranking or AP is computed.
    """

    def __init__(self, names: Sequence[str], match_iou: float = 0.5) -> None:
        super().__init__(names)
        if (
            isinstance(match_iou, bool) or not isinstance(match_iou, (int, float))
            or not np.isfinite(match_iou) or not 0 < match_iou <= 1
        ):
            raise ValueError("match_iou must be finite and in (0, 1]")
        self.match_iou = float(match_iou)
        self.tp = np.zeros(len(self.names), dtype=np.int64)
        self.fp = np.zeros(len(self.names), dtype=np.int64)
        self.fn = np.zeros(len(self.names), dtype=np.int64)

    def update(
        self,
        target_labels: Sequence[str],
        target_boxes: np.ndarray,
        prediction_labels: Sequence[str],
        prediction_boxes: np.ndarray,
    ) -> None:
        targets = np.asarray(target_labels)
        predictions = np.asarray(prediction_labels)
        target_boxes = np.asarray(target_boxes, dtype=np.float64).reshape(-1, 4)
        prediction_boxes = np.asarray(prediction_boxes, dtype=np.float64).reshape(-1, 4)
        if len(targets) != len(target_boxes) or len(predictions) != len(prediction_boxes):
            raise ValueError("Each label must have exactly one box")
        if not set(target_labels).union(prediction_labels).issubset(self.names):
            raise ValueError("Detection labels must belong to the evaluated classes")
        iou = _box_iou(target_boxes, prediction_boxes)
        target_indices, prediction_indices = np.nonzero(
            (iou >= self.match_iou) & (targets[:, None] == predictions[None, :])
        )
        order = np.lexsort((prediction_indices, target_indices, -iou[target_indices, prediction_indices]))
        matched_targets: set[int] = set()
        matched_predictions: set[int] = set()
        for index in order:
            target, prediction = int(target_indices[index]), int(prediction_indices[index])
            if target not in matched_targets and prediction not in matched_predictions:
                matched_targets.add(target)
                matched_predictions.add(prediction)
        for i, name in enumerate(self.names):
            true_positives = sum(targets[index] == name for index in matched_targets)
            self.tp[i] += true_positives
            self.fp[i] += int(np.count_nonzero(predictions == name)) - true_positives
            self.fn[i] += int(np.count_nonzero(targets == name)) - true_positives
        self.seen += 1

    @property
    def precision(self) -> float:
        return self._aggregate()["precision"]

    @property
    def recall(self) -> float:
        return self._aggregate()["recall"]

    @property
    def f1(self) -> float:
        return self._aggregate()["f1"]

    def _aggregate(self) -> dict[str, float]:
        return _ratios(int(self.tp.sum()), int(self.fp.sum()), int(self.fn.sum()))

    @property
    def results_dict(self) -> dict[str, int | float]:
        return {
            **self._run_results(), "match_iou": self.match_iou,
            "tp": int(self.tp.sum()), "fp": int(self.fp.sum()), "fn": int(self.fn.sum()),
            **self._aggregate(),
        }

    def summary(self) -> list[dict[str, str | int | float]]:
        result = []
        for name, tp, fp, fn in zip(self.names, self.tp.tolist(), self.fp.tolist(), self.fn.tolist()):
            result.append({
                "class": name, "support": tp + fn, "tp": tp, "fp": fp, "fn": fn,
                **_ratios(tp, fp, fn),
            })
        return result


class GroundingMetrics(_RunMetrics):
    """Box IoU or point-in-box accuracy, counting every evaluated expression.

    Point containment includes the target box edges. It is a coarse proxy for
    target-mask containment and does not measure the object's extent.
    """

    thresholds = (0.5, 0.75, 0.9)

    def __init__(self, names: Sequence[str], *, geometry: Literal["box", "point"] = "box") -> None:
        super().__init__(names)
        if geometry not in ("box", "point"):
            raise ValueError("geometry must be 'box' or 'point'")
        self.geometry = geometry
        self._ids = {name: i for i, name in enumerate(self.names)}
        self.support = np.zeros(len(self.names), dtype=np.int64)
        self._iou_sums = np.zeros(len(self.names), dtype=np.float64)
        self._hits = np.zeros((len(self.names), len(self.thresholds)), dtype=np.int64)
        self._point_hits = np.zeros(len(self.names), dtype=np.int64)
        self._images: set[Path] = set()
        self.status_counts = dict.fromkeys(("found", "not_found", "ambiguous"), 0)
        self.dataset_metadata: dict[str, object] = {}

    def update(
        self, target: str, target_box, prediction: GroundingResult | None, im_file: str,
    ) -> None:
        index = self._ids[target]
        box = np.asarray(target_box, dtype=np.float64)
        if (
            box.shape != (4,) or not np.isfinite(box).all()
            or (box[2:] <= box[:2]).any()
        ):
            raise ValueError("Grounding metrics require one finite positive-area target box")
        iou = 0.0
        point_hit = False
        if prediction is not None:
            if not isinstance(prediction, GroundingResult):
                raise TypeError("GroundingMetrics requires GroundingResult outputs")
            if prediction.geometry != self.geometry:
                raise ValueError("Grounding prediction geometry differs from metric geometry")
            location = prediction.box if self.geometry == "box" else prediction.point
            other = prediction.point if self.geometry == "box" else prediction.box
            if prediction.status not in self.status_counts or other is not None or (
                (prediction.status == "found") != (location is not None)
            ):
                raise ValueError("Invalid grounding status/geometry combination")
            if self.geometry == "box" and location is not None:
                predicted = np.array([
                    prediction.box.x1, prediction.box.y1, prediction.box.x2, prediction.box.y2,
                ], dtype=np.float64)
                if not np.isfinite(predicted).all() or (predicted[2:] <= predicted[:2]).any():
                    raise ValueError("Grounding predictions require finite positive-area boxes")
                iou = float(_box_iou(box[None, :], predicted[None, :])[0, 0])
            elif self.geometry == "point" and location is not None:
                point = np.array([location.x, location.y], dtype=np.float64)
                size = np.asarray(prediction.image_size, dtype=np.float64)
                if (
                    not np.isfinite(point).all() or size.shape != (2,)
                    or not np.isfinite(size).all() or (size <= 0).any()
                    or (point < 0).any() or (point > size).any()
                ):
                    raise ValueError("Grounding predictions require finite points inside the image")
                point_hit = bool((point >= box[:2]).all() and (point <= box[2:]).all())
            self.status_counts[prediction.status] += 1
        self.support[index] += 1
        self._iou_sums[index] += iou
        self._hits[index] += iou >= np.asarray(self.thresholds)
        self._point_hits[index] += point_hit
        self._images.add(Path(im_file).expanduser().resolve())
        self.seen += 1

    @property
    def unique_images(self) -> int:
        return len(self._images)

    @property
    def mean_iou(self) -> float:
        if self.geometry != "box":
            raise ValueError("mean_iou applies only to box grounding")
        return float(self._iou_sums.sum()) / self.seen if self.seen else 0.0

    def _scores(self, hits, iou_sum: float, support: int, point_hits: int = 0) -> dict[str, float]:
        if self.geometry == "point":
            return {"point_in_box_accuracy": int(point_hits) / support if support else 0.0}
        return {
            **{
                f"accuracy_iou_{threshold}": int(hit) / support if support else 0.0
                for threshold, hit in zip(self.thresholds, hits)
            },
            "mean_iou": float(iou_sum) / support if support else 0.0,
        }

    @property
    def results_dict(self) -> dict[str, int | float]:
        return {
            **self._run_results(), "unique_images": self.unique_images,
            **self._scores(self._hits.sum(axis=0), self._iou_sums.sum(), self.seen, self._point_hits.sum()),
        }

    def summary(self) -> list[dict[str, str | int | float]]:
        return [
            {
                "class": name, "support": int(self.support[i]),
                **self._scores(self._hits[i], self._iou_sums[i], int(self.support[i]), self._point_hits[i]),
            }
            for i, name in enumerate(self.names)
        ]
