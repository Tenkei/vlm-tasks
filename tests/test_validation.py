"""Dataset validation with local images and deterministic task backends."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from ml_pipes.vlm import (
    Classify, Detect, Ground, InvalidModelResponseError,
    ModelExecutionError, ModelResponse,
)
from ml_pipes.vlm.validation import (
    BaseValidator, ClassificationDataset, ClassificationMetrics,
    ClassificationValidator, ClassificationDataLoader, DetectionMetrics, DetectionValidator,
    YOLODataset,
)


def write_image(path: Path, size=(60, 120)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.zeros((*size, 3), dtype=np.uint8))
    return path


def classification_data(root: Path, classes=("cat", "dog"), per_class=1) -> Path:
    for name in classes:
        for index in range(per_class):
            write_image(root / "val" / name / f"{index}.png")
    return root


def detection_data(root: Path, annotations: list[str | None], names=("person", "dog")) -> Path:
    for index, annotation in enumerate(annotations):
        write_image(root / "images" / "val" / f"{index}.png")
        if annotation is not None:
            label = root / "labels" / "val" / f"{index}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text(annotation)
    # JSON is also valid YAML; exercise the reader without a YAML test helper.
    config = root / "data.yaml"
    config.write_text(json.dumps({"path": ".", "val": "images/val", "names": list(names)}))
    return config


class Backend:
    def __init__(self, *responses):
        self.responses = responses
        self.calls = []

    def __call__(self, request):
        response = self.responses[min(len(self.calls), len(self.responses) - 1)]
        self.calls.append(request)
        if isinstance(response, Exception):
            raise response
        return ModelResponse(response)


def box(label, x1=0.25, y1=0.25, x2=0.75, y2=0.75):
    return {"label": label, "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}


def test_classification_aliases_failures_and_out_of_subset_predictions(tmp_path):
    data = classification_data(tmp_path, classes=("cat", "dog", "excluded"), per_class=2)
    backend = Backend(
        {"labels": ["feline"]}, {"labels": ["dog"]},
        ModelExecutionError("unavailable"), {"labels": ["other"]},
    )
    task = Classify(backend, ["feline", "dog", "other"])
    metrics = task.val(
        data=data, class_map={"cat": "feline", "dog": "dog"}, verbose=False,
    )
    assert isinstance(metrics, ClassificationMetrics)
    assert task.metrics is metrics
    assert metrics.results_dict == {"samples": 4, "failures": 1, "failure_rate": 0.25, "accuracy_top1": 0.25}
    np.testing.assert_array_equal(metrics.confusion_matrix, [[1, 1, 0, 0], [0, 0, 1, 1]])
    assert metrics.prediction_names == ("feline", "dog", "other")
    assert metrics.summary()[0] == {"class": "feline", "support": 2, "precision": 1.0, "recall": 0.5, "f1": 2 / 3}
    assert metrics.failures[0]["type"] == "ModelExecutionError"
    assert metrics.failures[0]["message"] == "unavailable"
    assert metrics.failures[0]["im_file"].endswith("dog/0.png")
    assert len(backend.calls) == 4
    assert all(call.image.size == (120, 60) for call in backend.calls)
    assert set(metrics.speed) == {"load", "inference", "metrics"}
    assert all(value >= 0 for value in metrics.speed.values())


def test_detection_counts_duplicates_wrong_classes_and_failed_images(tmp_path):
    data = detection_data(tmp_path, [
        "0 0.25 0.25 0.5 0.5\n1 0.75 0.75 0.5 0.5\n",
        "", "0 0.5 0.5 0.5 0.5\n", None,
    ])
    backend = Backend(
        {"detections": [
            box("pedestrian", 0, 0, 0.5, 0.5),
            box("pedestrian", 0, 0, 0.5, 0.5),
            box("pedestrian", 0.5, 0.5, 1, 1),
            box("ignored"),
        ]},
        {"detections": [box("dog")]},
        InvalidModelResponseError("bad response"), ModelExecutionError("unavailable"),
    )
    metrics = Detect(backend, ["pedestrian", "dog", "ignored"]).val(
        data=data, class_map={"person": "pedestrian", "dog": "dog"}, verbose=False,
    )
    assert metrics.results_dict == {
        "samples": 4, "failures": 2, "failure_rate": 0.5, "match_iou": 0.5,
        "tp": 1, "fp": 3, "fn": 2, "precision": 0.25, "recall": 1 / 3, "f1": 2 / 7,
    }
    assert metrics.summary()[0] == {
        "class": "pedestrian", "support": 2, "tp": 1, "fp": 2, "fn": 1,
        "precision": 1 / 3, "recall": 0.5, "f1": 0.4,
    }
    assert metrics.dataset_counts == {"labeled_images": 2, "empty_labels": 1, "missing_labels": 1}
    assert len(backend.calls) == 4


def test_original_detection_coordinates_and_metadata_are_preserved(tmp_path):
    data = detection_data(tmp_path, ["0 0.5 0.5 0.5 0.5\n"])
    dataset = YOLODataset(data)
    sample = dataset[0]
    assert sample["img"].size == (120, 60)
    np.testing.assert_array_equal(sample["bboxes"], [[30, 15, 90, 45]])
    np.testing.assert_array_equal(dataset[0]["bboxes"], sample["bboxes"])
    np.testing.assert_array_equal(dataset.samples[0]["bboxes"], [[0.25, 0.25, 0.75, 0.75]])
    backend = Backend({"detections": [box("person")]})
    metrics = Detect(backend, ["person"]).val(data=data, verbose=False)
    assert (metrics.precision, metrics.recall, metrics.f1) == (1, 1, 1)


def test_detection_keeps_images_with_only_unselected_annotations(tmp_path):
    data = detection_data(tmp_path, ["1 0.5 0.5 0.5 0.5\n"])
    metrics = Detect(Backend({"detections": [box("person")]}), ["person"]).val(data=data, verbose=False)
    assert metrics.results_dict["samples"] == 1
    assert metrics.results_dict["fp"] == 1
    assert metrics.results_dict["fn"] == 0


def test_many_dataset_classes_can_map_to_one_task_label(tmp_path):
    data = classification_data(tmp_path)
    metrics = Classify(Backend({"labels": ["animal"]}), ["animal"]).val(
        data=data, class_map={"cat": "animal", "dog": "animal"}, verbose=False,
    )
    assert metrics.names == ("animal",)
    assert metrics.top1 == 1
    assert metrics.summary()[0]["support"] == 2


@pytest.mark.parametrize("task_kind", ["classify", "detect"])
def test_all_failed_runs_keep_denominators(tmp_path, task_kind):
    backend = Backend(InvalidModelResponseError("invalid"))
    if task_kind == "classify":
        data = classification_data(tmp_path, classes=("person",))
        task = Classify(backend, ["person"])
    else:
        data = detection_data(tmp_path, ["0 0.5 0.5 0.5 0.5\n"])
        task = Detect(backend, ["person"])
    metrics = task.val(data=data, verbose=False)
    assert metrics.seen == 1
    assert metrics.failure_rate == 1
    if task_kind == "classify":
        assert metrics.top1 == 0
        np.testing.assert_array_equal(metrics.confusion_matrix, [[0, 1]])
        assert metrics.summary()[0]["recall"] == 0
    else:
        assert metrics.results_dict["fn"] == 1
        assert (metrics.precision, metrics.recall, metrics.f1) == (0, 0, 0)


def test_sample_limit_and_repeated_validator_runs(tmp_path):
    data = classification_data(tmp_path, per_class=2)
    backend = Backend({"labels": ["cat"]})
    task = Classify(backend, ["cat", "dog"])
    validator = ClassificationValidator(task, data=data, max_samples=1, verbose=False)
    assert validator.task is task
    first = validator()
    second = validator()
    assert first is not second
    assert first.seen == second.seen == 1
    assert first.top1 == second.top1 == 1
    assert len(backend.calls) == 2
    assert len(second.failures) == 0
    third = task.val(data=data, max_samples=1, verbose=False)
    assert third is task.metrics and third is not second


@pytest.mark.parametrize("custom_loader", [False, True])
def test_classification_subset_does_not_decode_excluded_images(tmp_path, custom_loader):
    data = classification_data(tmp_path)
    (data / "val" / "cat" / "0.png").write_bytes(b"corrupt but outside selected classes")
    settings = {"data": data}
    if custom_loader:
        dataset = ClassificationDataset(data)
        dog_ids = [class_id for class_id, name in dataset.names.items() if name == "dog"]
        loader = ClassificationDataLoader(dataset, class_ids=dog_ids)
        assert len(loader) == 1
        settings = {"dataloader": loader}
    metrics = Classify(Backend({"labels": ["dog"]}), ["dog"]).val(**settings, verbose=False)
    assert metrics.seen == 1 and metrics.top1 == 1


def test_custom_validator_class():
    class CustomValidator:
        def __init__(self, *, task, marker):
            assert isinstance(task, Ground)
            self.task = task
            self.metrics = marker

        def __call__(self):
            assert self.task is task

    task = Ground(Backend(), "the person")
    assert task.val(validator=CustomValidator, marker="custom metrics") == "custom metrics"


def test_injected_loader_and_custom_dataset_construction(tmp_path):
    data = classification_data(tmp_path)
    dataset = ClassificationDataset(data)
    dog_ids = [class_id for class_id, name in dataset.names.items() if name == "dog"]
    loader = ClassificationDataLoader(dataset, class_ids=dog_ids)
    task = Classify(Backend({"labels": ["dog"]}), ["dog"])
    metrics = task.val(dataloader=loader, verbose=False)
    assert metrics.seen == 1 and metrics.top1 == 1

    class CustomDatasetValidator(ClassificationValidator):
        def build_dataset(self):
            return loader.dataset

    assert task.val(validator=CustomDatasetValidator, data="unused", verbose=False).top1 == 1


@pytest.mark.parametrize("mapping", [{}, {"missing": "cat"}, {"cat": "unknown"}, [], {"cat": 1}])
def test_invalid_class_alignment_fails_before_execution(tmp_path, mapping):
    data = classification_data(tmp_path)
    backend = Backend({"labels": ["cat"]})
    with pytest.raises(ValueError):
        Classify(backend, ["cat"]).val(data=data, class_map=mapping, verbose=False)
    assert not backend.calls


def test_default_alignment_requires_all_task_labels(tmp_path):
    data = classification_data(tmp_path)
    backend = Backend({"labels": ["cat"]})
    with pytest.raises(ValueError, match="missing dataset names"):
        Classify(backend, ["cat", "other"]).val(data=data, verbose=False)
    assert not backend.calls


@pytest.mark.parametrize("max_samples", [0, -1, True, 1.5])
def test_invalid_sample_limits(max_samples):
    with pytest.raises(ValueError, match="max_samples"):
        BaseValidator(Classify(Backend(), ["cat"]), data="unused", max_samples=max_samples)


@pytest.mark.parametrize("iou", [0, -0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_matching_thresholds(iou):
    with pytest.raises(ValueError, match="match_iou"):
        DetectionValidator(Detect(Backend(), ["person"]), data="unused", match_iou=iou)
    with pytest.raises(ValueError, match="match_iou"):
        DetectionMetrics(["person"], match_iou=iou)


def test_exact_split_and_empty_evaluation_errors(tmp_path):
    data = classification_data(tmp_path)
    with pytest.raises(FileNotFoundError, match="split not found"):
        ClassificationDataset(data, split="test")
    (tmp_path / "val" / "empty").mkdir()
    backend = Backend({"labels": ["empty"]})
    with pytest.raises(ValueError, match="No samples"):
        Classify(backend, ["empty"]).val(data=data, verbose=False)
    assert not backend.calls
    with pytest.raises(ValueError, match="split name"):
        ClassificationDataset(data, split="../val")


@pytest.mark.parametrize("annotation", [
    "0 0.5 0.5 0.2", "1.5 0.5 0.5 0.2 0.2", "9 0.5 0.5 0.2 0.2",
    "0 nan 0.5 0.2 0.2", "0 0.5 0.5 0 0.2", "0 0.9 0.5 0.8 0.2",
    "0 0.5 0.5 -0.2 0.2", "not a box",
    "0 0.9000025 0.5 0.2 0.2",
])
def test_invalid_annotations_are_rejected_before_task_calls(tmp_path, annotation):
    data = detection_data(tmp_path, [annotation])
    backend = Backend({"detections": []})
    with pytest.raises(ValueError, match="Invalid detection annotation"):
        Detect(backend, ["person"]).val(data=data, verbose=False)
    assert not backend.calls


def test_detection_relative_roots_lists_and_split_entries(tmp_path):
    root = tmp_path / "dataset"
    detection_data(root, ["0 0.5 0.5 0.5 0.5", ""])
    (root / "val.txt").write_text("images/val/1.png\nimages/val/0.png\n")
    config = tmp_path / "configs" / "eval.yaml"
    config.parent.mkdir()
    config.write_text("path: ../dataset\nval: [val.txt, images/val]\nnames:\n  0: person\n  1: dog\n")
    dataset = YOLODataset(config)
    assert len(dataset) == 2
    assert dataset.names == {0: "person", 1: "dog"}
    assert dataset[0]["im_file"].endswith("0.png")
    with pytest.raises(ValueError, match="split"):
        YOLODataset(config, split="test")
    (root / "val.txt").write_text("images/val/missing.png\n")
    with pytest.raises(ValueError, match="Invalid image entry"):
        YOLODataset(config)


def test_yolo_annotation_rounding_at_image_edges_is_preserved(tmp_path):
    # COCO128 contains this valid six-decimal annotation at the right edge.
    data = detection_data(tmp_path, ["0 0.662641 0.494385 0.674719 0.988771"])
    dataset = YOLODataset(data)
    box = dataset.samples[0]["bboxes"][0]
    assert box[2] == pytest.approx(1.0000005)
    assert box[1] == pytest.approx(-0.0000005)
    assert dataset[0]["bboxes"][0, 2] > 120


def test_corrupt_images_and_unexpected_exceptions_stop_the_run(tmp_path):
    data = classification_data(tmp_path)
    backend = Backend(RuntimeError("programming error"))
    with pytest.raises(RuntimeError, match="programming error"):
        Classify(backend, ["cat"]).val(data=data, verbose=False)
    (data / "val" / "cat" / "0.png").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Could not read image.*0.png"):
        Classify(backend, ["cat"]).val(data=data, verbose=False)
    (data / "val" / "cat" / "0.png").write_bytes(b"")
    with pytest.raises(ValueError, match="Could not read image.*0.png"):
        Classify(backend, ["cat"]).val(data=data, verbose=False)


@pytest.mark.parametrize("threshold,expected_tp", [(0.5, 1), (0.50001, 0), (1.0, 0)])
def test_detection_matching_iou_boundary(threshold, expected_tp):
    metrics = DetectionMetrics(["person"], match_iou=threshold)
    metrics.update(["person"], [[0, 0, 10, 10]], ["person"], [[0, 0, 5, 10]])
    assert metrics.results_dict["tp"] == expected_tp
    assert metrics.results_dict["fp"] == 1 - expected_tp
    assert metrics.results_dict["fn"] == 1 - expected_tp


def test_detection_matching_uses_largest_iou_and_unique_targets():
    metrics = DetectionMetrics(["person"])
    metrics.update(
        ["person", "person"], [[0, 0, 10, 10], [2, 0, 12, 10]],
        ["person", "person"], [[2, 0, 12, 10], [0, 0, 10, 10]],
    )
    assert metrics.results_dict["tp"] == 2
    assert metrics.results_dict["fp"] == metrics.results_dict["fn"] == 0


def test_all_negative_detection_dataset_returns_defined_metrics(tmp_path):
    data = detection_data(tmp_path, [None, ""])
    metrics = Detect(Backend({"detections": []}), ["person"]).val(data=data, verbose=False)
    assert metrics.seen == 2
    assert (metrics.precision, metrics.recall, metrics.f1) == (0, 0, 0)


def test_verbose_output_stays_on_stderr(tmp_path, capsys):
    data = classification_data(tmp_path, classes=("cat",))
    Classify(Backend({"labels": ["cat"]}), ["cat"]).val(data=data)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Validated 1 samples" in captured.err
    assert '"accuracy_top1": 1.0' in captured.err


def test_validation_runs_without_optional_model_or_yaml_dependencies(tmp_path):
    data = classification_data(tmp_path, classes=("cat",))
    script = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'ultralytics', 'yaml', 'transformers'}:
            raise ImportError('blocked optional dependency: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
from ml_pipes.vlm import Classify, ModelResponse
from ml_pipes.vlm.validation import YOLODataset
task = Classify(lambda request: ModelResponse({'labels': ['cat']}), ['cat'])
assert task.val(data=sys.argv[1], verbose=False).top1 == 1
try:
    YOLODataset('unused.yaml')
except ImportError as exc:
    assert 'ml-pipes-vlm[validation]' in str(exc)
else:
    raise AssertionError('Missing optional YAML dependency was not reported')
assert all(name not in sys.modules for name in ('torch', 'torchvision', 'ultralytics', 'yaml'))
"""
    subprocess.run([sys.executable, "-c", script, str(data)], check=True, capture_output=True, text=True)


def test_validation_example_with_local_backend(tmp_path, monkeypatch, capsys):
    from examples import run_validation

    data = classification_data(tmp_path, classes=("cat",))
    config = tmp_path / "tasks.json"
    config.write_text(json.dumps({"classify": {"labels": ["cat"]}}))
    monkeypatch.setattr(run_validation, "create_backend", lambda args: Backend({"labels": ["cat"]}))
    assert run_validation.main([
        "--task", "classify", "--data", str(data), "--config", str(config),
        "--model", "test", "--no-verbose",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["metrics"]["accuracy_top1"] == 1
    assert report["per_class"][0]["class"] == "cat"


def test_detection_example_and_execution_failure_exit_code(tmp_path, monkeypatch, capsys):
    from examples import run_validation

    data = detection_data(tmp_path, ["0 0.5 0.5 0.5 0.5"])
    config = tmp_path / "tasks.json"
    config.write_text(json.dumps({"detect": {"labels": ["pedestrian"]}}))
    mapping = tmp_path / "mapping.json"
    mapping.write_text('{"person": "pedestrian"}')
    args = [
        "--task", "detect", "--data", str(data), "--config", str(config),
        "--class-map", str(mapping), "--match-iou", "0.75",
        "--model", "test", "--no-verbose",
    ]
    monkeypatch.setattr(run_validation, "create_backend", lambda args: Backend({"detections": [box("pedestrian")]}))
    assert run_validation.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["metrics"]["f1"] == 1
    assert report["metrics"]["match_iou"] == 0.75

    monkeypatch.setattr(run_validation, "create_backend", lambda args: Backend(ModelExecutionError("offline")))
    assert run_validation.main(args) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["metrics"]["failure_rate"] == 1
    assert report["failures"][0]["message"] == "offline"
