"""Dataset validation with local images and deterministic task backends."""
from __future__ import annotations

from ml_pipes.vlm.validation import load_classification, load_yolo

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
from ml_pipes.vlm.validation import BaseValidator, ClassificationMetrics, ClassificationValidator, DetectionMetrics, DetectionValidator


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


def test_classification_uses_dataset_labels_and_keeps_failures(tmp_path):
    data = classification_data(tmp_path, per_class=2)
    backend = Backend(
        {"labels": ["cat"]}, {"labels": ["dog"]},
        ModelExecutionError("unavailable"), {"labels": ["cat"]},
    )
    task = Classify(backend, ["feline", "dog", "other"])
    metrics = task.val(dataset=load_local(data), verbose=False)
    assert isinstance(metrics, ClassificationMetrics)
    assert task.metrics is metrics
    assert metrics.results_dict == {"samples": 4, "failures": 1, "failure_rate": 0.25, "accuracy_top1": 0.25}
    np.testing.assert_array_equal(metrics.confusion_matrix, [[1, 1, 0], [1, 0, 1]])
    assert metrics.names == ("cat", "dog")
    assert metrics.summary()[0] == {"class": "cat", "support": 2, "precision": 0.5, "recall": 0.5, "f1": 0.5}
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
            box("person", 0, 0, 0.5, 0.5),
            box("person", 0, 0, 0.5, 0.5),
            box("person", 0.5, 0.5, 1, 1),
        ]},
        {"detections": [box("dog")]},
        InvalidModelResponseError("bad response"), ModelExecutionError("unavailable"),
    )
    metrics = Detect(backend, ["pedestrian", "dog", "ignored"]).val(dataset=load_local(data), verbose=False)
    assert metrics.results_dict == {
        "samples": 4, "failures": 2, "failure_rate": 0.5, "match_iou": 0.5,
        "tp": 1, "fp": 3, "fn": 2, "precision": 0.25, "recall": 1 / 3, "f1": 2 / 7,
    }
    assert metrics.summary()[0] == {
        "class": "person", "support": 2, "tp": 1, "fp": 2, "fn": 1,
        "precision": 1 / 3, "recall": 0.5, "f1": 0.4,
    }
    assert metrics.dataset_counts == {"labeled_images": 2, "empty_labels": 1, "missing_labels": 1}
    assert len(backend.calls) == 4


def test_original_detection_coordinates_and_metadata_are_preserved(tmp_path):
    data = detection_data(tmp_path, ["0 0.5 0.5 0.5 0.5\n"])
    dataset = load_yolo(data)
    sample = dataset[0]
    assert sample["image"].shape == (60, 120, 3)
    np.testing.assert_array_equal(sample["objects"]["bbox"], [[30, 15, 90, 45]])
    np.testing.assert_array_equal(dataset[0]["objects"]["bbox"], sample["objects"]["bbox"])
    sample["objects"]["bbox"][0][0] = 999
    assert dataset[0]["objects"]["bbox"] == [[30, 15, 90, 45]]
    backend = Backend({"detections": [box("person")]})
    metrics = Detect(backend, ["person"]).val(dataset=load_local(data), verbose=False)
    assert (metrics.precision, metrics.recall, metrics.f1) == (1, 1, 1)


def test_detection_scores_classes_absent_from_constructor_labels(tmp_path):
    data = detection_data(tmp_path, ["1 0.5 0.5 0.5 0.5\n"])
    metrics = Detect(Backend({"detections": [box("person")]}), ["person"]).val(dataset=load_local(data), verbose=False)
    assert metrics.results_dict["samples"] == 1
    assert metrics.results_dict["fp"] == 1
    assert metrics.results_dict["fn"] == 1


def test_constructor_only_labels_are_invalid_validation_predictions(tmp_path):
    data = classification_data(tmp_path)
    metrics = Classify(Backend({"labels": ["animal"]}), ["animal"]).val(dataset=load_local(data), verbose=False)
    assert metrics.names == ("cat", "dog")
    assert metrics.top1 == 0
    assert len(metrics.failures) == 2
    assert all(failure["type"] == "InvalidModelResponseError" for failure in metrics.failures)


@pytest.mark.parametrize("task_kind", ["classify", "detect"])
def test_all_failed_runs_keep_denominators(tmp_path, task_kind):
    backend = Backend(InvalidModelResponseError("invalid"))
    if task_kind == "classify":
        data = classification_data(tmp_path, classes=("person",))
        task = Classify(backend, ["person"])
    else:
        data = detection_data(tmp_path, ["0 0.5 0.5 0.5 0.5\n"])
        task = Detect(backend, ["person"])
    metrics = task.val(dataset=load_local(data), verbose=False)
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
    validator = ClassificationValidator(task, dataset=load_local(data), max_samples=1, verbose=False)
    assert validator.task is task
    first = validator()
    second = validator()
    assert first is not second
    assert first.seen == second.seen == 1
    assert first.top1 == second.top1 == 1
    assert len(backend.calls) == 2
    assert len(second.failures) == 0
    third = task.val(dataset=load_local(data), max_samples=1, verbose=False)
    assert third is task.metrics and third is not second


@pytest.mark.parametrize("class_ids", [[2, 0, 2], {0, 2}, np.array([2, 0], dtype=np.int64)])
def test_classification_dataset_filters_classes_and_preserves_ids(tmp_path, monkeypatch, class_ids):
    from ml_pipes.vlm.validation import default_datasets as readers

    root = classification_data(tmp_path, classes=("cat", "dog", "fox"), per_class=2)
    image_files = readers._image_files
    visited = []

    def selected_image_files(directory):
        visited.append(directory.name)
        assert directory.name != "dog", "Unselected classes must not be scanned for images"
        return image_files(directory)

    monkeypatch.setattr(readers, "_image_files", selected_image_files)
    dataset = load_classification(root, class_ids=class_ids)
    assert dataset.features["label"].names == ["cat", "dog", "fox"]
    assert len(dataset) == 4
    assert visited == ["cat", "fox"]
    assert [sample["label"] for sample in dataset] == [0, 0, 2, 2]


@pytest.mark.parametrize("class_ids", [[], [-1], [2], [1.0], [0, True], [np.bool_(False)], 1, "1", b"\x01"])
def test_classification_dataset_rejects_invalid_class_ids(tmp_path, class_ids):
    root = classification_data(tmp_path)
    with pytest.raises(ValueError, match="class_ids"):
        load_classification(root, class_ids=class_ids)


def test_classification_dataset_selection_does_not_decode_excluded_images(tmp_path):
    data = classification_data(tmp_path, per_class=2)
    (data / "val" / "cat" / "0.png").write_bytes(b"corrupt but outside selected classes")
    (data / "val" / "dog" / "1.png").write_bytes(b"corrupt but past sample limit")
    dataset = load_classification(data, class_ids=[1])
    assert len(dataset) == 2
    metrics = Classify(Backend({"labels": ["dog"]})).val(dataset=load_classification(data, class_ids=[1]), max_samples=1, verbose=False)
    assert metrics.seen == 1 and metrics.top1 == 1
    assert metrics.names == ("cat", "dog")


def test_custom_validator_class():
    class CustomValidator:
        def __init__(self, *, task, dataset, marker):
            assert isinstance(task, Ground)
            self.task = task
            self.metrics = marker

        def __call__(self):
            assert self.task is task

    task = Ground(Backend(), "the person")
    assert task.val(dataset=(), validator=CustomValidator, marker="custom metrics") == "custom metrics"


def test_injected_loader_and_custom_dataset_construction(tmp_path):
    dataset = load_classification(classification_data(tmp_path), class_ids=[1])
    task = Classify(Backend({"labels": ["dog"]}), ["dog"])
    assert task.val(dataset=dataset, verbose=False).top1 == 1

    class CustomDataset:
        features = dataset.features
        def __iter__(self):
            return iter(dataset)

    assert task.val(dataset=CustomDataset(), verbose=False).top1 == 1


@pytest.mark.parametrize("task_type", [Classify, Detect, Ground])
def test_removed_class_map_fails_before_execution(tmp_path, task_type):
    data = classification_data(tmp_path)
    backend = Backend({"labels": ["cat"]})
    with pytest.raises(TypeError, match="class_map"):
        task_type(backend).val(dataset=load_local(data), class_map={"cat": "cat"}, verbose=False)
    assert not backend.calls


@pytest.mark.parametrize("custom_loader", [False, True])
@pytest.mark.parametrize("task_type", [Classify, Detect])
def test_validation_overrides_labels_without_mutating_task(tmp_path, custom_loader, task_type):
    if task_type is Classify:
        data = classification_data(tmp_path)
        dataset = load_classification(data)
        payload = {"labels": ["cat"]}
        labels = ("cat", "dog")
        original_payload = {"labels": ["original"]}
    else:
        data = detection_data(tmp_path, ["0 0.5 0.5 0.5 0.5"])
        dataset = load_yolo(data)
        payload = {"detections": [box("person")]}
        labels = ("person", "dog")
        original_payload = {"detections": [box("original")]}
    backend = Backend(payload, original_payload)
    task = task_type(backend, {"original": "Original description"}, context="Keep this context.")
    spec, instruction = task._spec, task._instruction
    settings = {"dataset": load_local(data).select([0]) if custom_loader else load_local(data)}
    metrics = task.val(**settings, max_samples=1, verbose=False)
    assert metrics.names == labels
    assert metrics.summary()[1]["support"] == 0
    request = backend.calls[0]
    assert "Keep this context." in request.instruction
    assert "Original description" not in request.instruction
    assert "original" not in json.dumps(request.response_spec.schema)
    assert all(label in json.dumps(request.response_spec.schema) for label in labels)
    assert task.labels == {"original": "Original description"}
    assert task._spec is spec and task._instruction == instruction
    task(dataset[0]["image"])
    assert backend.calls[1].response_spec is spec


@pytest.mark.parametrize("task_type", [Classify, Detect])
def test_validation_accepts_tasks_without_constructor_labels(tmp_path, task_type):
    if task_type is Classify:
        data = classification_data(tmp_path)
        payload = {"labels": ["cat"]}
    else:
        data = detection_data(tmp_path, ["0 0.5 0.5 0.5 0.5"])
        payload = {"detections": [box("person")]}
    task = task_type(Backend(payload))
    metrics = task.val(dataset=load_local(data), max_samples=1, verbose=False)
    assert metrics.seen == 1 and not metrics.failures
    assert task.labels is task._spec is task._instruction is None


@pytest.mark.parametrize("max_samples", [0, -1, True, 1.5])
def test_invalid_sample_limits(max_samples):
    with pytest.raises(ValueError, match="max_samples"):
        BaseValidator(Classify(Backend(), ["cat"]), dataset=(), max_samples=max_samples)


@pytest.mark.parametrize("iou", [0, -0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_matching_thresholds(iou):
    with pytest.raises(ValueError, match="match_iou"):
        DetectionValidator(Detect(Backend(), ["person"]), dataset=(), match_iou=iou)
    with pytest.raises(ValueError, match="match_iou"):
        DetectionMetrics(["person"], match_iou=iou)


def test_exact_split_and_empty_evaluation_errors(tmp_path):
    data = classification_data(tmp_path)
    with pytest.raises(FileNotFoundError, match="split not found"):
        load_classification(data, split="test")
    empty = tmp_path / "empty_dataset"
    (empty / "val" / "empty").mkdir(parents=True)
    backend = Backend({"labels": ["empty"]})
    with pytest.raises(ValueError, match="No classification images"):
        Classify(backend, ["empty"]).val(dataset=load_local(empty), verbose=False)

    with pytest.raises(ValueError, match="No samples to evaluate"):
        Classify(backend).val(dataset=load_classification(data).select([]), verbose=False)
    assert not backend.calls
    with pytest.raises(ValueError, match="split name"):
        load_classification(data, split="../val")


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
        Detect(backend, ["person"]).val(dataset=load_local(data), verbose=False)
    assert not backend.calls


def test_detection_relative_roots_lists_and_split_entries(tmp_path):
    root = tmp_path / "dataset"
    detection_data(root, ["0 0.5 0.5 0.5 0.5", ""])
    (root / "val.txt").write_text("images/val/1.png\nimages/val/0.png\n")
    config = tmp_path / "configs" / "eval.yaml"
    config.parent.mkdir()
    config.write_text("path: ../dataset\nval: [val.txt, images/val]\nnames:\n  0: person\n  1: dog\n")
    dataset = load_yolo(config)
    assert len(dataset) == 2
    assert dataset.features["objects"]["category"].feature.names == ["person", "dog"]
    assert dataset[0]["im_file"].endswith("0.png")
    with pytest.raises(ValueError, match="split"):
        load_yolo(config, split="test")
    (root / "val.txt").write_text("images/val/missing.png\n")
    with pytest.raises(ValueError, match="Invalid image entry"):
        load_yolo(config)


def test_yolo_annotation_rounding_at_image_edges_is_preserved(tmp_path):
    # COCO128 contains this valid six-decimal annotation at the right edge.
    data = detection_data(tmp_path, ["0 0.662641 0.494385 0.674719 0.988771"])
    dataset = load_yolo(data)
    box = dataset[0]["objects"]["bbox"][0]
    assert box[2] == pytest.approx(120 * 1.0000005)
    assert box[1] == pytest.approx(60 * -0.0000005)
    assert dataset[0]["objects"]["bbox"][0][2] > 120


def test_corrupt_images_and_unexpected_exceptions_stop_the_run(tmp_path):
    data = classification_data(tmp_path)
    backend = Backend(RuntimeError("programming error"))
    with pytest.raises(RuntimeError, match="programming error"):
        Classify(backend, ["cat"]).val(dataset=load_local(data), verbose=False)
    (data / "val" / "cat" / "0.png").write_bytes(b"not an image")
    with pytest.raises(OSError, match="0.png"):
        Classify(backend, ["cat"]).val(dataset=load_local(data), verbose=False)
    (data / "val" / "cat" / "0.png").write_bytes(b"")
    with pytest.raises(OSError, match="0.png"):
        Classify(backend, ["cat"]).val(dataset=load_local(data), verbose=False)


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
    metrics = Detect(Backend({"detections": []}), ["person"]).val(dataset=load_local(data), verbose=False)
    assert metrics.seen == 2
    assert (metrics.precision, metrics.recall, metrics.f1) == (0, 0, 0)


def test_verbose_output_stays_on_stderr(tmp_path, capsys):
    data = classification_data(tmp_path, classes=("cat",))
    Classify(Backend({"labels": ["cat"]}), ["cat"]).val(dataset=load_local(data))
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
from ml_pipes.vlm.validation import load_yolo
from types import SimpleNamespace
from pathlib import Path
import cv2
class Dataset:
    features = {'image': object(), 'label': SimpleNamespace(names=['cat'])}
    def __iter__(self):
        yield {'image': cv2.imread(str(next(Path(sys.argv[1]).rglob('*.png'))))[:, :, ::-1], 'label': 0}
task = Classify(lambda request: ModelResponse({'labels': ['cat']}), ['cat'])
assert task.val(dataset=Dataset(), verbose=False).top1 == 1
try:
    load_yolo('unused.yaml')
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
        "--model", "test",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["metrics"]["accuracy_top1"] == 1
    assert report["per_class"][0]["class"] == "cat"


def test_detection_example_and_execution_failure_exit_code(tmp_path, monkeypatch, capsys):
    from examples import run_validation

    data = detection_data(tmp_path, ["0 0.5 0.5 0.5 0.5"])
    config = tmp_path / "tasks.json"
    config.write_text(json.dumps({"detect": {"labels": ["pedestrian"]}}))
    args = [
        "--task", "detect", "--data", str(data), "--config", str(config),
        "--match-iou", "0.75",
        "--model", "test",
    ]
    monkeypatch.setattr(run_validation, "create_backend", lambda args: Backend({"detections": [box("person")]}))
    assert run_validation.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["metrics"]["f1"] == 1
    assert report["metrics"]["match_iou"] == 0.75

    monkeypatch.setattr(run_validation, "create_backend", lambda args: Backend(ModelExecutionError("offline")))
    assert run_validation.main(args) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["metrics"]["failure_rate"] == 1
    assert report["failures"][0]["message"] == "offline"


def load_local(path):
    return load_yolo(path) if Path(path).suffix == ".yaml" else load_classification(path)
