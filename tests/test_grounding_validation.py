"""Grounding evaluation against tiny local MDETR annotations."""
from __future__ import annotations

from ml_pipes.vlm.validation import load_refcoco

import json
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from ml_pipes.core import Pipeline
from ml_pipes.vision import ImagePayload
from ml_pipes.vlm import (
    BoundingBox, Ground, GroundingResult, InvalidModelResponseError,
    ModelExecutionError, ModelResponse, Point,
)
from ml_pipes.vlm.validation import GroundingMetrics, GroundingValidator


def grounding_data(root, *, dataset="refcocog", split="val", count=4):
    annotations, images = root / "annotations", root / "train2014"
    annotations.mkdir(parents=True)
    images.mkdir()
    for filename in ("a.png", "b.png"):
        assert cv2.imwrite(str(images / filename), np.zeros((60, 120, 3), np.uint8))
    ids = [91, 7, 440, 42, 130]
    data = {
        "categories": [{"id": 3, "name": "cat"}, {"id": 17, "name": "dog"}],
        "images": [
            {"id": ids[i], "file_name": "a.png" if i < 2 else "b.png",
             "width": 120, "height": 60, "caption": f"the animal numbered {i}"}
            for i in range(count)
        ],
        "annotations": [
            {"id": 1000 + i, "image_id": ids[i], "category_id": 3 if i % 2 == 0 else 17,
             "bbox": [30, 15, 60, 30]}
            for i in range(count)
        ],
    }
    path = annotations / f"finetune_{dataset}_{split}.json"
    path.write_text(json.dumps(data))
    return annotations, images, path, data


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


FOUND = {"status": "found", "box": {"x1": .25, "y1": .25, "x2": .75, "y2": .75}}
POINT_FOUND = {"status": "found", "point": {"x": .5, "y": .5}}


@pytest.mark.parametrize("default", [None, "the cat"])
def test_point_grounding_outputs_pixels_and_preserves_target_state(default):
    backend = Backend(POINT_FOUND)
    task = Ground(backend, default, geometry="point", context="Choose the visible body.")
    image = ImagePayload(np.zeros((60, 120, 3), np.uint8))
    pipeline = Pipeline([task])
    pipeline.validate(strict=True)
    assert pipeline((image, "the dog")) == GroundingResult(
        "found", None, (120, 60), geometry="point", point=Point(60, 30),
    )
    assert task.target == default
    assert "visible torso" in backend.calls[0].instruction
    assert 'Target: "the dog"' in backend.calls[0].instruction
    assert "Choose the visible body." in backend.calls[0].instruction
    schema = backend.calls[0].response_spec.schema
    assert set(schema["properties"]) == {"status", "point"}
    if default is not None:
        assert task(image=image).point == Point(60, 30)


@pytest.mark.parametrize("data", [
    {"status": "found", "point": None},
    {"status": "not_found", "point": {"x": .5, "y": .5}},
    {"status": "ambiguous", "point": {"x": .5, "y": .5}},
    {"status": "found", "point": {"x": float("nan"), "y": .5}},
    {"status": "found", "point": {"x": float("inf"), "y": .5}},
    {"status": "found", "point": {"x": -.1, "y": .5}},
    {"status": "found", "point": {"x": .5, "y": 1.01}},
    {"status": "found", "point": {"x": True, "y": .5}},
    {"status": "found", "point": {"x": ".5", "y": .5}},
    {"status": "found", "point": {"x": .5}},
    {"status": "found", "point": {"x": .5, "y": .5}, "box": None},
    FOUND,
])
def test_point_schema_rejects_invalid_responses(data):
    image = ImagePayload(np.zeros((60, 120, 3), np.uint8))
    with pytest.raises(InvalidModelResponseError):
        Ground(Backend(data), "animal", geometry="point")(image)


@pytest.mark.parametrize("geometry", [None, "points", "polygon", 1, []])
def test_ground_rejects_unsupported_geometry(geometry):
    backend = Backend(POINT_FOUND)
    with pytest.raises(ValueError, match="geometry"):
        Ground(backend, geometry=geometry)
    assert not backend.calls


@pytest.mark.parametrize("xy, expected", [
    ((60, 30), 1), ((30, 15), 1), ((90, 45), 1),
    ((29.999, 30), 0), ((60, 45.001), 0), ((0, 0), 0), ((120, 60), 0),
])
def test_point_metrics_use_inclusive_target_box_containment(xy, expected):
    metrics = GroundingMetrics(["animal"], geometry="point")
    metrics.update("animal", [30, 15, 90, 45], GroundingResult(
        "found", None, (120, 60), geometry="point", point=Point(*xy),
    ), "image.png")
    assert metrics.results_dict == {
        "samples": 1, "failures": 0, "failure_rate": 0,
        "unique_images": 1, "point_in_box_accuracy": expected,
    }
    assert metrics.summary() == [{"class": "animal", "support": 1, "point_in_box_accuracy": expected}]


@pytest.mark.parametrize("prediction", [
    GroundingResult("found", BoundingBox(0, 0, 1, 1), (120, 60)),
    GroundingResult("found", BoundingBox(0, 0, 1, 1), (120, 60), "point", Point(60, 30)),
    GroundingResult("found", None, (120, 60), "point", Point(float("nan"), 30)),
    GroundingResult("found", None, (120, 60), "point", Point(121, 30)),
    GroundingResult("not_found", None, (120, 60), "point", Point(60, 30)),
])
def test_point_metrics_reject_inconsistent_public_outputs(prediction):
    metrics = GroundingMetrics(["animal"], geometry="point")
    with pytest.raises(ValueError):
        metrics.update("animal", [30, 15, 90, 45], prediction, "image.png")
    assert metrics.seen == 0


def test_point_validation_counts_all_statuses_failures_and_repeated_images(tmp_path):
    annotations, images, _, _ = grounding_data(tmp_path, count=5)
    backend = Backend(
        POINT_FOUND, {"status": "not_found", "point": None},
        {"status": "ambiguous", "point": None}, ModelExecutionError("offline"),
        {"status": "found", "point": {"x": 0, "y": 0}},
    )
    task = Ground(backend, geometry="point")
    metrics = task.val(dataset=load_refcoco(annotations, images), verbose=False)
    assert metrics.results_dict == {
        "samples": 5, "failures": 1, "failure_rate": .2,
        "unique_images": 2, "point_in_box_accuracy": .2,
    }
    assert metrics.status_counts == {"found": 2, "not_found": 1, "ambiguous": 1}
    assert metrics.summary()[0] == {"class": "cat", "support": 3, "point_in_box_accuracy": 1 / 3}
    assert metrics.failures[0]["target"] == "the animal numbered 3"
    task.model = Backend(ModelExecutionError("offline"))
    again = task.val(dataset=load_refcoco(annotations, images), max_samples=2, verbose=False)
    assert again is not metrics and again.results_dict["point_in_box_accuracy"] == 0
    assert again.results_dict["failure_rate"] == 1 and again.unique_images == 1


def test_point_validation_example_selects_metrics_from_task_config(tmp_path, monkeypatch, capsys):
    from examples import run_validation

    annotations, images, _, _ = grounding_data(tmp_path)
    monkeypatch.setattr(run_validation, "create_backend", lambda args: Backend(POINT_FOUND))
    config = tmp_path / "point.json"
    config.write_text(json.dumps({"ground": {"geometry": "point"}}))
    assert run_validation.main([
        "--task", "ground", "--data", str(annotations), "--images", str(images),
        "--model", "test", "--config", str(config),
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["task_config"] == {"geometry": "point"}
    assert report["metrics"]["point_in_box_accuracy"] == 1
    assert "mean_iou" not in report["metrics"]


def test_loader_preserves_expression_order_sparse_ids_and_original_pixels(tmp_path):
    annotations, images, _, _ = grounding_data(tmp_path)
    dataset = load_refcoco(annotations, images)
    assert dataset.features["label"].names == ["cat", "dog"]
    assert [sample["sample_id"] for sample in dataset] == [91, 7, 440, 42]
    assert dataset.counts == {"expressions": 4, "unique_images": 2}
    assert dataset.metadata["expected_split_by"] == "umd"
    sample = dataset[1]
    assert sample["image"].shape == (60, 120, 3)
    assert sample["text"] == "the animal numbered 1"
    assert sample["label"] == 1
    np.testing.assert_array_equal(sample["bbox"], [30, 15, 90, 45])
    sample["bbox"][:] = [0] * 4
    np.testing.assert_array_equal(dataset[1]["bbox"], [30, 15, 90, 45])


def test_expression_scoring_statuses_and_failures_keep_all_denominators(tmp_path):
    annotations, images, _, _ = grounding_data(tmp_path, count=5)
    backend = Backend(
        FOUND, {"status": "not_found", "box": None}, {"status": "ambiguous", "box": None},
        ModelExecutionError("unavailable"), InvalidModelResponseError("bad JSON"),
    )
    task = Ground(backend, "unused default", context="Resolve the whole animal.")
    metrics = task.val(dataset=load_refcoco(annotations, images), verbose=False)
    assert isinstance(metrics, GroundingMetrics)
    assert task.metrics is metrics
    assert metrics.results_dict == {
        "samples": 5, "failures": 2, "failure_rate": .4, "unique_images": 2,
        "accuracy_iou_0.5": .2, "accuracy_iou_0.75": .2, "accuracy_iou_0.9": .2,
        "mean_iou": .2,
    }
    assert metrics.status_counts == {"found": 1, "not_found": 1, "ambiguous": 1}
    assert metrics.dataset_counts == {"expressions": 5, "unique_images": 2}
    assert metrics.dataset_metadata["dataset"] == "refcocog"
    assert metrics.summary()[0] == {
        "class": "cat", "support": 3, "accuracy_iou_0.5": 1 / 3,
        "accuracy_iou_0.75": 1 / 3, "accuracy_iou_0.9": 1 / 3, "mean_iou": 1 / 3,
    }
    assert metrics.failures[0]["sample_id"] == 42
    assert metrics.failures[1]["sample_id"] == 130
    assert metrics.failures[0]["target"] == "the animal numbered 3"
    assert metrics.failures[0]["type"] == "ModelExecutionError"
    assert metrics.failures[1]["type"] == "InvalidModelResponseError"
    assert len(backend.calls) == 5
    for i, request in enumerate(backend.calls):
        assert f'Target: "the animal numbered {i}"' in request.instruction
        assert "Resolve the whole animal." in request.instruction
        assert "unused default" not in request.instruction
    assert set(metrics.speed) == {"load", "inference", "metrics"}
    assert all(value >= 0 for value in metrics.speed.values())


@pytest.mark.parametrize("overlap", [0, .5, .75, .9, 1])
def test_grounding_uses_ordinary_iou_and_inclusive_thresholds(overlap):
    metrics = GroundingMetrics(["animal"])
    prediction = GroundingResult("found", BoundingBox(0, 0, overlap * 100, 10), (100, 10))
    if overlap == 0:
        prediction = GroundingResult("found", BoundingBox(101, 0, 110, 10), (120, 10))
    metrics.update("animal", [0, 0, 100, 10], prediction, "same.png")
    assert metrics.mean_iou == pytest.approx(overlap)
    for threshold in metrics.thresholds:
        assert metrics.results_dict[f"accuracy_iou_{threshold}"] == int(overlap >= threshold)


def test_limit_counts_evaluated_images_and_repeated_runs_reset(tmp_path):
    annotations, images, _, _ = grounding_data(tmp_path)
    backend = Backend(ModelExecutionError("unavailable"))
    task = Ground(backend)
    validator = GroundingValidator(task, dataset=load_refcoco(annotations, images), max_samples=2, verbose=False)
    assert validator.task is task
    first, second = validator(), validator()
    assert first is not second
    for metrics in (first, second):
        assert metrics.seen == 2 and metrics.unique_images == 1
        assert metrics.failure_rate == 1
        assert metrics.mean_iou == 0
        assert metrics.results_dict["accuracy_iou_0.5"] == 0
        assert metrics.status_counts == {"found": 0, "not_found": 0, "ambiguous": 0}
        assert metrics.dataset_counts == {"expressions": 4, "unique_images": 2}
    assert len(backend.calls) == 4


@pytest.mark.parametrize("dataset,split", [
    ("refcocog", "train"), ("refcocog", "test"), ("refcoco", "testA"), ("refcoco+", "testB"),
])
def test_dataset_variants_and_exact_splits(tmp_path, dataset, split):
    annotations, images, _, _ = grounding_data(tmp_path, dataset=dataset, split=split)
    loaded = load_refcoco(annotations, images, dataset=dataset, split=split)
    assert loaded.metadata["expected_split_by"] == ("umd" if dataset == "refcocog" else "unc")
    with pytest.raises(FileNotFoundError):
        load_refcoco(annotations, images, dataset=dataset, split="val")


@pytest.mark.parametrize("change,match", [
    (lambda d: d["images"].append(dict(d["images"][0])), "Duplicate.*images"),
    (lambda d: d["annotations"].append(dict(d["annotations"][0])), "Duplicate.*annotations"),
    (lambda d: d["categories"].append(dict(d["categories"][0])), "Duplicate.*categories"),
    (lambda d: d["annotations"][0].update(image_id=999), "Orphan"),
    (lambda d: d["annotations"][0].update(image_id=7), "Multiple target"),
    (lambda d: d["annotations"].pop(), "Missing target"),
    (lambda d: d["annotations"][0].update(category_id=999), "Unknown category"),
    (lambda d: d["images"][0].update(caption=" "), "Invalid expression"),
    (lambda d: d["images"][0].update(caption=["animal"]), "Invalid expression"),
    (lambda d: d["images"][0].update(width=0), "Invalid image dimensions"),
    (lambda d: d["images"][0].update(file_name="missing.jpg"), "Grounding image not found"),
    (lambda d: d["annotations"][0].update(bbox=[0, 0, float("nan"), 1]), "Invalid box"),
    (lambda d: d["annotations"][0].update(bbox=[0, 0, 1, float("inf")]), "Invalid box"),
    (lambda d: d["annotations"][0].update(bbox=[0, 0, -1, 1]), "Invalid box"),
    (lambda d: d["annotations"][0].update(bbox=[0, 0, 1, 0]), "Invalid box"),
    (lambda d: d["annotations"][0].update(bbox=[0, 0, 1]), "Invalid box"),
    (lambda d: d["annotations"][0].update(bbox=[0, 0, "1", 1]), "Invalid box"),
    (lambda d: d.update(images=[], annotations=[]), "No grounding expressions"),
])
def test_invalid_annotations_fail_before_inference(tmp_path, change, match):
    annotations, images, path, data = grounding_data(tmp_path)
    change(data)
    path.write_text(json.dumps(data))
    backend = Backend(FOUND)
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        Ground(backend).val(dataset=load_refcoco(annotations, images), verbose=False)
    assert not backend.calls


def test_original_annotation_coordinates_are_not_clipped(tmp_path):
    annotations, images, path, data = grounding_data(tmp_path)
    data["annotations"][0]["bbox"] = [-1.5, 0, 130, 60]
    path.write_text(json.dumps(data))
    np.testing.assert_array_equal(load_refcoco(annotations, images)[0]["bbox"], [-1.5, 0, 128.5, 60])


def test_image_dimensions_corruption_and_invalid_options(tmp_path):
    annotations, images, path, data = grounding_data(tmp_path)
    data["images"][0]["height"] = 61
    path.write_text(json.dumps(data))
    backend = Backend(FOUND)
    with pytest.raises(ValueError, match="dimensions differ"):
        Ground(backend).val(dataset=load_refcoco(annotations, images), verbose=False)
    (images / "a.png").write_bytes(b"not an image")
    with pytest.raises(OSError, match="cannot identify image"):
        load_refcoco(annotations, images)[0]
    for dataset, split in [("unknown", "val"), ("refcocog", "testA"), ("refcoco", "test"), ("refcocog", "../val")]:
        with pytest.raises(ValueError):
            load_refcoco(annotations, images, dataset=dataset, split=split)
    with pytest.raises(TypeError, match="loaded dataset"):
        Ground(backend).val(dataset=annotations)
    with pytest.raises(TypeError, match="class_map"):
        Ground(backend).val(dataset=load_refcoco(annotations, images),  class_map={"cat": "cat"})
    assert not backend.calls


def test_custom_loader_and_dataset_hook(tmp_path):
    annotations, images, _, _ = grounding_data(tmp_path)
    dataset = load_refcoco(annotations, images).select([1, 0])
    task = Ground(Backend(FOUND))
    metrics = task.val(dataset=dataset, verbose=False)
    assert metrics.seen == 2 and metrics.unique_images == 1 and metrics.mean_iou == 1
    assert GroundingValidator(task, dataset, verbose=False)().seen == 2


@pytest.mark.parametrize("payload", [(), (42, "cat"), (None, "cat"), (1, 2, 3)])
def test_ground_rejects_invalid_tuple_inputs_before_inference(payload):
    backend = Backend(FOUND)
    with pytest.raises(ValueError):
        Ground(backend)(payload)
    assert not backend.calls


def test_ground_pipeline_with_declared_tuple_input():
    backend = Backend(FOUND)
    task = Ground(backend)
    pipeline = Pipeline([task])
    contract = pipeline.validate(pipeline_input_type=tuple[ImagePayload, str], strict=True)
    assert contract.input_type == tuple[ImagePayload, str]
    image = ImagePayload(np.zeros((60, 120, 3), np.uint8))
    assert pipeline((image, "the animal")).box == BoundingBox(30, 15, 90, 45)


def test_grounding_example_defaults_to_neutral_context_and_reports_settings(tmp_path, monkeypatch, capsys):
    from examples import run_validation

    annotations, images, _, _ = grounding_data(tmp_path)
    backend = Backend(FOUND)
    monkeypatch.setattr(run_validation, "create_backend", lambda args: backend)
    args = ["--task", "ground", "--data", str(annotations), "--images", str(images),
            "--max-samples", "2", "--model", "test"]
    assert run_validation.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["task_config"] == {}
    assert report["metrics"]["samples"] == 2 and report["metrics"]["unique_images"] == 1
    assert report["metrics"]["mean_iou"] == 1
    assert report["status_counts"]["found"] == 2
    assert report["dataset_metadata"]["expected_split_by"] == "umd"
    assert report["model_settings"] == {"temperature": 0.0, "max_tokens": 4096}
    assert all("Context:" not in request.instruction for request in backend.calls)

    config = tmp_path / "tasks.json"
    config.write_text(json.dumps({"ground": {"context": "Resolve whole animals."}}))
    assert run_validation.main(args + ["--config", str(config)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["task_config"]["context"] == "Resolve whole animals."
    assert "Resolve whole animals." in backend.calls[-1].instruction
    monkeypatch.setattr(run_validation, "create_backend", lambda args: Backend(ModelExecutionError("offline")))
    assert run_validation.main(args) == 1
    assert json.loads(capsys.readouterr().out)["metrics"]["failure_rate"] == 1


def test_transformers_validation_runner_saves_report(tmp_path, monkeypatch, capsys):
    from examples import run_transformers, run_validation

    annotations, images, _, _ = grounding_data(tmp_path)
    backend = Backend(FOUND)
    backend.model = SimpleNamespace(weights=SimpleNamespace(device="mps", dtype="torch.float16"))
    received = []

    def create(args):
        received.append(args)
        return backend

    monkeypatch.setattr(run_transformers, "create_backend", create)
    monkeypatch.setattr(run_validation, "create_backend", lambda args: pytest.fail("Unexpected Bedrock load"))
    output = tmp_path / "reports" / "ground.json"
    assert run_validation.main([
        "--backend", "transformers", "--task", "ground", "--data", str(annotations),
        "--images", str(images), "--max-samples", "2",
        "--device", "mps", "--dtype", "float16", "--output", str(output),
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert json.loads(output.read_text()) == report
    assert report["backend"] == "transformers"
    assert report["model"] == "Qwen/Qwen3-VL-2B-Instruct"
    assert report["device"] == "mps" and report["dtype"] == "torch.float16"
    assert report["output_mode"] == "prompted" and report["retries"] == 0
    assert report["do_sample"] is False and report["load_seconds"] >= 0
    assert report["metrics"]["mean_iou"] == 1 and report["task_config"] == {}
    assert received[0].device == "mps" and received[0].dtype == "float16"


@pytest.mark.parametrize("options, message", [
    (["--backend", "transformers", "--profile", "aws"], "only to Bedrock"),
    (["--model", "bedrock", "--device", "mps"], "only to Transformers"),
    ([], "--model is required"),
])
def test_validation_rejects_incompatible_backend_options(options, message, capsys):
    from examples import run_validation

    with pytest.raises(SystemExit) as exc:
        run_validation.main(["--task", "ground", "--data", "unused", "--images", "unused", *options])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_grounding_without_optional_dependencies(tmp_path):
    annotations, images, _, _ = grounding_data(tmp_path)
    script = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'ultralytics', 'yaml', 'transformers', 'datasets', 'pycocotools'}:
            raise ImportError('blocked optional dependency: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
from ml_pipes.vlm import Ground, ModelResponse
task = Ground(lambda request: ModelResponse({'status': 'not_found', 'box': None}))
import cv2
from types import SimpleNamespace
from ml_pipes.vlm.validation.default_datasets import _refcoco_selection
import json
from pathlib import Path
rows, names = _refcoco_selection(json.loads((Path(sys.argv[1]) / 'finetune_refcocog_val.json').read_text()))
class Dataset:
    features = {'image': object(), 'text': object(), 'bbox': object(),
                'label': SimpleNamespace(names=names)}
    def __iter__(self):
        for row in rows:
            yield {**row, 'image': cv2.imread(str(Path(sys.argv[2]) / row['file_name']))[:, :, ::-1]}
metrics = task.val(dataset=Dataset(), verbose=False)
assert metrics.seen == 4 and metrics.mean_iou == 0
"""
    subprocess.run([sys.executable, "-c", script, str(annotations), str(images)], check=True, capture_output=True, text=True)
