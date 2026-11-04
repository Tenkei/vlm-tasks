"""OVAD categorical adaptation, masked scoring, and fixed task requests."""
from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import hashlib
import json
from types import SimpleNamespace
from zipfile import ZipFile

import cv2
import numpy as np
import pytest

from ml_pipes.core import Pipeline
from ml_pipes.validation import PipelineValidationError
from ml_pipes.vision import ImagePayload
from ml_pipes.vlm import (
    Attributes, AttributePredictionResult, InvalidModelResponseError, ModelExecutionError, ModelResponse,
)
from ml_pipes.vlm.ops import AttributeChoices
from ml_pipes.vlm.validation import (
    AttributesMetrics, AttributesValidator, DataLoader, OVADDataset, prepare_dataset,
)
from ml_pipes.vlm.validation import prepare as preparation
from ml_pipes.vlm.validation.data import OVAD_GROUPS, _ovad_selection


def ovad_data():
    names = [
        ("cleanliness", "clean/neat"), ("cleanliness", "unclean/dirt/dirty/muddy"),
        ("material", "wood/wooden"), ("material", "metal/metallic"),
        ("color", "red"), ("color", "blue"),
        ("clothes color", "red"), ("clothes color", "blue"),
    ]
    return {
        "attributes": [{"id": i, "type": group, "name": f"{group}:{value}"} for i, (group, value) in enumerate(names)],
        "categories": [{"id": 7, "name": "chair"}],
        "images": [
            {"id": i, "file_name": f"{i:012d}.jpg", "width": 120, "height": 60}
            for i in (13, 24, 35)
        ],
        "annotations": [
            {"id": i, "image_id": image_id, "category_id": 7, "area": area,
             "bbox": [-1.2, 2.3, 31.3, 20.2], "att_vec": vector}
            for i, image_id, area, vector in [
                (91, 35, 49.9, [1, 0, 1, 0, 1, 1, 1, 1]),  # too small
                (2, 35, 50, [0, -1, 0, 0, 1, 1, 1, 1]),  # no categorical targets
                (70, 13, 50, [1, 0, -1, -1, 0, 0, 0, 0]),
                (3, 13, 80, [0, 0, 0, 1, 0, 0, 0, 0]),
                (88, 24, 90, [0, 1, 1, 0, 0, 0, 0, 0]),
            ]
        ],
    }


def write_local(root, data=None):
    data = ovad_data() if data is None else data
    root.mkdir(parents=True, exist_ok=True)
    images = root / "val2017"
    images.mkdir(exist_ok=True)
    for image in data["images"]:
        pixels = np.arange(60 * 120 * 3, dtype=np.uint8).reshape(60, 120, 3)
        assert cv2.imwrite(str(images / image["file_name"]), pixels)
    path = root / "ovad2000.json"
    path.write_text(json.dumps(data))
    return path, images


class Backend:
    def __init__(self, *responses):
        self.calls = []
        self.responses = responses or ({"attributes": {"cleanliness": "clean", "material": "wood"}},)

    def __call__(self, request):
        response = self.responses[min(len(self.calls), len(self.responses) - 1)]
        self.calls.append(request)
        if isinstance(response, Exception):
            raise response
        return ModelResponse(response)


def test_group_schema_synonyms_selection_and_crop(tmp_path):
    path, images = write_local(tmp_path)
    dataset = OVADDataset(path, images)
    assert dataset.attributes == {
        "cleanliness": {"clean": "clean/neat", "unclean": "unclean/dirt/dirty/muddy"},
        "material": {"wood": "wood/wooden", "metal": "metal/metallic"},
    }
    assert [s["sample_id"] for s in dataset.samples] == [70, 3, 88]
    assert dataset.counts == {
        "source_objects": 5, "source_images": 3, "excluded_small_objects": 1,
        "excluded_unscored_objects": 1, "excluded_objects": 2, "eligible_objects": 3,
        "prepared_objects": 3, "prepared_images": 2,
    }
    sample = dataset[0]
    assert sample["attributes"] == {"cleanliness": "clean"}
    assert sample["image_id"] == 13 and sample["bbox"] == [-1.2, 2.3, 31.3, 20.2]
    assert sample["crop_box"] == (0, 2, 31, 23)
    assert sample["img"].size == (31, 21)
    np.testing.assert_array_equal(sample["img"].array, cv2.imread(sample["im_file"])[2:23, 0:31])
    sample["attributes"].clear()
    sample["bbox"][0] = 999
    assert dataset[0]["attributes"] == {"cleanliness": "clean"}
    assert dataset[0]["bbox"][0] == -1.2
    subset = OVADDataset(path, images, groups=("material",), max_samples=1)
    assert [s["sample_id"] for s in subset.samples] == [3]
    assert subset.counts["excluded_unscored_objects"] == 2
    assert subset.counts["eligible_objects"] == 2
    assert OVADDataset(path, images, split="test").samples == dataset.samples


def test_default_schema_covers_all_22_types():
    data = ovad_data()
    data["attributes"] = [
        {"id": i, "type": group, "name": f"{group}:a/b"} for i, group in enumerate(OVAD_GROUPS)
    ]
    for annotation in data["annotations"]:
        annotation["att_vec"] = [1] * 22
    schema, samples, _ = _ovad_selection(data)
    assert len(schema) == 22 and set(schema) == set(OVAD_GROUPS)
    assert len(samples[0]["attributes"]) == 22


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d["attributes"].append(d["attributes"][0]), "Duplicate.*attributes"),
    (lambda d: d["attributes"][0].update(id=20), "consecutive"),
    (lambda d: d["attributes"][0].update(name="cleanliness:"), "synonyms"),
    (lambda d: d["attributes"][1].update(name="cleanliness:clean/dirty"), "canonical"),
    (lambda d: d["attributes"][0].update(type="unsupported"), "type"),
    (lambda d: d["images"].append(d["images"][0]), "Duplicate.*images"),
    (lambda d: d["images"][0].update(id=True), "integer IDs"),
    (lambda d: d["images"][0].update(width=0), "dimensions"),
    (lambda d: d["images"][0].update(height=1.5), "dimensions"),
    (lambda d: d["images"][0].update(file_name=""), "filename"),
    (lambda d: d["annotations"].append(d["annotations"][0]), "Duplicate.*annotations"),
    (lambda d: d["annotations"][0].update(image_id=999), "image join"),
    (lambda d: d["annotations"][0].update(category_id=True), "category"),
    (lambda d: d["annotations"][0].update(att_vec=[1]), "att_vec"),
    (lambda d: d["annotations"][0]["att_vec"].__setitem__(0, True), "att_vec"),
    (lambda d: d["annotations"][0]["att_vec"].__setitem__(0, 2), "att_vec"),
    (lambda d: d["annotations"][0]["att_vec"].__setitem__(1, 1), "Multiple positive"),
    (lambda d: d["annotations"][0].update(area=float("nan")), "area"),
    (lambda d: d["annotations"][0].update(bbox=[0, 0, 0, 5]), "box"),
    (lambda d: d["annotations"][0].update(bbox=[0, 0, 5, float("inf")]), "box"),
    (lambda d: d["annotations"][0].update(bbox=[130, 0, 5, 5]), "Empty clipped"),
])
def test_malformed_source_annotations_fail_even_outside_selection(mutation, match):
    data = ovad_data()
    mutation(data)
    with pytest.raises(ValueError, match=match):
        _ovad_selection(data, groups=("material",), max_samples=1)


@pytest.mark.parametrize("settings", [{"split": "train"}, {"groups": ("color",)}, {"groups": ("cooked",)}, {"groups": ()}, {"max_samples": True}])
def test_invalid_dataset_configuration(tmp_path, settings):
    path, images = write_local(tmp_path)
    with pytest.raises(ValueError):
        OVADDataset(path, images, **settings)


def test_unreadable_wrong_dimensions_and_empty_selection(tmp_path):
    path, images = write_local(tmp_path)
    dataset = OVADDataset(path, images)
    selected_image = images / "000000000013.jpg"
    assert cv2.imwrite(str(selected_image), np.zeros((10, 20, 3), np.uint8))
    with pytest.raises(ValueError, match="dimensions differ"):
        dataset[0]
    selected_image.write_bytes(b"bad image")
    with pytest.raises(ValueError, match="Could not read image"):
        dataset[0]
    selected_image.unlink()
    with pytest.raises(FileNotFoundError, match="image not found"):
        OVADDataset(path, images)
    data = ovad_data()
    for annotation in data["annotations"]:
        annotation["att_vec"] = [-1] * 8
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="No eligible"):
        OVADDataset(path, images)


def test_masked_scoring_fixed_schema_and_failures(tmp_path):
    path, images = write_local(tmp_path)
    backend = Backend(
        {"attributes": {"cleanliness": "clean", "material": "wood"}},
        {"attributes": {"cleanliness": "unclean", "material": "wood"}},
        ModelExecutionError("offline"),
    )
    schema = OVADDataset(path, images).attributes
    schema["cleanliness"]["clean"] = "Custom description"
    task = Attributes(backend, schema, context="Describe the cropped object.")
    state = (deepcopy(task.attributes), task.context, task._spec, task._instruction)
    metrics = task.val(data=path, images=images, verbose=False)
    assert task.metrics is metrics and isinstance(metrics, AttributesMetrics)
    assert metrics.results_dict == {
        "samples": 3, "failures": 1, "failure_rate": 1 / 3,
        "attribute_decisions": 4, "unique_images": 2, "accuracy": .25,
        "mean_attribute_accuracy": .25, "macro_f1": .25,
    }
    np.testing.assert_array_equal(metrics.confusion_matrices["cleanliness"], [[1, 0, 0], [0, 0, 1]])
    np.testing.assert_array_equal(metrics.confusion_matrices["material"], [[0, 0, 1], [1, 0, 0]])
    assert metrics.summary()[0]["values"][0]["f1"] == 1
    assert metrics.summary()[1]["values"][0]["precision"] == 0
    assert len(backend.calls) == 3
    assert all(call.response_spec.schema == task._spec.schema for call in backend.calls)
    assert len({call.instruction for call in backend.calls}) == 1
    assert all("Custom description" in call.instruction for call in backend.calls)
    assert all(set(call.response_spec.schema["$defs"]["AttributeValues"]["required"]) == set(schema) for call in backend.calls)
    assert (task.attributes, task.context, task._spec, task._instruction) == state
    assert metrics.failures[0]["sample_id"] == 88
    assert metrics.dataset_counts["evaluated_objects"] == 3
    assert metrics.dataset_counts["evaluated_attribute_decisions"] == 4
    assert set(metrics.speed) == {"load", "inference", "metrics"}


def test_all_failed_runs_and_repeated_resets(tmp_path):
    path, images = write_local(tmp_path)
    backend = Backend({"attributes": {"cleanliness": "unsupported", "material": "wood"}})
    task = Attributes(backend, OVADDataset(path, images).attributes)
    validator = AttributesValidator(task, data=path, images=images, verbose=False)
    first, second = validator(), validator()
    assert first is not second
    for metrics in (first, second):
        assert metrics.seen == 3 and metrics.attribute_decisions == 4
        assert metrics.failure_rate == 1 and metrics.accuracy == metrics.macro_f1 == 0
        assert metrics.failures[0]["type"] == "InvalidModelResponseError"
        assert all(matrix[:, -1].sum() == 2 for matrix in metrics.confusion_matrices.values())


@pytest.mark.parametrize("schema", [
    {"color": ["red", "blue"]}, {"material": ["wood"]},
    {"material": ["wood", "metal", "none"]},
])
def test_incompatible_task_schema_fails_before_inference(tmp_path, schema):
    path, images = write_local(tmp_path)
    backend = Backend()
    with pytest.raises(ValueError):
        Attributes(backend, schema).val(data=path, images=images, verbose=False)
    assert not backend.calls


def test_class_map_rejected_and_custom_loader_never_accesses_labels(tmp_path):
    path, images = write_local(tmp_path)
    dataset = OVADDataset(path, images, groups=("material",))
    backend = Backend({"attributes": {"material": "metal"}})
    task = Attributes(backend, dataset.attributes)
    with pytest.raises(ValueError, match="class_map"):
        task.val(data=path, images=images, class_map={"material": "material"})
    assert not backend.calls
    # This dataset exposes neither names nor class IDs; the loader owns crops.
    loader = SimpleNamespace(dataset=SimpleNamespace(attributes=dataset.attributes))

    class Loader:
        dataset = loader.dataset

        def __iter__(self):
            yield dataset[0]

    metrics = task.val(dataloader=Loader(), verbose=False)
    assert metrics.seen == 1 and metrics.accuracy == 1
    assert backend.calls[0].image.size == (31, 21)
    assert not hasattr(task, "labels")
    with pytest.raises(ValueError, match="attributes schema"):
        task.val(dataloader=SimpleNamespace(dataset=object()), verbose=False)
    validator = AttributesValidator(task, dataloader=DataLoader(dataset), verbose=False)
    validator.dataset = dataset
    validator.attributes = dataset.attributes
    validator.init_metrics()
    with pytest.raises(TypeError, match="AttributePredictionResult"):
        validator.update_metrics({}, dataset[0])
    validator.update_metrics(AttributePredictionResult({"material": "metal"}), dataset[0])
    assert validator.metrics.accuracy == 1


def test_metrics_group_weighting_prediction_support_and_zero_support():
    metrics = AttributesMetrics({"material": ["wood", "metal", "glass"], "cleanliness": ["clean", "unclean"], "size": ["small", "large"]})
    metrics.update({"material": "wood", "cleanliness": "clean"}, {"material": "metal", "cleanliness": "clean", "size": "large"}, 1)
    metrics.update({"material": "wood"}, {"material": "wood", "cleanliness": "unclean", "size": "small"}, 1)
    assert metrics.accuracy == 2 / 3
    assert metrics.mean_attribute_accuracy == .75
    # material: wood F1 2/3, predicted-only metal F1 0, absent glass excluded.
    assert metrics.macro_f1 == pytest.approx((1 / 3 + 1) / 2)
    assert metrics.summary()[2]["accuracy"] is None
    assert metrics.summary()[2]["macro_f1"] is None
    assert metrics.summary()[2]["values"][0]["f1"] is None
    assert metrics.confusion_matrices["size"].sum() == 0
    assert metrics.unique_images == 1
    for prediction in ({"material": "wood"}, {"material": "paper", "cleanliness": "clean", "size": "small"}):
        with pytest.raises(ValueError):
            metrics.update({"material": "wood"}, prediction, 2)
    assert metrics.seen == 2  # rejected updates are atomic


@pytest.fixture
def http(monkeypatch):
    sources, calls = {}, []
    data = ovad_data()
    stream = BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr("ovad2000.json", json.dumps(data))
        archive.writestr("../../outside", b"never extract")
    content = stream.getvalue()
    sources[preparation._OVAD_URL] = content
    monkeypatch.setattr(preparation, "_OVAD_CHECKSUM", "sha256:" + hashlib.sha256(content).hexdigest())
    for image in data["images"]:
        ok, pixels = cv2.imencode(".jpg", np.zeros((60, 120, 3), np.uint8))
        assert ok
        sources[preparation._OVAD_IMAGE_URL + image["file_name"]] = pixels.tobytes()

    def open_url(url):
        calls.append(url)
        return BytesIO(sources[url])

    monkeypatch.setattr(preparation, "_open_url", open_url)
    return sources, calls


def test_named_preparation_filters_before_limit_and_reuses_cache(tmp_path, http):
    prepared = prepare_dataset("ovad", max_samples=2, cache_dir=tmp_path)
    dataset = OVADDataset(prepared.data, prepared.images)
    assert [s["sample_id"] for s in dataset.samples] == [70, 3]
    assert prepared.attributes == dataset.attributes
    assert prepared.metadata["counts"]["source_objects"] == 5
    assert prepared.metadata["counts"]["excluded_objects"] == 2
    assert prepared.metadata["counts"]["prepared_objects"] == 2
    assert len(http[1]) == 2  # archive + one distinct selected image
    assert not (tmp_path / "outside").exists()
    assert prepare_dataset("ovad", max_samples=2, cache_dir=tmp_path) == prepared
    assert len(http[1]) == 2
    image = next(prepared.images.glob("*.jpg"))
    image.write_bytes(b"corrupt")
    assert prepare_dataset("ovad", max_samples=2, cache_dir=tmp_path) == prepared
    assert len(http[1]) == 3
    full = prepare_dataset("ovad", split="test", cache_dir=tmp_path)
    assert full.metadata["counts"]["prepared_objects"] == 3
    assert len(http[1]) == 4
    assert not any("000000000035.jpg" in url for url in http[1])


def test_named_group_selection_cache_identity_and_validation(tmp_path, http):
    material = prepare_dataset("ovad", groups=("material",), max_samples=1, cache_dir=tmp_path)
    cleanliness = prepare_dataset("ovad", groups=("cleanliness",), max_samples=1, cache_dir=tmp_path)
    assert material.data != cleanliness.data
    assert material.metadata["sample_ids"] == [3]
    assert cleanliness.metadata["sample_ids"] == [70]
    assert material.attributes == {"material": {"wood": "wood/wooden", "metal": "metal/metallic"}}
    backend = Backend({"attributes": {"material": "metal"}})
    metrics = Attributes(backend, material.attributes).val(max_samples=1, cache_dir=tmp_path, verbose=False)
    assert metrics.accuracy == 1 and metrics.seen == 1
    assert metrics.dataset_counts["source_objects"] == 5
    assert metrics.dataset_counts["excluded_unscored_objects"] == 2
    assert metrics.dataset_counts["prepared_objects"] == 1
    assert metrics.dataset_metadata["preparation"]["selected_groups"] == ["material"]
    assert len(http[1]) == 2


@pytest.mark.parametrize("settings", [{"split": "train"}, {"groups": ("color",)}, {"groups": []}, {"classes": ("chair",)}])
def test_invalid_named_requests_do_not_download(tmp_path, http, settings):
    with pytest.raises(ValueError):
        prepare_dataset("ovad", cache_dir=tmp_path, **settings)
    assert not http[1]


def test_runner_schema_discovery_explicit_config_and_prepare_only(tmp_path, http, monkeypatch, capsys):
    from examples import run_validation

    monkeypatch.setattr(run_validation, "create_backend", lambda args: pytest.fail("Must not load a model"))
    assert run_validation.main(["--task", "attributes", "--prepare-only", "--max-samples", "2", "--cache-dir", str(tmp_path), "--no-verbose"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["attributes"]["material"]["wood"] == "wood/wooden"
    assert "categorical" in report["metadata"]["protocol"]
    monkeypatch.setattr(run_validation, "create_backend", lambda args: Backend())
    assert run_validation.main(["--task", "attributes", "--model", "test", "--max-samples", "2", "--cache-dir", str(tmp_path), "--no-verbose"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["selected_groups"] == ["cleanliness", "material"]
    assert report["metrics"]["attribute_decisions"] == 2
    assert report["per_attribute"][0]["attribute"] == "cleanliness"
    config = tmp_path / "config.json"
    settings = {"attributes": {"material": {"wood": "custom wood", "metal": "custom metal"}}, "context": "Inspect the crop."}
    config.write_text(json.dumps({"attributes": settings}))
    monkeypatch.setattr(run_validation, "create_backend", lambda args: Backend({"attributes": {"material": "metal"}}))
    assert run_validation.main(["--task", "attributes", "--model", "test", "--config", str(config), "--max-samples", "1", "--cache-dir", str(tmp_path), "--no-verbose"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["task_config"] == settings
    assert report["selected_groups"] == ["material"]


def test_runner_local_automatic_schema(tmp_path, monkeypatch, capsys):
    from examples import run_validation

    path, images = write_local(tmp_path)
    monkeypatch.setattr(run_validation, "create_backend", lambda args: Backend())
    assert run_validation.main(["--task", "attributes", "--data", str(path), "--images", str(images), "--model", "test", "--no-verbose"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["task_config"]["attributes"] == OVADDataset(path, images).attributes
    assert report["dataset_counts"]["source_objects"] == 5


@pytest.mark.parametrize("source", ["named", "local"])
def test_runner_discovers_explicit_null_attribute_schema(tmp_path, http, monkeypatch, capsys, source):
    from examples import run_validation

    backend = Backend()
    monkeypatch.setattr(run_validation, "create_backend", lambda args: backend)
    config = tmp_path / "task.json"
    config.write_text(json.dumps({"attributes": {"attributes": None, "context": "Keep this context."}}))
    args = ["--task", "attributes", "--config", str(config), "--model", "test", "--max-samples", "1", "--no-verbose"]
    if source == "named":
        args.extend(["--cache-dir", str(tmp_path)])
    else:
        path, images = write_local(tmp_path)
        args.extend(["--data", str(path), "--images", str(images)])
    assert run_validation.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report["task_config"]["attributes"]) == {"cleanliness", "material"}
    assert report["task_config"]["context"] == "Keep this context."
    assert "Keep this context." in backend.calls[0].instruction


@pytest.mark.parametrize("configured", [False, True])
def test_call_attributes_override_defaults_without_mutation(configured):
    backend = Backend(
        {"attributes": {"window color": "blue", "model_config": "healthy"}},
        {"attributes": {"size": "large"}},
        {"attributes": {"material": "wood"}},
    )
    defaults = {"material": {"wood": "Constructor description"}} if configured else None
    task = Attributes(backend, defaults, context="Inspect the cropped subject.")
    state = (deepcopy(task.attributes), task.context, task._spec, task._instruction)
    image = ImagePayload(np.zeros((20, 30, 3), np.uint8))
    override = {
        "window color": {"blue": "A custom blue description", "red": ""},
        "model_config": ["healthy", "damaged"],
    }
    pipeline = Pipeline([task])
    contract = pipeline.validate(strict=True)
    tuple_type = tuple[ImagePayload, AttributeChoices]
    assert contract.input_type == (ImagePayload | tuple_type if configured else tuple_type)
    assert contract.output_type is AttributePredictionResult
    assert pipeline((image, override)).attributes == {"window color": "blue", "model_config": "healthy"}
    assert task((image, {"size": ["small", "large"]})).attributes == {"size": "large"}
    assert "custom blue description" in backend.calls[0].instruction
    assert "Inspect the cropped subject." in backend.calls[0].instruction
    assert set(backend.calls[0].response_spec.schema["$defs"]["AttributeValues"]["required"]) == set(override)
    assert (task.attributes, task.context, task._spec, task._instruction) == state
    if configured:
        assert pipeline(image).attributes == {"material": "wood"}
        assert "Constructor description" in backend.calls[-1].instruction
        assert backend.calls[-1].response_spec is state[2]
    else:
        with pytest.raises(ValueError, match="at construction or when called"):
            task(image)
        assert len(backend.calls) == 2


@pytest.mark.parametrize("override", [None, {}, [], "material", {"": ["wood"]}, {"material": []}, {"material": "wood"}, {"material": ["wood", "wood"]}, {"material": {"wood": 1}}])
def test_invalid_call_attributes_rejected_before_inference(override):
    backend = Backend()
    task = Attributes(backend, {"material": ["wood"]})
    with pytest.raises(ValueError):
        task((ImagePayload(np.zeros((2, 2, 3), np.uint8)), override))
    assert not backend.calls
    assert task.attributes == {"material": {"wood": ""}}


@pytest.mark.parametrize("length", [0, 1, 3])
def test_attributes_rejects_malformed_call_tuples(length):
    backend = Backend()
    with pytest.raises(ValueError, match="tuple"):
        Attributes(backend)(tuple(None for _ in range(length)))
    assert not backend.calls


def test_call_specific_response_validation_and_failure_preserve_defaults():
    backend = Backend({"attributes": {"material": "wood"}})
    task = Attributes(backend, {"material": ["wood"]})
    image = ImagePayload(np.zeros((2, 2, 3), np.uint8))
    original_spec, original_instruction = task._spec, task._instruction
    with pytest.raises(InvalidModelResponseError):
        task((image, {"material": ["metal"]}))
    assert task._spec is original_spec and task._instruction == original_instruction
    assert task.attributes == {"material": {"wood": ""}}
    assert task(image).attributes == {"material": "wood"}
    assert backend.calls[0].response_spec.schema != backend.calls[1].response_spec.schema


def test_dynamic_attributes_pipeline_composition():
    backend = Backend({"attributes": {"material": "wood"}})
    task = Attributes(backend)

    def supply_attributes(image: ImagePayload) -> tuple[ImagePayload, dict[str, list[str]]]:
        return image, {"material": ["wood"]}

    pipeline = Pipeline([supply_attributes, task])
    assert pipeline.validate(strict=True).input_type is ImagePayload
    assert pipeline(ImagePayload(np.zeros((2, 2, 3), np.uint8))).attributes == {"material": "wood"}

    def supply_image(image: ImagePayload) -> ImagePayload:
        return image

    with pytest.raises(PipelineValidationError):
        Pipeline([supply_image, task]).validate(strict=True)


@pytest.mark.parametrize("source", ["named", "local", "loader"])
def test_val_discovers_schema_and_keeps_fixed_call_overrides(tmp_path, http, source):
    inputs = []

    class RecordingAttributes(Attributes):
        def __call__(self, value):
            inputs.append(value)
            return super().__call__(value)

    backend = Backend()
    task = RecordingAttributes(backend, context="Describe this crop.")
    if source == "named":
        settings = {"data": "ovad", "cache_dir": tmp_path}
    else:
        path, images = write_local(tmp_path)
        settings = {"data": path, "images": images}
        if source == "loader":
            settings = {"dataloader": DataLoader(OVADDataset(path, images))}
    validator = AttributesValidator(task, max_samples=2, verbose=False, **settings)
    first, second = validator(), validator()
    assert first is not second and first.seen == second.seen == 2
    assert first.attribute_decisions == second.attribute_decisions == 2
    assert first.accuracy == second.accuracy == .5
    assert task.attributes is task._spec is task._instruction is None
    assert task.context == "Describe this crop."
    assert all(isinstance(value, tuple) and len(value) == 2 for value in inputs)
    assert all(value[1] == inputs[0][1] for value in inputs)
    assert set(inputs[0][1]) == {"cleanliness", "material"}
    assert all(call.response_spec.schema == backend.calls[0].response_spec.schema for call in backend.calls)
    assert len({call.instruction for call in backend.calls}) == 1
    assert len(http[1]) == (2 if source == "named" else 0)
    metrics = task.val(max_samples=1, verbose=False, **settings)
    assert task.metrics is metrics and metrics.seen == 1
    assert task.attributes is None


@pytest.mark.parametrize("error", [ModelExecutionError("offline"), RuntimeError("programming error")])
def test_discovered_validation_schema_is_not_stored_after_errors(tmp_path, error):
    path, images = write_local(tmp_path)
    task = Attributes(Backend(error), context="Original context.")
    if isinstance(error, ModelExecutionError):
        metrics = task.val(data=path, images=images, verbose=False)
        assert metrics.failure_rate == 1 and metrics.accuracy == 0
    else:
        with pytest.raises(RuntimeError, match="programming error"):
            task.val(data=path, images=images, verbose=False)
    assert task.attributes is task._spec is task._instruction is None
    assert task.context == "Original context."
