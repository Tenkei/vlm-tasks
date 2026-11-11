"""Integration with ordinary HF datasets, independent of benchmark preparation."""
from __future__ import annotations

from types import SimpleNamespace

import datasets
import numpy as np
from PIL import Image
import pytest

from ml_pipes.vlm import Answer, Attributes, Classify, Describe, Detect, Ground, ModelResponse
from ml_pipes.vlm.validation import ClassificationValidator
from ml_pipes.vlm.validation._caption_tokenizer import CaptionTokenizer


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (120, 60), (200, 30, 10)).save(path)
    return str(path)


class Backend:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, request):
        self.calls.append(request)
        return ModelResponse(self.response)


def classification(image):
    return datasets.Dataset.from_dict(
        {"img": [image, image], "label": [1, 0]},
        features=datasets.Features({"img": datasets.Image(), "label": datasets.ClassLabel(names=["cat", "dog", "fox"])}),
    ).rename_column("img", "image")


def test_hf_default_selection_keeps_vocabulary_and_rgb_pixels(image):
    dataset = classification(image).select([0])
    assert isinstance(dataset[0]["image"], Image.Image) and dataset.format["type"] is None
    backend = Backend({"labels": ["dog"]})
    task = Classify(backend, {"animal": "Original description"}, context="Use the whole image.")
    metrics = task.val(dataset=dataset, verbose=False)
    assert metrics.top1 == 1 and metrics.names == ("cat", "dog", "fox")
    assert metrics.summary()[2]["support"] == 0
    request = backend.calls[0]
    assert request.image.color_space == "RGB" and request.image.size == (120, 60)
    np.testing.assert_array_equal(request.image.array[0, 0], [200, 30, 10])
    assert "Use the whole image." in request.instruction
    assert "Original description" not in request.instruction
    assert task.labels == {"animal": "Original description"}


@pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
def test_hf_default_image_modes_are_converted_only_at_task_input(mode):
    source = Image.new("RGB", (5, 3), (200, 30, 10)).convert(mode)
    dataset = datasets.Dataset.from_dict({"image": [source], "label": [0]}, features=datasets.Features({
        "image": datasets.Image(), "label": datasets.ClassLabel(names=["cat"]),
    }))
    backend = Backend({"labels": ["cat"]})
    assert Classify(backend).val(dataset=dataset, verbose=False).top1 == 1
    np.testing.assert_array_equal(backend.calls[0].image.array, np.asarray(source.convert("RGB")))
    assert dataset[0]["image"].mode == mode and dataset.format["type"] is None


def test_hf_streaming_needs_no_length_or_indexing(image, tmp_path, monkeypatch):
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf-cache")
    monkeypatch.setattr(datasets.config, "TORCH_AVAILABLE", False)
    visited = []
    def examples():
        for label in [1, 0, 2]:
            visited.append(label)
            yield {"image": image, "label": label}
    dataset = datasets.IterableDataset.from_generator(examples, features=datasets.Features({
        "image": datasets.Image(), "label": datasets.ClassLabel(names=["cat", "dog", "fox"]),
    }))
    backend = Backend({"labels": ["dog"]})
    validator = ClassificationValidator(Classify(backend), dataset, max_samples=1, verbose=False)
    first, second = validator(), validator()
    assert first is not second and first.top1 == second.top1 == 1
    assert visited == [1, 1] and len(backend.calls) == 2


def test_custom_dataset_only_needs_iteration_and_features(image):
    class CustomDataset:
        features = {"image": object(), "label": SimpleNamespace(names=["cat", "dog"])}
        def __iter__(self):
            yield {"image": np.asarray(Image.open(image).convert("RGB")), "label": 1}
    metrics = Classify(Backend({"labels": ["dog"]})).val(dataset=CustomDataset(), verbose=False)
    assert metrics.top1 == 1 and metrics.names == ("cat", "dog")


def test_hf_image_bytes_use_default_decoding(image):
    with open(image, "rb") as stream:
        content = stream.read()
    dataset = datasets.Dataset.from_dict({"image": [{"path": None, "bytes": content}], "label": [0]},
        features=datasets.Features({"image": datasets.Image(), "label": datasets.ClassLabel(names=["cat"])}))
    assert Classify(Backend({"labels": ["cat"]})).val(dataset=dataset, verbose=False).top1 == 1


def test_hf_detection_adapts_records_and_xywh_before_validation(image):
    category = datasets.ClassLabel(names=["cat", "dog", "fox"])
    box = datasets.List(datasets.Value("float64"), length=4)
    dataset = datasets.Dataset.from_dict(
        {"image": [image], "objects": [[{"bbox": [30, 15, 60, 30], "category": 1}]]},
        features=datasets.Features({"image": datasets.Image(), "objects": datasets.List({"bbox": box, "category": category})}),
    )

    def adapt(row):
        return {"objects": {
            "bbox": [[x, y, x + w, y + h] for x, y, w, h in [obj["bbox"] for obj in row["objects"]]],
            "category": [obj["category"] for obj in row["objects"]],
        }}

    features = datasets.Features({"image": datasets.Image(), "objects": {
        "bbox": datasets.List(box), "category": datasets.List(category),
    }})
    dataset = dataset.map(adapt, features=features)
    backend = Backend({"detections": [{"label": "dog", "box": {"x1": .25, "y1": .25, "x2": .75, "y2": .75}}]})
    metrics = Detect(backend, ["animal"]).val(dataset=dataset, verbose=False)
    assert metrics.f1 == 1 and metrics.names == ("cat", "dog", "fox")


def test_hf_grounding_rows_use_expression_and_pixel_xyxy(image):
    dataset = datasets.Dataset.from_dict({"image": [image], "text": ["the dog"], "label": [0], "bbox": [[30, 15, 90, 45]]},
        features=datasets.Features({"image": datasets.Image(), "text": datasets.Value("string"),
            "label": datasets.ClassLabel(names=["dog"]), "bbox": datasets.List(datasets.Value("float64"), length=4)}))
    backend = Backend({"status": "found", "box": {"x1": .25, "y1": .25, "x2": .75, "y2": .75}})
    metrics = Ground(backend, "stored default").val(dataset=dataset, verbose=False)
    assert metrics.mean_iou == 1 and 'Target: "the dog"' in backend.calls[0].instruction


def test_hf_attributes_uses_group_features_and_null_targets(image):
    dataset = datasets.Dataset.from_dict({"image": [image], "attributes": [{"material": 1, "size": None}]},
        features=datasets.Features({"image": datasets.Image(), "attributes": {
            "material": datasets.ClassLabel(names=["wood", "metal"]), "size": datasets.ClassLabel(names=["big", "small"]),
        }}))
    backend = Backend({"attributes": {"material": "metal", "size": "small"}})
    task = Attributes(backend)
    metrics = task.val(dataset=dataset, verbose=False)
    assert metrics.accuracy == 1 and metrics.attribute_decisions == 1
    assert task.attributes is None
    assert "big" in str(backend.calls[0].response_spec.schema)


def test_hf_vqa_answer_records_preserve_all_votes(image):
    with open(image, "rb") as stream:
        content = stream.read()
    dataset = datasets.Dataset.from_dict({
        "image": [{"path": None, "bytes": content}] * 2,
        "image_id": [7, 7], "question_id": [100, 101],
        "question": ["How many dogs?", "How many animals?"],
        "answers": [[{"answer": "2", "answer_id": i} for i in range(10)]] * 2,
        "answer_type": ["number"] * 2, "question_type": ["how many"] * 2,
    }).cast_column("image", datasets.Image()).rename_column("question_id", "sample_id")
    dataset = dataset.map(lambda row: {"answers": [vote["answer"] for vote in row["answers"]]})
    backend = Backend({"status": "answered", "answer": "two"})
    metrics = Answer(backend).val(dataset=dataset, verbose=False)
    assert metrics.vqa_accuracy == 1 and metrics.seen == 2 and metrics.unique_images == 1


def test_hf_caption_dataset_preserves_multiple_references(image, monkeypatch):
    def prepare(self):
        self.metadata = {"tokenizer_version": self.version}
        return self
    monkeypatch.setattr(CaptionTokenizer, "prepare", prepare)
    monkeypatch.setattr(CaptionTokenizer, "tokenize", lambda self, texts: [text.lower() for text in texts])
    dataset = datasets.Dataset.from_dict({"image": [image], "captions": [["a cat sits on a mat", "a cat on a mat"]]}).cast_column("image", datasets.Image())
    metrics = Describe(Backend({"description": "a cat sits on a mat"})).val(dataset=dataset, verbose=False)
    assert metrics.reference_captions == 2 and metrics.bleu_4 == pytest.approx(1)


@pytest.mark.parametrize("bad_data", ["cifar10", {"test": []}, datasets.DatasetDict()])
def test_validation_rejects_unloaded_sources_and_split_collections(bad_data):
    backend = Backend({"labels": ["cat"]})
    with pytest.raises(TypeError, match="loaded dataset"):
        Classify(backend).val(dataset=bad_data, verbose=False)
    assert not backend.calls


def test_validation_rejects_split_before_loading(monkeypatch):
    from ml_pipes.vlm import validation
    monkeypatch.setattr(validation, "load_cifar10", lambda **kwargs: pytest.fail("Unexpected loading"))
    with pytest.raises(TypeError, match="split"):
        Classify(Backend({"labels": ["cat"]})).val(split="test")


@pytest.mark.parametrize("label", [-1, 3, True, 0.5])
def test_custom_dataset_invalid_ids_fail_before_inference(image, label):
    class Dataset:
        features = {"image": object(), "label": SimpleNamespace(names=["cat", "dog"])}
        def __iter__(self):
            yield {"image": Image.open(image), "label": label}
    backend = Backend({"labels": ["cat"]})
    with pytest.raises(ValueError, match="label ID"):
        Classify(backend).val(dataset=Dataset(), verbose=False)
    assert not backend.calls


def test_missing_class_vocabulary_is_not_inferred_from_samples(image):
    dataset = datasets.Dataset.from_dict({"image": [image], "label": [0]}).cast_column("image", datasets.Image())
    backend = Backend({"labels": ["cat"]})
    with pytest.raises(ValueError, match="ClassLabel"):
        Classify(backend).val(dataset=dataset, verbose=False)
    assert not backend.calls


def test_custom_validator_receives_the_public_sample_schema(image):
    from types import MappingProxyType

    row = MappingProxyType({"image": np.asarray(Image.open(image).convert("RGB")), "label": 1})
    observed = []

    class Dataset:
        features = {"image": object(), "label": SimpleNamespace(names=["cat", "dog"])}

        def __iter__(self):
            yield row

    class Validator(ClassificationValidator):
        def predict_sample(self, sample):
            observed.append("predict")
            assert sample is row
            return super().predict_sample(sample)

        def update_metrics(self, prediction, sample):
            observed.append("score")
            assert sample is row
            super().update_metrics(prediction, sample)

    task = Classify(Backend({"labels": ["dog"]}))
    assert task.val(dataset=Dataset(), validator=Validator, verbose=False).top1 == 1
    assert observed == ["predict", "score"]


@pytest.mark.parametrize("kind", ["path", "mapping", "float", "chw", "gray", "alpha", "empty", "empty_pil"])
def test_validator_rejects_invalid_dataset_images_before_inference(image, kind):
    pixels = np.asarray(Image.open(image))
    value = {
        "path": image, "mapping": {"path": image, "bytes": None},
        "float": pixels.astype(np.float32), "chw": pixels.transpose(2, 0, 1),
        "gray": pixels[:, :, 0], "alpha": np.zeros((60, 120, 4), np.uint8),
        "empty": pixels[:0], "empty_pil": Image.new("RGB", (0, 10)),
    }[kind]

    class Dataset:
        features = {"image": object(), "label": SimpleNamespace(names=["cat"])}

        def __iter__(self):
            yield {"image": value, "label": 0}

    backend = Backend({"labels": ["cat"]})
    with pytest.raises(ValueError, match="HWC uint8 RGB"):
        Classify(backend).val(dataset=Dataset(), verbose=False)
    assert not backend.calls


@pytest.mark.parametrize("objects", [
    [{"bbox": [30, 15, 90, 45], "category": 0}],
    {"bbox": [[30, 15, 20, 45]], "category": [0]},
    {"bbox": [[30, 15, float("nan"), 45]], "category": [0]},
    {"bbox": [[30, 15, 90, 45]], "category": []},
])
def test_detection_requires_canonical_objects_before_inference(image, objects):
    class Dataset:
        features = {"image": object(), "objects": {
            "category": SimpleNamespace(feature=SimpleNamespace(names=["cat"])),
        }}

        def __iter__(self):
            yield {"image": np.asarray(Image.open(image)), "objects": objects}

    backend = Backend({"detections": []})
    with pytest.raises(ValueError):
        Detect(backend).val(dataset=Dataset(), verbose=False)
    assert not backend.calls


def test_answer_records_require_dataset_adaptation_before_inference(image):
    dataset = datasets.Dataset.from_dict({
        "image": [image], "question": ["How many?"],
        "answers": [[{"answer": "2"}] * 10],
        "answer_type": ["number"], "question_type": ["how many"],
    }).cast_column("image", datasets.Image())
    backend = Backend({"status": "answered", "answer": "two"})
    with pytest.raises(ValueError, match="string"):
        Answer(backend).val(dataset=dataset, verbose=False)
    assert not backend.calls
