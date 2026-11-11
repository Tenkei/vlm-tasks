"""COCO caption adaptation, independent metric goldens, and validation wiring."""
from __future__ import annotations

from ml_pipes.vlm.validation import load_coco_captions2017

from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import pytest

from ml_pipes.vlm import Describe, DescriptionResult, ModelExecutionError, ModelResponse
from ml_pipes.vlm.validation import DescribeMetrics, DescribeValidator, ClassificationMetrics, DetectionMetrics, GroundingMetrics
from ml_pipes.vlm.validation import default_datasets as constants
from ml_pipes.vlm.validation._caption import score_captions
from ml_pipes.vlm.validation._caption_tokenizer import CaptionTokenizer


FIXTURE = json.loads((Path(__file__).parent / "fixtures/coco_caption_scores.json").read_text())


def caption_data():
    return {"images": [
        {"id": i, "file_name": f"{i:012d}.jpg", "width": 20, "height": 12} for i in (13, 7, 24)
    ], "annotations": [
        {"id": i, "image_id": image_id, "caption": text} for i, image_id, text in [
            (0, 7, "two dogs play outside"), (1, 13, "a cat sits on a mat"),
            (2, 24, "an airplane flies above water"), (3, 13, "a cat on a mat"),
            (4, 7, "two dogs play outside"), (5, 13, "a cat sits on a mat"),
        ]
    ]}


def jpeg():
    ok, pixels = cv2.imencode(".jpg", np.zeros((12, 20, 3), np.uint8))
    assert ok
    return pixels.tobytes()


def write_local(root, data=None):
    data = caption_data() if data is None else data
    path, images = root / "captions_val2017.json", root / "images"
    images.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    for image_id in (13, 7, 24):
        (images / f"{image_id:012d}.jpg").write_bytes(jpeg())
    return path, images


class Backend:
    def __init__(self, *responses):
        self.responses = responses or ({"description": "a cat sits on a mat"},)
        self.calls = []

    def __call__(self, request):
        response = self.responses[min(len(self.calls), len(self.responses) - 1)]
        self.calls.append(request)
        if isinstance(response, Exception):
            raise response
        return ModelResponse(response)


@pytest.fixture
def tokenizer(monkeypatch):
    calls = []

    def prepare(self):
        self.metadata = {"tokenizer_version": self.version, "java_version": "offline test"}
        return self

    def tokenize(self, texts):
        calls.append(list(texts))
        return [text.lower().replace(".", "").strip() for text in texts]

    monkeypatch.setattr(CaptionTokenizer, "prepare", prepare)
    monkeypatch.setattr(CaptionTokenizer, "tokenize", tokenize)
    return calls


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["name"])
def test_local_metrics_match_independent_pinned_reference(case):
    assert score_captions(case["candidates"], case["references"]) == pytest.approx(case["scores"], abs=1e-13)


def test_empty_tokenized_references_are_safe_and_failed_candidates_score_zero():
    scores = score_captions(["", "a cat"], [[""], ["a cat"]])
    assert scores["rouge_l"] == .5
    assert all(np.isfinite(value) for value in scores.values())
    assert score_captions([""], [[""]])["rouge_l"] == 0


def test_metrics_finalize_once_and_use_all_reference_images_in_idf(tokenizer):
    case = next(case for case in FIXTURE["cases"] if case["name"] == "failure_retains_reference")
    metrics = DescribeMetrics()
    for image_id, (candidate, refs) in enumerate(zip(case["candidates"], case["references"])):
        metrics.update(refs, DescriptionResult(candidate) if candidate else None, image_id=image_id, im_file=f"{image_id}.jpg")
    assert not tokenizer and metrics.names == ()
    metrics.finalize()
    assert len(tokenizer) == 1 and tokenizer[0][1] == ""
    assert metrics.cider > 0  # Dropping failures or using a per-image IDF changes this.
    assert {key: metrics.results_dict[key] for key in case["scores"]} == pytest.approx(case["scores"])
    assert len(metrics.summary()) == 1 and metrics.summary()[0]["support"] == 3
    assert metrics.reference_captions == 4 and metrics.unique_images == 3
    assert metrics.dataset_metadata["idf_reference_scope"] == "selected_images"
    assert metrics.dataset_metadata["idf_reference_images"] == 3
    assert metrics.dataset_metadata["degenerate_idf"] is False
    assert len(tokenizer) == 1


def test_one_image_idf_is_zero_with_diagnostic(tokenizer):
    metrics = DescribeMetrics()
    metrics.update(["a cat sits on a mat"], DescriptionResult("a cat sits on a mat"), image_id=3, im_file="3.jpg")
    assert metrics.cider == 0 and metrics.bleu_4 == pytest.approx(1)
    assert metrics.dataset_metadata["degenerate_idf"] is True
    assert "zero IDF" in metrics.dataset_metadata["idf_diagnostic"]


@pytest.mark.parametrize("prediction", ["a cat", DescriptionResult(""), DescriptionResult(None)])
def test_invalid_description_outputs_do_not_change_metrics(tokenizer, prediction):
    metrics = DescribeMetrics()
    with pytest.raises((ValueError, TypeError)):
        metrics.update(["a cat"], prediction, image_id=1, im_file="a.jpg")
    assert metrics.seen == 0


@pytest.mark.parametrize("metric", [ClassificationMetrics, DetectionMetrics, GroundingMetrics])
@pytest.mark.parametrize("names", [None, [], ()])
def test_existing_metrics_still_reject_empty_vocabularies(metric, names):
    with pytest.raises(ValueError, match="nonempty"):
        metric(names)


@pytest.mark.parametrize("split", ["train", "val"])
def test_dataset_groups_all_references_before_limit_in_source_image_order(tmp_path, split):
    annotations, images = write_local(tmp_path)
    (images / f"{24:012d}.jpg").unlink()
    dataset = load_coco_captions2017(annotations, images, split=split, max_samples=2)
    assert [sample["image_id"] for sample in dataset] == [13, 7]
    assert dataset[0]["captions"] == ["a cat sits on a mat", "a cat on a mat", "a cat sits on a mat"]
    assert dataset[1]["captions"] == ["two dogs play outside"] * 2
    assert dataset[0]["image"].size == (20, 12) and "label" not in dataset.features
    with pytest.raises(FileNotFoundError, match="image not found"):
        load_coco_captions2017(annotations, images)
    (images / f"{13:012d}.jpg").write_bytes(b"corrupt")
    from datasets.exceptions import DatasetGenerationError
    with pytest.raises(DatasetGenerationError) as error:
        load_coco_captions2017(annotations, images, max_samples=1)
    assert isinstance(error.value.__cause__, OSError)


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d["images"].append(deepcopy(d["images"][0])), "Duplicate.*images ID"),
    (lambda d: d["annotations"].append(deepcopy(d["annotations"][0])), "Duplicate.*annotations ID"),
    (lambda d: d["annotations"][0].update(image_id=999), "Unknown.*image ID"),
    (lambda d: d["annotations"][0].update(image_id=True), "Unknown.*image ID"),
    (lambda d: d["annotations"][0].update(caption=" "), "nonempty strings"),
    (lambda d: d["images"][2].update(width=0), "dimensions"),
    (lambda d: d["images"][2].update(file_name="../outside.jpg"), "filename"),
    (lambda d: d["images"][2].update(file_name="/outside.jpg"), "filename"),
    (lambda d: d["images"][2].update(file_name="000000000013.jpg"), "Duplicate.*filename"),
    (lambda d: d["annotations"].pop(2), "nonempty sequence"),
])
def test_dataset_rejects_invalid_joins_and_metadata_even_beyond_limit(tmp_path, mutate, match):
    data = caption_data()
    mutate(data)
    annotations, images = write_local(tmp_path, data)
    with pytest.raises(ValueError, match=match):
        load_coco_captions2017(annotations, images, max_samples=1)


def test_dataset_checks_dimensions_on_decode_and_rejects_symlink_escape(tmp_path):
    data = caption_data()
    data["images"][0]["width"] += 1
    annotations, images = write_local(tmp_path, data)
    from datasets.exceptions import DatasetGenerationError
    with pytest.raises(DatasetGenerationError) as error:
        load_coco_captions2017(annotations, images, max_samples=1)
    assert isinstance(error.value.__cause__, ValueError)
    assert 'dimensions differ' in str(error.value.__cause__)
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(jpeg())
    image = images / f"{13:012d}.jpg"
    image.unlink()
    image.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        load_coco_captions2017(annotations, images, max_samples=1)


def test_validator_preserves_context_calls_once_per_image_and_keeps_failures(tmp_path, tokenizer):
    annotations, images = write_local(tmp_path)
    backend = Backend({"description": "a cat sits on a mat"}, ModelExecutionError("offline"), {"description": " "})
    task = Describe(backend, context="Describe visible objects briefly.")
    metrics = task.val(dataset=load_coco_captions2017(annotations, images), verbose=False)
    assert task.metrics is metrics and isinstance(metrics, DescribeMetrics)
    assert metrics.seen == 3 and len(backend.calls) == 3
    assert len(metrics.failures) == 2 and metrics.failures[0]["sample_id"] == 7
    assert tokenizer[0][:3] == ["a cat sits on a mat", "", ""]
    assert metrics.unique_images == 3 and metrics.reference_captions == 6
    assert metrics.dataset_counts["evaluated_captions"] == 6
    assert metrics.dataset_counts["evaluated_images"] == 3
    assert metrics.speed["metrics"] > 0 and metrics.elapsed_seconds > 0
    for request in backend.calls:
        assert "Describe visible objects briefly." in request.instruction
        assert "a cat sits on a mat" not in request.instruction
        assert "two dogs play outside" not in request.instruction
    assert metrics.summary()[0]["support"] == 3


def test_runs_and_reused_validator_start_fresh(tmp_path, tokenizer):
    annotations, images = write_local(tmp_path)
    task = Describe(Backend())
    first = task.val(dataset=load_coco_captions2017(annotations, images), max_samples=1, verbose=False)
    second = task.val(dataset=load_coco_captions2017(annotations, images), max_samples=2, verbose=False)
    assert first is not second and first.seen == 1 and second.seen == 2
    validator = DescribeValidator(task, load_coco_captions2017(annotations, images), max_samples=1, verbose=False)
    assert validator() is not validator()


def test_custom_loader_and_validator_need_no_vocabulary(tmp_path, tokenizer):
    annotations, images = write_local(tmp_path)
    dataset = load_coco_captions2017(annotations, images)
    task = Describe(Backend())
    metrics = task.val(dataset=dataset.select([1, 0]), max_samples=1, verbose=False)
    assert metrics.seen == 1 and metrics.reference_captions == 2
    assert DescribeValidator(task, dataset, verbose=False)().seen == 3

    class Dataset:
        features = None
        def __iter__(self):
            sample = dataset[0]
            sample.pop("image_id")
            yield sample

    assert task.val(dataset=Dataset(), verbose=False).unique_images == 1


def test_custom_loader_bad_targets_stop_before_inference(tmp_path, tokenizer):
    annotations, images = write_local(tmp_path)
    dataset = load_coco_captions2017(annotations, images)

    class Loader:
        dataset = object()

        def __iter__(self):
            yield {**dataset[0], "captions": []}

    backend = Backend()
    with pytest.raises(ValueError, match="captions"):
        Describe(backend).val(dataset=Loader(), verbose=False)
    assert not backend.calls
    with pytest.raises(RuntimeError, match="programming bug"):
        Describe(Backend(RuntimeError("programming bug"))).val(dataset=load_coco_captions2017(annotations, images), verbose=False)


@pytest.mark.parametrize("duplicate_id", [True, False])
def test_custom_loader_duplicate_images_stop_before_repeated_inference(tmp_path, tokenizer, duplicate_id):
    annotations, images = write_local(tmp_path)
    dataset = load_coco_captions2017(annotations, images)

    class Loader:
        dataset = object()

        def __iter__(self):
            yield dataset[0]
            yield {**dataset[1], "image_id": 13} if duplicate_id else {**dataset[0], "image_id": 999}

    backend = Backend()
    with pytest.raises(ValueError, match="Duplicate caption image"):
        Describe(backend).val(dataset=Loader(), verbose=False)
    assert len(backend.calls) == 1


@pytest.mark.parametrize("settings", [
    {"split": "test"}, {"tokenizer_version": "latest"},
    {"images": "local"}, {"dataset": "local"}, {"max_samples": 0},
])
def test_invalid_settings_fail_without_model_execution(settings, tokenizer):
    backend = Backend()
    settings = {"dataset": [], **settings}
    with pytest.raises((ValueError, TypeError)):
        Describe(backend).val(**settings)
    assert not backend.calls


@pytest.fixture
def http(hf_downloads):
    return hf_downloads


def caption_source(http):
    archive = BytesIO()
    with ZipFile(archive, "w") as stream:
        for split in ("train", "val"):
            stream.writestr(f"annotations/captions_{split}2017.json", json.dumps(caption_data()))
        stream.writestr("../../outside", "must not be extracted")
    http[0][constants.CAPTIONS_URL] = archive.getvalue()
    for split in ("train", "val"):
        for image_id in (13, 7, 24):
            http[0][constants.CAPTIONS_IMAGE_URL.format(split=split) + f"{image_id:012d}.jpg"] = jpeg()


@pytest.mark.parametrize("split", ["train", "val"])
def test_preparation_fetches_selected_images_keeps_references_and_repairs_cache(tmp_path, http, split):
    caption_source(http)
    first = load_coco_captions2017(split=split, max_samples=1, cache_dir=tmp_path)
    assert first[0]['captions'] == ['a cat sits on a mat', 'a cat on a mat', 'a cat sits on a mat']
    assert len(http[1]) == 2 and not (tmp_path / 'outside').exists()
    again = load_coco_captions2017(split=split, max_samples=1, cache_dir=tmp_path)
    assert first._fingerprint == again._fingerprint and len(http[1]) == 2
    assert first.info.dataset_name == 'coco_captions2017'
    assert not list(tmp_path.rglob('captions_*.json'))


def test_preparation_reuses_archive_for_larger_subsets_and_other_split(tmp_path, http, monkeypatch):
    caption_source(http)
    monkeypatch.setattr(CaptionTokenizer, 'prepare', lambda self: pytest.fail('Data preparation must not require Java'))
    load_coco_captions2017(max_samples=1, cache_dir=tmp_path)
    larger = load_coco_captions2017(max_samples=2, cache_dir=tmp_path)
    train = load_coco_captions2017(split='train', max_samples=1, cache_dir=tmp_path)
    assert len(http[1]) == 4
    assert len(larger) == 2 and str(train.split) == 'train'


def test_automatic_validation_reports_evaluated_counts(tmp_path, http, tokenizer):
    caption_source(http)
    metrics = Describe(Backend()).val(max_samples=1, cache_dir=tmp_path, verbose=False)
    assert metrics.dataset_counts == {'evaluated_samples': 1, 'evaluated_images': 1, 'evaluated_captions': 3}
    assert metrics.dataset_metadata['dataset_name'] == 'coco_captions2017'


@pytest.mark.parametrize("automatic", [False, True])
def test_cli_describe_reports_corpus_summary_and_context(tmp_path, http, tokenizer, monkeypatch, capsys, automatic):
    from examples import run_validation

    caption_source(http)
    backend = Backend()
    monkeypatch.setattr(run_validation, "create_backend", lambda args: backend)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"describe": {"context": "A concise image description."}}))
    args = ["--task", "describe", "--model", "test", "--max-samples", "1", "--config", str(config), "--tokenizer-version", "4.6.0"]
    if not automatic:
        annotations, images = write_local(tmp_path)
        args += ["--data", str(annotations), "--images", str(images)]
    assert run_validation.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["metrics"]["cider"] == 0 and report["summary"][0]["support"] == 1
    assert "per_class" not in report
    assert report["dataset_metadata"]["tokenizer_version"] == "4.6.0"
    assert "A concise image description." in backend.calls[0].instruction


def test_cli_records_failures_with_nonzero_exit(tmp_path, tokenizer, monkeypatch, capsys):
    from examples import run_validation

    annotations, images = write_local(tmp_path)
    monkeypatch.setattr(run_validation, "create_backend", lambda args: Backend(ModelExecutionError("failed")))
    assert run_validation.main(["--task", "describe", "--model", "test", "--max-samples", "1",
                                "--data", str(annotations), "--images", str(images)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["metrics"]["failures"] == 1
    assert report["metrics"]["rouge_l"] == 0


@pytest.mark.parametrize("args", [
    ["--task", "answer", "--tokenizer-version", "3.4.1"],
    ["--task", "describe", "--tokenizer-version", "latest"],
    ["--task", "describe", "--tokenizer-version", ""],
    ["--task", "describe", "--split", "test"],
    ["--task", "describe", "--class-map", "map.json"],
    ["--task", "describe", "--data", "local.json"],
])
def test_cli_rejects_invalid_settings_before_model_loading(args, monkeypatch):
    from examples import run_validation

    monkeypatch.setattr(run_validation, "create_backend", lambda args: pytest.fail("Model must not load"))
    with pytest.raises(SystemExit) as error:
        run_validation.main(["--model", "test", *args])
    assert error.value.code == 2
