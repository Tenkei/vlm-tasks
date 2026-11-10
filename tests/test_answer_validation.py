"""Offline VQA v2 adaptation, consensus scoring, preparation, and execution."""
from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import json
from zipfile import ZipFile

import cv2
import numpy as np
import pytest

from ml_pipes.vlm import Answer, AnswerResult, ModelExecutionError, ModelResponse
from ml_pipes.vlm.validation import (
    AnswerMetrics, AnswerValidator, DataLoader, DEFAULT_DATASETS, VQAv2Dataset, prepare_dataset,
)
from ml_pipes.vlm.validation import prepare as preparation
from ml_pipes.vlm.validation._vqa import normalize_answer, score_answer


def vqa_data():
    questions = {"questions": [
        {"question_id": 91, "image_id": 13, "question": "How many cats?"},
        {"question_id": 7, "image_id": 13, "question": "Is it raining?"},
        {"question_id": 440, "image_id": 24, "question": "What color is the chair?"},
    ]}
    annotations = {"annotations": [
        {"question_id": question_id, "image_id": image_id,
         "answer_type": answer_type, "question_type": question_type,
         "multiple_choice_answer": "MULTIPLE_CHOICE_SECRET",
         "answers": [{"answer_id": i + 1, "answer": answer, "answer_confidence": "yes"} for i in range(10)]}
        for question_id, image_id, answer_type, question_type, answer in [
            (440, 24, "other", "what color", "blue"),
            (91, 13, "number", "how many", "2"),
            (7, 13, "yes/no", "is it", "no"),
        ]
    ]}
    return questions, annotations


def write_local(root, *, split="val", data=None):
    questions, targets = vqa_data() if data is None else data
    annotations, images = root / "annotations", root / f"{split}2014"
    annotations.mkdir(parents=True, exist_ok=True)
    images.mkdir(exist_ok=True)
    for image_id in (13, 24):
        assert cv2.imwrite(str(images / f"COCO_{split}2014_{image_id:012d}.jpg"), np.zeros((12, 20, 3), np.uint8))
    (annotations / f"v2_OpenEnded_mscoco_{split}2014_questions.json").write_text(json.dumps(questions))
    (annotations / f"v2_mscoco_{split}2014_annotations.json").write_text(json.dumps(targets))
    return annotations, images


class Backend:
    def __init__(self, *responses):
        self.responses = responses or ({"status": "answered", "answer": "two"},)
        self.calls = []

    def __call__(self, request):
        response = self.responses[min(len(self.calls), len(self.responses) - 1)]
        self.calls.append(request)
        if isinstance(response, Exception):
            raise response
        return ModelResponse(response)


# Expected strings obtained using process_punctuation then process_digit_article
# from lmms-eval revision c7eaef373f4b1bd8a6b5f3e85c3391962c30068c.
# In particular, the VQA v2 wrapper does not call word_tokenize/__call__.
@pytest.mark.parametrize("text,expected", [
    (" Two!\n", "2"), ("THE cat", "cat"), ("dont", "don't"),
    ("Im", "im"), ("Ive", "ive"), ("somebody'd", "somebodyd"),
    ("a 1,000.25 cats!", "1000.25 cats"), ("red-blue", "red blue"),
    ("red - blue", "red blue"), ("red-blue 1,000", "redblue 1000"),
    ("3.14", "3.14"), (".5", ".5"), ("1. 2.", "1 2"), ("a/an/the", ""),
    ("John's", "john's"), ("2:30", "2:30"),
    ("The answer is blue.", "answer is blue"), ("yes\t\nno", "yes no"),
    ("none", "0"), ("twenty one", "twenty 1"),
])
def test_normalization_matches_pinned_vqav2_processor(text, expected):
    assert normalize_answer(text) == expected


@pytest.mark.parametrize("votes,expected", list(enumerate([0, .3, .6, .9] + [1] * 7)))
def test_consensus_keeps_duplicate_votes_and_holds_out_one_index(votes, expected):
    references = ["red"] * votes + ["blue"] * (10 - votes)
    assert score_answer("red", references) == pytest.approx(expected)
    assert score_answer("red", references[::-1]) == pytest.approx(expected)
    assert references == ["red"] * votes + ["blue"] * (10 - votes)


def test_both_sides_are_always_normalized_without_answer_extraction():
    references = ["Two."] * 10
    assert score_answer("2", references) == 1
    assert score_answer("Two", ["2"] * 10) == 1
    assert score_answer("The answer is blue.", ["blue"] * 10) == 0
    assert references == ["Two."] * 10


def test_metrics_weight_questions_and_report_both_groupings():
    metrics = AnswerMetrics()
    for answers, prediction, kind, question_type, image in [
        (["no"] * 10, AnswerResult("answered", "NO!"), "yes/no", "is it", "a.jpg"),
        (["2"] * 2 + ["3"] * 8, AnswerResult("answered", "two"), "number", "how many", "a.jpg"),
        (["0"] * 10, AnswerResult("unanswerable", None), "number", "how many", "b.jpg"),
    ]:
        metrics.update(answers, prediction, answer_type=kind, question_type=question_type, im_file=image)
    assert metrics.vqa_accuracy == pytest.approx(1.6 / 3)
    assert metrics.results_dict == {
        "samples": 3, "failures": 0, "failure_rate": 0, "unique_images": 2,
        "vqa_accuracy": pytest.approx(1.6 / 3),
    }
    assert metrics.status_counts == {"answered": 2, "unanswerable": 1}
    assert metrics.summary() == [
        {"group_by": "answer_type", "group": "yes/no", "support": 1, "vqa_accuracy": 1},
        {"group_by": "answer_type", "group": "number", "support": 2, "vqa_accuracy": pytest.approx(.3)},
        {"group_by": "answer_type", "group": "other", "support": 0, "vqa_accuracy": None},
        {"group_by": "question_type", "group": "how many", "support": 2, "vqa_accuracy": pytest.approx(.3)},
        {"group_by": "question_type", "group": "is it", "support": 1, "vqa_accuracy": 1},
    ]
    assert metrics.dataset_metadata["normalization"] == "lmms-eval VQA v2 (always normalized)"


@pytest.mark.parametrize("prediction", [
    "blue", AnswerResult("answered", ""), AnswerResult("answered", None),
    AnswerResult("unanswerable", "no"), AnswerResult("missing", None),
])
def test_metrics_reject_invalid_results_without_changing_counts(prediction):
    metrics = AnswerMetrics()
    with pytest.raises((ValueError, TypeError)):
        metrics.update(["blue"] * 10, prediction, answer_type="other", question_type="what color", im_file="a.jpg")
    assert metrics.seen == 0
    assert metrics.status_counts == {"answered": 0, "unanswerable": 0}


@pytest.mark.parametrize("split", ["train", "val"])
def test_local_dataset_joins_by_id_and_limits_questions_before_image_io(tmp_path, split):
    annotations, images = write_local(tmp_path, split=split)
    (images / f"COCO_{split}2014_{24:012d}.jpg").unlink()
    dataset = VQAv2Dataset(annotations, images, split=split, max_samples=2)
    assert [sample["sample_id"] for sample in dataset.samples] == [91, 7]
    assert dataset.counts == {"source_questions": 3, "source_images": 2, "selected_questions": 2, "selected_images": 1}
    assert dataset[0]["answers"] == ("2",) * 10
    assert dataset[1]["answers"] == ("no",) * 10
    assert dataset[0]["img"].size == (20, 12)
    assert "multiple_choice_answer" not in dataset[0]
    assert dataset.metadata["split"] == split
    with pytest.raises(FileNotFoundError, match="image not found"):
        VQAv2Dataset(annotations, images, split=split)
    (images / f"COCO_{split}2014_{13:012d}.jpg").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Could not read image"):
        dataset[0]


@pytest.mark.parametrize("change,match", [
    (lambda q, a: q["questions"].append(dict(q["questions"][0])), "Duplicate.*questions"),
    (lambda q, a: a["annotations"].append(dict(a["annotations"][0])), "Duplicate.*annotations"),
    (lambda q, a: a["annotations"].pop(), "matching question IDs"),
    (lambda q, a: a["annotations"][0].update(question_id=999), "matching question IDs"),
    (lambda q, a: a["annotations"][0].update(image_id=999), "image IDs disagree"),
    (lambda q, a: q["questions"][0].update(question_id=True), "integer"),
    (lambda q, a: q["questions"][0].update(image_id=-1), "integer"),
    (lambda q, a: q["questions"][0].update(question=" "), "Invalid.*question"),
    (lambda q, a: a["annotations"][0]["answers"].pop(), "ten string"),
    (lambda q, a: a["annotations"][0]["answers"][0].update(answer=None), "ten string"),
    (lambda q, a: a["annotations"][0].update(answers="blue"), "list of answer"),
    (lambda q, a: a["annotations"][0].update(answer_type="unknown"), "answer_type"),
    (lambda q, a: a["annotations"][0].update(question_type=None), "question_type"),
    (lambda q, a: q.update(questions=[]), "nonempty questions"),
    (lambda q, a: a.update(annotations=[]), "nonempty annotations"),
])
def test_invalid_dataset_stops_before_inference_even_outside_limit(tmp_path, change, match):
    questions, targets = deepcopy(vqa_data())
    change(questions, targets)
    annotations, images = write_local(tmp_path, data=(questions, targets))
    backend = Backend()
    with pytest.raises(ValueError, match=match):
        Answer(backend).val(data=annotations, images=images, max_samples=1, verbose=False)
    assert not backend.calls


def test_validator_preserves_questions_context_schema_and_gold_isolation(tmp_path):
    questions, targets = vqa_data()
    for target in targets["annotations"]:
        target["question_type"] = "QUESTION_TYPE_SECRET"
        for vote in target["answers"]:
            vote["answer"] = "GOLD_ANSWER_SECRET"
    annotations, images = write_local(tmp_path, data=(questions, targets))
    backend = Backend()
    task = Answer(backend, "DEFAULT_QUESTION", context="Assess visible evidence.")
    original_spec = task._spec
    metrics = task.val(data=annotations, images=images, verbose=False)
    assert isinstance(metrics, AnswerMetrics) and task.metrics is metrics
    assert task.question == "DEFAULT_QUESTION" and task.context == "Assess visible evidence."
    assert task._spec is original_spec
    for call, query in zip(backend.calls, questions["questions"], strict=True):
        assert "Question: " + json.dumps(query["question"]) in call.instruction
        assert "Assess visible evidence." in call.instruction
        assert not any(secret in call.instruction for secret in (
            "DEFAULT_QUESTION", "GOLD_ANSWER_SECRET", "QUESTION_TYPE_SECRET", "MULTIPLE_CHOICE_SECRET",
        ))
        assert call.response_spec is original_spec
    assert metrics.seen == 3 and metrics.unique_images == 2
    assert set(metrics.speed) == {"load", "inference", "metrics"}
    assert metrics.elapsed_seconds > 0
    assert not hasattr(metrics, "predictions")


def test_failures_and_abstentions_keep_denominators_and_statuses(tmp_path):
    annotations, images = write_local(tmp_path)
    backend = Backend(
        {"status": "unanswerable", "answer": None},
        ModelExecutionError("offline"),
        {"status": "answered", "answer": " "},
    )
    metrics = Answer(backend).val(data=annotations, images=images, verbose=False)
    assert metrics.vqa_accuracy == 0 and metrics.seen == 3
    assert metrics.failure_rate == pytest.approx(2 / 3)
    assert metrics.status_counts == {"answered": 0, "unanswerable": 1}
    assert [failure["sample_id"] for failure in metrics.failures] == [7, 440]
    assert [failure["type"] for failure in metrics.failures] == ["ModelExecutionError", "InvalidModelResponseError"]
    assert [row["support"] for row in metrics.summary()[:3]] == [1, 1, 1]


def test_negative_answers_zero_and_repeated_runs(tmp_path):
    questions, targets = vqa_data()
    for vote in targets["annotations"][1]["answers"]:
        vote["answer"] = "0"
    annotations, images = write_local(tmp_path, data=(questions, targets))
    backend = Backend({"status": "answered", "answer": "zero"}, {"status": "answered", "answer": "no"})
    task = Answer(backend)
    first = task.val(data=annotations, images=images, max_samples=2, verbose=False)
    assert first.vqa_accuracy == 1 and first.status_counts == {"answered": 2, "unanswerable": 0}
    backend.responses = (ModelExecutionError("offline"),)
    validator = AnswerValidator(task, data=annotations, images=images, max_samples=2, verbose=False)
    for metrics in (validator(), validator()):
        assert metrics is not first and metrics.seen == 2 and metrics.unique_images == 1
        assert metrics.vqa_accuracy == 0 and metrics.failure_rate == 1
        assert metrics.status_counts == {"answered": 0, "unanswerable": 0}


def test_custom_loader_without_class_metadata_and_dataset_hook(tmp_path):
    annotations, images = write_local(tmp_path)
    dataset = VQAv2Dataset(annotations, images)
    loader = DataLoader(dataset, indices=[1, 0])
    backend = Backend({"status": "answered", "answer": "no"}, {"status": "answered", "answer": "two"})
    task = Answer(backend)
    metrics = task.val(dataloader=loader, verbose=False)
    assert metrics.vqa_accuracy == 1
    assert metrics.dataset_counts == {
        "source_questions": 3, "source_images": 2,
        "selected_questions": 3, "selected_images": 2,
        "evaluated_questions": 2, "evaluated_images": 1,
    }
    assert 'Question: "Is it raining?"' in backend.calls[0].instruction

    class CustomValidator(AnswerValidator):
        def build_dataset(self):
            return dataset

    assert task.val(validator=CustomValidator, data="unused", images=images, verbose=False).seen == 3


def test_custom_loader_bad_targets_and_programming_errors_propagate(tmp_path):
    annotations, images = write_local(tmp_path)
    sample = VQAv2Dataset(annotations, images)[0]

    class Loader:
        dataset = object()

        def __iter__(self):
            yield {**sample, "answers": ["2"] * 9}

    backend = Backend()
    task = Answer(backend)
    with pytest.raises(ValueError, match="ten string"):
        task.val(dataloader=Loader(), verbose=False)
    assert not backend.calls
    with pytest.raises(RuntimeError, match="bug"):
        Answer(Backend(RuntimeError("bug"))).val(data=annotations, images=images, verbose=False)


@pytest.mark.parametrize("settings,match", [
    ({"data": "local"}, "requires images"),
    ({"data": "vqav2", "images": "images"}, "supplies images"),
    ({"class_map": {"cat": "cat"}}, "class_map"),
    ({"split": "test"}, "labeled train and val"),
    ({"max_samples": 0}, "max_samples"),
])
def test_invalid_validator_settings_fail_before_inference(settings, match):
    backend = Backend()
    with pytest.raises(ValueError, match=match):
        Answer(backend).val(**settings)
    assert not backend.calls


@pytest.fixture
def http(monkeypatch):
    sources, calls = {}, []

    def open_url(url):
        calls.append(url)
        return BytesIO(sources[url])

    monkeypatch.setattr(preparation, "_open_url", open_url)
    return sources, calls


def add_sources(http, split="val", data=None):
    questions, annotations = vqa_data() if data is None else data
    for kind, filename, content in (
        ("Questions", f"v2_OpenEnded_mscoco_{split}2014_questions.json", questions),
        ("Annotations", f"v2_mscoco_{split}2014_annotations.json", annotations),
    ):
        archive = BytesIO()
        with ZipFile(archive, "w") as source:
            source.writestr(filename, json.dumps(content))
            source.writestr("../../unwanted.txt", "never extract")
        http[0][preparation._VQA_URL.format(kind=kind, split=split.title())] = archive.getvalue()
    ok, image = cv2.imencode(".jpg", np.zeros((12, 20, 3), np.uint8))
    assert ok
    for image_id in (13, 24):
        http[0][preparation._VQA_IMAGE_URL.format(split=split) + f"COCO_{split}2014_{image_id:012d}.jpg"] = image.tobytes()


@pytest.mark.parametrize("split", ["train", "val"])
def test_preparation_downloads_distinct_selected_images_and_reuses_cache(tmp_path, http, split):
    add_sources(http, split)
    first = prepare_dataset("vqav2", split=split, max_samples=2, cache_dir=tmp_path)
    assert DEFAULT_DATASETS["answer"] == "vqav2" and first.names == ()
    dataset = VQAv2Dataset(first.data, first.images, split=split)
    assert [sample["sample_id"] for sample in dataset.samples] == [91, 7]
    assert dataset.samples[0]["answers"] == ("2",) * 10
    assert first.metadata["source_questions"] == 3 and first.metadata["selected_images"] == 1
    assert len(http[1]) == 3  # two archives, one shared image
    assert prepare_dataset("vqav2", split=split, max_samples=2, cache_dir=tmp_path) == first
    assert len(http[1]) == 3
    larger = prepare_dataset("vqav2", split=split, max_samples=3, cache_dir=tmp_path)
    assert larger.images == first.images and larger.data != first.data
    assert len(http[1]) == 4
    assert larger.metadata["selected_questions"] == 3
    assert not list(tmp_path.rglob("unwanted.txt"))


@pytest.mark.parametrize("corrupt", ["image", "questions", "archive", "missing_annotations"])
def test_cache_repairs_selected_images_annotations_and_source_archives(tmp_path, http, corrupt):
    add_sources(http)
    first = prepare_dataset("vqav2", max_samples=2, cache_dir=tmp_path)
    if corrupt == "image":
        path = next(first.images.glob("*.jpg"))
    elif corrupt == "archive":
        path = next((tmp_path / "vqav2" / "sources").glob("*.zip"))
    elif corrupt == "questions":
        path = next(first.data.glob("*questions.json"))
    else:
        path = next(first.data.glob("*annotations.json"))
    if corrupt == "missing_annotations":
        path.unlink()
    else:
        path.write_bytes(b"corrupt")
    assert prepare_dataset("vqav2", max_samples=2, cache_dir=tmp_path) == first
    assert VQAv2Dataset(first.data, first.images)[0]["answers"] == ("2",) * 10
    assert len(http[1]) == (4 if corrupt in {"image", "archive"} else 3)


def test_default_val_prepares_and_retains_source_and_evaluated_counts(tmp_path, http):
    add_sources(http)
    metrics = Answer(Backend()).val(max_samples=1, cache_dir=tmp_path, verbose=False)
    assert metrics.vqa_accuracy == 1 and metrics.seen == 1
    assert metrics.dataset_counts == {
        "source_questions": 3, "source_images": 2,
        "selected_questions": 1, "selected_images": 1,
        "evaluated_questions": 1, "evaluated_images": 1,
    }
    assert metrics.dataset_metadata["preparation"]["source_questions"] == 3
    assert metrics.dataset_metadata["preparation"]["question_ids"] == [91]


def test_invalid_source_joins_fail_before_any_image_download(tmp_path, http):
    questions, annotations = vqa_data()
    annotations["annotations"].pop()
    add_sources(http, data=(questions, annotations))
    with pytest.raises(ValueError, match="matching question IDs"):
        prepare_dataset("vqav2", max_samples=1, cache_dir=tmp_path)
    assert len(http[1]) == 2 and not list(tmp_path.rglob("prepared.json"))


@pytest.mark.parametrize("automatic", [False, True])
def test_cli_answer_reports_local_or_automatic_data(tmp_path, http, monkeypatch, capsys, automatic):
    from examples import run_validation

    backend = Backend()
    monkeypatch.setattr(run_validation, "create_backend", lambda args: backend)
    args = ["--task", "answer", "--max-samples", "1", "--model", "test"]
    if automatic:
        add_sources(http)
        monkeypatch.setenv("VLM_TASKS_DATASETS_DIR", str(tmp_path / "cache"))
    else:
        annotations, images = write_local(tmp_path)
        args += ["--data", str(annotations), "--images", str(images)]
    output = tmp_path / "report.json"
    assert run_validation.main(args + ["--output", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert json.loads(output.read_text()) == report
    assert report["task_config"] == {} and "per_class" not in report
    assert report["metrics"]["vqa_accuracy"] == 1
    assert report["per_type"][1]["support"] == 1
    assert report["status_counts"] == {"answered": 1, "unanswerable": 0}
    assert report["dataset_metadata"]["upstream_revision"] == "c7eaef373f4b1bd8a6b5f3e85c3391962c30068c"
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"answer": {"context": "Use visible evidence."}}))
    assert run_validation.main(args + ["--config", str(config)]) == 0
    assert json.loads(capsys.readouterr().out)["task_config"] == {"context": "Use visible evidence."}
    assert "Use visible evidence." in backend.calls[-1].instruction
    monkeypatch.setattr(run_validation, "create_backend", lambda args: Backend(ModelExecutionError("offline")))
    assert run_validation.main(args) == 1


@pytest.mark.parametrize("extra", [
    ["--class-map", "unused"], ["--match-iou", ".5"], ["--dataset", "refcoco"],
    ["--split", "test"], ["--data", "local"], ["--images", "images"], ["--data", "cifar10"],
])
def test_cli_rejects_invalid_answer_settings_before_backend_setup(monkeypatch, extra):
    from examples import run_validation

    def unexpected(args):
        pytest.fail("backend must not be loaded for invalid settings")

    monkeypatch.setattr(run_validation, "create_backend", unexpected)
    with pytest.raises(SystemExit) as exc:
        run_validation.main(["--task", "answer", "--model", "test", *extra])
    assert exc.value.code == 2
