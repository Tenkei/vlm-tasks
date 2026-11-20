"""Check benchmark accounting across parsing failures without model downloads."""
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.usage import RequestUsage
import pytest

from benchmarks.src.evaluation import RecordingBackend, RecordingModel, ReplayBackend, usage_summary
from ml_pipes.vlm import (
    Classify, InvalidModelResponseError, PydanticAIBackend, ResponseConstraintError,
    ResponseSchemaError, ResponseSyntaxError, TransformersModel,
)


def test_provenance_finds_repository_sources_after_layout_change(monkeypatch):
    from benchmarks.src import artifacts

    root = Path(__file__).resolve().parents[1]

    def git_revision(command, *, cwd, text):
        assert command == ["git", "rev-parse", "HEAD"]
        assert cwd == root and text is True
        return "revision\n"

    monkeypatch.setattr(artifacts.subprocess, "check_output", git_revision)
    provenance = artifacts.provenance()
    assert artifacts.ROOT == root
    assert provenance["git_head"] == "revision"
    assert {"benchmarks/__init__.py", "benchmarks/src/artifacts.py", "benchmarks/src/run_transformers.py",
            "benchmarks/src/task_registry.py", "src/ml_pipes/vlm/ops.py"} <= provenance["source_sha256"].keys()
    for filename, expected in provenance["source_sha256"].items():
        assert "source_snapshots" not in filename and "/runs/" not in filename
        assert artifacts.digest((root / filename).read_bytes()) == expected


@pytest.mark.parametrize(("raw_text", "error_type"), [
    ("not JSON", ResponseSyntaxError),
    ("{}", ResponseSchemaError),
    ('{"labels": ["unknown"]}', ResponseConstraintError),
])
def test_benchmark_retains_usage_and_raw_text_for_invalid_predictions(monkeypatch, raw_text, error_type):
    responses = iter([raw_text, '{"labels": ["cat"]}'])

    def generate(self, messages, info):
        return ModelResponse(parts=[TextPart(next(responses))],
                             usage=RequestUsage(input_tokens=11, output_tokens=7))

    monkeypatch.setattr(TransformersModel, "_generate", generate)
    weights = SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=False))
    weights.eval = lambda: weights
    model = RecordingModel(weights, None, model_name="test")
    stream = StringIO()
    backend = RecordingBackend(PydanticAIBackend(model, retries=0), stream)
    task = Classify(backend, labels=["cat", "dog"])
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    with pytest.raises(error_type):
        task(image)
    assert task(image).labels == ("cat",)
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert records[0]["error"]["type"] == error_type.__name__
    assert records[0]["generations"][0]["raw_text"] == raw_text
    assert records[1]["prediction"] == {"labels": ["cat"]}
    usage = usage_summary(model.generations, backend.index)
    assert usage["input_tokens"] == 22
    assert usage["output_tokens"] == 14
    assert usage["measured_requests"] == 2
    assert usage["unmeasured_requests"] == 0
    assert usage["estimated_cost_usd"] is None
    replay = Classify(ReplayBackend(records), labels=["cat", "dog"])
    with pytest.raises(error_type):
        replay(image)
    assert replay(image).labels == ("cat",)
    # Original benchmark journals keep their historical base exception type.
    records[0]["error"]["type"] = "InvalidModelResponseError"
    with pytest.raises(InvalidModelResponseError):
        Classify(ReplayBackend(records), labels=["cat", "dog"])(image)
    if error_type is ResponseConstraintError:
        records[0]["error"]["type"] = "ResponseContentError"
        with pytest.raises(ResponseConstraintError):
            Classify(ReplayBackend(records), labels=["cat", "dog"])(image)


def test_benchmark_marks_generation_errors_unmeasured(monkeypatch):
    def generate(self, messages, info):
        raise RuntimeError("generation interrupted")

    monkeypatch.setattr(TransformersModel, "_generate", generate)
    weights = SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=False))
    weights.eval = lambda: weights
    model = RecordingModel(weights, None, model_name="test")
    stream = StringIO()
    backend = RecordingBackend(PydanticAIBackend(model, retries=0), stream)
    from ml_pipes.vlm import ModelExecutionError
    with pytest.raises(ModelExecutionError):
        Classify(backend, labels=["cat"])(np.zeros((2, 3, 3), dtype=np.uint8))
    assert json.loads(stream.getvalue())["generations"] == []
    assert usage_summary(model.generations, backend.index)["unmeasured_requests"] == 1


@pytest.mark.parametrize("succeeds", [True, False])
def test_local_benchmark_counts_every_correction_request(monkeypatch, succeeds):
    outputs = iter(["not JSON", '{"labels":["unknown"]}',
                    '{"labels":["cat"]}' if succeeds else "not JSON"])

    def generate(self, messages, info):
        return ModelResponse(parts=[TextPart(next(outputs))],
                             usage=RequestUsage(input_tokens=11, output_tokens=7))

    monkeypatch.setattr(TransformersModel, "_generate", generate)
    weights = SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=False))
    weights.eval = lambda: weights
    model = RecordingModel(weights, None, model_name="test")
    stream = StringIO()
    backend = RecordingBackend(PydanticAIBackend(model, retries=2), stream)
    task = Classify(backend, labels=["cat"])
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    if succeeds:
        assert task(image).labels == ("cat",)
    else:
        with pytest.raises(ResponseSyntaxError):
            task(image)
    record = json.loads(stream.getvalue())
    usage = usage_summary(model.generations, backend.attempted_requests)
    assert backend.index == 1
    assert record["attempted_requests"] == usage["attempted_requests"] == 3
    assert len(record["generations"]) == usage["measured_requests"] == 3
    assert usage["unmeasured_requests"] == 0
    assert usage["input_tokens"] == 33 and usage["output_tokens"] == 21


def test_local_benchmark_counts_unmeasured_correction_attempt(monkeypatch):
    from ml_pipes.vlm import ModelExecutionError

    outputs = iter(["not JSON", RuntimeError("generation interrupted")])

    def generate(self, messages, info):
        output = next(outputs)
        if isinstance(output, Exception):
            raise output
        return ModelResponse(parts=[TextPart(output)], usage=RequestUsage(input_tokens=11, output_tokens=7))

    monkeypatch.setattr(TransformersModel, "_generate", generate)
    weights = SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=False))
    weights.eval = lambda: weights
    model = RecordingModel(weights, None, model_name="test")
    backend = RecordingBackend(PydanticAIBackend(model, retries=2), StringIO())
    with pytest.raises(ModelExecutionError):
        Classify(backend, labels=["cat"])(np.zeros((2, 3, 3), dtype=np.uint8))
    usage = usage_summary(model.generations, backend.attempted_requests)
    assert usage["attempted_requests"] == 2
    assert usage["measured_requests"] == usage["unmeasured_requests"] == 1


@pytest.mark.parametrize("value", [-1, True, 1.5, "2"])
def test_benchmark_rejects_invalid_retry_configs(value):
    from benchmarks.src.evaluation import max_retries

    assert max_retries({}) == 0
    assert max_retries({"retries": 2}) == 2
    with pytest.raises(ValueError, match="nonnegative integer"):
        max_retries({"retries": value})


def test_retry_report_normalizes_requests_and_counts_capped_samples_once(tmp_path):
    from benchmarks.src.artifacts import write_json
    from benchmarks.src.generate_report import generate_report

    run = tmp_path / "runs/retry-model"
    write_json(run / "config.json", {"tasks": ["classify"], "max_new_tokens": 4096, "retries": 2})
    write_json(run / "classify.json", {
        "task": "classify", "dataset": "cifar10", "model": "test", "revision": "pinned",
        "device": "cpu", "dtype": "float32", "retries": 2,
        "metrics": {"samples": 2, "failures": 1, "accuracy_top1": .5},
        "usage": {"input_tokens": 60, "output_tokens": 16394, "attempted_requests": 6,
                  "estimated_cost_usd": "0.006"},
        "speed_ms_per_sample": {"inference": 1000}, "elapsed_seconds": 2,
        "dataset_fingerprint_sha256": "data",
    })
    call = {"attempted_requests": 3, "generations": [{"output_tokens": n} for n in (4096, 4096, 5)]}
    (run / "classify.jsonl").write_text((json.dumps(call) + "\n") * 2)
    row, = generate_report(run.parent)
    assert row["usage_requests_per_sample"] == 3
    assert row["responses_at_token_cap"] == 4
    assert row["samples_at_token_cap"] == 2
    assert row["responses_at_token_cap_per_sample"] == 2
    assert row["samples_at_token_cap_per_sample"] == 1
    page = (tmp_path / "README.md").read_text()
    assert "| test (2 retries) |" in page
    assert "50% (100%) | 3 | 30 | 8,197 | 3 | 1 |" in page


@pytest.mark.parametrize("sample_scale", [1, 3])
def test_comparison_normalizes_usage_and_rates_by_evaluated_samples(tmp_path, sample_scale):
    from benchmarks.src.artifacts import write_json
    from benchmarks.src.generate_report import generate_report

    runs = tmp_path / "runs"
    run = runs / "model-run"
    write_json(run / "config.json", {"tasks": ["classify"], "max_new_tokens": 4096})
    write_json(run / "classify.json", {
        "task": "classify", "dataset": "cifar10", "model": "test", "revision": "pinned",
        "device": "cpu", "dtype": "float32",
        "metrics": {"samples": 100 * sample_scale, "failures": 2 * sample_scale, "accuracy_top1": .9},
        "usage": {"input_tokens": 20 * sample_scale, "output_tokens": 4101 * sample_scale,
                  "estimated_cost_usd": str(.02 * sample_scale)},
        "speed_ms_per_sample": {"inference": 1000}, "elapsed_seconds": 100 * sample_scale,
        "dataset_fingerprint_sha256": "data",
    })
    calls = [{"generations": [{"output_tokens": 4096}]}, {"generations": [{"output_tokens": 5}]}]
    (run / "classify.jsonl").write_text("\n".join(json.dumps(call) for call in calls * sample_scale) + "\n")
    row, = generate_report(runs)
    assert row["failures"] == 2 * sample_scale
    assert row["responses_at_token_cap"] == sample_scale
    assert row["max_new_tokens"] == 4096
    assert row["usage_input_tokens"] == 20 * sample_scale
    assert row["usage_input_tokens_per_sample"] == pytest.approx(.2)
    assert row["usage_output_tokens_per_sample"] == pytest.approx(41.01)
    assert row["usage_estimated_cost_usd_per_sample"] == pytest.approx(.0002)
    assert row["usage_estimated_cost_usd_per_1k_samples"] == pytest.approx(.2)
    assert row["failure_rate"] == pytest.approx(.02)
    assert row["responses_at_token_cap_per_sample"] == pytest.approx(.01)
    table = (tmp_path / "README.md").read_text()
    assert "Input tokens / sample | Output tokens / sample | Estimated cost / 1,000 samples | Mean latency (s)" in table
    assert "Failure rate (at cap)" in table
    assert "| At cap |" not in table
    assert row["usage_requests_per_sample"] == 1
    assert f"| {100 * sample_scale} | Top-1 Acc 90% | 2% (1%) | 1 | 0 | 41 | 0.2 | 1 |" in table
    assert "Each row evaluates 100" not in table
    assert (tmp_path / "reports/report-manifest.json").is_file()


def test_capped_rerun_selects_all_and_only_responses_at_limit(tmp_path):
    from benchmarks.src.run_token_budget import selected_calls

    records = [{"index": 0, "generations": [{"output_tokens": 4095}]},
               {"index": 1, "generations": [{"output_tokens": 4096}]},
               {"index": 2, "generations": [], "error": {"type": "ModelExecutionError"}}]
    (tmp_path / "detect.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    result = selected_calls(tmp_path, {"tasks": ["detect"], "max_new_tokens": 4096})
    assert [r["index"] for r in result["detect"]] == [1]


@pytest.mark.parametrize("task_name, operator, geometry, scoring", [
    ("classify", "Classify", None, {}),
    ("detect", "Detect", None, {"match_iou": .7}),
    ("ground_box", "Ground", "box", {}),
    ("ground_point", "Ground", "point", {}),
    ("attributes", "Attributes", None, {}),
    ("answer", "Answer", None, {}),
    ("describe", "Describe", None, {"tokenizer_version": "3.4.1"}),
])
def test_shared_evaluation_preserves_task_variants_and_scoring(monkeypatch, tmp_path, task_name, operator, geometry, scoring):
    from benchmarks.src.evaluation import evaluate
    from ml_pipes import vlm

    calls = []
    task_class = getattr(vlm, operator)

    def validate(task, **options):
        calls.append((task, options))
        return "metrics"

    monkeypatch.setattr(task_class, "val", validate)
    def backend(request):
        raise AssertionError("Validation dispatch must not run inference in this test")

    dataset = object()
    config = {"match_iou": .7, "tokenizer_version": "3.4.1"}
    assert evaluate(task_name, backend, dataset, config, tmp_path) == "metrics"
    (task, options), = calls
    assert type(task) is task_class
    if geometry:
        assert set(task._spec.schema["properties"]) == {"status", geometry}
    assert options == {"dataset": dataset, "cache_dir": tmp_path, **scoring}


@pytest.mark.parametrize("overrides, context", [
    ({}, "Write exactly one sentence of 10 to 15 words describing the main visible "
         "subject and action. Do not add additional sentences."),
    ({"describe": None}, None),
    ({"describe": "Write one detailed paragraph.", "classify": "Unrelated classification context"},
     "Write one detailed paragraph."),
])
def test_benchmark_caption_context_changes_only_instructions(tmp_path, monkeypatch, overrides, context):
    from benchmarks.src.evaluation import evaluate
    from ml_pipes import vlm
    from ml_pipes.vlm.validation._caption_tokenizer import CaptionTokenizer

    def prepare(tokenizer):
        tokenizer.metadata = {"tokenizer_version": tokenizer.version, "java_version": "offline test"}
        return tokenizer

    monkeypatch.setattr(CaptionTokenizer, "prepare", prepare)
    monkeypatch.setattr(CaptionTokenizer, "tokenize",
                        lambda self, texts: [text.lower().replace(".", "").strip() for text in texts])
    dataset = [
        {"image": np.zeros((2, 3, 3), dtype=np.uint8), "image_id": index, "captions": references}
        for index, references in enumerate([
            ["a cat sits on a mat", "a cat on a mat"],
            ["two dogs play outside", "two dogs play outside"],
            ["an airplane flies above water"],
        ])
    ]

    def backend(calls):
        def predict(request):
            calls.append(request)
            return vlm.ModelResponse({"description": "a cat sits on a mat"})
        return predict

    original_calls, benchmark_calls = [], []
    config = {"tokenizer_version": "3.4.1"}
    baseline = vlm.Describe(backend(original_calls)).val(
        dataset=dataset, cache_dir=tmp_path / "cache", tokenizer_version="3.4.1",
    )
    changed = evaluate("describe", backend(benchmark_calls), dataset, {
        **config, "task_context": overrides,
    }, tmp_path / "cache")
    assert changed.results_dict == baseline.results_dict
    assert changed.dataset_counts == baseline.dataset_counts
    assert len(benchmark_calls) == len(original_calls) == 3
    for before, after in zip(original_calls, benchmark_calls):
        expected = before.instruction
        if context is not None:
            expected = expected.replace("\nProvide a prediction", f"\nContext: {context}\nProvide a prediction")
        assert after.instruction == expected
        assert after.response_spec.schema == before.response_spec.schema
        assert "Unrelated classification context" not in after.instruction
        assert "a cat sits on a mat" not in after.instruction
        assert "two dogs play outside" not in after.instruction


@pytest.mark.parametrize("changed", ["image", "target", "features"])
def test_frozen_data_verification_detects_changes_and_loads_shared_data_once(monkeypatch, tmp_path, changed):
    import datasets
    from PIL import Image

    from benchmarks.src.artifacts import digest, encoded, sample_record
    from benchmarks.src.evaluation import load_prepared

    def dataset(*, color="red", target=0, names=("cat", "dog")):
        return datasets.Dataset.from_dict(
            {"image": [Image.new("RGB", (2, 3), color)], "label": [target]},
            features=datasets.Features({"image": datasets.Image(), "label": datasets.ClassLabel(names=list(names))}),
        )

    frozen = dataset()
    identity = {"features": frozen.features.to_dict(), "samples": [sample_record(frozen[0], 0)]}
    manifest = {"datasets": {"refcocog": {"fingerprint_sha256": digest(encoded(identity).encode())}}}
    loads = []

    def load(path):
        loads.append(path)
        return frozen

    monkeypatch.setattr(datasets, "load_from_disk", load)
    assert load_prepared(tmp_path, manifest, ["ground_box", "ground_point"]) == {"refcocog": frozen}
    assert loads == [str(tmp_path / "refcocog")]
    frozen = dataset(**{"image": {"color": "blue"}, "target": {"target": 1},
                        "features": {"names": ("dog", "cat")}}[changed])
    with pytest.raises(ValueError, match="Prepared samples changed: refcocog"):
        load_prepared(tmp_path, manifest, ["ground_box"])


def test_report_repeats_the_same_observations_in_model_and_task_views(tmp_path):
    import re

    from benchmarks.src.artifacts import digest, write_json
    from benchmarks.src.generate_report import generate_report
    from benchmarks.src.task_registry import TASKS

    runs = tmp_path / "runs"
    hashes, profiles = {}, []
    comparisons = [{"task": name, "title": spec.chart_title, "metric": spec.primary_metric,
                    "figure": f"task-comparisons/{name}", "value_format": "native" if name == "describe" else "percent",
                    "models": []} for name, spec in TASKS.items()]
    labels = {"first": "Kimi K3.1", "second": "Qwen3-VL-2B"}
    for index, run in enumerate(("first", "second")):
        write_json(runs / run / "config.json", {"tasks": list(TASKS), "max_new_tokens": 4096})
        profiles.append({"run": run, "model": run, "figure": f"model-profiles/{run}"})
        for comparison, (name, spec) in zip(comparisons, TASKS.items()):
            score = .5 + index * .1
            comparison["models"].append({"run": run, "label": labels[run], "score": score})
            path = runs / run / f"{name}.json"
            write_json(path, {
                "task": name, "dataset": spec.dataset, "model": run, "revision": "pinned",
                "device": "cpu", "dtype": "float32",
                "metrics": {"samples": 2, "failures": 0, **{metric: score for metric, _ in spec.metrics}},
                "usage": {"input_tokens": 20, "output_tokens": 10, "estimated_cost_usd": None},
                "speed_ms_per_sample": {"inference": 1000}, "elapsed_seconds": 2,
                "dataset_fingerprint_sha256": "frozen",
            })
            hashes[f"{run}/{name}.json"] = digest(path.read_bytes())
    write_json(tmp_path / "reports/charts/manifest.json", {
        "models": profiles, "task_comparisons": comparisons, "reports_sha256": hashes,
    })

    assert len(generate_report(runs)) == 14
    page = (tmp_path / "README.md").read_text()
    detail_rows = [line for line in page.splitlines() if re.match(r"\| (first|second) \|", line)]
    assert len(detail_rows) == 28
    assert len(set(detail_rows)) == 14
    assert all(detail_rows.count(row) == 2 for row in detail_rows)
    assert page.count("<details>") == page.count("</details>") == 10
    assert page.count("<img ") == 18  # Nine overview thumbnails and nine detail charts.
    anchors = set(re.findall(r'<a id="([^"]+)"', page)) | {
        "kimi-k31", "qwen3-vl-2b", "classification", "detection", "grounding",
        "box", "point", "attributes", "vqa", "captioning",
    }
    assert set(re.findall(r'href="#([^"]+)"', page)) <= anchors
    assert "[Kimi K3.1](#kimi-k31)" in page
    assert "[Qwen3-VL-2B](#qwen3-vl-2b)" in page
    assert "### Kimi K3.1\n\n<details>\n<summary>View chart, metrics and usage</summary>" in page
    assert "### Classification\n\n<details>\n<summary>View chart, metrics and usage</summary>" in page
    assert "### Grounding\n\n<details>\n<summary>View variations</summary>" in page
    assert "#### Box\n\n<details>\n<summary>View chart, metrics and usage</summary>" in page
    assert "#### Point\n\n<details>\n<summary>View chart, metrics and usage</summary>" in page
    assert page.rfind("## Table field guide") > page.rfind("</details>")
    metadata = json.loads((tmp_path / "reports/report-manifest.json").read_text())
    assert "benchmarks/src/task_registry.py" in metadata["source_sha256"]
    assert (tmp_path / "reports/metrics.csv").is_file()
    assert not (runs / "metrics.csv").exists()
    assert "reports/charts/model-profiles/first.png" in page
    assert "reports/charts/task-comparisons/classify.png" in page
    assert "runs/first/classify.json" in page


def test_token_budget_report_accepts_a_relocated_baseline(tmp_path):
    from benchmarks.src.artifacts import write_json
    from benchmarks.src.report_token_budget import main

    baseline = tmp_path / "runs/model"
    experiment = tmp_path / "experiments/token-budget/model"
    write_json(baseline / "config.json", {"max_new_tokens": 4096})
    write_json(experiment / "selection.json", {
        "baseline": "original/path/no-longer-present", "baseline_limit": 4096,
        "rerun_limit": 8192, "samples": {},
    })
    assert main(["--experiment", str(experiment), "--baseline", str(baseline),
                 "--interrupted-requests", "0"]) == 0
    assert json.loads((experiment / "status.json").read_text())["completed_samples"] == 0
    assert "# Token budget experiment (partial)" in (experiment / "README.md").read_text()


@pytest.fixture
def comparison_runs(tmp_path):
    from benchmarks.src.artifacts import write_json
    from benchmarks.src.task_registry import TASKS

    runs = tmp_path / "runs"
    for index, (run, model) in enumerate([
        ("small", "Qwen/Qwen3-VL-2B-Instruct"),
        ("gemma", "google.gemma-3-27b-it"),
        ("large", "qwen.qwen3-vl-235b-a22b"),
    ]):
        write_json(runs / run / "config.json", {"tasks": list(TASKS), "max_new_tokens": 4096})
        for axis, (task, spec) in enumerate(TASKS.items()):
            write_json(runs / run / f"{task}.json", {
                "task": task, "dataset": spec.dataset, "model": model, "revision": "pinned",
                "device": "cpu", "dtype": "float32",
                "metrics": {"samples": 2, "failures": 1 if index == 0 else 0,
                            spec.primary_metric: .1 + index * .2 + axis * .01},
                "usage": {"input_tokens": 20, "output_tokens": 4101,
                          "estimated_cost_usd": None if index == 0 else "0.004"},
                "speed_ms_per_sample": {"inference": 1000}, "elapsed_seconds": 2,
                "dataset_fingerprint_sha256": "frozen",
            })
            (runs / run / f"{task}.jsonl").write_text(
                json.dumps({"generations": [{"output_tokens": 4096}]}) + "\n" +
                json.dumps({"generations": [{"output_tokens": 5}]}) + "\n")
    return runs


def test_report_views_share_newest_run_order_using_recorded_dates(comparison_runs, tmp_path):
    pytest.importorskip("matplotlib")
    import csv

    from benchmarks.src.artifacts import write_json
    from benchmarks.src.generate_charts import build_profiles, generate_charts
    from benchmarks.src.generate_report import load_task_reports, ordered_runs

    # UTC normalization, latest-task fallback and completion metadata all matter.
    for name, date in [("small", "2026-11-19T15:00:00+09:00"),
                       ("gemma", "2026-11-19T07:00:00Z"),
                       ("large", "2026-11-19T05:00:00+00:00")]:
        for path in (comparison_runs / name).glob("*.json"):
            if path.name == "config.json":
                continue
            report = json.loads(path.read_text())
            report["completed_at"] = date
            write_json(path, report)
    write_json(comparison_runs / "large/completed.json", {"completed_at": "2026-11-20T00:00:00Z"})
    latest = comparison_runs / "small/describe.json"
    report = json.loads(latest.read_text())
    report["completed_at"] = "2026-11-19T08:00:00Z"
    write_json(latest, report)
    expected = ["large", "small", "gemma"]
    reports = load_task_reports(comparison_runs)
    assert ordered_runs(reports) == expected
    assert [profile["run"] for profile in build_profiles(reports)] == expected
    generate_charts(comparison_runs)
    charts = json.loads((tmp_path / "reports/charts/manifest.json").read_text())
    assert [model["run"] for model in charts["models"]] == expected
    assert all([model["run"] for model in task["models"]] == expected for task in charts["task_comparisons"])
    rows = list(csv.DictReader((tmp_path / "reports/metrics.csv").open()))
    for task in {row["task"] for row in rows}:
        assert [row["run"] for row in rows if row["task"] == task] == expected
    page = (tmp_path / "README.md").read_text()
    assert page.index("### Qwen3-VL-235B") < page.index("### Qwen3-VL-2B") < page.index("### Gemma 3 27B")
    # Changing completion metadata alone must not reuse charts in the old order.
    from benchmarks.src.generate_report import generate_report
    write_json(comparison_runs / "large/completed.json", {"completed_at": "2026-11-18T00:00:00Z"})
    generate_report(comparison_runs)
    assert "reports/charts/model-profiles/" not in (tmp_path / "README.md").read_text()


def test_undated_runs_sort_last_with_stable_ties(tmp_path):
    from benchmarks.src.generate_report import ordered_runs

    reports = [(tmp_path / run / "classify.json", {"completed_at": date}) for run, date in [
        ("unknown", None), ("zulu", "2026-11-19T00:00:00Z"), ("alpha", "2026-11-19T09:00:00+09:00")]]
    assert ordered_runs(reports) == ["alpha", "zulu", "unknown"]


def test_local_comparison_selects_only_two_models_and_preserves_overview(comparison_runs, tmp_path):
    pytest.importorskip("matplotlib")
    import re
    from urllib.parse import unquote

    from benchmarks.src.artifacts import digest
    from benchmarks.src.generate_comparison import main
    from benchmarks.src.task_registry import TASKS

    overview = tmp_path / "README.md"
    overview.write_text("Existing benchmark overview")
    # Unrelated incomplete runs must not prevent comparing the requested pair.
    (comparison_runs / "gemma/describe.json").unlink()
    before = {path: path.read_bytes() for path in comparison_runs.rglob("*") if path.is_file()}
    output = tmp_path / "local comparison"
    assert main(["qwen3-vl-235b", "Qwen3-VL-2B-Instruct", "--runs", str(comparison_runs),
                 "--output", str(output)]) == 0
    page = (output / "README.md").read_text()
    assert page.startswith("# Qwen3-VL-235B vs Qwen3-VL-2B")
    assert "Gemma" not in page
    rows = [line for line in page.splitlines() if line.startswith(("| Qwen/", "| qwen."))]
    assert len(rows) == 2 * len(TASKS)
    assert "| 2 | Top-1 Acc 10% | 50% (50%) | 1 | 10 | 2,050 | — | 1 |" in page
    assert "| 2 | Top-1 Acc 50% | 0% (50%) | 1 | 10 | 2,050 | 2 | 1 |" in page
    assert overview.read_text() == "Existing benchmark overview"
    assert all(path.read_bytes() == content for path, content in before.items())
    assert not (tmp_path / "reports/charts").exists()
    for target in re.findall(r"\]\(([^)]+)\)", page):
        assert (output / unquote(target.split("#", 1)[0])).exists(), target
    manifest = json.loads((output / "manifest.json").read_text())
    assert [model["run"] for model in manifest["models"]] == ["large", "small"]
    for model, index in zip(manifest["models"], (2, 0)):
        assert model["scores"] == pytest.approx([.1 + index * .2 + axis * .01 for axis in range(len(TASKS))])
    assert {key.split("/")[0] for key in manifest["inputs_sha256"]} == {"large", "small"}
    for name, expected in manifest["inputs_sha256"].items():
        assert digest((comparison_runs / name).read_bytes()) == expected
    for name, expected in manifest["artifacts_sha256"].items():
        assert digest((output / name).read_bytes()) == expected
    svg = (output / "comparison.svg").read_text()
    assert "Qwen3-VL-2B" in svg and "Qwen3-VL-235B" in svg


@pytest.mark.parametrize("names, message", [
    (["Unknown", "Qwen3-VL-235B"], "Unknown model"),
    (["Qwen3-VL-2B", "Qwen/Qwen3-VL-2B-Instruct"], "two different model runs"),
    (["Qwen3-VL-2B"], "exactly two model names"),
    (["Qwen3-VL-2B", "Qwen3-VL-235B", "Gemma 3 27B"], "exactly two model names"),
])
def test_local_comparison_rejects_invalid_selections_before_writing(comparison_runs, tmp_path, names, message):
    from benchmarks.src.generate_comparison import generate_comparison

    output = tmp_path / "comparison"
    with pytest.raises(ValueError, match=message):
        generate_comparison(comparison_runs, names, output)
    assert not output.exists()


def test_local_comparison_requires_run_id_when_model_is_ambiguous():
    from benchmarks.src.generate_comparison import select_runs

    catalog = {"small-first": "Qwen/Qwen3-VL-2B-Instruct", "small-second": "Qwen/Qwen3-VL-2B-Instruct",
               "large": "qwen.qwen3-vl-235b-a22b"}
    with pytest.raises(ValueError, match="matches multiple runs.*small-first, small-second"):
        select_runs(catalog, ["Qwen3-VL-2B", "Qwen3-VL-235B"])
    assert select_runs(catalog, ["small-second", "large"]) == ["small-second", "large"]


def test_local_comparison_lists_available_names_without_generating_reports(comparison_runs, tmp_path, capsys):
    from benchmarks.src.generate_comparison import main

    assert main(["--list-models", "--runs", str(comparison_runs)]) == 0
    listing = capsys.readouterr().out
    assert "Qwen3-VL-2B  [small]" in listing and "Qwen3-VL-235B  [large]" in listing
    assert not (tmp_path / "reports").exists()


@pytest.mark.parametrize("settings, suffix", [
    ({"retries": 2}, "2 retries"),
    ({"output_mode": "native"}, "JSON Schema"),
    ({"output_mode": "native", "retries": 2}, "JSON Schema, 2 retries"),
])
def test_configured_profiles_can_be_selected_by_their_displayed_names(comparison_runs, tmp_path, settings, suffix):
    pytest.importorskip("matplotlib")
    from benchmarks.src.chart_style import model_color, next_color
    from benchmarks.src.generate_comparison import generate_comparison

    original = comparison_runs / "small"
    retry = comparison_runs / "small-retries"
    retry.mkdir()
    for path in original.iterdir():
        if path.suffix == ".json":
            value = json.loads(path.read_text())
            value.update(settings)
            (retry / path.name).write_text(json.dumps(value))
        else:
            (retry / path.name).write_bytes(path.read_bytes())
    output = generate_comparison(comparison_runs, ["Qwen3-VL-2B", f"Qwen3-VL-2B ({suffix})"], tmp_path / "comparison").parent
    manifest = json.loads((output / "manifest.json").read_text())
    assert [model["run"] for model in manifest["models"]] == ["small", "small-retries"]
    page = (output / "README.md").read_text()
    assert page.startswith(f"# Qwen3-VL-2B vs Qwen3-VL-2B ({suffix})")
    assert f"Qwen/Qwen3-VL-2B-Instruct ({suffix})" in page
    base = model_color("Qwen/Qwen3-VL-2B-Instruct")
    assert [model["base_color"] for model in manifest["models"]] == [base, base]
    assert [model["color"] for model in manifest["models"]] == [base, next_color(base)]
    svg = (output / "comparison.svg").read_text()
    assert all(f"stroke: {model['color']}" in svg for model in manifest["models"])


def test_checksum_palette_preserves_qwen_colors_and_accepts_new_models():
    from benchmarks.src.chart_style import COLORS, model_color, next_color

    assert len(COLORS) == len(set(COLORS)) == 10
    assert model_color("Qwen/Qwen3-VL-2B-Instruct") == "#2563ad"
    assert model_color("qwen.qwen3-vl-235b-a22b") == "#864b9a"
    assert model_color("a-new-vlm-without-a-registered-color") in COLORS
    assert next_color(COLORS[-1]) == COLORS[0]


def test_comparison_colors_match_overview_after_adding_an_unrelated_model(comparison_runs, tmp_path):
    pytest.importorskip("matplotlib")
    from benchmarks.src.artifacts import write_json
    from benchmarks.src.generate_charts import generate_charts
    from benchmarks.src.generate_comparison import generate_comparison

    generate_charts(comparison_runs)
    manifest_path = tmp_path / "reports/charts/manifest.json"
    before = manifest_path.read_bytes()
    charts = json.loads(before)
    colors = {model["run"]: model["color"] for model in charts["models"]}
    assert all(model["color"] == colors[model["run"]]
               for task in charts["task_comparisons"] for model in task["models"])

    # A new, incomplete run would shift a positional palette assignment.
    new = comparison_runs / "new-model"
    write_json(new / "config.json", json.loads((comparison_runs / "small/config.json").read_text()))
    sample = json.loads((comparison_runs / "small/classify.json").read_text())
    write_json(new / "classify.json", {**sample, "model": "a-new-vlm"})
    for index, names in enumerate((["large", "small"], ["small", "large"])):
        output = generate_comparison(comparison_runs, names, tmp_path / f"comparison-{index}").parent
        manifest = json.loads((output / "manifest.json").read_text())
        assert {model["run"]: model["color"] for model in manifest["models"]} == {
            run: colors[run] for run in names
        }
        assert "benchmarks/src/chart_style.py" in manifest["source_sha256"]
        svg = (output / "comparison.svg").read_text()
        assert all(f"stroke: {colors[run]}" in svg for run in names)
    assert manifest_path.read_bytes() == before


def test_standard_charts_do_not_generate_pairwise_reports(comparison_runs, tmp_path):
    pytest.importorskip("matplotlib")
    from benchmarks.src.generate_charts import generate_charts

    generate_charts(comparison_runs)
    manifest = json.loads((tmp_path / "reports/charts/manifest.json").read_text())
    assert len(manifest["models"]) == 3
    assert "model_comparisons" not in manifest
    assert not (tmp_path / "reports/charts/model-comparisons").exists()
    assert not (tmp_path / "reports/model-comparisons").exists()
    assert "## Model comparisons" not in (tmp_path / "README.md").read_text()
