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
    assert f"| {100 * sample_scale} | Top-1 Acc 90% | 2% (1%) | 0 | 41 | 0.2 | 1 |" in table
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
    assert "| 2 | Top-1 Acc 10% | 50% (50%) | 10 | 2,050 | — | 1 |" in page
    assert "| 2 | Top-1 Acc 50% | 0% (50%) | 10 | 2,050 | 2 | 1 |" in page
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
