"""Preserve sample identity, failures, live latency and billing when joining runs."""
import json

import datasets
from PIL import Image
import pytest

from benchmarks.src.artifacts import digest, encoded, revision_fields, sample_record, save_run_setup, write_json
from benchmarks.src.combine_runs import checked_segments, combine, global_sample
from benchmarks.src.evaluation import evaluate, usage_summary as local_usage_summary
from benchmarks.src.usage import combine_usage
from ml_pipes.vlm import ModelResponse, ResponseSyntaxError


@pytest.fixture(params=["bedrock", "transformers"])
def consecutive_runs(tmp_path, request):
    backend_name = request.param
    if backend_name == "bedrock":
        from benchmarks.src.run_bedrock import usage_summary
    else:
        usage_summary = local_usage_summary
    dataset = datasets.Dataset.from_dict(
        {"image": [Image.new("RGB", (2, 3), color) for color in ("red", "blue", "green")],
         "label": [0, 1, 1]},
        features=datasets.Features({"image": datasets.Image(), "label": datasets.ClassLabel(names=["cat", "dog"])}),
    )

    def manifest(data):
        identity = {"features": data.features.to_dict(),
                    "samples": [sample_record(row, i) for i, row in enumerate(data)]}
        return {"samples_per_task": len(data), "datasets": {
            "cifar10": {**identity, "fingerprint_sha256": digest(encoded(identity).encode())}}}

    frozen = manifest(dataset)
    prepared = tmp_path / "prepared"
    dataset.save_to_disk(str(prepared / "cifar10"))
    write_json(prepared / "manifest.json", frozen)
    sources = []
    for index, indices in enumerate(([0], [1, 2])):
        data = dataset.select(indices)
        source = tmp_path / f"segment-{index}"
        config = {"model": "test", "revision": 1, "tasks": ["classify"],
                  "samples_per_task": len(data), "max_new_tokens": 4096}
        save_run_setup(source, config)
        local_manifest = manifest(data)
        write_json(source / "manifest.json", local_manifest)
        model = ({"settings": {"temperature": 0}} if backend_name == "bedrock" else {
            "load_seconds": index + 1, "generation_config": {"do_sample": False},
            "image_processor": {"size": 32}, "dtype": "torch.float16", "device": "mps:0",
        })
        write_json(source / "model.json", model)
        write_json(source / "environment.json", {"source_sha256": {}})
        write_json(source / "completed.json", {"tasks": ["classify"]})
        rows = []

        def backend(request):
            invalid = index == 1 and not rows
            generation = {"input_tokens": 100, "output_tokens": 10,
                          "raw_text": "not JSON" if invalid else '{"labels":["cat"]}'}
            if backend_name == "bedrock":
                generation.update(cache_read_tokens=0, cache_write_tokens=0,
                                  output_reasoning_tokens=0, unmeasured_requests=0,
                                  estimated_cost_usd="0.01", estimated_cost_source="test")
            row = {"index": len(rows), "instruction": request.instruction,
                   "response_schema": request.response_spec.schema, "generations": [generation],
                   "elapsed_seconds": 1 + index * 3}
            if backend_name == "bedrock":
                row["attempted_requests"] = 1
            rows.append(row)
            if invalid:
                row["error"] = {"type": "ResponseSyntaxError", "message": "not JSON"}
                raise ResponseSyntaxError("not JSON")
            row["prediction"] = {"labels": ["cat"]}
            return ModelResponse(data=row["prediction"])

        metrics = evaluate("classify", backend, data, config, tmp_path / "cache")
        (source / "classify.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        write_json(source / "classify.json", {
            **({"backend": backend_name} if backend_name == "bedrock" else {}),
            **({"shared_model_load_seconds": model["load_seconds"]} if backend_name == "transformers" else {}),
            "model": "test", "task": "classify", "dataset": "cifar10",
            **revision_fields("classify", config),
            "metrics": metrics.results_dict, "usage": usage_summary([row["generations"][0] for row in rows], len(rows)),
            "dataset_fingerprint_sha256": local_manifest["datasets"]["cifar10"]["fingerprint_sha256"],
            "speed_ms_per_sample": {"inference": 1000 + index * 3000},
            "elapsed_seconds": len(rows) * (1 + index * 3), "wall_seconds": len(rows) * (1 + index * 3),
        })
        sources.append(source)
    return sources, prepared, frozen


def test_combination_rescores_failures_without_requests_and_preserves_original_usage(consecutive_runs, tmp_path):
    sources, prepared, _ = consecutive_runs
    output = combine(sources, prepared, tmp_path / "combined")
    report = json.loads((output / "classify.json").read_text())
    assert report["metrics"]["samples"] == 3
    assert report["task_spec_revision"] == report["model_config_revision"] == 1
    assert (output / "task-specs/classify.json").read_bytes() == (sources[0] / "task-specs/classify.json").read_bytes()
    assert report["metrics"]["accuracy_top1"] == pytest.approx(1 / 3)
    assert report["metrics"]["failures"] == 1
    assert report["speed_ms_per_sample"]["inference"] == 3000
    assert report["usage"]["attempted_requests"] == 3
    assert report["usage"]["input_tokens"] == 300
    assert report["usage"]["output_tokens"] == 30
    if report["backend"] == "bedrock":
        assert report["usage"]["estimated_cost_usd"] == "0.03"
        assert report["usage"]["estimated_cost_complete"]
    else:
        assert report["backend"] == "transformers"
        assert report["usage"]["estimated_cost_usd"] is None
        assert report["usage"]["reported_cost_usd"] is None
        assert "shared_model_load_seconds" not in report
        assert "load_seconds" not in json.loads((output / "model.json").read_text())
    assert json.loads((output / "assembly.json").read_text())["model_requests"] == 0
    calls = [json.loads(line) for line in (output / "classify.jsonl").read_text().splitlines()]
    assert [row["index"] for row in calls] == [0, 1, 2]
    assert calls[1]["error"]["type"] == "ResponseSyntaxError"
    for index, source in enumerate(sources):
        assert (output / f"segments/{index:03d}/classify.jsonl").read_bytes() == (source / "classify.jsonl").read_bytes()


def test_explicit_dataset_ids_are_not_replaced_by_combined_row_numbers():
    sample = {"index": 0, "sample_id": "expression-17", "targets": {"sample_id": "expression-17"}}
    assert global_sample(sample, 10) == {**sample, "index": 10}
    assert sample["index"] == 0


def test_combination_preserves_unversioned_legacy_records(consecutive_runs, tmp_path):
    sources, prepared, _ = consecutive_runs
    checkpoint = None
    for source in sources:
        config = json.loads((source / "config.json").read_text())
        del config["revision"]
        report = json.loads((source / "classify.json").read_text())
        if report.get("backend", "transformers") == "transformers":
            checkpoint = "hf-checkpoint"
            config["revision"] = checkpoint
            report["revision"] = checkpoint
        write_json(source / "config.json", config)
        del report["model_config_revision"], report["task_spec_revision"]
        write_json(source / "classify.json", report)
        (source / "task-specs/classify.json").unlink()
        (source / "task-specs").rmdir()
    output = combine(sources, prepared, tmp_path / "legacy-combined")
    assert json.loads((output / "config.json").read_text()).get("revision") == checkpoint
    assert json.loads((output / "classify.json").read_text()).get("revision") == checkpoint
    assert "task_spec_revision" not in json.loads((output / "classify.json").read_text())
    assert not (output / "task-specs").exists()


def test_combination_does_not_write_inside_an_original_run(consecutive_runs):
    sources, prepared, _ = consecutive_runs
    with pytest.raises(ValueError, match="outside the source runs"):
        combine(sources, prepared, sources[0] / "combined")


@pytest.mark.parametrize("defect", ["order", "config", "usage", "indices", "model", "processor", "backend", "incomplete", "scorer", "task_spec"])
def test_combination_refuses_incompatible_or_unverifiable_segments(consecutive_runs, defect):
    sources, _, manifest = consecutive_runs
    if defect == "order":
        sources.reverse()
    else:
        name = {"config": "config", "usage": "classify", "indices": "classify",
                "model": "model", "processor": "model", "backend": "classify",
                "incomplete": "completed", "scorer": "environment", "task_spec": "environment"}[defect]
        path = sources[1] / f"{name}.json"
        content = json.loads(path.read_text())
        if defect == "config":
            content["max_new_tokens"] = 8192
        elif defect == "usage":
            content["usage"]["input_tokens"] += 100
        elif defect == "model":
            if "settings" in content:
                content["settings"]["temperature"] = 1
            else:
                content["generation_config"]["do_sample"] = True
        elif defect == "processor":
            content["image_processor"] = {"size": 64}
        elif defect == "backend":
            content["backend"] = "transformers" if content.get("backend") == "bedrock" else "bedrock"
        elif defect == "incomplete":
            content["tasks"] = []
        elif defect == "scorer":
            content["source_sha256"]["benchmarks/src/evaluation.py"] = "changed"
        elif defect == "task_spec":
            content["source_sha256"]["benchmarks/setup/task-specs/describe.json"] = "changed"
        elif defect == "indices":
            journal = sources[1] / "classify.jsonl"
            calls = [json.loads(line) for line in journal.read_text().splitlines()]
            calls[0]["index"] = 1
            journal.write_text("".join(json.dumps(row) + "\n" for row in calls))
        write_json(path, content)
    with pytest.raises(ValueError):
        checked_segments(sources, manifest)


def test_partial_costs_remain_incomplete_and_missing_usage_remains_unknown():
    result = combine_usage([
        {"input_tokens": 100, "output_tokens": 20, "attempted_requests": 1,
         "estimated_cost_usd": "0.01", "estimated_cost_complete": True,
         "cost_sources": ["source"], "reported_cost_usd": None},
        {"input_tokens": 200, "output_tokens": 30,
         "estimated_cost_usd": None, "estimated_cost_complete": False,
         "reported_cost_usd": None},
    ])
    assert result["input_tokens"] == 300 and result["output_tokens"] == 50
    assert result["attempted_requests"] is None
    assert result["estimated_cost_usd"] == "0.01"
    assert result["reported_cost_usd"] is None
    assert result["estimated_cost_complete"] is False
    assert result["cost_sources"] == ["source"]
