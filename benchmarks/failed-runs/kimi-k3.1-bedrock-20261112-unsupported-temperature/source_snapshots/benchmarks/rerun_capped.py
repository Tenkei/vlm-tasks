"""Rerun baseline responses at their token cap with twice the output budget."""
from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
from time import perf_counter

from ml_pipes import vlm

from .common import DATASETS, digest, encoded, provenance, sample_record, utc_now, write_json
from .run import RecordingBackend, RecordingModel, usage_summary
from .summarize import METRICS


CLASSES = {
    "classify": vlm.Classify, "detect": vlm.Detect, "ground_box": vlm.Ground,
    "ground_point": vlm.Ground, "attributes": vlm.Attributes, "answer": vlm.Answer,
    "describe": vlm.Describe,
}


class ReplayBackend:
    """Re-score recorded predictions on exactly the selected subset, without inference."""

    def __init__(self, records):
        self.records = iter(records)

    def __call__(self, request):
        record = next(self.records)
        if encoded(request.response_spec.schema) != encoded(record["response_schema"]) or request.instruction != record["instruction"]:
            raise ValueError("Replay task instructions/schema differ from the baseline")
        if "error" in record:
            kind = {"ModelExecutionError": vlm.ModelExecutionError,
                    "InvalidModelResponseError": vlm.InvalidModelResponseError}[record["error"]["type"]]
            raise kind(record["error"]["message"])
        return vlm.ModelResponse(data=record["prediction"])


def selected_calls(baseline, config):
    selected = {}
    for task in config["tasks"]:
        records = [json.loads(line) for line in (baseline / f"{task}.jsonl").read_text().splitlines()]
        selected[task] = [record for record in records
                          if any(g["output_tokens"] >= config["max_new_tokens"] for g in record["generations"])]
    return {task: records for task, records in selected.items() if records}


def evaluate(task_name, backend, dataset, config, cache):
    settings = {"geometry": task_name.removeprefix("ground_")} if task_name.startswith("ground_") else {}
    task = CLASSES[task_name](backend, **settings)
    options = {"dataset": dataset, "cache_dir": cache}
    if task_name == "detect":
        options["match_iou"] = config["match_iou"]
    if task_name == "describe":
        options["tokenizer_version"] = config["tokenizer_version"]
    return task.val(**options)


def write_comparison(directory, comparison):
    write_json(directory / "comparison.json", comparison)
    lines = ["# Token budget followup", "",
             "Only samples reaching the baseline cap are included. Both conditions are scored on "
             "the same selected samples. [Paired raw responses and metrics](comparison.json) · "
             "[Selected sample identities](selection.json)", "",
             "| Task | N | Token budget | Validation metrics | Failures | At cap | Input tokens | Output tokens | Mean s/sample |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for task, report in comparison["tasks"].items():
        for condition, limit in (("baseline", comparison["baseline_limit"]), ("rerun", comparison["rerun_limit"])):
            metrics, usage = report[f"{condition}_metrics"], report[f"{condition}_usage"]
            scores = []
            for key, label in METRICS[task]:
                if key in metrics:
                    value = f"{metrics[key]:.4f}" if task == "describe" else f"{100 * metrics[key]:.2f}%"
                    scores.append(f"{label} {value}")
            capped = report["samples"] if condition == "baseline" else sum(pair["rerun_at_cap"] for pair in report["pairs"])
            elapsed = sum(pair[f"{condition}_seconds"] for pair in report["pairs"]) / report["samples"]
            lines.append(f"| {task} | {report['samples']} | {limit:,} | {'; '.join(scores)} | "
                         f"{metrics['failures']} | {capped} | {usage['input_tokens']:,} | {usage['output_tokens']:,} | {elapsed:.3f} |")
    lines += ["", "'At cap' identifies possible truncation, not proof of truncation. "
              "This selected subset does not replace the 100-sample model comparison. "
              "Local monetary cost is unmeasured.", "", "## Captured samples", "",
              "| Task / baseline index | Image | Baseline response | Doubled-budget response | Output tokens before → after |",
              "|---|---|---|---|---:|"]
    for task, report in comparison["tasks"].items():
        for pair in report["pairs"]:
            index = pair["baseline_index"]
            old = "valid" if pair["baseline_valid"] else "invalid"
            new = "valid" if pair["rerun_valid"] else "invalid"
            if pair["rerun_at_cap"]:
                new += "; at cap"
            lines.append(f"| {task} / {index} | [PNG](images/{task}/{index:03d}.png) | {old} | {new} | "
                         f"{pair['baseline_output_tokens']:,} → {pair['rerun_output_tokens']:,} |")
    lines.append("")
    (directory / "README.md").write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, default=Path("benchmarks/.cache/prepared100"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Use a new rerun directory: {args.output}")
    if not (args.baseline / "completed.json").is_file():
        raise ValueError("Complete the baseline first so all capped samples are captured")
    config = json.loads((args.baseline / "config.json").read_text())
    selected = selected_calls(args.baseline, config)
    manifest = json.loads((args.baseline / "manifest.json").read_text())
    limit = config["max_new_tokens"] * 2
    write_json(args.output / "selection.json", {
        "baseline": str(args.baseline), "baseline_limit": config["max_new_tokens"], "rerun_limit": limit,
        "rule": "all baseline responses with output_tokens >= baseline limit; no other filtering",
        "samples": {task: [{"baseline_index": r["index"],
                             "image": f"images/{task}/{r['index']:03d}.png",
                             "sample": manifest["datasets"][DATASETS[task]]["samples"][r["index"]]}
                            for r in records] for task, records in selected.items()},
    })
    write_json(args.output / "config.json", {**config, "max_new_tokens": limit, "tasks": list(selected),
                                             "baseline_samples_per_task": config["samples_per_task"],
                                             "samples_per_task": {task: len(records) for task, records in selected.items()},
                                             "experiment": "double token budget on capped baseline samples"})
    if not selected:
        write_json(args.output / "comparison.json", {"tasks": {}, "note": "No baseline responses reached the token cap"})
        write_json(args.output / "completed.json", {"completed_at": utc_now(), "tasks": []})
        return 0
    import datasets
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed

    set_seed(config["seed"])
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this process")
    loaded = {}
    for name in dict.fromkeys(DATASETS[task] for task in selected):
        dataset = datasets.load_from_disk(str(args.prepared / name))
        identity = {"features": dataset.features.to_dict(),
                    "samples": [sample_record(row, i) for i, row in enumerate(dataset)]}
        if digest(encoded(identity).encode()) != manifest["datasets"][name]["fingerprint_sha256"]:
            raise ValueError(f"Prepared samples changed: {name}")
        loaded[name] = dataset
    for task_name, records in selected.items():
        directory = args.output / "images" / task_name
        directory.mkdir(parents=True, exist_ok=True)
        for record in records:
            loaded[DATASETS[task_name]][record["index"]]["image"].convert("RGB").save(
                directory / f"{record['index']:03d}.png",
            )
    write_json(args.output / "environment.json", {
        "started_at": utc_now(), "python": platform.python_version(), "platform": platform.platform(),
        "device": device, "packages": {d.metadata["Name"]: d.version for d in metadata.distributions() if d.metadata["Name"]},
        **provenance(),
    })
    options = {"revision": config["revision"], "local_files_only": args.offline}
    started = perf_counter()
    processor = AutoProcessor.from_pretrained(config["model"], **options)
    weights = AutoModelForImageTextToText.from_pretrained(
        config["model"], dtype=getattr(torch, config["dtype"]), **options,
    ).to(device)
    model = RecordingModel(weights, processor, model_name=config["model"], max_new_tokens=limit)
    load_seconds = perf_counter() - started
    comparison = {"baseline_limit": config["max_new_tokens"], "rerun_limit": limit,
                  "model_load_seconds": load_seconds, "tasks": {}}
    for task_name, originals in selected.items():
        print(f"Rerunning {task_name}: {len(originals)} capped samples at {limit} tokens", flush=True)
        indices = [r["index"] for r in originals]
        dataset = loaded[DATASETS[task_name]].select(indices)
        cache = args.prepared.parent / "downloads"
        baseline_metrics = evaluate(task_name, ReplayBackend(originals), dataset, config, cache)
        before = len(model.generations)
        with (args.output / f"{task_name}.jsonl").open("w") as stream:
            backend = RecordingBackend(vlm.PydanticAIBackend(model, retries=0), stream)
            rerun_metrics = evaluate(task_name, backend, dataset, config, cache)
        reruns = [json.loads(line) for line in (args.output / f"{task_name}.jsonl").read_text().splitlines()]
        if len(reruns) != len(originals) or rerun_metrics.seen != len(originals):
            raise RuntimeError("Rerun did not cover every selected sample")
        pairs = []
        for original, rerun in zip(originals, reruns):
            pairs.append({
                "baseline_index": original["index"], "rerun_index": rerun["index"],
                "baseline_valid": "error" not in original, "rerun_valid": "error" not in rerun,
                "baseline_output_tokens": sum(g["output_tokens"] for g in original["generations"]),
                "rerun_output_tokens": sum(g["output_tokens"] for g in rerun["generations"]),
                "rerun_at_cap": any(g["output_tokens"] >= limit for g in rerun["generations"]),
                "baseline_seconds": original["elapsed_seconds"], "rerun_seconds": rerun["elapsed_seconds"],
                "prediction_changed": original.get("prediction") != rerun.get("prediction"),
                "baseline_raw_text": [g["raw_text"] for g in original["generations"]],
                "rerun_raw_text": [g["raw_text"] for g in rerun["generations"]],
            })
        report = {
            "samples": len(originals), "baseline_indices": indices,
            "baseline_metrics": baseline_metrics.results_dict, "rerun_metrics": rerun_metrics.results_dict,
            "baseline_summary": baseline_metrics.summary(), "rerun_summary": rerun_metrics.summary(),
            "baseline_usage": usage_summary([g for r in originals for g in r["generations"]], len(originals)),
            "rerun_usage": usage_summary(model.generations[before:], backend.index),
            "rerun_failures": rerun_metrics.failures,
            "rerun_speed_ms_per_sample": rerun_metrics.speed, "pairs": pairs,
        }
        comparison["tasks"][task_name] = report
        write_comparison(args.output, comparison)
        print(f"Completed {task_name}: {report['baseline_metrics']} -> {report['rerun_metrics']}", flush=True)
    write_json(args.output / "completed.json", {"completed_at": utc_now(), "tasks": list(selected)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
