"""Combine consecutive benchmark subsets and rescore them without model requests."""
from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
from time import perf_counter

from .artifacts import ROOT, digest, encoded, provenance, utc_now, write_json
from .evaluation import ReplayBackend, evaluate, load_prepared
from .task_registry import DATASETS
from .usage import check_journal_usage, combine_usage


def global_sample(sample, offset):
    """Restore row-based IDs; explicit dataset IDs are preserved in targets."""
    sample = {**sample, "index": sample["index"] + offset}
    if not {"sample_id", "image_id"}.intersection(sample["targets"]):
        sample["sample_id"] = sample["index"]
    return sample


def checked_segments(sources, manifest):
    """Reject changed settings, missing responses, gaps and reordered data."""
    segments, offset, reference = [], 0, None
    reference_backend = None
    for source in sources:
        source = Path(source)
        config = json.loads((source / "config.json").read_text())
        count = config["samples_per_task"]
        settings = {key: value for key, value in config.items() if key != "samples_per_task"}
        if reference is None:
            reference = settings
        if settings != reference:
            raise ValueError("Segment configurations differ beyond samples_per_task")
        if count <= 0 or offset + count > manifest["samples_per_task"]:
            raise ValueError("Segment sample counts exceed the prepared suite")
        complete = json.loads((source / "completed.json").read_text())
        if complete["tasks"] != config["tasks"]:
            raise ValueError("Segment has not completed all configured tasks")
        frozen = json.loads((source / "manifest.json").read_text())
        if frozen["samples_per_task"] != count:
            raise ValueError("Segment configuration and manifest counts differ")
        for dataset in dict.fromkeys(DATASETS[task] for task in config["tasks"]):
            actual, expected = frozen["datasets"][dataset], manifest["datasets"][dataset]
            identity = {key: actual[key] for key in ("features", "samples")}
            if (actual["fingerprint_sha256"] != digest(encoded(identity).encode())
                    or actual["features"] != expected["features"]
                    or len(actual["samples"]) != count
                    or [global_sample(row, offset) for row in actual["samples"]]
                    != expected["samples"][offset:offset + count]):
                raise ValueError(f"Segment is not the expected consecutive subset: {dataset}")
        model = json.loads((source / "model.json").read_text())
        # Both hosted and local runners save their effective settings here.
        # Loading duration varies independently of the inference configuration.
        effective = {key: value for key, value in model.items() if key != "load_seconds"}
        if segments and effective != segments[0]["effective_model"]:
            raise ValueError("Segment model settings or inference routes differ")
        environment = json.loads((source / "environment.json").read_text())
        for name, expected_hash in environment["source_sha256"].items():
            if (name.startswith("src/ml_pipes/vlm/")
                    or name.startswith("benchmarks/setup/task-specs/")
                    or name in ("benchmarks/src/evaluation.py", "benchmarks/src/task_registry.py")):
                if digest((ROOT / name).read_bytes()) != expected_hash:
                    raise ValueError(f"Scoring source changed since segment inference: {name}")
        reports, records = {}, {}
        for task in config["tasks"]:
            report = json.loads((source / f"{task}.json").read_text())
            backend = report.get("backend", "transformers")
            if reference_backend is None:
                reference_backend = backend
            if backend != reference_backend:
                raise ValueError("Segment backends differ")
            calls = [json.loads(line) for line in (source / f"{task}.jsonl").read_text().splitlines()]
            if (report["model"] != config["model"]
                    or report["task"] != task or report["metrics"]["samples"] != count
                    or len(calls) != count or [row["index"] for row in calls] != list(range(count))
                    or report["dataset_fingerprint_sha256"]
                    != frozen["datasets"][DATASETS[task]]["fingerprint_sha256"]):
                raise ValueError(f"Cannot verify segment task report: {task}")
            check_journal_usage(calls, report["usage"], task=task)
            reports[task], records[task] = report, calls
        segments.append({"source": source, "count": count, "offset": offset,
                         "config": config, "model": model, "effective_model": effective,
                         "backend": backend, "reports": reports, "records": records})
        offset += count
    if not segments or offset != manifest["samples_per_task"]:
        raise ValueError("Segments must cover the full prepared suite exactly once")
    return segments


def combine(sources, prepared, output):
    output, prepared = Path(output), Path(prepared)
    if output.exists():
        raise FileExistsError(f"Use a new combined run directory: {output}")
    if any(output.resolve().is_relative_to(Path(source).resolve()) for source in sources):
        raise ValueError("Combined output must be outside the source runs")
    manifest = json.loads((prepared / "manifest.json").read_text())
    segments = checked_segments(sources, manifest)
    config = {**segments[0]["config"], "samples_per_task": manifest["samples_per_task"]}
    loaded = load_prepared(prepared, manifest, config["tasks"])
    write_json(output / "config.json", config)
    write_json(output / "manifest.json", manifest)
    packages = {d.metadata["Name"]: d.version for d in metadata.distributions() if d.metadata["Name"]}
    write_json(output / "environment.json", {
        "started_at": utc_now(), "python": platform.python_version(), "platform": platform.platform(),
        "phase": "offline rescoring; inference environments retained in segments/",
        "backend": segments[0]["backend"],
        "packages": packages, **provenance(),
    })
    (output / "requirements.txt").write_text("\n".join(f"{k}=={v}" for k, v in sorted(packages.items())) + "\n")
    write_json(output / "model.json", segments[0]["effective_model"])
    assembly = {"created_at": utc_now(), "model_requests": 0, "segments": [],
                "method": "Consecutive frozen subsets; recompute all metrics on the full suite; retain live usage and latency"}
    for index, segment in enumerate(segments):
        copy = output / "segments" / f"{index:03d}"
        shutil.copytree(segment["source"], copy)
        assembly["segments"].append({
            "source": str(segment["source"]), "saved_copy": str(copy.relative_to(output)),
            "start_index": segment["offset"], "samples_per_task": segment["count"],
            "files_sha256": {str(path.relative_to(copy)): digest(path.read_bytes())
                             for path in sorted(copy.rglob("*")) if path.is_file()},
        })
    write_json(output / "assembly.json", assembly)
    for task in config["tasks"]:
        calls = [{**row, "index": row["index"] + segment["offset"]}
                 for segment in segments for row in segment["records"][task]]
        (output / f"{task}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in calls))
        started = perf_counter()
        metrics = evaluate(task, ReplayBackend(calls), loaded[DATASETS[task]], config, prepared.parent / "downloads")
        if metrics.seen != config["samples_per_task"]:
            raise RuntimeError("Rescoring did not cover the full suite")
        source_reports = [segment["reports"][task] for segment in segments]
        speed = dict(metrics.speed)
        speed["inference"] = sum(report["speed_ms_per_sample"]["inference"] * segment["count"]
                                 for segment, report in zip(segments, source_reports)) / metrics.seen
        report = {
            **source_reports[0], "completed_at": utc_now(),
            "backend": segments[0]["backend"],
            "execution_environment": "environment.json",
            "inference_environments": [f"segments/{index:03d}/environment.json" for index in range(len(segments))],
            "assembly": "assembly.json", "metrics": metrics.results_dict, "summary": metrics.summary(),
            "failures": metrics.failures, "dataset_counts": metrics.dataset_counts,
            "dataset_metadata": metrics.dataset_metadata, "status_counts": getattr(metrics, "status_counts", {}),
            "dataset_fingerprint_sha256": manifest["datasets"][DATASETS[task]]["fingerprint_sha256"],
            "speed_ms_per_sample": speed,
            "timing_note": "Inference is the weighted original live latency; load/metrics are offline rescoring timings; elapsed/wall sum source runs",
            "elapsed_seconds": sum(row["elapsed_seconds"] for row in source_reports),
            "wall_seconds": sum(row["wall_seconds"] for row in source_reports),
            "offline_rescoring_seconds": perf_counter() - started,
            "usage": combine_usage([row["usage"] for row in source_reports]),
        }
        # No model is loaded for assembly; setup timings remain in the sources.
        report.pop("shared_model_load_seconds", None)
        write_json(output / f"{task}.json", report)
        print(f"Combined {task}: {report['metrics']}", flush=True)
    write_json(output / "completed.json", {"completed_at": utc_now(), "tasks": config["tasks"]})
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", nargs="+", type=Path, required=True, help="Completed runs in sample order")
    parser.add_argument("--prepared", type=Path, default=Path("benchmarks/.cache/prepared-datasets-100"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    combine(args.segments, args.prepared, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
