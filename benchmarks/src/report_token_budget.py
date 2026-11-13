"""Summarize completed capped-sample reruns after stopping inference; no model calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_from_disk

from .artifacts import digest, encoded, sample_record, utc_now, write_json
from .evaluation import ReplayBackend, evaluate, usage_summary
from .run_token_budget import write_comparison
from .task_registry import DATASETS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, help="Override the recorded baseline path after moving a run")
    parser.add_argument("--prepared", type=Path, default=Path("benchmarks/.cache/prepared-datasets-100"))
    parser.add_argument("--interrupted-requests", type=int, default=1)
    args = parser.parse_args(argv)
    selection = json.loads((args.experiment / "selection.json").read_text())
    baseline = args.baseline or Path(selection["baseline"])
    config = json.loads((baseline / "config.json").read_text())
    comparison = {
        "status": "stopped_by_user", "recorded_at": utc_now(),
        "baseline_limit": selection["baseline_limit"], "rerun_limit": selection["rerun_limit"],
        "selected_samples": sum(len(rows) for rows in selection["samples"].values()),
        "interrupted_requests": args.interrupted_requests,
        "note": "Scores and token totals cover completed reruns only; interrupted generation usage is unknown.",
        "postprocessor_sha256": digest(Path(__file__).read_bytes()), "tasks": {},
    }
    for task, entries in selection["samples"].items():
        path = args.experiment / f"{task}.jsonl"
        if not path.exists():
            continue
        records = [json.loads(line) for line in path.read_text().splitlines()]
        reruns = [r for r in records if "prediction" in r or "error" in r]
        if not reruns:
            continue
        indices = [entries[r["index"]]["baseline_index"] for r in reruns]
        all_originals = [json.loads(line) for line in (baseline / f"{task}.jsonl").read_text().splitlines()]
        originals = [all_originals[i] for i in indices]
        dataset = load_from_disk(str(args.prepared / DATASETS[task])).select(indices)
        for row, index, rerun in zip(dataset, indices, reruns):
            if encoded(sample_record(row, index)) != encoded(entries[rerun["index"]]["sample"]):
                raise ValueError("Captured sample identity changed")
        cache = args.prepared.parent / "downloads"
        old_metrics = evaluate(task, ReplayBackend(originals), dataset, config, cache)
        new_metrics = evaluate(task, ReplayBackend(reruns), dataset, config, cache)
        pairs = []
        for old, new in zip(originals, reruns):
            a, b = [g["raw_text"] for g in old["generations"]], [g["raw_text"] for g in new["generations"]]
            pairs.append({
                "baseline_index": old["index"], "rerun_index": new["index"],
                "baseline_valid": "error" not in old, "rerun_valid": "error" not in new,
                "baseline_output_tokens": sum(g["output_tokens"] for g in old["generations"]),
                "rerun_output_tokens": sum(g["output_tokens"] for g in new["generations"]),
                "rerun_at_cap": any(g["output_tokens"] >= selection["rerun_limit"] for g in new["generations"]),
                "baseline_seconds": old["elapsed_seconds"], "rerun_seconds": new["elapsed_seconds"],
                "prediction_changed": old.get("prediction") != new.get("prediction"),
                "raw_prefix_preserved": len(a) == len(b) and all(y.startswith(x) for x, y in zip(a, b)),
                "baseline_raw_text": a, "rerun_raw_text": b,
            })
        comparison["tasks"][task] = {
            "samples": len(reruns), "baseline_indices": indices,
            "baseline_metrics": old_metrics.results_dict, "rerun_metrics": new_metrics.results_dict,
            "baseline_summary": old_metrics.summary(), "rerun_summary": new_metrics.summary(),
            "baseline_usage": usage_summary([g for r in originals for g in r["generations"]], len(originals)),
            "rerun_usage": usage_summary([g for r in reruns for g in r["generations"]], len(reruns)),
            "pairs": pairs,
        }
    comparison["completed_samples"] = sum(r["samples"] for r in comparison["tasks"].values())
    comparison["not_started_samples"] = comparison["selected_samples"] - comparison["completed_samples"] - args.interrupted_requests
    write_comparison(args.experiment, comparison)
    readme = args.experiment / "README.md"
    note = (f"**Stopped at the user's request after {comparison['completed_samples']} of "
            f"{comparison['selected_samples']} selected samples finished.** "
            f"{args.interrupted_requests} in-flight request was interrupted; its token usage is unknown. "
            "Scores, token totals and latency below cover completed cases only.\n\n")
    readme.write_text(readme.read_text().replace("# Token budget experiment\n\n", "# Token budget experiment (partial)\n\n" + note, 1))
    write_json(args.experiment / "status.json", {key: value for key, value in comparison.items() if key != "tasks"})
    print(f"Saved partial token budget report: {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
