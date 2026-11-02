"""Validate a configured classification or detection task on a labeled dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.run_bedrock import DEFAULT_CONFIG, TASKS, create_backend


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["classify", "detect"], required=True)
    parser.add_argument("--data", type=Path, required=True, help="Classification root or detection YAML")
    parser.add_argument("--split", default="val")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Task constructor settings as JSON")
    parser.add_argument("--class-map", type=Path, help="JSON mapping of dataset class names to task labels")
    parser.add_argument("--max-samples", type=int, help="Limit the number of evaluated images")
    parser.add_argument("--match-iou", type=float, help="Detection matching IoU; default 0.5")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model", required=True, help="Vision-capable Bedrock model or inference profile ID")
    parser.add_argument("--profile", help="AWS profile; defaults to the normal credential chain")
    parser.add_argument("--region", help="AWS region; defaults to the environment or profile")
    parser.add_argument("--output-mode", choices=["prompted", "native", "tool"], default="prompted")
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args(argv)
    if args.retries < 0 or args.max_tokens < 1:
        parser.error("--retries must be nonnegative and --max-tokens must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if args.task != "detect" and args.match_iou is not None:
        parser.error("--match-iou applies only to detection")

    try:
        config = json.loads(args.config.read_text())
        if not isinstance(config, dict) or not isinstance(config.get(args.task), dict):
            raise ValueError(f"Configuration must contain an object for task {args.task!r}")
        class_map = None if args.class_map is None else json.loads(args.class_map.read_text())
        task = TASKS[args.task](create_backend(args), **config[args.task])
        settings = {
            "data": args.data, "split": args.split, "class_map": class_map,
            "max_samples": args.max_samples, "verbose": args.verbose,
        }
        if args.match_iou is not None:
            settings["match_iou"] = args.match_iou
        metrics = task.val(**settings)
    except Exception as exc:
        parser.exit(2, f"Validation failed: {exc}\n")

    print(json.dumps({
        "metrics": metrics.results_dict,
        "per_class": metrics.summary(),
        "failures": metrics.failures,
        "dataset_counts": metrics.dataset_counts,
        "speed_ms_per_sample": metrics.speed,
        "elapsed_seconds": metrics.elapsed_seconds,
    }, indent=2))
    # Execution errors are visible in the report and exit code. This is not an
    # accuracy threshold: a completed run can have any measured accuracy.
    return 1 if metrics.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
