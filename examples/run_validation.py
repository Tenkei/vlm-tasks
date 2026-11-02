"""Validate a configured classification, detection, or grounding task."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from examples.run_bedrock import DEFAULT_CONFIG, TASKS, create_backend


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["bedrock", "transformers"], default="bedrock")
    parser.add_argument("--task", choices=["classify", "detect", "ground"], required=True)
    parser.add_argument("--data", type=Path, required=True, help="Classification root, detection YAML, or MDETR annotation directory")
    parser.add_argument("--images", type=Path, help="Grounding COCO train2014 image directory")
    parser.add_argument("--dataset", choices=["refcocog", "refcoco", "refcoco+"], help="Grounding dataset; default refcocog")
    parser.add_argument("--split", default="val")
    parser.add_argument("--config", type=Path, help="Task settings JSON; grounding defaults to empty settings, other tasks to image_tasks.json")
    parser.add_argument("--class-map", type=Path, help="JSON mapping of dataset class names to task labels")
    parser.add_argument("--max-samples", type=int, help="Limit evaluated samples (expressions for grounding)")
    parser.add_argument("--match-iou", type=float, help="Detection matching IoU; default 0.5")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model", help="Bedrock model ID or HF model/local directory; Transformers defaults to Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--device", default="auto", help="Transformers device: auto, mps, cuda, cpu, etc.")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--revision", help="Transformers model and processor revision")
    parser.add_argument("--local-files-only", action="store_true", help="Transformers: use only cached/local weights")
    parser.add_argument("--output", type=Path, help="Also save the JSON report to this file")
    parser.add_argument("--profile", help="AWS profile; defaults to the normal credential chain")
    parser.add_argument("--region", help="AWS region; defaults to the environment or profile")
    parser.add_argument("--output-mode", choices=["prompted", "native", "tool"], default="prompted")
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args(argv)
    if args.backend == "transformers":
        args.model = args.model or "Qwen/Qwen3-VL-2B-Instruct"
        if args.output_mode != "prompted":
            parser.error("The Transformers backend supports only prompted output")
        if args.profile is not None or args.region is not None:
            parser.error("--profile and --region apply only to Bedrock")
    else:
        if not args.model:
            parser.error("--model is required for Bedrock")
        if args.device != "auto" or args.dtype != "auto" or args.revision or args.local_files_only:
            parser.error("--device, --dtype, --revision and --local-files-only apply only to Transformers")
    if args.retries < 0 or args.max_tokens < 1:
        parser.error("--retries must be nonnegative and --max-tokens must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if args.task != "detect" and args.match_iou is not None:
        parser.error("--match-iou applies only to detection")
    if args.task == "ground":
        if args.images is None:
            parser.error("--images is required for grounding")
        if args.class_map is not None:
            parser.error("--class-map is not supported for grounding")
    elif args.images is not None or args.dataset is not None:
        parser.error("--images and --dataset apply only to grounding")

    try:
        task_settings = {}
        if args.config is not None or args.task != "ground":
            config = json.loads((args.config or DEFAULT_CONFIG).read_text())
            if not isinstance(config, dict) or not isinstance(config.get(args.task), dict):
                raise ValueError(f"Configuration must contain an object for task {args.task!r}")
            task_settings = config[args.task]
        class_map = None if args.class_map is None else json.loads(args.class_map.read_text())
        started = perf_counter()
        if args.backend == "transformers":
            from examples.run_transformers import create_backend as create_transformers_backend

            backend = create_transformers_backend(args)
        else:
            backend = create_backend(args)
        load_seconds = perf_counter() - started
        task = TASKS[args.task](backend, **task_settings)
        settings = {
            "data": args.data, "split": args.split, "class_map": class_map,
            "max_samples": args.max_samples, "verbose": args.verbose,
        }
        if args.match_iou is not None:
            settings["match_iou"] = args.match_iou
        if args.task == "ground":
            settings.update(images=args.images, dataset=args.dataset or "refcocog")
        metrics = task.val(**settings)
    except Exception as exc:
        parser.exit(2, f"Validation failed: {exc}\n")

    report = {
        "metrics": metrics.results_dict,
        "per_class": metrics.summary(),
        "failures": metrics.failures,
        "dataset_counts": metrics.dataset_counts,
        "dataset_metadata": getattr(metrics, "dataset_metadata", {}),
        "status_counts": getattr(metrics, "status_counts", {}),
        "task": args.task, "task_config": task_settings, "model": args.model,
        "model_settings": {"temperature": 0.0, "max_tokens": args.max_tokens},
        "output_mode": args.output_mode, "retries": args.retries,
        "backend": args.backend, "load_seconds": load_seconds,
        "speed_ms_per_sample": metrics.speed,
        "elapsed_seconds": metrics.elapsed_seconds,
    }
    if args.backend == "transformers":
        report.update(
            revision=args.revision, device=str(backend.model.weights.device),
            dtype=str(backend.model.weights.dtype), local_files_only=args.local_files_only,
            do_sample=False,
        )
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    # Execution errors are visible in the report and exit code. This is not an
    # accuracy threshold: a completed run can have any measured accuracy.
    return 1 if metrics.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
