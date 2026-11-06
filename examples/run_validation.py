"""Validate a configured classification, detection, grounding, or attributes task."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

from examples.run_bedrock import DEFAULT_CONFIG, TASKS, create_backend
from ml_pipes.vlm.validation import DATASETS, DEFAULT_DATASETS, OVADDataset, prepare_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["bedrock", "transformers"], default="bedrock")
    parser.add_argument("--task", choices=["classify", "detect", "ground", "attributes"], required=True)
    parser.add_argument("--data", help="Registered dataset name or local path; defaults: cifar10, coco128, refcocog, ovad")
    parser.add_argument("--images", type=Path, help="Local grounding (COCO train2014) or OVAD (COCO val2017) image directory")
    parser.add_argument("--dataset", choices=["refcocog", "refcoco", "refcoco+"], help="Grounding dataset; default refcocog")
    parser.add_argument("--split", default="val")
    parser.add_argument("--cache-dir", type=Path, help="Dataset cache; defaults to VLM_TASKS_DATASETS_DIR or ~/.cache/vlm-tasks/datasets")
    parser.add_argument("--config", type=Path, help="Task settings JSON; attributes discovers the dataset schema, named classify/detect use dataset labels, local classify/detect use image_tasks.json, ground uses empty settings")
    parser.add_argument("--class-map", type=Path, help="JSON mapping of dataset class names to task labels")
    parser.add_argument("--max-samples", type=int, help="Limit evaluated samples (expressions for grounding; eligible object crops for attributes)")
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
    args.data = args.data or DEFAULT_DATASETS[args.task]
    automatic = args.data in DATASETS
    if automatic and DATASETS[args.data].task != args.task:
        parser.error(f"Dataset {args.data!r} does not support {args.task}")
    if args.task == "ground":
        if args.images is None and not automatic:
            parser.error("--images is required for local grounding data")
        if automatic and (args.images is not None or args.dataset not in (None, "refcocog")):
            parser.error("Automatic grounding preparation supports refcocog; provide local --data and --images for other sources")
        if args.class_map is not None:
            parser.error("--class-map is not supported for grounding")
    elif args.task == "attributes":
        if args.images is None and not automatic:
            parser.error("--images is required for local attributes data")
        if automatic and args.images is not None:
            parser.error("Automatic OVAD preparation supplies images; provide local --data with --images")
        if args.dataset is not None:
            parser.error("--dataset applies only to grounding")
        if args.class_map is not None:
            parser.error("--class-map is not supported for attributes")
    elif args.images is not None or args.dataset is not None:
        parser.error("--images applies only to grounding/attributes; --dataset applies only to grounding")

    try:
        task_settings = {}
        if args.config is None and automatic and args.task in {"classify", "detect"}:
            task_settings = {"labels": list(DATASETS[args.data].names)}
        elif args.config is not None or args.task in {"classify", "detect"}:
            config = json.loads((args.config or DEFAULT_CONFIG).read_text())
            if not isinstance(config, dict) or not isinstance(config.get(args.task), dict):
                raise ValueError(f"Configuration must contain an object for task {args.task!r}")
            task_settings = config[args.task]
        class_map = None if args.class_map is None else json.loads(args.class_map.read_text())
        preparation_seconds = 0.0
        if automatic:
            classes = tuple(class_map or task_settings["labels"]) if args.task == "classify" else None
            started = perf_counter()
            if args.verbose:
                print(f"Preparing {args.data} split {args.split}...", file=sys.stderr, flush=True)
            prepared = prepare_dataset(
                args.data, split=args.split, max_samples=args.max_samples,
                cache_dir=args.cache_dir, classes=classes,
                **({"groups": tuple(task_settings["attributes"])} if args.task == "attributes" and task_settings.get("attributes") is not None else {}),
            )
            preparation_seconds = perf_counter() - started
            if args.task == "attributes" and task_settings.get("attributes") is None:
                task_settings["attributes"] = prepared.attributes
        elif args.task == "attributes" and task_settings.get("attributes") is None:
            task_settings["attributes"] = OVADDataset(
                args.data, args.images, split=args.split, max_samples=args.max_samples,
            ).attributes
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
            "cache_dir": args.cache_dir,
        }
        if args.match_iou is not None:
            settings["match_iou"] = args.match_iou
        if args.task == "ground":
            settings.update(images=args.images, dataset=args.dataset or "refcocog")
        elif args.task == "attributes":
            settings["images"] = args.images
        metrics = task.val(**settings)
    except Exception as exc:
        parser.exit(2, f"Validation failed: {exc}\n")

    report = {
        "metrics": metrics.results_dict,
        "per_attribute" if args.task == "attributes" else "per_class": metrics.summary(),
        "failures": metrics.failures,
        "dataset_counts": metrics.dataset_counts,
        "dataset_metadata": getattr(metrics, "dataset_metadata", {}),
        "status_counts": getattr(metrics, "status_counts", {}),
        "task": args.task, "task_config": task_settings, "model": args.model,
        "model_settings": {"temperature": 0.0, "max_tokens": args.max_tokens},
        "output_mode": args.output_mode, "retries": args.retries,
        "backend": args.backend, "load_seconds": load_seconds,
        "preparation_seconds": preparation_seconds,
        "speed_ms_per_sample": metrics.speed,
        "elapsed_seconds": metrics.elapsed_seconds,
    }
    if args.task == "attributes":
        report.update(
            protocol=metrics.dataset_metadata["protocol"],
            selected_groups=list(task.attributes),
        )
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
