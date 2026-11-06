"""Run VLM tasks with directly loaded Transformers weights and PydanticAI."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

from ml_pipes.core import Pipeline
from ml_pipes.vision import Decode, LoadFile
from ml_pipes.vlm import PydanticAIBackend, TransformersModel

from examples.run_bedrock import DEFAULT_CONFIG, TASKS, build_pipelines


def create_backend(args: argparse.Namespace) -> PydanticAIBackend:
    # Load optional dependencies only when this backend is selected.
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    print(f"Loading {args.model} on {device}...", file=sys.stderr, flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    weights = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=dtype,
    ).to(device)
    model = TransformersModel(
        weights, processor, model_name=args.model, max_new_tokens=args.max_tokens,
    )
    return PydanticAIBackend(model)


def error_details(exc: Exception) -> dict:
    result = {"type": type(exc).__name__, "message": str(exc), "causes": []}
    cause = exc.__cause__
    while cause is not None:
        result["causes"].append({"type": type(cause).__name__, "message": str(cause)})
        cause = cause.__cause__
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct", help="HF model ID or local weights directory")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps, etc.; one device")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")

    def unused_backend(request):
        raise RuntimeError("Validation must not call the model")

    try:
        config = json.loads(args.config.read_text())
        if not isinstance(config, dict):
            raise ValueError("Task configuration must be a JSON object")
        pipelines = build_pipelines(config, args.tasks, unused_backend)
        image_pipeline = Pipeline([LoadFile(), Decode()])
        image_pipeline.validate(strict=True)
        image = image_pipeline(str(args.input))
        started = perf_counter()
        backend = create_backend(args)
        load_seconds = perf_counter() - started
        pipelines = build_pipelines(config, args.tasks, backend)
    except Exception as exc:
        parser.exit(2, f"Setup failed: {exc}\n")

    report = {
        "input": str(args.input.resolve()), "image_size": image.size,
        "model": args.model,
        "device": str(backend.model.weights.device), "dtype": str(backend.model.weights.dtype),
        "load_seconds": load_seconds,
        "output_mode": "prompted", "max_tokens": args.max_tokens,
        "retries": 0, "do_sample": False,
        "task_config": {name: config[name] for name in pipelines}, "results": {},
    }
    failed = False
    for name, pipeline in pipelines.items():
        print(f"Running {name}...", file=sys.stderr, flush=True)
        started = perf_counter()
        try:
            result = {"prediction": asdict(pipeline(image))}
        except Exception as exc:
            failed = True
            result = {"error": error_details(exc)}
        result["elapsed_seconds"] = perf_counter() - started
        report["results"][name] = result
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
