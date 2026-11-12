"""Run VLM task pipelines on one image using an application's AWS credentials."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from ml_pipes.core import Pipeline
from ml_pipes.vision import Decode, ImagePayload, LoadFile
from ml_pipes.vlm import (
    Answer, Attributes, Classify, ClassifyMultiLabel, Describe, Detect, Ground, ModelBackend,
    PydanticAIBackend,
)

TASKS = {
    "classify": Classify,
    "multilabel": ClassifyMultiLabel,
    "detect": Detect,
    "ground": Ground,
    "attributes": Attributes,
    "describe": Describe,
    "answer": Answer,
}
DEFAULT_CONFIG = Path(__file__).with_name("image_tasks.json")


def create_backend(args: argparse.Namespace) -> PydanticAIBackend:
    # Load optional dependencies only when this backend is selected.
    import boto3
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    client = session.client("bedrock-runtime")
    model = BedrockConverseModel(
        args.model,
        provider=BedrockProvider(bedrock_client=client),
        settings={"temperature": 0.0, "max_tokens": args.max_tokens},
    )
    return PydanticAIBackend(model)


def build_pipelines(config: dict, names: list[str], backend: ModelBackend) -> dict[str, Pipeline]:
    pipelines = {}
    for name in names:
        settings = config.get(name)
        if not isinstance(settings, dict):
            raise ValueError(f"Configuration must contain an object for task {name!r}")
        pipeline = Pipeline([TASKS[name](backend, **settings)])
        pipeline.validate(pipeline_input_type=ImagePayload, strict=True)
        pipelines[name] = pipeline
    return pipelines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Image file to evaluate")
    parser.add_argument("--model", required=True, help="Vision-capable Bedrock model or inference profile ID")
    parser.add_argument("--profile", help="AWS profile; defaults to the normal boto3 credential chain")
    parser.add_argument("--region", help="AWS region; defaults to the environment or AWS profile")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Task constructor settings as JSON")
    parser.add_argument("--output", type=Path, help="Also save the JSON report to this path")
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args(argv)
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")

    def unused_backend(request):
        raise RuntimeError("Validation must not call the model")

    try:
        config = json.loads(args.config.read_text())
        if not isinstance(config, dict):
            raise ValueError("Task configuration must be a JSON object")
        # Validate all local inputs before discovering credentials or making requests.
        pipelines = build_pipelines(config, args.tasks, unused_backend)
        image_pipeline = Pipeline([LoadFile(), Decode()])
        image_pipeline.validate(strict=True)
        image = image_pipeline(str(args.input))
        backend = create_backend(args)
        pipelines = build_pipelines(config, args.tasks, backend)
    except Exception as exc:
        parser.exit(2, f"Setup failed: {exc}\n")

    report = {
        "input": str(args.input.resolve()), "image_size": image.size,
        "model": args.model, "profile": args.profile, "requested_region": args.region,
        "output_mode": "prompted", "retries": 0,
        "max_tokens": args.max_tokens, "task_config": {name: config[name] for name in pipelines},
        "results": {},
    }
    failed = False
    for name, pipeline in pipelines.items():
        print(f"Running {name}...", file=sys.stderr, flush=True)
        started = perf_counter()
        try:
            prediction = asdict(pipeline(image))
            result = {"prediction": prediction}
        except Exception as exc:
            # Keep other tasks' results when one model response fails validation.
            failed = True
            result = {"error": {"type": type(exc).__name__, "message": str(exc)}}
        result["elapsed_seconds"] = perf_counter() - started
        report["results"][name] = result

    serialized = json.dumps(report, indent=2, default=str)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
