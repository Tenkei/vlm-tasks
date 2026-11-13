"""Run a frozen suite through public task validation and record every generation."""
from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
from time import perf_counter

from ml_pipes import vlm

from .common import DATASETS, digest, encoded, provenance, sample_record, serialize, utc_now, write_json


class RecordingModel(vlm.TransformersModel):
    """Observe the existing adapter before PydanticAI parses the response."""

    def __init__(self, *args, **kwargs):
        self.generations = []
        super().__init__(*args, **kwargs)

    def _generate(self, messages, info):
        started = perf_counter()
        response = super()._generate(messages, info)
        self.generations.append({
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "raw_text": "\n".join(part.content for part in response.parts),
            "generation_seconds": perf_counter() - started,
        })
        return response


class RecordingBackend:
    def __init__(self, backend, stream):
        self.backend, self.stream = backend, stream
        self.index = 0

    def __call__(self, request):
        model = self.backend.model
        before = len(model.generations)
        started = perf_counter()
        record = {"index": self.index, "instruction": request.instruction,
                  "response_schema": request.response_spec.schema}
        self.index += 1
        try:
            response = self.backend(request)
            record["prediction"] = response.data
            return response
        except Exception as exc:
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            record["elapsed_seconds"] = perf_counter() - started
            record["generations"] = model.generations[before:]
            self.stream.write(json.dumps(record, default=serialize) + "\n")
            self.stream.flush()


def usage_summary(records, attempted):
    return {
        "attempted_requests": attempted, "measured_requests": len(records),
        "unmeasured_requests": attempted - len(records),
        "input_tokens": sum(row["input_tokens"] for row in records),
        "output_tokens": sum(row["output_tokens"] for row in records),
        "total_tokens": sum(row["input_tokens"] + row["output_tokens"] for row in records),
        "estimated_cost_usd": None, "reported_cost_usd": None,
        "cost_note": "Local inference; electricity/hardware costs not measured",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("benchmarks/qwen3.json"))
    parser.add_argument("--prepared", type=Path, default=Path("benchmarks/.cache/prepared100"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=DATASETS, help="Run selected tasks in a fresh output directory")
    parser.add_argument("--offline", action="store_true", help="Require already cached model files")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    tasks = args.tasks or config["tasks"]
    if args.output.exists():
        raise FileExistsError(f"Use a new run directory: {args.output}")
    import datasets
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed

    set_seed(config["seed"])
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    if config["samples_per_task"] != manifest["samples_per_task"]:
        raise ValueError("Configured sample count differs from the frozen datasets")
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this process")
    loaded = {}
    for name in dict.fromkeys(DATASETS[task] for task in tasks):
        dataset = datasets.load_from_disk(str(args.prepared / name))
        identity = {"features": dataset.features.to_dict(),
                    "samples": [sample_record(row, i) for i, row in enumerate(dataset)]}
        if digest(encoded(identity).encode()) != manifest["datasets"][name]["fingerprint_sha256"]:
            raise ValueError(f"Prepared samples changed: {name}")
        loaded[name] = dataset
    write_json(args.output / "config.json", {**config, "tasks": tasks})
    write_json(args.output / "manifest.json", manifest)
    packages = {dist.metadata["Name"]: dist.version for dist in metadata.distributions() if dist.metadata["Name"]}
    environment = {
        "started_at": utc_now(), "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "packages": packages, "device": device,
        "torch_threads": torch.get_num_threads(), "mps_available": torch.backends.mps.is_available(),
        **provenance(),
    }
    write_json(args.output / "environment.json", environment)
    (args.output / "requirements.txt").write_text("\n".join(f"{k}=={v}" for k, v in sorted(packages.items())) + "\n")
    print(f"Loading {config['model']}@{config['revision']} on {device}", flush=True)
    started = perf_counter()
    options = {"revision": config["revision"], "local_files_only": args.offline}
    processor = AutoProcessor.from_pretrained(config["model"], **options)
    weights = AutoModelForImageTextToText.from_pretrained(
        config["model"], dtype=getattr(torch, config["dtype"]), **options,
    ).to(device)
    model = RecordingModel(weights, processor, model_name=config["model"], max_new_tokens=config["max_new_tokens"])
    load_seconds = perf_counter() - started
    write_json(args.output / "model.json", {
        "load_seconds": load_seconds, "config": weights.config.to_dict(),
        "generation_config": weights.generation_config.to_dict(),
        "image_processor": processor.image_processor.to_dict(),
        "chat_template": processor.chat_template,
        "prompted_output_template": model.profile["prompted_output_template"],
        "dtype": str(weights.dtype), "device": str(weights.device),
        "do_sample": False, "output_mode": "prompted", "retries": 0,
    })
    classes = {
        "classify": vlm.Classify, "detect": vlm.Detect, "ground_box": vlm.Ground,
        "ground_point": vlm.Ground, "attributes": vlm.Attributes, "answer": vlm.Answer, "describe": vlm.Describe,
    }
    for task_name in tasks:
        print(f"Running {task_name}: {config['samples_per_task']} samples", flush=True)
        dataset_name = DATASETS[task_name]
        dataset = loaded[dataset_name]
        before = len(model.generations)
        with (args.output / f"{task_name}.jsonl").open("w") as stream:
            backend = RecordingBackend(vlm.PydanticAIBackend(model, retries=0), stream)
            settings = {"geometry": task_name.removeprefix("ground_")} if task_name.startswith("ground_") else {}
            task = classes[task_name](backend, **settings)
            val_settings = {"dataset": dataset, "cache_dir": args.prepared.parent / "downloads"}
            if task_name == "detect":
                val_settings["match_iou"] = config["match_iou"]
            if task_name == "describe":
                val_settings["tokenizer_version"] = config["tokenizer_version"]
            started = perf_counter()
            metrics = task.val(**val_settings)
            wall_seconds = perf_counter() - started
        if metrics.seen != config["samples_per_task"] or backend.index != metrics.seen:
            raise RuntimeError("Benchmark did not evaluate the requested number of samples")
        report = {
            "task": task_name, "dataset": dataset_name, "model": config["model"],
            "revision": config["revision"], "device": device, "dtype": config["dtype"],
            "completed_at": utc_now(), "metrics": metrics.results_dict,
            "summary": metrics.summary(), "failures": metrics.failures,
            "dataset_counts": metrics.dataset_counts, "dataset_metadata": metrics.dataset_metadata,
            "status_counts": getattr(metrics, "status_counts", {}), "task_settings": settings,
            "dataset_fingerprint_sha256": manifest["datasets"][dataset_name]["fingerprint_sha256"],
            "speed_ms_per_sample": metrics.speed, "elapsed_seconds": metrics.elapsed_seconds,
            "wall_seconds": wall_seconds, "shared_model_load_seconds": load_seconds,
            "usage": usage_summary(model.generations[before:], backend.index),
        }
        write_json(args.output / f"{task_name}.json", report)
        from .summarize import summarize
        summarize(args.output.parent)
        print(f"Completed {task_name}: {report['metrics']}; usage={report['usage']}", flush=True)
    write_json(args.output / "completed.json", {"completed_at": utc_now(), "tasks": tasks})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
