"""Run the frozen comparison suite on Bedrock, retaining usage before parsing."""
from __future__ import annotations

import argparse
from decimal import Decimal
from importlib import metadata
import json
from pathlib import Path
import platform
from time import perf_counter

from pydantic_ai.messages import TextPart, ThinkingPart
from pydantic_ai.models.bedrock import BedrockConverseModel

from ml_pipes import vlm
from ml_pipes.vlm.transformer import _PROMPTED_OUTPUT_TEMPLATE

from .artifacts import provenance, utc_now, write_json
from .evaluation import RecordingBackend, evaluate, load_prepared
from .task_registry import DATASETS


class RecordingBedrockModel(BedrockConverseModel):
    """Observe provider responses before agent output validation can fail."""

    def __init__(self, *args, pricing=None, **kwargs):
        self.generations = []
        self.pricing = pricing
        super().__init__(*args, **kwargs)

    async def request(self, messages, model_settings, model_request_parameters):
        started = perf_counter()
        response = await super().request(messages, model_settings, model_request_parameters)
        if self.pricing is not None:
            estimated_cost = str(priced_usage(response.usage, self.pricing))
        else:
            try:
                estimated_cost = str(response.cost().total_price)
            except (LookupError, ValueError, AssertionError):
                estimated_cost = None
        self.generations.append({
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_read_tokens": response.usage.cache_read_tokens,
            "cache_write_tokens": response.usage.cache_write_tokens,
            "output_reasoning_tokens": response.usage.output_reasoning_tokens,
            "unmeasured_requests": response.usage.unmeasured_requests,
            "raw_text": "\n".join(part.content for part in response.parts if isinstance(part, TextPart)),
            "reasoning_text": "\n".join(part.content for part in response.parts if isinstance(part, ThinkingPart)),
            "generation_seconds": perf_counter() - started,
            "finish_reason": response.finish_reason,
            "provider_details": response.provider_details,
            "provider_response_id": response.provider_response_id,
            "estimated_cost_usd": estimated_cost,
            "estimated_cost_source": self.pricing["source"] if self.pricing else "genai-prices",
        })
        return response


class InfrastructureCheckedBackend(RecordingBackend):
    def __call__(self, request):
        try:
            return super().__call__(request)
        except vlm.ModelExecutionError as exc:
            # Authentication, permissions and transport errors must not turn an
            # entire dataset into misleading model-quality failures.
            raise RuntimeError(f"Bedrock execution stopped; inspect saved request: {exc}") from exc


def priced_usage(usage, pricing):
    """Price total usage, charging cached input at its own documented rates."""
    rates = {key: Decimal(value) for key, value in pricing["rates_usd_per_million_tokens"].items()}
    if not all(value.is_finite() and value >= 0 for value in rates.values()):
        raise ValueError("Token prices must be finite and nonnegative")
    uncached = usage.input_tokens - usage.cache_read_tokens - usage.cache_write_tokens
    if uncached < 0:
        raise ValueError("Cached input exceeds total input tokens")
    return (uncached * rates["input"] + usage.cache_read_tokens * rates["cache_read"]
            + usage.cache_write_tokens * rates["cache_write"]
            + usage.output_tokens * rates["output"]) / Decimal(1_000_000)


def usage_summary(records, attempted):
    unmeasured = attempted - len(records) + sum(r["unmeasured_requests"] for r in records)
    costs = [Decimal(r["estimated_cost_usd"]) for r in records if r["estimated_cost_usd"] is not None]
    result = {
        "attempted_requests": attempted, "measured_requests": attempted - unmeasured,
        "unmeasured_requests": unmeasured,
        **{key: sum(r[key] for r in records) for key in (
            "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "output_reasoning_tokens",
        )},
        "estimated_cost_usd": str(sum(costs, Decimal(0))) if costs else None,
        "costed_requests": len(costs),
        "estimated_cost_complete": len(costs) == attempted and unmeasured == 0,
        "reported_cost_usd": None,
        "cost_note": "Estimate from recorded pricing sources; Bedrock does not return invoiced charges",
        "cost_sources": sorted({r.get("estimated_cost_source", "genai-prices") for r in records}),
    }
    result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    return result


def inference_target(client, config, details):
    """Resolve a configured route without silently substituting another model."""
    profile_id = config.get("inference_profile")
    if profile_id is None:
        if "ON_DEMAND" not in details["inferenceTypesSupported"]:
            raise ValueError("This model requires an inference_profile in the benchmark config")
        return config["model"], None
    profile = client.get_inference_profile(inferenceProfileIdentifier=profile_id)
    profile.pop("ResponseMetadata", None)
    if profile["status"] != "ACTIVE":
        raise ValueError("Configured inference profile is not active")
    targets = {model["modelArn"].rsplit("/", 1)[-1] for model in profile["models"]}
    if targets != {config["model"]}:
        raise ValueError("Inference profile does not exclusively route to the configured model")
    return profile_id, json.loads(json.dumps(profile, default=str))


def remaining_tasks(output, config, manifest):
    """Resume at a task boundary; never repeat a possibly billed request."""
    if (output / "completed.json").exists():
        raise ValueError("This benchmark is already complete")
    if json.loads((output / "config.json").read_text()) != config:
        raise ValueError("Resume configuration differs from the saved run")
    if json.loads((output / "manifest.json").read_text()) != manifest:
        raise ValueError("Resume prepared data differs from the saved run")
    pending = []
    for task in config["tasks"]:
        report_path = output / f"{task}.json"
        journal = output / f"{task}.jsonl"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            calls = [json.loads(line) for line in journal.read_text().splitlines()]
            if (pending or report["model"] != config["model"]
                    or report["task"] != task
                    or report["metrics"]["samples"] != config["samples_per_task"]
                    or len(calls) != config["samples_per_task"]
                    or [call["index"] for call in calls] != list(range(config["samples_per_task"]))
                    or report["dataset_fingerprint_sha256"] != manifest["datasets"][DATASETS[task]]["fingerprint_sha256"]):
                raise ValueError(f"Cannot verify completed task: {task}")
        else:
            if journal.exists() and journal.stat().st_size:
                raise ValueError(f"Task {task} has an interrupted request journal; preserve and inspect it before resuming")
            pending.append(task)
    return pending


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("benchmarks/configs/qwen3-vl-235b-a22b-bedrock.json"))
    parser.add_argument("--prepared", type=Path, default=Path("benchmarks/.cache/prepared-datasets-100"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Continue untouched tasks after a task-boundary interruption")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Use a new run directory: {args.output}")
    import boto3
    from botocore.config import Config
    from pydantic_ai.providers.bedrock import BedrockProvider

    manifest = json.loads((args.prepared / "manifest.json").read_text())
    if config["samples_per_task"] != manifest["samples_per_task"]:
        raise ValueError("Configured sample count differs from the frozen datasets")
    tasks = remaining_tasks(args.output, config, manifest) if args.resume else config["tasks"]
    loaded = load_prepared(args.prepared, manifest, config["tasks"])
    session = boto3.Session(profile_name=config["profile"], region_name=config["region"])
    sdk_config = Config(connect_timeout=10, read_timeout=600,
                        retries={"total_max_attempts": config["sdk_total_max_attempts"], "mode": "standard"})
    control = session.client("bedrock", config=sdk_config)
    details = control.get_foundation_model(modelIdentifier=config["model"])["modelDetails"]
    details = json.loads(json.dumps(details, default=str))
    if "IMAGE" not in details["inputModalities"]:
        raise ValueError("The selected model does not support image input")
    target, inference_profile = inference_target(control, config, details)
    client = session.client("bedrock-runtime", config=sdk_config)
    model = RecordingBedrockModel(
        target, provider=BedrockProvider(bedrock_client=client),
        pricing=config.get("pricing"),
        settings={"temperature": config["temperature"], "max_tokens": config["max_new_tokens"]},
        profile={"prompted_output_template": _PROMPTED_OUTPUT_TEMPLATE, "json_schema_transformer": None},
    )
    metadata_output = args.output
    if args.resume:
        metadata_output = args.output / "continuations" / utc_now().replace(":", "-")
    else:
        write_json(args.output / "config.json", config)
        write_json(args.output / "manifest.json", manifest)
    packages = {d.metadata["Name"]: d.version for d in metadata.distributions() if d.metadata["Name"]}
    write_json(metadata_output / "environment.json", {
        "started_at": utc_now(), "python": platform.python_version(), "platform": platform.platform(),
        "backend": "bedrock", "region": config["region"], "profile": config["profile"],
        "packages": packages, **provenance(),
    })
    (metadata_output / "requirements.txt").write_text("\n".join(f"{k}=={v}" for k, v in sorted(packages.items())) + "\n")
    model_metadata = {
        "model_details": details, "endpoint": client.meta.endpoint_url,
        "inference_model_id": target, "inference_profile": inference_profile,
        "pricing": config.get("pricing"),
        "settings": model.settings, "prompted_output_template": _PROMPTED_OUTPUT_TEMPLATE,
        "json_schema_transformer": None, "output_mode": "prompted", "retries": 0,
        "sdk_total_max_attempts": config["sdk_total_max_attempts"], "read_timeout_seconds": 600,
        "checkpoint_revision": None, "dtype": None, "device": None, "seed": None,
        "note": "Bedrock manages checkpoint revision, hardware, dtype, image processing and sampling seed",
    }
    if args.resume:
        original = json.loads((args.output / "model.json").read_text())
        for key in ("inference_model_id", "settings", "prompted_output_template", "pricing", "output_mode", "retries"):
            if original.get(key) != model_metadata.get(key):
                raise ValueError(f"Resume model setting differs: {key}")
        write_json(metadata_output / "resume.json", {"tasks": tasks, "reason": "Continue untouched tasks; completed task artifacts retained"})
    write_json(metadata_output / "model.json", model_metadata)
    for task_name in tasks:
        print(f"Running {task_name}: {config['samples_per_task']} frozen samples on {config['model']}", flush=True)
        before = len(model.generations)
        with (args.output / f"{task_name}.jsonl").open("w") as stream:
            backend = InfrastructureCheckedBackend(vlm.PydanticAIBackend(model, retries=0), stream)
            started = perf_counter()
            metrics = evaluate(task_name, backend, loaded[DATASETS[task_name]], config, args.prepared.parent / "downloads")
            wall_seconds = perf_counter() - started
        if metrics.seen != config["samples_per_task"] or backend.index != metrics.seen:
            raise RuntimeError("Benchmark did not evaluate the requested number of samples")
        report = {
            "task": task_name, "dataset": DATASETS[task_name], "model": config["model"],
            "inference_model_id": target,
            "revision": None, "device": f"bedrock/{config['region']}", "dtype": "provider-managed",
            "backend": "bedrock", "region": config["region"], "completed_at": utc_now(),
            "execution_environment": str((metadata_output / "environment.json").relative_to(args.output)),
            "metrics": metrics.results_dict, "summary": metrics.summary(), "failures": metrics.failures,
            "dataset_counts": metrics.dataset_counts, "dataset_metadata": metrics.dataset_metadata,
            "status_counts": getattr(metrics, "status_counts", {}),
            "dataset_fingerprint_sha256": manifest["datasets"][DATASETS[task_name]]["fingerprint_sha256"],
            "speed_ms_per_sample": metrics.speed, "elapsed_seconds": metrics.elapsed_seconds,
            "wall_seconds": wall_seconds, "usage": usage_summary(model.generations[before:], backend.index),
        }
        write_json(args.output / f"{task_name}.json", report)
        from .generate_report import generate_report
        generate_report(args.output.parent)
        print(f"Completed {task_name}: {report['metrics']}; usage={report['usage']}", flush=True)
    write_json(args.output / "completed.json", {"completed_at": utc_now(), "tasks": config["tasks"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
