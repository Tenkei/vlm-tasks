"""Shared validation, dataset verification and response recording for benchmark runners."""
from __future__ import annotations

import json
from time import perf_counter

from ml_pipes import vlm

from .artifacts import (
    check_run_setup, digest, encoded, revision_fields, run_config_path,
    run_setup_directory, sample_record, serialize,
)
from .task_registry import DATASETS, TASKS


class RecordingModel(vlm.TransformersModel):
    """Observe the existing adapter before PydanticAI parses the response."""

    def __init__(self, *args, **kwargs):
        self.generations = []
        self.attempted_requests = 0
        super().__init__(*args, **kwargs)

    def _generate(self, messages, info):
        self.attempted_requests += 1
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
    """Journal each task call, including responses that fail output validation."""

    def __init__(self, backend, stream):
        self.backend, self.stream = backend, stream
        self.index = 0
        self.attempted_requests = 0

    def __call__(self, request):
        model = self.backend.model
        before = len(model.generations)
        attempts_before = model.attempted_requests
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
            record["attempted_requests"] = model.attempted_requests - attempts_before
            self.attempted_requests += record["attempted_requests"]
            record["elapsed_seconds"] = perf_counter() - started
            record["generations"] = model.generations[before:]
            self.stream.write(json.dumps(record, default=serialize) + "\n")
            self.stream.flush()


def usage_summary(records, attempted):
    """Summarize local usage; hardware and electricity costs are unmeasured."""
    return {
        "attempted_requests": attempted, "measured_requests": len(records),
        "unmeasured_requests": attempted - len(records),
        "input_tokens": sum(row["input_tokens"] for row in records),
        "output_tokens": sum(row["output_tokens"] for row in records),
        "total_tokens": sum(row["input_tokens"] + row["output_tokens"] for row in records),
        "estimated_cost_usd": None, "reported_cost_usd": None,
        "cost_note": "Local inference; electricity/hardware costs not measured",
    }


def max_retries(config):
    retries = config.get("retries", 0)
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("retries must be a nonnegative integer")
    return retries


def remaining_tasks(output, config, manifest):
    """Resume at a task boundary; never repeat recorded model requests."""
    if (output / "completed.json").exists():
        raise ValueError("This benchmark is already complete")
    if json.loads(run_config_path(output).read_text()) != config:
        raise ValueError("Resume configuration differs from the saved run")
    if json.loads((output / "manifest.json").read_text()) != manifest:
        raise ValueError("Resume prepared data differs from the saved run")
    versioned = (run_setup_directory(output) / "task-specs").is_dir() or isinstance(config.get("revision"), int)
    if versioned:
        check_run_setup(output, config)
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
            if versioned and any(
                report.get(key) != value for key, value in revision_fields(task, config).items()
            ):
                raise ValueError(f"Cannot verify completed task revisions: {task}")
        else:
            if journal.exists() and journal.stat().st_size:
                raise ValueError(f"Task {task} has an interrupted request journal; preserve and inspect it before resuming")
            pending.append(task)
    return pending


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
                    "InvalidModelResponseError": vlm.InvalidModelResponseError,
                    "ResponseSyntaxError": vlm.ResponseSyntaxError,
                    "ResponseSchemaError": vlm.ResponseSchemaError,
                    # Preserve replay of journals written before the rename.
                    "ResponseContentError": vlm.ResponseConstraintError,
                    "ResponseConstraintError": vlm.ResponseConstraintError}[record["error"]["type"]]
            raise kind(record["error"]["message"])
        return vlm.ModelResponse(data=record["prediction"])


def task_settings(task_name):
    spec = TASKS[task_name]
    settings = {}
    if spec.geometry:
        settings["geometry"] = spec.geometry
    if spec.context is not None:
        settings["context"] = spec.context
    return settings


def evaluate(task_name, backend, dataset, config, cache):
    """Use the same public validator and scoring settings for every backend."""
    settings = task_settings(task_name)
    if task_name in config.get("task_context", {}):
        settings["context"] = config["task_context"][task_name]
    task = getattr(vlm, TASKS[task_name].operator)(backend, **settings)
    options = {"dataset": dataset, "cache_dir": cache}
    if task_name == "detect":
        options["match_iou"] = config["match_iou"]
    if task_name == "describe":
        options["tokenizer_version"] = config["tokenizer_version"]
    return task.val(**options)


def load_prepared(prepared, manifest, tasks):
    """Load each frozen dataset once and verify its images, targets and features."""
    from datasets import load_from_disk

    loaded = {}
    for name in dict.fromkeys(DATASETS[task] for task in tasks):
        dataset = load_from_disk(str(prepared / name))
        identity = {"features": dataset.features.to_dict(),
                    "samples": [sample_record(row, i) for i, row in enumerate(dataset)]}
        if digest(encoded(identity).encode()) != manifest["datasets"][name]["fingerprint_sha256"]:
            raise ValueError(f"Prepared samples changed: {name}")
        loaded[name] = dataset
    return loaded
