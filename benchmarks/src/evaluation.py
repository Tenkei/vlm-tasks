"""Shared validation, dataset verification and response recording for benchmark runners."""
from __future__ import annotations

import json
from time import perf_counter

from ml_pipes import vlm

from .artifacts import digest, encoded, sample_record, serialize
from .task_registry import DATASETS, TASKS


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
    """Journal each task call, including responses that fail output validation."""

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
                    "InvalidModelResponseError": vlm.InvalidModelResponseError}[record["error"]["type"]]
            raise kind(record["error"]["message"])
        return vlm.ModelResponse(data=record["prediction"])


def task_settings(task_name):
    spec = TASKS[task_name]
    return {"geometry": spec.geometry} if spec.geometry else {}


def evaluate(task_name, backend, dataset, config, cache):
    """Use the same public validator and scoring settings for every backend."""
    task = getattr(vlm, TASKS[task_name].operator)(backend, **task_settings(task_name))
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
