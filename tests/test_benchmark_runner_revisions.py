"""Exercise run artifact writing with fake inference and no provider calls."""
import json
import sys
from types import SimpleNamespace

import datasets
from PIL import Image
import pytest

from benchmarks.src.artifacts import digest, encoded, sample_record, write_json
from ml_pipes import vlm


@pytest.mark.parametrize("backend", ["bedrock", "transformers"])
def test_runners_save_config_and_task_revisions_before_inference(tmp_path, monkeypatch, backend):
    from benchmarks.src import run_bedrock, run_transformers

    runner = run_bedrock if backend == "bedrock" else run_transformers
    dataset = datasets.Dataset.from_dict(
        {"image": [Image.new("RGB", (2, 3), "red"), Image.new("RGB", (2, 3), "blue")], "label": [0, 1]},
        features=datasets.Features({"image": datasets.Image(), "label": datasets.ClassLabel(names=["cat", "dog"])}),
    )
    identity = {"features": dataset.features.to_dict(),
                "samples": [sample_record(row, index) for index, row in enumerate(dataset)]}
    prepared, output = tmp_path / "prepared", tmp_path / "runs/test"
    dataset.save_to_disk(str(prepared / "cifar10"))
    write_json(prepared / "manifest.json", {
        "samples_per_task": 2, "datasets": {"cifar10": {
            **identity, "fingerprint_sha256": digest(encoded(identity).encode()),
        }},
    })
    config = {"model": "test-model", "revision": 4, "tasks": ["classify"],
              "samples_per_task": 2, "max_new_tokens": 4096, "temperature": 0,
              "profile": "test", "region": "us-east-1", "sdk_total_max_attempts": 1,
              "seed": 42, "device": "cpu", "dtype": "float32", "checkpoint_revision": "hf-checkpoint"}
    config_path = tmp_path / "model-config.json"
    write_json(config_path, config)
    model = SimpleNamespace(generations=[], attempted_requests=0, settings={},
                            profile={"prompted_output_template": "test template"})
    checkpoint_requests = []

    class Backend:
        def __init__(self, model):
            self.model = model

        def __call__(self, request):
            assert (output / "config.json").is_file()
            assert (output / "task-specs/classify.json").is_file()
            model.attempted_requests += 1
            model.generations.append({
                "input_tokens": 10, "output_tokens": 5, "raw_text": '{"labels":["cat"]}',
                "cache_read_tokens": 0, "cache_write_tokens": 0,
                "output_reasoning_tokens": 0, "unmeasured_requests": 0,
                "estimated_cost_usd": "0.001", "estimated_cost_source": "offline test",
            })
            return vlm.ModelResponse({"labels": ["cat"]})

    monkeypatch.setattr(vlm, "PydanticAIBackend", lambda model, **kwargs: Backend(model))
    monkeypatch.setattr(runner.metadata, "distributions", lambda: [])
    if backend == "bedrock":
        import boto3

        control = SimpleNamespace(get_foundation_model=lambda **kwargs: {"modelDetails": {
            "inputModalities": ["IMAGE"], "inferenceTypesSupported": ["ON_DEMAND"],
        }})
        runtime = SimpleNamespace(meta=SimpleNamespace(endpoint_url="https://bedrock.example"))
        monkeypatch.setattr(boto3, "Session", lambda **kwargs: SimpleNamespace(
            client=lambda service, **options: control if service == "bedrock" else runtime,
        ))
        monkeypatch.setattr(runner, "build_model", lambda *args: model)
    else:
        mps = SimpleNamespace(is_available=lambda: False)
        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
            backends=SimpleNamespace(mps=mps), get_num_threads=lambda: 1, float32="float32",
        ))
        weights = SimpleNamespace(
            config=SimpleNamespace(to_dict=lambda: {}), generation_config=SimpleNamespace(to_dict=lambda: {}),
            dtype="float32", device="cpu",
        )
        weights.to = lambda device: weights
        processor = SimpleNamespace(image_processor=SimpleNamespace(to_dict=lambda: {}), chat_template="test")

        def from_pretrained(value):
            def load(*args, **kwargs):
                checkpoint_requests.append(kwargs["revision"])
                return value
            return load

        monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
            set_seed=lambda seed: None,
            AutoProcessor=SimpleNamespace(from_pretrained=from_pretrained(processor)),
            AutoModelForImageTextToText=SimpleNamespace(from_pretrained=from_pretrained(weights)),
        ))
        monkeypatch.setattr(runner, "RecordingModel", lambda *args, **kwargs: model)

    assert runner.main(["--config", str(config_path), "--prepared", str(prepared), "--output", str(output)]) == 0
    assert json.loads((output / "config.json").read_text()) == config
    task_spec = json.loads((output / "task-specs/classify.json").read_text())
    report = json.loads((output / "classify.json").read_text())
    assert report["task_spec_revision"] == task_spec["revision"] == 1
    assert report["model_config_revision"] == 4
    assert report["revision"] == ("hf-checkpoint" if backend == "transformers" else None)
    if backend == "transformers":
        assert checkpoint_requests == ["hf-checkpoint", "hf-checkpoint"]
    assert model.attempted_requests == 2
    assert (output / "completed.json").is_file()
