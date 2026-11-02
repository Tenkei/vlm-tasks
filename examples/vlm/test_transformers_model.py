"""Exercise the real PydanticAI path with fake weights; no model downloads."""
from __future__ import annotations

from contextlib import nullcontext
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pydantic_ai")
pytest.importorskip("PIL")

from ml_pipes.core import Pipeline
from ml_pipes.vision import ImagePayload
from ml_pipes.vlm import (
    Classify, ClassifyMultiLabel, Detect, Ground, PredictAttributes,
    InvalidModelResponseError, ModelExecutionError, PydanticAIBackend,
)
from examples.vlm.transformers_model import TransformersModel


class Batch(dict):
    def to(self, device):
        assert device == "cpu"
        return self


class Processor:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.chats = []

    def apply_chat_template(self, chat, **kwargs):
        assert kwargs == dict(tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        # Transformers' multimodal processor iterates content blocks for every role.
        assert all(isinstance(message["content"], list) for message in chat)
        self.chats.append(chat)
        return Batch(input_ids=np.array([[101, 102, 103]]))

    def batch_decode(self, ids, **kwargs):
        # The prompt must have been stripped, and raw output must be unmodified.
        assert ids.tolist() == [[201, 202]]
        assert kwargs == dict(skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [next(self.outputs)]


class Weights:
    config = SimpleNamespace(is_encoder_decoder=False)
    device = "cpu"
    dtype = "float32"

    def eval(self):
        return self

    def generate(self, **kwargs):
        assert kwargs["max_new_tokens"] == 128
        assert kwargs["do_sample"] is False
        return np.array([[101, 102, 103, 201, 202]])


@pytest.fixture
def image():
    return ImagePayload(np.full((10, 20, 3), [10, 20, 30], dtype=np.uint8))


@pytest.fixture
def adapter(monkeypatch):
    # Exercise chat translation, parsing and validation without requiring PyTorch.
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))

    def make(outputs, retries=0, weights=None, output_mode="prompted"):
        processor = Processor(outputs)
        model = TransformersModel(weights or Weights(), processor, model_name="local-test", max_new_tokens=128)
        return PydanticAIBackend(model, retries=retries, output_mode=output_mode), processor

    return make


def test_all_tasks_receive_schema_and_rgb_image_with_fresh_history(adapter, image):
    outputs = [
        {"labels": ["clean"]}, {"labels": ["clean", "intact"]},
        {"detections": [{"label": "window", "box": {"x1": .1, "y1": .2, "x2": .8, "y2": .9}}]},
        {"status": "found", "box": {"x1": .1, "y1": .2, "x2": .8, "y2": .9}},
        {"attributes": {"state": "clean"}},
    ]
    backend, processor = adapter([json.dumps(value) for value in outputs])
    tasks = [Classify(backend, ["clean", "dirty"]), ClassifyMultiLabel(backend, ["clean", "intact"]),
             Detect(backend, ["window"]), Ground(backend, "the window"),
             PredictAttributes(backend, {"state": ["clean", "dirty"]})]
    results = []
    for task in tasks:
        pipeline = Pipeline([task])
        pipeline.validate(strict=True)
        results.append(pipeline(image))
    assert results[0].labels == ("clean",)
    assert results[1].labels == ("clean", "intact")
    assert results[2].detections[0].box.x2 == 16
    assert results[3].box.y2 == 9
    assert results[4].attributes == {"state": "clean"}
    for chat in processor.chats:
        assert [message["role"] for message in chat] == ["system", "user"]
        assert '"properties"' in chat[0]["content"][0]["text"]
        assert '"required"' in chat[0]["content"][0]["text"]
        content = chat[1]["content"]
        assert content[0]["type"] == "text"
        assert content[1]["image"].size == (20, 10)
        assert content[1]["image"].mode == "RGB"
        assert content[1]["image"].getpixel((0, 0)) == (30, 20, 10)


def test_schema_failure_retries_with_response_and_feedback(adapter, image):
    backend, processor = adapter(['{"labels": ["unknown"]}', '{"labels": ["clean"]}'], retries=1)
    assert Pipeline([Classify(backend, ["clean", "dirty"])])(image).labels == ("clean",)
    assert len(processor.chats) == 2
    retried = processor.chats[1]
    assert [message["role"] for message in retried] == ["system", "user", "assistant", "user"]
    assert '"unknown"' in retried[2]["content"][0]["text"]
    feedback = retried[3]["content"]
    if isinstance(feedback, list):
        feedback = "\n".join(item["text"] for item in feedback)
    assert "labels" in feedback
    assert "clean" in feedback


@pytest.mark.parametrize("response", ['not JSON', '{"labels": ["unknown"]}'])
def test_invalid_output_is_not_repaired(adapter, image, response):
    backend, processor = adapter([response])
    with pytest.raises(InvalidModelResponseError) as caught:
        Pipeline([Classify(backend, ["clean", "dirty"])])(image)
    assert caught.value.__cause__ is not None
    assert len(processor.chats) == 1


def test_generation_failure_preserves_original_cause(adapter, image):
    class BrokenWeights(Weights):
        def generate(self, **kwargs):
            raise RuntimeError("generation failed")

    backend, _ = adapter([], weights=BrokenWeights())
    with pytest.raises(ModelExecutionError) as caught:
        Pipeline([Classify(backend, ["clean", "dirty"])])(image)
    assert str(caught.value.__cause__) == "generation failed"


@pytest.mark.parametrize("mode", ["native", "tool"])
def test_unsupported_output_modes_do_not_call_weights(adapter, image, mode):
    backend, processor = adapter([], output_mode=mode)
    with pytest.raises(ModelExecutionError):
        Pipeline([Classify(backend, ["clean", "dirty"])])(image)
    assert processor.chats == []


def test_cli_continues_after_invalid_task_and_saves_causes(adapter, tmp_path, monkeypatch):
    import cv2
    from examples.vlm import run_transformers

    image = tmp_path / "input.png"
    cv2.imwrite(str(image), np.zeros((10, 20, 3), dtype=np.uint8))
    config = tmp_path / "tasks.json"
    config.write_text(json.dumps({"classify": {"labels": ["clean", "dirty"]},
                                  "ground": {"target": "the window"}}))
    backend, processor = adapter(['{"labels": ["unknown"]}', '{"status": "not_found", "box": null}'])
    loads = []

    def create(args):
        loads.append(args.model)
        return backend

    monkeypatch.setattr(run_transformers, "create_backend", create)
    output = tmp_path / "results.json"
    status = run_transformers.main([
        "--input", str(image), "--config", str(config), "--tasks", "classify", "ground",
        "--output", str(output),
    ])
    report = json.loads(output.read_text())
    assert status == 1
    assert len(loads) == 1
    assert report["results"]["classify"]["error"]["type"] == "InvalidModelResponseError"
    assert report["results"]["classify"]["error"]["causes"]
    assert report["results"]["ground"]["prediction"]["status"] == "not_found"
    assert len(processor.chats) == 2


def test_cli_dry_run_never_loads_weights(tmp_path, monkeypatch):
    import cv2
    from examples.vlm import run_transformers

    image = tmp_path / "input.png"
    cv2.imwrite(str(image), np.zeros((10, 20, 3), dtype=np.uint8))

    def unexpected_load(args):
        pytest.fail("Dry run must not load model dependencies or weights")

    monkeypatch.setattr(run_transformers, "create_backend", unexpected_load)
    assert run_transformers.main(["--input", str(image), "--dry-run"]) == 0
