"""Bedrock benchmark usage accounting through the real parser, without AWS calls."""
from io import StringIO
import json

import boto3
import numpy as np
import pytest

from botocore.stub import Stubber
from pydantic_ai.providers.bedrock import BedrockProvider

from benchmarks.src.run_bedrock import InfrastructureCheckedBackend, RecordingBedrockModel, inference_target, priced_usage, remaining_tasks, usage_summary
from ml_pipes.vlm import Classify, InvalidModelResponseError, PydanticAIBackend
from ml_pipes.vlm.transformer import _PROMPTED_OUTPUT_TEMPLATE


@pytest.mark.parametrize("model_id, temperature", [
    ("qwen.qwen3-vl-235b-a22b", 0),
    ("google.gemma-3-27b-it", 0),
    ("global.moonshotai.kimi-k3.1", None),
])
def test_bedrock_benchmark_counts_invalid_output_and_preserves_prompt_schema(model_id, temperature):
    client = boto3.client("bedrock-runtime", region_name="ap-northeast-1",
                          aws_access_key_id="test", aws_secret_access_key="test")
    stubber = Stubber(client)
    for text in ("not JSON", '{"labels":["cat"]}'):
        stubber.add_response("converse", {
            "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
            "stopReason": "end_turn", "usage": {"inputTokens": 100, "outputTokens": 10, "totalTokens": 110},
            "metrics": {"latencyMs": 5},
        })
    requests = []
    client.meta.events.register("before-parameter-build.bedrock-runtime.Converse",
                                lambda params, **kwargs: requests.append(params))
    pricing = None
    if "kimi-k3.1" in model_id:
        from pathlib import Path
        pricing = json.loads(Path("benchmarks/configs/kimi-k3.1-bedrock.json").read_text())["pricing"]
    model = RecordingBedrockModel(
        model_id, provider=BedrockProvider(bedrock_client=client), pricing=pricing,
        settings={"temperature": temperature, "max_tokens": 4096},
        profile={"prompted_output_template": _PROMPTED_OUTPUT_TEMPLATE, "json_schema_transformer": None},
    )
    stream = StringIO()
    backend = InfrastructureCheckedBackend(PydanticAIBackend(model, retries=0), stream)
    task = Classify(backend, labels=["cat", "dog"])
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    with stubber:
        with pytest.raises(InvalidModelResponseError):
            task(image)
        assert task(image).labels == ("cat",)
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert records[0]["generations"][0]["raw_text"] == "not JSON"
    usage = usage_summary(model.generations, backend.index)
    assert usage["input_tokens"] == 200 and usage["output_tokens"] == 20
    assert usage["measured_requests"] == 2 and usage["unmeasured_requests"] == 0
    assert usage["estimated_cost_usd"] is not None and usage["estimated_cost_complete"]
    assert usage["reported_cost_usd"] is None
    for request in requests:
        assert request["inferenceConfig"]["maxTokens"] == 4096
        assert request["modelId"] == model_id
        if temperature is None:
            assert "temperature" not in request["inferenceConfig"]
        else:
            assert request["inferenceConfig"]["temperature"] == temperature
        instructions = "\n".join(part["text"] for part in request["system"])
        assert "Produce an instance of the schema" in instructions
        schema = json.loads(instructions.split("JSON schema:\n", 1)[1])
        assert schema["title"] == task._spec.schema["title"]
        assert schema["properties"] == task._spec.schema["properties"]


def test_bedrock_benchmark_aborts_on_infrastructure_errors():
    client = boto3.client("bedrock-runtime", region_name="ap-northeast-1",
                          aws_access_key_id="test", aws_secret_access_key="test")
    stubber = Stubber(client)
    stubber.add_client_error("converse", service_error_code="AccessDeniedException", http_status_code=403)
    model = RecordingBedrockModel("qwen.qwen3-vl-235b-a22b", provider=BedrockProvider(bedrock_client=client),
                                  settings={"max_tokens": 4096})
    backend = InfrastructureCheckedBackend(PydanticAIBackend(model, retries=0), StringIO())
    with stubber, pytest.raises(RuntimeError, match="Bedrock execution stopped"):
        Classify(backend, labels=["cat"])(np.zeros((2, 3, 3), dtype=np.uint8))
    assert usage_summary(model.generations, backend.index)["unmeasured_requests"] == 1


def test_bedrock_profile_preserves_model_identity():
    from types import SimpleNamespace

    model = "moonshotai.kimi-k3.1"
    config = {"model": model, "inference_profile": f"global.{model}"}
    profile = {"status": "ACTIVE", "models": [{"modelArn": f"arn:aws:bedrock:::foundation-model/{model}"}]}
    calls = []

    def get_profile(**kwargs):
        calls.append(kwargs)
        return dict(profile)

    client = SimpleNamespace(get_inference_profile=get_profile)
    target, metadata = inference_target(client, config, {"inferenceTypesSupported": ["INFERENCE_PROFILE"]})
    assert target == f"global.{model}"
    assert metadata == profile
    assert calls == [{"inferenceProfileIdentifier": target}]
    profile["models"] = [{"modelArn": "arn:aws:bedrock:::foundation-model/moonshotai.kimi-k3"}]
    with pytest.raises(ValueError, match="configured model"):
        inference_target(client, config, {"inferenceTypesSupported": ["INFERENCE_PROFILE"]})
    with pytest.raises(ValueError, match="requires an inference_profile"):
        inference_target(client, {"model": model}, {"inferenceTypesSupported": ["INFERENCE_PROFILE"]})
    assert inference_target(client, {"model": "google.gemma-3-27b-it"},
                            {"inferenceTypesSupported": ["ON_DEMAND"]}) == ("google.gemma-3-27b-it", None)


def test_official_rate_override_prices_cached_input_and_total_output():
    from decimal import Decimal
    from pydantic_ai.usage import RequestUsage

    usage = RequestUsage(input_tokens=1000, output_tokens=200, cache_read_tokens=300,
                         cache_write_tokens=100, output_reasoning_tokens=50)
    pricing = {"rates_usd_per_million_tokens": {
        "input": "4", "output": "20", "cache_read": "0.2", "cache_write": "5",
    }}
    # Reasoning is already included in output_tokens and must not be charged twice.
    assert priced_usage(usage, pricing) == Decimal("0.00696")


def test_resume_skips_completed_tasks_and_refuses_to_repeat_interrupted_requests(tmp_path):
    config = {"model": "google.gemma-3-27b-it", "tasks": ["classify", "describe"], "samples_per_task": 2}
    manifest = {"datasets": {"cifar10": {"fingerprint_sha256": "frozen"}}}
    for name, value in (("config", config), ("manifest", manifest), ("classify", {
        "model": config["model"], "task": "classify", "metrics": {"samples": 2},
        "dataset_fingerprint_sha256": "frozen",
    })):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    (tmp_path / "classify.jsonl").write_text('{"index":0}\n{"index":1}\n')
    (tmp_path / "describe.jsonl").write_text("")
    assert remaining_tasks(tmp_path, config, manifest) == ["describe"]
    with pytest.raises(ValueError, match="configuration differs"):
        remaining_tasks(tmp_path, {**config, "model": "different"}, manifest)
    with pytest.raises(ValueError, match="prepared data differs"):
        remaining_tasks(tmp_path, config, {})
    (tmp_path / "describe.jsonl").write_text('{"index":0}\n')
    with pytest.raises(ValueError, match="interrupted request journal"):
        remaining_tasks(tmp_path, config, manifest)
