"""Bedrock benchmark usage accounting through the real parser, without AWS calls."""
from io import StringIO
import json

import boto3
import numpy as np
import pytest

from botocore.stub import Stubber
from pydantic_ai.providers.bedrock import BedrockProvider

from benchmarks.src.run_bedrock import InfrastructureCheckedBackend, RecordingBedrockModel, build_model, inference_target, output_mode, priced_usage, remaining_tasks, usage_summary
from ml_pipes.vlm import Classify, Ground, InvalidModelResponseError, PydanticAIBackend, ResponseConstraintError


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
    model = build_model({"temperature": temperature, "max_new_tokens": 4096, "pricing": pricing}, client, model_id)
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
        assert "outputConfig" not in request


@pytest.mark.parametrize("geometry, field, coordinates", [
    ("point", "point", {"x": 0.25, "y": 0.5}),
    ("box", "box", {"x1": 0.1, "y1": 0.2, "x2": 0.7, "y2": 0.8}),
])
def test_native_benchmark_sends_schema_and_keeps_original_value_validation(geometry, field, coordinates):
    client = boto3.client("bedrock-runtime", region_name="ap-northeast-1",
                          aws_access_key_id="test", aws_secret_access_key="test")
    stubber = Stubber(client)
    outside_bounds = {**coordinates, next(iter(coordinates)): -0.1}
    for values in (coordinates, outside_bounds):
        stubber.add_response("converse", {
            "output": {"message": {"role": "assistant", "content": [
                {"text": json.dumps({"status": "found", field: values})},
            ]}},
            "stopReason": "end_turn", "usage": {"inputTokens": 100, "outputTokens": 30, "totalTokens": 130},
            "metrics": {"latencyMs": 5},
        })
    model = build_model({"output_mode": "native", "temperature": 0, "max_new_tokens": 4096},
                        client, "qwen.qwen3-vl-235b-a22b")
    stream = StringIO()
    backend = InfrastructureCheckedBackend(PydanticAIBackend(model, output_mode="native", retries=0), stream)
    task = Ground(backend, target="the cat", geometry=geometry)
    with stubber:
        assert task(np.zeros((2, 3, 3), dtype=np.uint8)).status == "found"
        with pytest.raises(ResponseConstraintError):
            task(np.zeros((2, 3, 3), dtype=np.uint8))
    calls = [json.loads(line) for line in stream.getvalue().splitlines()]
    for call in calls:
        assert call["attempted_requests"] == 1
        output = call["generations"][0]["output_config"]["textFormat"]
        assert output["type"] == "json_schema"
        schema = json.loads(output["structure"]["jsonSchema"]["schema"])
        geometry_schema = schema["properties"][field]["anyOf"][0]
        assert set(geometry_schema["required"]) == set(coordinates)
        assert geometry_schema["additionalProperties"] is False
        for key in coordinates:
            assert geometry_schema["properties"][key]["type"] == "number"
            assert call["response_schema"]["$defs"]["_Point" if geometry == "point" else "_Box"]["properties"][key]["minimum"] == 0
    usage = usage_summary(model.generations, backend.attempted_requests)
    assert usage["attempted_requests"] == usage["measured_requests"] == 2


def test_bedrock_output_mode_defaults_and_validation():
    assert output_mode({}) == "prompted"
    assert output_mode({"output_mode": "native"}) == "native"
    with pytest.raises(ValueError, match="output_mode"):
        output_mode({"output_mode": "json_schema"})


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


@pytest.mark.parametrize("succeeds", [True, False])
def test_bedrock_correction_requests_include_all_usage_and_history(succeeds):
    client = boto3.client("bedrock-runtime", region_name="ap-northeast-1",
                          aws_access_key_id="test", aws_secret_access_key="test")
    stubber = Stubber(client)
    for text in ("not JSON", '{"labels":["unknown"]}',
                 '{"labels":["cat"]}' if succeeds else "not JSON"):
        stubber.add_response("converse", {
            "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
            "stopReason": "end_turn", "usage": {"inputTokens": 100, "outputTokens": 10, "totalTokens": 110},
            "metrics": {"latencyMs": 5},
        })
    requests = []
    client.meta.events.register("before-parameter-build.bedrock-runtime.Converse",
                                lambda params, **kwargs: requests.append(params))
    model = RecordingBedrockModel("qwen.qwen3-vl-235b-a22b", provider=BedrockProvider(bedrock_client=client),
                                  settings={"max_tokens": 4096})
    stream = StringIO()
    backend = InfrastructureCheckedBackend(PydanticAIBackend(model, retries=2), stream)
    with stubber:
        task = Classify(backend, labels=["cat"])
        if succeeds:
            assert task(np.zeros((2, 3, 3), dtype=np.uint8)).labels == ("cat",)
        else:
            with pytest.raises(InvalidModelResponseError):
                task(np.zeros((2, 3, 3), dtype=np.uint8))
    record = json.loads(stream.getvalue())
    usage = usage_summary(model.generations, backend.attempted_requests)
    assert backend.index == 1
    assert record["attempted_requests"] == usage["attempted_requests"] == 3
    assert len(record["generations"]) == usage["measured_requests"] == 3
    assert usage["unmeasured_requests"] == 0
    assert usage["input_tokens"] == 300 and usage["output_tokens"] == 30
    assert usage["estimated_cost_complete"]
    assert [len(request["messages"]) for request in requests] == [1, 3, 5]


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
