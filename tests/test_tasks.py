from __future__ import annotations

import json
from dataclasses import is_dataclass

import numpy as np
import pytest


from pydantic_ai import models
from pydantic_ai.messages import ModelResponse as AIResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.profiles import ModelProfile

from ml_pipes.core import Pipeline
from ml_pipes.vision import ImagePayload
from ml_pipes.validation import PipelineValidationError
from ml_pipes.vlm import (
    AttributePredictionResult, Attributes, BoundingBox, ClassificationResult, Classify,
    ClassifyMultiLabel, Detect, DetectionResult, Ground, GroundingResult,
    InvalidModelResponseError, ModelExecutionError, ModelResponse,
    PydanticAIBackend,
)
from ml_pipes.vlm.ops import AttributeChoices


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)


@pytest.fixture
def image():
    return ImagePayload(np.zeros((100, 200, 3), dtype=np.uint8))


def backend_for(*responses, retries=0, output_mode="prompted"):
    calls = []

    def respond(messages, info):
        calls.append((list(messages), info))
        value = responses[min(len(calls) - 1, len(responses) - 1)]
        if isinstance(value, Exception):
            raise value
        if info.output_tools:
            return AIResponse(parts=[
                ToolCallPart(info.output_tools[0].name, value, tool_call_id="prediction")
            ])
        text = value if isinstance(value, str) else json.dumps(value)
        return AIResponse(parts=[TextPart(text)])

    model = FunctionModel(
        respond,
        profile=ModelProfile(supports_tools=True, supports_json_schema_output=True),
    )
    return PydanticAIBackend(model, retries=retries, output_mode=output_mode), calls


def test_public_exports():
    import ml_pipes.vlm as vlm

    expected = {
        "Classify", "ClassifyMultiLabel", "Detect", "Ground", "Attributes",
        "PydanticAIBackend", "TransformersModel", "BoundingBox", "Point", "ClassificationResult", "Detection",
        "DetectionResult", "GroundingResult", "AttributePredictionResult",
        "ModelBackend", "ModelRequest", "ModelResponse", "ResponseSpec",
        "InvalidModelResponseError", "ModelExecutionError",
    }
    assert set(vlm.__all__) == expected
    assert all(getattr(vlm, name) is not None for name in expected)


def test_classification_context_descriptions_and_scores(image):
    backend, calls = backend_for({"labels": ["dirty"]})
    op = Classify(
        backend,
        {"clean": "No visible residue", "dirty": "Visible dirt"},
        context="Assess the visible windows.",
    )
    result = op(image)
    assert result == ClassificationResult(("dirty",))
    assert result.scores is None
    from pydantic_ai.messages import BinaryContent, UserPromptPart
    content = [
        part.content for message in calls[0][0] for part in message.parts
        if isinstance(part, UserPromptPart)
    ][0]
    assert "visible windows" in content[0]
    assert "No visible residue" in content[0]
    assert isinstance(content[1], BinaryContent)
    assert content[1].media_type == "image/png"
    schema_text = json.dumps(op._spec.schema)
    assert '"enum": ["clean", "dirty"]' in schema_text


@pytest.mark.parametrize("labels", [[], ["clean", "dirty"], ["absent"], ["clean", "clean"]])
def test_single_label_requires_exactly_one_allowed_label(image, labels):
    backend, calls = backend_for({"labels": labels})
    with pytest.raises(InvalidModelResponseError):
        Classify(backend, ["clean", "dirty"])(image)
    assert len(calls) == 1


@pytest.mark.parametrize("labels", [[], ["scratched"], ["scratched", "stained"]])
def test_multi_label_accepts_independent_selections(image, labels):
    backend, _ = backend_for({"labels": labels})
    result = ClassifyMultiLabel(backend, ["scratched", "stained"])(image)
    assert result.labels == tuple(labels)


@pytest.mark.parametrize("labels", [["scratched", "scratched"], ["unknown"]])
def test_multi_label_rejects_duplicates_and_unknown_labels(image, labels):
    backend, _ = backend_for({"labels": labels})
    with pytest.raises(InvalidModelResponseError):
        ClassifyMultiLabel(backend, ["scratched", "stained"])(image)


def test_detect_overlapping_instances_and_pixel_conversion(image):
    first = {"x1": 0.1, "y1": 0.2, "x2": 0.6, "y2": 0.8}
    second = {"x1": 0.2, "y1": 0.3, "x2": 0.8, "y2": 0.9}
    backend, _ = backend_for({"detections": [
        {"label": "window", "box": first},
        {"label": "window", "box": second},
    ]})
    result = Detect(backend, ["window"])(image)
    assert result.image_size == (200, 100)
    assert len(result.detections) == 2
    assert result.detections[0].box == BoundingBox(20, 20, 120, 80)
    assert result.detections[1].box == BoundingBox(40, 30, 160, 90)
    assert all(item.score is None for item in result.detections)


def test_empty_detection_is_valid(image):
    backend, _ = backend_for({"detections": []})
    assert Detect(backend, ["window"])(image) == DetectionResult((), (200, 100))


@pytest.mark.parametrize("box", [
    {"x1": -0.1, "y1": 0, "x2": 1, "y2": 1},
    {"x1": 0, "y1": 0, "x2": 1000, "y2": 1000},
    {"x1": 0.8, "y1": 0, "x2": 0.2, "y2": 1},
    {"x1": 0, "y1": 0.5, "x2": 1, "y2": 0.5},
    {"x1": 0, "y1": 0, "x2": float("inf"), "y2": 1},
    {"x1": float("nan"), "y1": 0, "x2": 1, "y2": 1},
])
def test_invalid_boxes_are_rejected_without_repair(image, box):
    backend, _ = backend_for({"detections": [{"label": "window", "box": box}]})
    with pytest.raises(InvalidModelResponseError):
        Detect(backend, ["window"])(image)


def test_detection_rejects_unrequested_class(image):
    backend, _ = backend_for({"detections": [{
        "label": "door", "box": {"x1": 0, "y1": 0, "x2": 1, "y2": 1},
    }]})
    with pytest.raises(InvalidModelResponseError):
        Detect(backend, ["window"])(image)


@pytest.mark.parametrize("status", ["not_found", "ambiguous"])
def test_ground_absence_and_ambiguity(image, status):
    backend, _ = backend_for({"status": status, "box": None})
    assert Ground(backend, "the leftmost window")(image) == GroundingResult(
        status, None, (200, 100)
    )


def test_ground_found_returns_one_box(image):
    backend, _ = backend_for({
        "status": "found", "box": {"x1": 0, "y1": 0, "x2": 0.5, "y2": 1},
    })
    assert Ground(backend, "left window")(image).box == BoundingBox(0, 0, 100, 100)


@pytest.mark.parametrize("default", [None, "the left window"])
def test_ground_call_targets_do_not_change_configuration(image, default):
    requests = []

    def backend(request):
        requests.append(request)
        return ModelResponse({
            "status": "found", "box": {"x1": 0, "y1": 0, "x2": 0.5, "y2": 1},
        })

    task = Ground(backend, default, context="Resolve a whole window.")
    instruction = task._instruction
    targets = ["the right window", "the middle window"]
    for target in targets:
        assert task((image, target)).box == BoundingBox(0, 0, 100, 100)
    if default is not None:
        task(image=image)
        targets.append(default)
    else:
        with pytest.raises(ValueError, match="requires a target"):
            task(image)

    assert task.target == default
    assert task._instruction == instruction
    assert len(requests) == len(targets)
    for request, target in zip(requests, targets):
        assert request.image is image
        assert "Target: " + json.dumps(target) + "\nContext:" in request.instruction
        assert request.instruction.count("Target: ") == 1
        assert "Resolve a whole window." in request.instruction


@pytest.mark.parametrize("target", ["", " ", 42])
def test_ground_invalid_call_target_fails_before_inference(image, target):
    backend, calls = backend_for({"status": "not_found", "box": None})
    task = Ground(backend, "the window")
    with pytest.raises(ValueError, match="target"):
        task((image, target))
    assert calls == []


@pytest.mark.parametrize("target", ["", " ", 42])
def test_ground_invalid_constructor_target_fails_before_inference(target):
    backend, calls = backend_for({"status": "not_found", "box": None})
    with pytest.raises(ValueError, match="target"):
        Ground(backend, target)
    assert calls == []


def test_ground_tuple_requires_an_explicit_string_target(image):
    backend, calls = backend_for({"status": "not_found", "box": None})
    assert Ground(backend, "the cat")(image=image).status == "not_found"
    with pytest.raises(ValueError, match="requires a target"):
        Ground(backend)((image, None))
    with pytest.raises(ValueError, match="requires a target"):
        Ground(backend, "the cat")((image, None))
    assert len(calls) == 1


def test_ground_default_target_in_pipeline(image):
    backend, _ = backend_for({"status": "not_found", "box": None})
    pipeline = Pipeline([Ground(backend, "the cat")])
    contract = pipeline.validate(strict=True)
    assert contract.input_type == ImagePayload | tuple[ImagePayload, str]
    assert contract.output_type is GroundingResult
    assert pipeline(image).status == "not_found"
    assert pipeline.validate(pipeline_input_type=ImagePayload, strict=True).input_type is ImagePayload
    with pytest.raises(PipelineValidationError):
        pipeline.validate(pipeline_input_type=tuple[ImagePayload, int], strict=True)


@pytest.mark.parametrize("default", [None, "the left window"])
def test_ground_dynamic_target_in_pipeline(image, default):
    backend, calls = backend_for({"status": "not_found", "box": None})

    def supply_target(image: ImagePayload) -> tuple[ImagePayload, str]:
        return image, "the right window"

    pipeline = Pipeline([supply_target, Ground(backend, default)])
    contract = pipeline.validate(strict=True)
    assert contract.input_type is ImagePayload
    assert contract.output_type is GroundingResult
    assert pipeline(image) == GroundingResult("not_found", None, (200, 100))
    assert len(calls) == 1


def test_ground_without_default_requires_target_in_pipeline(image):
    backend, calls = backend_for({"status": "ambiguous", "box": None})
    pipeline = Pipeline([Ground(backend)])
    contract = pipeline.validate(strict=True)
    assert contract.input_type == tuple[ImagePayload, str]
    assert contract.output_type is GroundingResult
    assert pipeline((image, "the window")) == GroundingResult("ambiguous", None, (200, 100))
    with pytest.raises(PipelineValidationError):
        pipeline.validate(pipeline_input_type=ImagePayload, strict=True)
    with pytest.raises(PipelineValidationError):
        pipeline.validate(pipeline_input_type=tuple[ImagePayload, None], strict=True)
    assert len(calls) == 1


@pytest.mark.parametrize("data", [
    {"status": "found", "box": None},
    {"status": "not_found", "box": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}},
    {"status": "ambiguous", "box": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}},
    {"status": "unknown", "box": None},
])
def test_ground_status_and_geometry_must_agree(image, data):
    backend, _ = backend_for(data)
    with pytest.raises(InvalidModelResponseError):
        Ground(backend, "window")(image)


def test_attributes_require_exact_choices_and_support_arbitrary_names(image):
    values = {"window color": "blue", "model_config": "healthy"}
    backend, _ = backend_for({"attributes": values})
    result = Attributes(backend, {
        "window color": ["blue", "red"],
        "model_config": {"healthy": "Undamaged", "damaged": "Visible damage"},
    }, context="Assess the window.")(image)
    assert result == AttributePredictionResult(values)


@pytest.mark.parametrize("values", [
    {"color": "blue"},
    {"color": "green", "condition": "healthy"},
    {"color": "blue", "condition": "healthy", "extra": "value"},
])
def test_attributes_reject_missing_extra_or_invalid_values(image, values):
    backend, _ = backend_for({"attributes": values})
    with pytest.raises(InvalidModelResponseError):
        Attributes(backend, {
            "color": ["blue", "red"], "condition": ["healthy", "damaged"],
        })(image)


@pytest.mark.parametrize("mode", ["prompted", "native", "tool"])
def test_dynamic_attributes_use_current_schema_with_fresh_backend_history(image, mode):
    backend, calls = backend_for(
        {"attributes": {"material": "wood"}},
        {"attributes": {"size": "large"}},
        output_mode=mode,
    )
    task = Attributes(backend)
    assert task((image, {"material": ["wood", "metal"]})).attributes == {"material": "wood"}
    assert task((image, {"size": ["small", "large"]})).attributes == {"size": "large"}
    assert [len(messages) for messages, _ in calls] == [1, 1]
    assert task.attributes is task._spec is task._instruction is None


@pytest.mark.parametrize("mode", ["prompted", "native", "tool"])
def test_output_modes_validate_task_constraints(image, mode):
    backend, calls = backend_for({"labels": ["clean"]}, output_mode=mode)
    assert Classify(backend, ["clean", "dirty"])(image).labels == ("clean",)
    assert len(calls) == 1


def test_validation_feedback_can_correct_response(image):
    backend, calls = backend_for({"labels": ["unknown"]}, {"labels": ["clean"]}, retries=1)
    assert Classify(backend, ["clean", "dirty"])(image).labels == ("clean",)
    assert len(calls) == 2
    assert len(calls[1][0]) > len(calls[0][0])


def test_correction_budget_is_bounded(image):
    backend, calls = backend_for({"labels": ["unknown"]}, retries=2)
    with pytest.raises(InvalidModelResponseError):
        Classify(backend, ["clean"])(image)
    assert len(calls) == 3


def test_malformed_output_is_an_invalid_response(image):
    backend, calls = backend_for("no structured answer")
    with pytest.raises(InvalidModelResponseError):
        Classify(backend, ["clean"])(image)
    assert len(calls) == 1


def test_execution_error_preserves_cause(image):
    failure = RuntimeError("inference service unavailable")
    backend, _ = backend_for(failure)
    with pytest.raises(ModelExecutionError) as caught:
        Classify(backend, ["clean"])(image)
    assert caught.value.__cause__ is failure


def test_shared_backend_does_not_share_task_schema_or_history(image):
    backend, calls = backend_for(
        {"labels": ["clean"]}, {"detections": []}, {"labels": ["dirty"]},
    )
    classify = Classify(backend, ["clean", "dirty"])
    detect = Detect(backend, ["window"])
    assert classify(image).labels == ("clean",)
    assert detect(image).detections == ()
    assert classify(image).labels == ("dirty",)
    assert [len(messages) for messages, _ in calls] == [1, 1, 1]


@pytest.mark.parametrize(("task", "payload", "result_type"), [
    (lambda b: Classify(b, ["clean"]), {"labels": ["clean"]}, ClassificationResult),
    (lambda b: ClassifyMultiLabel(b, ["clean"]), {"labels": []}, ClassificationResult),
    (lambda b: Detect(b, ["window"]), {"detections": []}, DetectionResult),
    (lambda b: Ground(b, "window"), {"status": "not_found", "box": None}, GroundingResult),
    (lambda b: Attributes(b, {"color": ["red"]}),
     {"attributes": {"color": "red"}}, AttributePredictionResult),
])
def test_operator_runs_and_validates_in_pipeline(image, task, payload, result_type):
    backend, _ = backend_for(payload)
    pipeline = Pipeline([task(backend)])
    contract = pipeline.validate(strict=True)
    expected_input = ImagePayload
    if result_type is GroundingResult:
        expected_input = ImagePayload | tuple[ImagePayload, str]
    elif result_type is AttributePredictionResult:
        expected_input = ImagePayload | tuple[ImagePayload, AttributeChoices]
    assert contract.input_type == expected_input
    assert contract.output_type is result_type
    result = pipeline(image)
    assert isinstance(result, result_type)
    assert is_dataclass(result)


@pytest.mark.parametrize("labels", [[], ["clean", "clean"], [" "], "clean", {"clean": 3}])
def test_invalid_vocabularies_fail_before_inference(labels):
    backend, calls = backend_for({"labels": ["clean"]})
    with pytest.raises(ValueError):
        Classify(backend, labels)
    assert calls == []


@pytest.mark.parametrize("image", [
    ImagePayload(np.zeros((10, 10, 3), dtype=np.float32)),
    ImagePayload(np.zeros((3, 10, 10), dtype=np.uint8), layout="CHW"),
    ImagePayload(np.zeros((10, 10, 4), dtype=np.uint8)),
    ImagePayload(np.zeros((0, 10, 3), dtype=np.uint8)),
])
def test_bad_images_fail_before_inference(image):
    backend, calls = backend_for({"labels": ["clean"]})
    with pytest.raises(ValueError):
        Classify(backend, ["clean"])(image)
    assert calls == []


def test_task_boundary_validates_other_backend_responses(image):
    def invalid_backend(request):
        return ModelResponse({"labels": ["outside-vocabulary"]})

    with pytest.raises(InvalidModelResponseError):
        Classify(invalid_backend, ["clean"])(image)


def test_backend_requires_initialized_model_and_valid_settings():
    with pytest.raises(TypeError, match="initialized"):
        PydanticAIBackend("provider:model")
    model = FunctionModel(lambda messages, info: AIResponse(parts=[TextPart("{}")]))
    with pytest.raises(ValueError, match="output_mode"):
        PydanticAIBackend(model, output_mode="unsupported")
    for retries in (-1, True, 1.5):
        with pytest.raises(ValueError, match="retries"):
            PydanticAIBackend(model, retries=retries)
