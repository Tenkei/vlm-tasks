from __future__ import annotations

import json
from dataclasses import is_dataclass
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest


from pydantic_ai import models
from pydantic_ai.messages import ModelResponse as AIResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.usage import RequestUsage

from ml_pipes.core import Pipeline
from ml_pipes.vision import ImagePayload
from ml_pipes.validation import PipelineValidationError
from ml_pipes.vlm import (
    Answer, AnswerResult, AttributePredictionResult, Attributes, BoundingBox,
    ClassificationResult, Classify, ClassifyMultiLabel, Describe, DescriptionResult,
    Detect, DetectionResult, Ground, GroundingResult,
    InvalidModelResponseError, ModelExecutionError, ModelResponse,
    PydanticAIBackend, Usage,
)
from ml_pipes.vlm._images import ImageInput
from ml_pipes.vlm.ops import AttributeChoices, LabelChoices


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
        if isinstance(value, AIResponse):
            return value
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


@pytest.mark.parametrize("input_kind", ["str", "path", "array", "pil"])
@pytest.mark.parametrize(("make_task", "payload"), [
    (lambda b: Classify(b, ["clean"]), {"labels": ["clean"]}),
    (lambda b: ClassifyMultiLabel(b, ["clean"]), {"labels": ["clean"]}),
    (lambda b: Detect(b, ["window"]), {"detections": [
        {"label": "window", "box": {"x1": 0.1, "y1": 0.2, "x2": 0.6, "y2": 0.8}},
    ]}),
    (lambda b: Ground(b, "window"), {
        "status": "found", "box": {"x1": 0.1, "y1": 0.2, "x2": 0.6, "y2": 0.8},
    }),
    (lambda b: Ground(b, "window", geometry="point"), {
        "status": "found", "point": {"x": 0.25, "y": 0.5},
    }),
    (lambda b: Attributes(b, {"color": ["red"]}), {"attributes": {"color": "red"}}),
    (lambda b: Describe(b), {"description": "A red chair."}),
    (lambda b: Answer(b, "What color is the chair?"), {"status": "answered", "answer": "red"}),
])
def test_tasks_accept_image_paths_arrays_and_pil(tmp_path, image, input_kind, make_task, payload):
    path = tmp_path / "image.png"
    assert cv2.imwrite(str(path), image.array)
    backend, _ = backend_for(payload)
    task = make_task(backend)
    expected = task(image)

    value, input_type = {
        "str": (str(path), str), "path": (path, Path),
        "array": (image.array[..., ::-1], np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]),
        "pil": (Image.fromarray(image.array[..., ::-1]), Image.Image),
    }[input_kind]
    assert task(value) == expected
    pipeline = Pipeline([task])
    pipeline.validate(pipeline_input_type=input_type, strict=True)
    assert pipeline(value) == expected


@pytest.mark.parametrize("path_type", [str, Path])
@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("kind", ["classify", "detect", "ground", "attributes", "answer"])
def test_paths_work_with_per_call_settings(tmp_path, image, path_type, configured, kind):
    path = tmp_path / "image.png"
    assert cv2.imwrite(str(path), image.array)
    if kind in {"classify", "detect"}:
        payload = {"labels": ["window"]} if kind == "classify" else {"detections": []}
        backend, _ = backend_for(payload)
        task_type = Classify if kind == "classify" else Detect
        task = task_type(backend, ["old label"] if configured else None)
        settings = {"window": "A visible window"}
        input_type = tuple[path_type, dict[str, str]]
    elif kind == "ground":
        backend, _ = backend_for({
            "status": "found", "box": {"x1": 0, "y1": 0, "x2": 0.5, "y2": 1},
        })
        task = Ground(backend, "old target" if configured else None)
        settings = "the window"
        input_type = tuple[path_type, str]
    elif kind == "attributes":
        backend, _ = backend_for({"attributes": {"color": "red"}})
        task = Attributes(backend, {"material": ["wood"]} if configured else None)
        settings = {"color": ["red"]}
        input_type = tuple[path_type, dict[str, list[str]]]
    else:
        backend, _ = backend_for({"status": "answered", "answer": "red"})
        task = Answer(backend, "Old question?" if configured else None)
        settings = "What color is the chair?"
        input_type = tuple[path_type, str]
    expected = task((image, settings))

    assert task((path_type(path), settings)) == expected
    pipeline = Pipeline([task])
    pipeline.validate(pipeline_input_type=input_type, strict=True)
    assert pipeline((path_type(path), settings)) == expected


@pytest.mark.parametrize(("contents", "error"), [
    (None, FileNotFoundError),
    (b"not an image", ValueError),
    (b"", ValueError),
])
def test_invalid_image_files_fail_before_inference(tmp_path, contents, error):
    path = tmp_path / "image.png"
    if contents is not None:
        path.write_bytes(contents)
    backend, calls = backend_for({"labels": ["clean"]})

    with pytest.raises(error, match=str(path)):
        Classify(backend, ["clean"])(path)
    assert calls == []


def test_public_exports():
    import ml_pipes.vlm as vlm

    expected = {
        "Classify", "ClassifyMultiLabel", "Detect", "Ground", "Attributes", "Describe", "Answer",
        "PydanticAIBackend", "TransformersModel", "BoundingBox", "Point", "ClassificationResult", "Detection",
        "DetectionResult", "GroundingResult", "AttributePredictionResult", "DescriptionResult", "AnswerResult",
        "ModelBackend", "ModelRequest", "ModelResponse", "ResponseSpec", "Usage",
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


@pytest.mark.parametrize("task_type", [Classify, Detect])
@pytest.mark.parametrize("configured", [False, True])
def test_label_overrides_preserve_defaults_and_context(image, task_type, configured):
    requests = []

    def backend(request):
        requests.append(request)
        if task_type is Classify:
            return ModelResponse({"labels": ["new" if len(requests) == 1 else "old"]})
        return ModelResponse({"detections": []})

    task = task_type(backend, {"old": "Old description"} if configured else None, context="The subject.")
    spec, instruction = task._spec, task._instruction
    task((image, {"new": "New description"}))
    assert "New description" in requests[0].instruction
    assert "Old description" not in requests[0].instruction
    assert "The subject." in requests[0].instruction
    assert "old" not in json.dumps(requests[0].response_spec.schema)
    assert task.labels == ({"old": "Old description"} if configured else None)
    assert task._spec is spec and task._instruction == instruction
    if configured:
        task(image)
        assert requests[1].response_spec is spec
        assert "Old description" in requests[1].instruction
    else:
        with pytest.raises(ValueError, match="requires labels"):
            task(image)
        assert len(requests) == 1


@pytest.mark.parametrize("task_type", [Classify, Detect])
@pytest.mark.parametrize("labels", [None, [], {}, ["new", "new"], "new", {"new": 1}])
def test_invalid_label_overrides_fail_before_inference(image, task_type, labels):
    backend, calls = backend_for({})
    task = task_type(backend, ["old"])
    with pytest.raises(ValueError):
        task((image, labels))
    assert task.labels == {"old": ""}
    assert not calls


@pytest.mark.parametrize("task_type", [Classify, Detect])
@pytest.mark.parametrize("length", [0, 1, 3])
def test_label_tasks_reject_malformed_tuples_before_inference(image, task_type, length):
    backend, calls = backend_for({})
    with pytest.raises(ValueError, match="image, labels"):
        task_type(backend)(tuple([image] * length))
    assert not calls


@pytest.mark.parametrize("task_type,result_type", [(Classify, ClassificationResult), (Detect, DetectionResult)])
@pytest.mark.parametrize("configured", [False, True])
def test_label_task_pipeline_contracts(task_type, result_type, configured):
    backend, _ = backend_for({})
    pipeline = Pipeline([task_type(backend, ["old"] if configured else None)])
    contract = pipeline.validate(strict=True)
    expected = ImageInput | tuple[ImageInput, LabelChoices] if configured else tuple[ImageInput, LabelChoices]
    assert contract.input_type == expected
    assert contract.output_type is result_type
    pipeline.validate(pipeline_input_type=tuple[ImagePayload, list[str]], strict=True)
    with pytest.raises(PipelineValidationError):
        pipeline.validate(pipeline_input_type=tuple[ImagePayload, int], strict=True)
    if not configured:
        with pytest.raises(PipelineValidationError):
            pipeline.validate(pipeline_input_type=ImagePayload, strict=True)


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
    assert contract.input_type == ImageInput | tuple[ImageInput, str]
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
    assert contract.input_type == tuple[ImageInput, str]
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


def test_describe_preserves_wording_and_context(image):
    requests = []
    description = "  A blue chair.\nThe label reads ‘椅子’.  "

    def backend(request):
        requests.append(request)
        return ModelResponse({"description": description})

    task = Describe(backend, context="Describe the chair and any visible text.")
    assert task(image) == DescriptionResult(description)
    assert requests[0].image is image
    assert "Context: Describe the chair and any visible text." in requests[0].instruction


@pytest.mark.parametrize("data", [
    {}, {"description": ""}, {"description": " \n\t"},
    {"description": None}, {"description": 42}, {"description": ["chair"]},
    {"description": "A chair.", "extra": "unexpected"},
])
def test_describe_validates_custom_backend_responses(image, data):
    with pytest.raises(InvalidModelResponseError):
        Describe(lambda request: ModelResponse(data))(image)


@pytest.mark.parametrize("answer", ["no", "zero", "0", "wood", "椅子", "  A blue chair.\n  "])
def test_answer_preserves_answer_text_including_negative_answers(image, answer):
    backend, _ = backend_for({"status": "answered", "answer": answer})
    assert Answer(backend, "What is visible?")(image) == AnswerResult("answered", answer)


def test_answer_unanswerable_is_a_valid_result(image):
    backend, calls = backend_for({"status": "unanswerable", "answer": None}, retries=1)
    assert Answer(backend, "What does the unreadable label say?")(image) == AnswerResult("unanswerable", None)
    assert len(calls) == 1


@pytest.mark.parametrize("data", [
    {}, {"status": "answered"}, {"answer": "wood"},
    {"status": "answered", "answer": None},
    {"status": "answered", "answer": ""},
    {"status": "answered", "answer": " \n\t"},
    {"status": "answered", "answer": 0},
    {"status": "answered", "answer": False},
    {"status": "answered", "answer": ["wood"]},
    {"status": "unanswerable", "answer": "unknown"},
    {"status": "ambiguous", "answer": None},
    {"status": "answered", "answer": "wood", "extra": "unexpected"},
])
def test_answer_validates_status_and_text_from_custom_backends(image, data):
    with pytest.raises(InvalidModelResponseError):
        Answer(lambda request: ModelResponse(data), "What material is the chair?")(image)


@pytest.mark.parametrize("default", [None, "What material is the chair?"])
def test_answer_call_questions_do_not_change_configuration(image, default):
    requests = []

    def backend(request):
        requests.append(request)
        return ModelResponse({"status": "answered", "answer": "wood"})

    task = Answer(backend, default, context="Assess the chair.")
    instruction = task._instruction
    questions = ["What color is the chair?", "What is resting on the chair?"]
    for question in questions:
        assert task((image, question)) == AnswerResult("answered", "wood")
    if default is not None:
        task(image=image)
        questions.append(default)
    else:
        with pytest.raises(ValueError, match="requires a question"):
            task(image)

    assert task.question == default
    assert task.context == "Assess the chair."
    assert task._instruction == instruction
    assert len(requests) == len(questions)
    for request, question in zip(requests, questions, strict=True):
        assert request.image is image
        assert "Question: " + json.dumps(question) + "\nContext:" in request.instruction
        assert request.instruction.count("Question: ") == 1
        assert "Assess the chair." in request.instruction


@pytest.mark.parametrize("question", ["", " \n\t", 42, True, []])
def test_invalid_constructor_questions_fail_before_inference(question):
    backend, calls = backend_for({"status": "answered", "answer": "wood"})
    with pytest.raises(ValueError, match="question"):
        Answer(backend, question)
    assert calls == []


@pytest.mark.parametrize("default", [None, "What material is the chair?"])
@pytest.mark.parametrize("question", [None, "", " \n\t", 42, True, []])
def test_invalid_question_overrides_fail_before_loading_the_image(default, question):
    backend, calls = backend_for({"status": "answered", "answer": "wood"})
    task = Answer(backend, default)
    with pytest.raises(ValueError, match="question"):
        task(("missing-image.png", question))
    assert task.question == default
    assert calls == []


@pytest.mark.parametrize("length", [0, 1, 3])
def test_answer_rejects_malformed_input_tuples_before_inference(length):
    backend, calls = backend_for({"status": "answered", "answer": "wood"})
    with pytest.raises(ValueError, match=r"\(image, question\)"):
        Answer(backend, "What is visible?")(("missing-image.png",) * length)
    assert calls == []


@pytest.mark.parametrize("mode", ["prompted", "native", "tool"])
def test_text_tasks_share_a_backend_without_sharing_questions_or_history(image, mode):
    backend, calls = backend_for(
        {"description": "A chair."},
        {"status": "answered", "answer": "wood"},
        {"status": "unanswerable", "answer": None},
        output_mode=mode,
    )
    assert Describe(backend)(image) == DescriptionResult("A chair.")
    task = Answer(backend)
    assert task((image, "What is the chair made of?")) == AnswerResult("answered", "wood")
    assert task((image, "Who manufactured it?")) == AnswerResult("unanswerable", None)
    assert [len(messages) for messages, _ in calls] == [1, 1, 1]


@pytest.mark.parametrize("mode", ["prompted", "native", "tool"])
def test_inconsistent_answers_can_be_corrected_and_overrides_stay_isolated(image, mode):
    backend, calls = backend_for(
        {"status": "unanswerable", "answer": "wood"},
        {"status": "answered", "answer": "wood"},
        {"status": "answered", "answer": "blue"},
        retries=1, output_mode=mode,
    )
    task = Answer(backend, "What color is the chair?", context="Assess the chair.")
    assert task((image, "What is the chair made of?")) == AnswerResult("answered", "wood")
    assert task(image) == AnswerResult("answered", "blue")
    assert task.question == "What color is the chair?"
    assert task.context == "Assess the chair."
    assert [len(messages) for messages, _ in calls] == [1, 3, 1]


def test_failed_answer_override_preserves_default_and_remains_an_error(image):
    backend, _ = backend_for(
        {"status": "answered", "answer": None},
        {"status": "answered", "answer": "wood"},
    )
    task = Answer(backend, "What is the chair made of?", context="Assess the chair.")
    with pytest.raises(InvalidModelResponseError):
        task((image, "Who manufactured the chair?"))
    assert task.question == "What is the chair made of?"
    assert task.context == "Assess the chair."
    assert task(image) == AnswerResult("answered", "wood")


@pytest.mark.parametrize("default", [None, "What material is the chair?"])
def test_answer_pipeline_publishes_and_checks_configured_inputs(image, default):
    backend, calls = backend_for({"status": "answered", "answer": "wood"})
    pipeline = Pipeline([Answer(backend, default)])
    contract = pipeline.validate(strict=True)
    expected = tuple[ImageInput, str] if default is None else ImageInput | tuple[ImageInput, str]
    assert contract.input_type == expected
    assert contract.output_type is AnswerResult
    assert pipeline((image, "What material is the chair?")) == AnswerResult("answered", "wood")
    if default is None:
        with pytest.raises(PipelineValidationError):
            pipeline.validate(pipeline_input_type=ImagePayload, strict=True)
    else:
        pipeline.validate(pipeline_input_type=ImagePayload, strict=True)
        assert pipeline(image) == AnswerResult("answered", "wood")
    with pytest.raises(PipelineValidationError):
        pipeline.validate(pipeline_input_type=tuple[ImagePayload, int], strict=True)
    assert len(calls) == (1 if default is None else 2)


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


def test_usage_includes_corrections_and_is_scoped_to_each_call(image):
    def response(label, *, cost, reported):
        return AIResponse(
            parts=[TextPart(json.dumps({"labels": [label]}))],
            usage=RequestUsage(input_tokens=100, output_tokens=20, cache_read_tokens=30,
                               cache_write_tokens=10, output_reasoning_tokens=5, cost=Decimal(cost)),
            provider_details={"cost": reported},
        )

    backend, _ = backend_for(
        response("unknown", cost="0.002", reported=0.001),
        response("clean", cost="0.003", reported=0.002),
        response("clean", cost="0", reported=0),
        retries=1,
    )
    task = Classify(backend, ["clean"])
    first = task(image).usage
    second = task(image).usage
    assert first == Usage(input_tokens=200, output_tokens=40, cache_read_tokens=60,
                          cache_write_tokens=20, output_reasoning_tokens=10, requests=2,
                          estimated_cost=Decimal("0.005"), reported_cost=Decimal("0.003"))
    assert first.total_tokens == 240
    assert second.requests == 1
    assert second.input_tokens == 100 and second.output_tokens == 20
    assert second.estimated_cost == second.reported_cost == Decimal(0)


def test_partial_provider_cost_is_not_reported_as_a_total(image):
    backend, _ = backend_for(
        AIResponse(parts=[TextPart('{"labels": ["unknown"]}')],
                   usage=RequestUsage(input_tokens=100, output_tokens=10),
                   provider_details={"cost": 0.001}),
        AIResponse(parts=[TextPart('{"labels": ["clean"]}')],
                   usage=RequestUsage(input_tokens=100, output_tokens=10)),
        retries=1,
    )
    usage = Classify(backend, ["clean"])(image).usage
    assert usage.requests == 2
    assert usage.reported_cost is None and usage.estimated_cost is None


def test_missing_usage_is_identified_and_unknown_cost_is_not_zero(image):
    backend, _ = backend_for(AIResponse(
        parts=[TextPart('{"labels": ["clean"]}')],
        usage=RequestUsage(unmeasured_requests=1),
    ))
    usage = Classify(backend, ["clean"])(image).usage
    assert usage.requests == usage.unmeasured_requests == 1
    assert usage.total_tokens == 0
    assert usage.estimated_cost is None and usage.reported_cost is None


def test_backend_preserves_pydantic_ai_price_estimates(image):
    model = FunctionModel(
        lambda messages, info: AIResponse(
            parts=[TextPart('{"labels": ["clean"]}')],
            usage=RequestUsage(input_tokens=100, output_tokens=10),
            provider_name="openai",
        ),
        model_name="gpt-4o-mini",
    )
    usage = Classify(PydanticAIBackend(model), ["clean"])(image).usage
    assert usage.estimated_cost is not None and usage.estimated_cost > 0
    assert usage.reported_cost is None


@pytest.mark.parametrize("usage", [None, Usage(input_tokens=100, output_tokens=20)])
@pytest.mark.parametrize(("task", "payload"), [
    (lambda b: Classify(b, ["clean"]), {"labels": ["clean"]}),
    (lambda b: ClassifyMultiLabel(b, ["clean"]), {"labels": []}),
    (lambda b: Detect(b, ["window"]), {"detections": []}),
    (lambda b: Ground(b, "window"), {"status": "not_found", "box": None}),
    (lambda b: Ground(b, "window", geometry="point"), {"status": "not_found", "point": None}),
    (lambda b: Attributes(b, {"color": ["red"]}), {"attributes": {"color": "red"}}),
    (lambda b: Describe(b), {"description": "A red chair."}),
    (lambda b: Answer(b, "What color?"), {"status": "answered", "answer": "red"}),
])
def test_task_results_preserve_optional_backend_usage(image, task, payload, usage):
    def backend(request):
        assert "usage" not in request.response_spec.schema["properties"]
        return ModelResponse(payload, usage=usage)

    assert Pipeline([task(backend)])(image).usage is usage


@pytest.mark.parametrize(("task", "payload", "result_type"), [
    (lambda b: Classify(b, ["clean"]), {"labels": ["clean"]}, ClassificationResult),
    (lambda b: ClassifyMultiLabel(b, ["clean"]), {"labels": []}, ClassificationResult),
    (lambda b: Detect(b, ["window"]), {"detections": []}, DetectionResult),
    (lambda b: Ground(b, "window"), {"status": "not_found", "box": None}, GroundingResult),
    (lambda b: Attributes(b, {"color": ["red"]}),
     {"attributes": {"color": "red"}}, AttributePredictionResult),
    (lambda b: Describe(b), {"description": "A red chair."}, DescriptionResult),
    (lambda b: Answer(b, "What color is the chair?"),
     {"status": "answered", "answer": "red"}, AnswerResult),
])
def test_operator_runs_and_validates_in_pipeline(image, task, payload, result_type):
    backend, _ = backend_for(payload)
    operator = task(backend)
    pipeline = Pipeline([operator])
    contract = pipeline.validate(strict=True)
    expected_input = ImageInput
    if result_type in (GroundingResult, AnswerResult):
        expected_input = ImageInput | tuple[ImageInput, str]
    elif result_type is AttributePredictionResult:
        expected_input = ImageInput | tuple[ImageInput, AttributeChoices]
    elif isinstance(operator, (Classify, Detect)):
        expected_input = ImageInput | tuple[ImageInput, LabelChoices]
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
