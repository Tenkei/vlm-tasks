from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias, overload

from ml_pipes.core import Operator
from ml_pipes.vision import ImagePayload

from ._images import ImageInput, load_image, validate_image
from ._schemas import (
    answer_spec, attributes_spec, classification_spec, description_spec,
    detection_spec, grounding_spec,
)
from .errors import InvalidModelResponseError
from .types import (
    AnswerResult, AttributePredictionResult, BoundingBox, ClassificationResult,
    DescriptionResult, Detection, DetectionResult, GroundingResult,
    ModelBackend, ModelRequest, ModelResponse,
    Point, ResponseSpec,
)

__all__ = ["Classify", "ClassifyMultiLabel", "Detect", "Ground", "Attributes", "Describe", "Answer"]

LabelChoices: TypeAlias = Sequence[str] | Mapping[str, str]
AttributeChoices: TypeAlias = Mapping[str, LabelChoices]


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _choices(labels: LabelChoices) -> dict[str, str]:
    if isinstance(labels, Mapping):
        result = dict(labels)
        for hint in result.values():
            if not isinstance(hint, str):
                raise ValueError("Label descriptions must be strings")
    elif isinstance(labels, Sequence) and not isinstance(labels, (str, bytes)):
        names = list(labels)
        for name in names:
            _text(name, "Label")
        if len(names) != len(set(names)):
            raise ValueError("Labels must be unique")
        result = dict.fromkeys(names, "")
    else:
        raise ValueError("Labels must be a sequence or a name-to-description mapping")
    if not result:
        raise ValueError("At least one label is required")
    for name in result:
        _text(name, "Label")
    return result


def _attribute_choices(attributes: AttributeChoices) -> dict[str, dict[str, str]]:
    if not isinstance(attributes, Mapping) or not attributes:
        raise ValueError("attributes must be a nonempty mapping")
    return {
        _text(name, "Attribute name"): _choices(labels)
        for name, labels in attributes.items()
    }


def _attributes_instruction(attributes: Mapping[str, Mapping[str, str]]) -> str:
    return (
        "Predict all requested categorical attributes of the subject identified "
        "by context. Select exactly one allowed value for every attribute. "
        "Include every requested attribute and no extra attributes. "
        "Attribute choices and descriptions: " + json.dumps(attributes)
    )


def _classification_instruction(labels: Mapping[str, str]) -> str:
    return (
        "Classify the image. Select exactly one allowed label for the subject "
        "and property defined by context. This is one image-level decision. "
        "Allowed labels and descriptions: " + json.dumps(labels)
    )


class _Task:
    def val(self, validator=None, **kwargs):
        """Evaluate this task and return its validator's metrics.

        ``validator`` accepts a custom validator class, constructed with this
        task and the other keyword arguments, then called without arguments.
        See ``ml_pipes.vlm.validation``.
        """
        if validator is None:
            from . import validation

            validators = {
                Classify: validation.ClassificationValidator,
                Detect: validation.DetectionValidator,
                Ground: validation.GroundingValidator,
                Attributes: validation.AttributesValidator,
                Answer: validation.AnswerValidator,
                Describe: validation.DescribeValidator,
            }
            validator = next(
                (candidate for task_type, candidate in validators.items() if isinstance(self, task_type)),
                None,
            )
            if validator is None:
                raise NotImplementedError(
                    f"{type(self).__name__} has no default validator; supply validator= explicitly"
                )
        runner = validator(task=self, **kwargs)
        runner()
        self.metrics = runner.metrics
        return self.metrics

    def _configure(self, model: ModelBackend, context: str | None) -> None:
        if not callable(model):
            raise TypeError("model must be a ModelBackend, such as PydanticAIBackend")
        if context is not None:
            _text(context, "context")
        self.model = model
        self.context = context

    def _predict(
        self, image: ImagePayload, *, instruction: str | None = None,
        response_spec: ResponseSpec | None = None,
    ) -> Mapping[str, object]:
        validate_image(image)
        if instruction is None:
            instruction = self._instruction
        if response_spec is None:
            response_spec = self._spec
        if self.context is not None:
            instruction += "\nContext: " + self.context
        instruction += "\nProvide a prediction matching the requested response schema."
        response = self.model(ModelRequest(image, instruction, response_spec))
        if not isinstance(response, ModelResponse):
            raise InvalidModelResponseError("The backend must return a ModelResponse")
        return response_spec.validate(response.data)


@Operator
class Classify(_Task):
    """Classify using labels supplied at construction or per call."""

    def __init__(
        self, model: ModelBackend, labels: LabelChoices | None = None, *, context: str | None = None
    ) -> None:
        self._configure(model, context)
        self.labels = None if labels is None else _choices(labels)
        self._spec = None if self.labels is None else classification_spec(self.labels, multiple=False)
        self._instruction = None if self.labels is None else _classification_instruction(self.labels)

    @overload
    def __call__(self, image: ImageInput) -> ClassificationResult: ...

    @overload
    def __call__(self, image: tuple[ImageInput, LabelChoices]) -> ClassificationResult: ...

    def __call__(self, image: Any) -> ClassificationResult:
        response_spec, instruction = self._spec, self._instruction
        if isinstance(image, tuple):
            if len(image) != 2:
                raise ValueError("Classify requires an image or an (image, labels) tuple")
            image, choices = image
            labels = _choices(choices)
            response_spec = classification_spec(labels, multiple=False)
            instruction = _classification_instruction(labels)
        elif self.labels is None:
            raise ValueError("Classify requires labels at construction or when called")
        image = load_image(image)
        data = self._predict(image, instruction=instruction, response_spec=response_spec)
        return ClassificationResult(labels=tuple(data["labels"]))

    def resolve_contract(
        self,
        upstream_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        del upstream_annotation, validation_error_type
        if self.labels is None:
            return (tuple[ImageInput, LabelChoices],), ClassificationResult
        return (ImageInput | tuple[ImageInput, LabelChoices],), ClassificationResult


@Operator
class ClassifyMultiLabel(_Task):
    def __init__(
        self, model: ModelBackend, labels: LabelChoices, *, context: str | None = None
    ) -> None:
        self._configure(model, context)
        self.labels = _choices(labels)
        self._spec = classification_spec(self.labels, multiple=True)
        self._instruction = (
            "Assess each label independently for the image and its context. "
            "Select every applicable label, once each; select none if none apply. "
            "Do not return a ranking or top-k alternatives to a single decision. "
            "Allowed labels and descriptions: " + json.dumps(self.labels)
        )

    def __call__(self, image: ImageInput) -> ClassificationResult:
        image = load_image(image)
        data = self._predict(image)
        return ClassificationResult(labels=tuple(data["labels"]))


def _pixel_box(box: Mapping[str, float], image: ImagePayload) -> BoundingBox:
    return BoundingBox(
        box["x1"] * image.width, box["y1"] * image.height,
        box["x2"] * image.width, box["y2"] * image.height,
    )


_GEOMETRY = (
    "Use xyxy boxes normalized to [0, 1] relative to the whole input image, "
    "with x1 < x2 and y1 < y2. Do not use pixels or a 0..1000 scale."
)


def _detection_instruction(labels: Mapping[str, str]) -> str:
    return (
        "Detect all instances matching the supplied labels and context. "
        "Search exhaustively; return an empty detections list if none match. "
        "Different instances may overlap. "
        + _GEOMETRY + "\nAllowed labels and descriptions: " + json.dumps(labels)
    )


@Operator
class Detect(_Task):
    """Detect objects using labels supplied at construction or per call."""

    def __init__(
        self, model: ModelBackend, labels: LabelChoices | None = None, *, context: str | None = None
    ) -> None:
        self._configure(model, context)
        self.labels = None if labels is None else _choices(labels)
        self._spec = None if self.labels is None else detection_spec(self.labels)
        self._instruction = None if self.labels is None else _detection_instruction(self.labels)

    @overload
    def __call__(self, image: ImageInput) -> DetectionResult: ...

    @overload
    def __call__(self, image: tuple[ImageInput, LabelChoices]) -> DetectionResult: ...

    def __call__(self, image: Any) -> DetectionResult:
        response_spec, instruction = self._spec, self._instruction
        if isinstance(image, tuple):
            if len(image) != 2:
                raise ValueError("Detect requires an image or an (image, labels) tuple")
            image, choices = image
            labels = _choices(choices)
            response_spec = detection_spec(labels)
            instruction = _detection_instruction(labels)
        elif self.labels is None:
            raise ValueError("Detect requires labels at construction or when called")
        image = load_image(image)
        data = self._predict(image, instruction=instruction, response_spec=response_spec)
        return DetectionResult(
            detections=tuple(
                Detection(label=item["label"], box=_pixel_box(item["box"], image))
                for item in data["detections"]
            ),
            image_size=image.size,
        )

    def resolve_contract(
        self,
        upstream_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        del upstream_annotation, validation_error_type
        if self.labels is None:
            return (tuple[ImageInput, LabelChoices],), DetectionResult
        return (ImageInput | tuple[ImageInput, LabelChoices],), DetectionResult


@Operator
class Ground(_Task):
    """Resolve a target supplied at construction or when calling the task."""

    def __init__(
        self, model: ModelBackend, target: str | None = None, *, context: str | None = None,
        geometry: Literal["box", "point"] = "box",
    ) -> None:
        self._configure(model, context)
        self.target = None if target is None else _text(target, "target")
        self._spec = grounding_spec(geometry)
        self.geometry = geometry
        location = "box" if geometry == "box" else "point"
        coordinates = _GEOMETRY if geometry == "box" else (
            "Predict one point inside the target, preferably near the center of its visible "
            "body (the visible torso for a person). Predict the point directly from the image "
            "and description. Use finite x and y normalized to [0, 1] relative to the whole "
            "input image. The origin is top-left; x increases rightward and y downward. "
            "Do not use pixels or a 0..1000 scale."
        )
        self._instruction = (
            "Resolve exactly one target in the image using the target description "
            f"and context. Return found with one {location} when uniquely resolved, "
            f"not_found with {location}=null when absent, or ambiguous with {location}=null "
            "when several plausible matches remain. Do not arbitrarily pick one. "
            + coordinates
        )

    @overload
    def __call__(self, image: ImageInput) -> GroundingResult: ...

    @overload
    def __call__(self, image: tuple[ImageInput, str]) -> GroundingResult: ...

    def __call__(self, image: Any) -> GroundingResult:
        target = self.target
        if isinstance(image, tuple):
            if len(image) != 2:
                raise ValueError("Ground requires an image or an (image, target) tuple")
            image, target = image
        if target is None:
            raise ValueError("Ground requires a target at construction or when called")
        target = _text(target, "target")
        image = load_image(image)
        data = self._predict(
            image, instruction=self._instruction + "\nTarget: " + json.dumps(target)
        )
        box = data.get("box")
        point = data.get("point")
        return GroundingResult(
            status=data["status"],
            box=None if box is None else _pixel_box(box, image),
            image_size=image.size,
            geometry=self.geometry,
            point=None if point is None else Point(
                point["x"] * image.size[0], point["y"] * image.size[1],
            ),
        )

    def resolve_contract(
        self,
        upstream_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        # Publish the configured boundary here; overloads describe direct calls.
        del upstream_annotation, validation_error_type
        if self.target is None:
            return (tuple[ImageInput, str],), GroundingResult
        return (ImageInput | tuple[ImageInput, str],), GroundingResult


@Operator
class Attributes(_Task):
    """Predict categorical attributes supplied at construction or per call."""

    def __init__(
        self,
        model: ModelBackend,
        attributes: AttributeChoices | None = None,
        *,
        context: str | None = None,
    ) -> None:
        self._configure(model, context)
        self.attributes = None if attributes is None else _attribute_choices(attributes)
        self._spec = None if self.attributes is None else attributes_spec(self.attributes)
        self._instruction = None if self.attributes is None else _attributes_instruction(self.attributes)

    @overload
    def __call__(self, image: ImageInput) -> AttributePredictionResult: ...

    @overload
    def __call__(self, image: tuple[ImageInput, AttributeChoices]) -> AttributePredictionResult: ...

    def __call__(self, image: Any) -> AttributePredictionResult:
        response_spec, instruction = self._spec, self._instruction
        if isinstance(image, tuple):
            if len(image) != 2:
                raise ValueError("Attributes requires an image or an (image, attributes) tuple")
            image, choices = image
            attributes = _attribute_choices(choices)
            response_spec = attributes_spec(attributes)
            instruction = _attributes_instruction(attributes)
        elif self.attributes is None:
            raise ValueError("Attributes requires attribute choices at construction or when called")
        image = load_image(image)
        data = self._predict(image, instruction=instruction, response_spec=response_spec)
        return AttributePredictionResult(attributes=dict(data["attributes"]))

    def resolve_contract(
        self,
        upstream_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        del upstream_annotation, validation_error_type
        if self.attributes is None:
            return (tuple[ImageInput, AttributeChoices],), AttributePredictionResult
        return (ImageInput | tuple[ImageInput, AttributeChoices],), AttributePredictionResult


@Operator
class Describe(_Task):
    """Describe visible image content, with optional focus and detail in context."""

    def __init__(self, model: ModelBackend, *, context: str | None = None) -> None:
        self._configure(model, context)
        self._spec = description_spec()
        self._instruction = (
            "Describe the image in natural language. Provide one concise, factual "
            "description of salient visible content. Use context to guide the "
            "subject, focus, language, and level of detail. Do not invent details "
            "unsupported by the image."
        )

    def __call__(self, image: ImageInput) -> DescriptionResult:
        image = load_image(image)
        data = self._predict(image)
        return DescriptionResult(description=data["description"])


@Operator
class Answer(_Task):
    """Answer a question supplied at construction or when calling the task."""

    def __init__(
        self, model: ModelBackend, question: str | None = None, *, context: str | None = None,
    ) -> None:
        self._configure(model, context)
        self.question = None if question is None else _text(question, "question")
        self._spec = answer_spec()
        self._instruction = (
            "Answer the supplied question about the image concisely and directly. "
            "Use visual evidence, relevant general knowledge, and any supplied context. "
            "Do not invent image-specific facts. Return answered with a nonempty answer "
            "when the answer can be determined, or unanswerable with answer=null when "
            "evidence is insufficient or the question remains ambiguous. A negative "
            "answer such as no or zero is still answered. Do not include a preamble "
            "or an unsolicited explanation."
        )

    @overload
    def __call__(self, image: ImageInput) -> AnswerResult: ...

    @overload
    def __call__(self, image: tuple[ImageInput, str]) -> AnswerResult: ...

    def __call__(self, image: Any) -> AnswerResult:
        question = self.question
        if isinstance(image, tuple):
            if len(image) != 2:
                raise ValueError("Answer requires an image or an (image, question) tuple")
            image, question = image
        if question is None:
            raise ValueError("Answer requires a question at construction or when called")
        question = _text(question, "question")
        image = load_image(image)
        data = self._predict(
            image, instruction=self._instruction + "\nQuestion: " + json.dumps(question)
        )
        return AnswerResult(status=data["status"], answer=data["answer"])

    def resolve_contract(
        self,
        upstream_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[tuple[Any, ...], Any]:
        del upstream_annotation, validation_error_type
        if self.question is None:
            return (tuple[ImageInput, str],), AnswerResult
        return (ImageInput | tuple[ImageInput, str],), AnswerResult
