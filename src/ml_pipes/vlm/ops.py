from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TypeAlias

from ml_pipes.core import Operator
from ml_pipes.vision import ImagePayload

from ._images import validate_image
from ._schemas import attributes_spec, classification_spec, detection_spec, grounding_spec
from .errors import InvalidModelResponseError
from .types import (
    AttributePredictionResult, BoundingBox, ClassificationResult, Detection,
    DetectionResult, GroundingResult, ModelBackend, ModelRequest, ModelResponse,
    ResponseSpec,
)

__all__ = ["Classify", "ClassifyMultiLabel", "Detect", "Ground", "PredictAttributes"]

LabelChoices: TypeAlias = Sequence[str] | Mapping[str, str]


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


class _Task:
    def val(self, validator=None, **kwargs):
        """Evaluate this configured task and return its validator's metrics.

        ``validator`` accepts a custom validator class. Other keyword arguments
        are passed to its constructor; see ``ml_pipes.vlm.validation``.
        """
        if validator is None:
            from .validation import ClassificationValidator, DetectionValidator

            if isinstance(self, Classify):
                validator = ClassificationValidator
            elif isinstance(self, Detect):
                validator = DetectionValidator
            else:
                raise NotImplementedError(
                    f"{type(self).__name__} has no default validator; supply validator= explicitly"
                )
        runner = validator(**kwargs)
        runner(task=self)
        self.metrics = runner.metrics
        return self.metrics

    def _configure(self, model: ModelBackend, context: str | None) -> None:
        if not callable(model):
            raise TypeError("model must be a ModelBackend, such as PydanticAIBackend")
        if context is not None:
            _text(context, "context")
        self.model = model
        self.context = context

    def _predict(self, image: ImagePayload) -> Mapping[str, object]:
        validate_image(image)
        instruction = self._instruction
        if self.context is not None:
            instruction += "\nContext: " + self.context
        instruction += "\nProvide a prediction matching the requested response schema."
        response = self.model(ModelRequest(image, instruction, self._spec))
        if not isinstance(response, ModelResponse):
            raise InvalidModelResponseError("The backend must return a ModelResponse")
        return self._spec.validate(response.data)


@Operator
class Classify(_Task):
    def __init__(
        self, model: ModelBackend, labels: LabelChoices, *, context: str | None = None
    ) -> None:
        self._configure(model, context)
        self.labels = _choices(labels)
        self._spec = classification_spec(self.labels, multiple=False)
        self._instruction = (
            "Classify the image. Select exactly one allowed label for the subject "
            "and property defined by context. This is one image-level decision. "
            "Allowed labels and descriptions: " + json.dumps(self.labels)
        )

    def __call__(self, image: ImagePayload) -> ClassificationResult:
        data = self._predict(image)
        return ClassificationResult(labels=tuple(data["labels"]))


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

    def __call__(self, image: ImagePayload) -> ClassificationResult:
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


@Operator
class Detect(_Task):
    def __init__(
        self, model: ModelBackend, labels: LabelChoices, *, context: str | None = None
    ) -> None:
        self._configure(model, context)
        self.labels = _choices(labels)
        self._spec = detection_spec(self.labels)
        self._instruction = (
            "Detect all instances matching the supplied labels and context. "
            "Search exhaustively; return an empty detections list if none match. "
            "Different instances may overlap. "
            + _GEOMETRY + "\nAllowed labels and descriptions: " + json.dumps(self.labels)
        )

    def __call__(self, image: ImagePayload) -> DetectionResult:
        data = self._predict(image)
        return DetectionResult(
            detections=tuple(
                Detection(label=item["label"], box=_pixel_box(item["box"], image))
                for item in data["detections"]
            ),
            image_size=image.size,
        )


@Operator
class Ground(_Task):
    def __init__(
        self, model: ModelBackend, target: str, *, context: str | None = None
    ) -> None:
        self._configure(model, context)
        self.target = _text(target, "target")
        self._spec = grounding_spec()
        self._instruction = (
            "Resolve exactly one target in the image using the target description "
            "and context. Return found with one box when uniquely resolved, "
            "not_found with box=null when absent, or ambiguous with box=null "
            "when several plausible matches remain. Do not arbitrarily pick one. "
            + _GEOMETRY + "\nTarget: " + json.dumps(self.target)
        )

    def __call__(self, image: ImagePayload) -> GroundingResult:
        data = self._predict(image)
        box = data["box"]
        return GroundingResult(
            status=data["status"],
            box=None if box is None else _pixel_box(box, image),
            image_size=image.size,
        )


@Operator
class PredictAttributes(_Task):
    def __init__(
        self,
        model: ModelBackend,
        attributes: Mapping[str, LabelChoices],
        *,
        context: str | None = None,
    ) -> None:
        self._configure(model, context)
        if not isinstance(attributes, Mapping) or not attributes:
            raise ValueError("attributes must be a nonempty mapping")
        self.attributes = {
            _text(name, "Attribute name"): _choices(labels)
            for name, labels in attributes.items()
        }
        self._spec = attributes_spec(self.attributes)
        self._instruction = (
            "Predict all requested categorical attributes of the subject identified "
            "by context. Select exactly one allowed value for every attribute. "
            "Include every requested attribute and no extra attributes. "
            "Attribute choices and descriptions: " + json.dumps(self.attributes)
        )

    def __call__(self, image: ImagePayload) -> AttributePredictionResult:
        data = self._predict(image)
        return AttributePredictionResult(attributes=dict(data["attributes"]))
