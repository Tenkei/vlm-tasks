"""Framework-independent task results and model communication contracts."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from ml_pipes.vision import ImagePayload

__all__ = [
    "BoundingBox", "ClassificationResult", "Detection", "DetectionResult",
    "GroundingResult", "AttributePredictionResult", "ResponseSpec",
    "ModelRequest", "ModelResponse", "ModelBackend",
]


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class ClassificationResult:
    labels: tuple[str, ...]
    scores: Mapping[str, float] | None = None


@dataclass(frozen=True)
class Detection:
    label: str
    box: BoundingBox
    score: float | None = None


@dataclass(frozen=True)
class DetectionResult:
    detections: tuple[Detection, ...]
    image_size: tuple[int, int]


@dataclass(frozen=True)
class GroundingResult:
    status: Literal["found", "not_found", "ambiguous"]
    box: BoundingBox | None
    image_size: tuple[int, int]


@dataclass(frozen=True)
class AttributePredictionResult:
    attributes: Mapping[str, str]


@dataclass(frozen=True)
class ResponseSpec:
    schema: Mapping[str, object]
    validate: Callable[[object], Mapping[str, object]] = field(repr=False)


@dataclass(frozen=True)
class ModelRequest:
    image: ImagePayload
    instruction: str
    response_spec: ResponseSpec


@dataclass(frozen=True)
class ModelResponse:
    data: Mapping[str, object]


class ModelBackend(Protocol):
    def __call__(self, request: ModelRequest) -> ModelResponse: ...
