"""Framework-independent task results and model communication contracts."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal, Protocol

from ml_pipes.vision import ImagePayload

__all__ = [
    "BoundingBox", "Point", "ClassificationResult", "Detection", "DetectionResult",
    "GroundingResult", "AttributePredictionResult", "DescriptionResult", "AnswerResult", "ResponseSpec",
    "Usage", "ModelRequest", "ModelResponse", "ModelBackend",
]


@dataclass(frozen=True)
class Usage:
    """Usage across a task call, including structured-output correction attempts.

    Token totals include cached and reasoning tokens; those fields are subsets.
    Costs are in USD. Estimated cost is a best-effort known subtotal; reported
    cost is available only when every response supplies a provider charge.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_reasoning_tokens: int = 0
    requests: int = 0
    unmeasured_requests: int = 0
    estimated_cost: Decimal | None = None
    reported_cost: Decimal | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class ClassificationResult:
    labels: tuple[str, ...]
    scores: Mapping[str, float] | None = None
    usage: Usage | None = field(default=None, kw_only=True, compare=False)


@dataclass(frozen=True)
class Detection:
    label: str
    box: BoundingBox
    score: float | None = None


@dataclass(frozen=True)
class DetectionResult:
    detections: tuple[Detection, ...]
    image_size: tuple[int, int]
    usage: Usage | None = field(default=None, kw_only=True, compare=False)


@dataclass(frozen=True)
class GroundingResult:
    status: Literal["found", "not_found", "ambiguous"]
    box: BoundingBox | None
    image_size: tuple[int, int]
    geometry: Literal["box", "point"] = "box"
    point: Point | None = None
    usage: Usage | None = field(default=None, kw_only=True, compare=False)


@dataclass(frozen=True)
class AttributePredictionResult:
    attributes: Mapping[str, str]
    usage: Usage | None = field(default=None, kw_only=True, compare=False)


@dataclass(frozen=True)
class DescriptionResult:
    description: str
    usage: Usage | None = field(default=None, kw_only=True, compare=False)


@dataclass(frozen=True)
class AnswerResult:
    status: Literal["answered", "unanswerable"]
    answer: str | None
    usage: Usage | None = field(default=None, kw_only=True, compare=False)


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
    usage: Usage | None = field(default=None, kw_only=True, compare=False)


class ModelBackend(Protocol):
    def __call__(self, request: ModelRequest) -> ModelResponse: ...
