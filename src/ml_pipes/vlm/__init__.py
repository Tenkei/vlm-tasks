"""Structured vision-language task operators."""
from .backend import PydanticAIBackend
from .errors import InvalidModelResponseError, ModelExecutionError
from .ops import Answer, Attributes, Classify, ClassifyMultiLabel, Describe, Detect, Ground
from .transformer import TransformersModel
from .types import (
    AnswerResult, AttributePredictionResult, BoundingBox, ClassificationResult,
    DescriptionResult, Detection, DetectionResult, GroundingResult,
    ModelBackend, ModelRequest, ModelResponse,
    Point, ResponseSpec,
)

__all__ = [
    "Classify", "ClassifyMultiLabel", "Detect", "Ground", "Attributes", "Describe", "Answer",
    "PydanticAIBackend", "TransformersModel", "BoundingBox", "Point", "ClassificationResult", "Detection",
    "DetectionResult", "GroundingResult", "AttributePredictionResult", "DescriptionResult", "AnswerResult",
    "ModelBackend", "ModelRequest", "ModelResponse", "ResponseSpec",
    "InvalidModelResponseError", "ModelExecutionError",
]
