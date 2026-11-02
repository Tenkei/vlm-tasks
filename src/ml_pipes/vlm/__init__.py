"""Structured vision-language task operators."""
from .backend import PydanticAIBackend
from .errors import InvalidModelResponseError, ModelExecutionError
from .ops import Classify, ClassifyMultiLabel, Detect, Ground, PredictAttributes
from .types import (
    AttributePredictionResult, BoundingBox, ClassificationResult, Detection,
    DetectionResult, GroundingResult, ModelBackend, ModelRequest, ModelResponse,
    ResponseSpec,
)

__all__ = [
    "Classify", "ClassifyMultiLabel", "Detect", "Ground", "PredictAttributes",
    "PydanticAIBackend", "BoundingBox", "ClassificationResult", "Detection",
    "DetectionResult", "GroundingResult", "AttributePredictionResult",
    "ModelBackend", "ModelRequest", "ModelResponse", "ResponseSpec",
    "InvalidModelResponseError", "ModelExecutionError",
]
