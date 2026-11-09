"""Check all task results with synthetic responses, including installed wheels."""
from __future__ import annotations

import json
import numpy as np
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from ml_pipes.core import Pipeline
from ml_pipes.vision import ImagePayload
from ml_pipes.vlm import (
    Answer, AnswerResult, AttributePredictionResult, Attributes, BoundingBox, ClassificationResult,
    Classify, ClassifyMultiLabel, Describe, DescriptionResult, Detect, Detection, DetectionResult, Ground,
    GroundingResult, PydanticAIBackend,
)


def test_tasks_with_synthetic_responses() -> None:
    predictions = iter([
        {"labels": ["clean"]},
        {"labels": ["scratched", "stained"]},
        {"detections": [{
            "label": "window",
            "box": {"x1": 0.1, "y1": 0.2, "x2": 0.8, "y2": 0.9},
        }]},
        {"status": "ambiguous", "box": None},
        {"attributes": {"condition": "healthy", "color": "blue"}},
        {"description": "A blue window."},
        {"status": "answered", "answer": "blue"},
    ])

    def respond(messages, info):
        return ModelResponse(parts=[TextPart(json.dumps(next(predictions)))])

    backend = PydanticAIBackend(FunctionModel(respond))
    image = ImagePayload(np.zeros((100, 200, 3), dtype=np.uint8))
    operators = [
        Classify(backend, ["clean", "dirty"], context="Assess window cleanliness."),
        ClassifyMultiLabel(backend, ["scratched", "cracked", "stained"]),
        Detect(backend, ["window"]),
        Ground(backend, "the window"),
        Attributes(backend, {
            "condition": ["healthy", "damaged"], "color": ["blue", "red"],
        }),
        Describe(backend),
        Answer(backend, "What color is the window?"),
    ]
    expected = [
        ClassificationResult(("clean",)),
        ClassificationResult(("scratched", "stained")),
        DetectionResult(
            (Detection("window", BoundingBox(20, 20, 160, 90)),),
            image_size=(200, 100),
        ),
        GroundingResult("ambiguous", None, image_size=(200, 100)),
        AttributePredictionResult({"condition": "healthy", "color": "blue"}),
        DescriptionResult("A blue window."),
        AnswerResult("answered", "blue"),
    ]
    for operator, result in zip(operators, expected, strict=True):
        pipeline = Pipeline([operator])
        pipeline.validate(strict=True)
        assert pipeline(image) == result


if __name__ == "__main__":
    test_tasks_with_synthetic_responses()
    print("Synthetic task checks passed.")
