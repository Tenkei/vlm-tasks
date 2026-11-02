"""Sanity check for VLM task execution with synthetic model responses."""
from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from ml_pipes.core import Pipeline
from ml_pipes.vision import ImagePayload
from ml_pipes.vlm import (
    Classify, ClassifyMultiLabel, Detect, Ground, PredictAttributes, PydanticAIBackend,
)


def main() -> None:
    predictions = iter([
        {"labels": ["clean"]},
        {"labels": ["scratched", "stained"]},
        {"detections": [{
            "label": "window",
            "box": {"x1": 0.1, "y1": 0.2, "x2": 0.8, "y2": 0.9},
        }]},
        {"status": "ambiguous", "box": None},
        {"attributes": {"condition": "healthy", "color": "blue"}},
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
        PredictAttributes(backend, {
            "condition": ["healthy", "damaged"], "color": ["blue", "red"],
        }),
    ]
    print("Sanity check: synthetic model responses.")
    for operator in operators:
        pipeline = Pipeline([operator])
        pipeline.validate(strict=True)
        print(type(operator).__name__, json.dumps(asdict(pipeline(image))))


if __name__ == "__main__":
    main()
