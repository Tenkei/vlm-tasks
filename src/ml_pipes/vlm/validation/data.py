"""Task columns and scoring identifiers for loaded datasets.

Rows may include additional reporting metadata such as sample_id and im_file.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Literal, NotRequired, Protocol, TypedDict

import numpy as np


class _Sample(TypedDict):
    # A nonempty HWC uint8 RGB image, produced by dataset formatting/transforms.
    image: np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]


class ClassificationSample(_Sample):
    label: int


class DetectionObjects(TypedDict):
    bbox: list[list[float]]  # Original-image pixel xyxy coordinates.
    category: list[int]


class DetectionSample(_Sample):
    objects: DetectionObjects


class GroundingSample(_Sample):
    text: str
    bbox: list[float]  # One original-image pixel xyxy box.
    label: int
    image_id: NotRequired[int]


class AttributesSample(_Sample):
    # image contains the selected subject crop.
    attributes: dict[str, int | None]
    image_id: NotRequired[int]


class AnswerSample(_Sample):
    question: str
    answers: list[str]  # All ten VQA reference votes, including duplicates.
    answer_type: Literal["yes/no", "number", "other"]
    question_type: str
    image_id: NotRequired[int]


class DescribeSample(_Sample):
    captions: list[str]
    image_id: NotRequired[int]


class ValidationDataset(Protocol):
    """A loaded split yielding sample mappings, with HF-style feature metadata.

    Categorical targets require ClassLabel-like features exposing complete
    ``names``. Other tasks may omit features. Length and indexing are optional.
    """

    @property
    def features(self) -> Mapping[str, Any] | None: ...

    def __iter__(self) -> Iterator[Mapping[str, Any]]: ...


def _limit(max_samples: int | None) -> None:
    if max_samples is not None and (type(max_samples) is not int or max_samples < 1):
        raise ValueError("max_samples must be a positive integer or None")
