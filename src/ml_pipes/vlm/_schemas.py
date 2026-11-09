"""Private Pydantic wire schemas; public task values are plain dataclasses."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import (
    BaseModel, ConfigDict, Field, ValidationError, create_model,
    field_validator, model_validator,
)

from .errors import InvalidModelResponseError
from .types import ResponseSpec


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _Selection(_WireModel):
    labels: list[str]

    @field_validator("labels")
    @classmethod
    def unique_labels(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Selected labels must be unique")
        return value


class _Description(_WireModel):
    description: str = Field(min_length=1)

    @field_validator("description")
    @classmethod
    def nonempty_description(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("description must be a nonempty string")
        return value


class _Answer(_WireModel):
    status: Literal["answered", "unanswerable"]
    answer: str | None = Field(min_length=1)

    @model_validator(mode="after")
    def consistent(self) -> _Answer:
        if self.answer is not None and not self.answer.strip():
            raise ValueError("answer must be a nonempty string or null")
        if (self.status == "answered") != (self.answer is not None):
            raise ValueError("An answer is required exactly when status is answered")
        return self


class _Box(_WireModel):
    x1: float = Field(ge=0, le=1, allow_inf_nan=False)
    y1: float = Field(ge=0, le=1, allow_inf_nan=False)
    x2: float = Field(ge=0, le=1, allow_inf_nan=False)
    y2: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered(self) -> _Box:
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError("Boxes require x1 < x2 and y1 < y2")
        return self


class _Grounding(_WireModel):
    status: Literal["found", "not_found", "ambiguous"]
    box: _Box | None

    @model_validator(mode="after")
    def consistent(self) -> _Grounding:
        if (self.status == "found") != (self.box is not None):
            raise ValueError("A box is required exactly when status is found")
        return self


class _Point(_WireModel):
    x: float = Field(ge=0, le=1, allow_inf_nan=False)
    y: float = Field(ge=0, le=1, allow_inf_nan=False)


class _PointGrounding(_WireModel):
    status: Literal["found", "not_found", "ambiguous"]
    point: _Point | None

    @model_validator(mode="after")
    def consistent(self) -> _PointGrounding:
        if (self.status == "found") != (self.point is not None):
            raise ValueError("A point is required exactly when status is found")
        return self


def _spec(model: type[BaseModel]) -> ResponseSpec:
    def validate(data: object) -> Mapping[str, object]:
        try:
            value = model.model_validate(data)
        except ValidationError as exc:
            raise InvalidModelResponseError(str(exc)) from exc
        return value.model_dump(mode="json", by_alias=True)

    return ResponseSpec(schema=model.model_json_schema(), validate=validate)


def _label_type(choices: Mapping[str, str]):
    return Literal.__getitem__(tuple(choices))


def description_spec() -> ResponseSpec:
    return _spec(_Description)


def answer_spec() -> ResponseSpec:
    return _spec(_Answer)


def classification_spec(choices: Mapping[str, str], *, multiple: bool) -> ResponseSpec:
    label = _label_type(choices)
    model = create_model(
        "MultiLabelPrediction" if multiple else "ClassificationPrediction",
        __base__=_Selection,
        labels=(
            list[label],
            Field(min_length=0 if multiple else 1, max_length=len(choices) if multiple else 1),
        ),
    )
    return _spec(model)


def detection_spec(choices: Mapping[str, str]) -> ResponseSpec:
    item = create_model(
        "DetectedInstance",
        __base__=_WireModel,
        label=(_label_type(choices), ...),
        box=(_Box, ...),
    )
    model = create_model(
        "DetectionPrediction", __base__=_WireModel, detections=(list[item], ...)
    )
    return _spec(model)


def grounding_spec(geometry: Literal["box", "point"] = "box") -> ResponseSpec:
    if geometry not in ("box", "point"):
        raise ValueError("geometry must be 'box' or 'point'")
    return _spec(_PointGrounding if geometry == "point" else _Grounding)


def attributes_spec(attributes: Mapping[str, Mapping[str, str]]) -> ResponseSpec:
    # Aliases permit arbitrary attribute names, including Pydantic-reserved names.
    fields = {
        f"attribute_{index}": (_label_type(choices), Field(alias=name))
        for index, (name, choices) in enumerate(attributes.items())
    }
    values = create_model("AttributeValues", __base__=_WireModel, **fields)
    model = create_model(
        "AttributePrediction", __base__=_WireModel, attributes=(values, ...)
    )
    return _spec(model)
