"""Errors independent of any model execution framework."""

__all__ = [
    "InvalidModelResponseError", "ResponseSyntaxError", "ResponseSchemaError",
    "ResponseConstraintError", "ModelExecutionError",
]


class InvalidModelResponseError(ValueError):
    """The model response failed parsing or the task's output contract."""


class ResponseSyntaxError(InvalidModelResponseError):
    """The response cannot be parsed as JSON."""


class ResponseSchemaError(InvalidModelResponseError):
    """The response has invalid fields, types, structure or item counts."""


class ResponseConstraintError(InvalidModelResponseError):
    """Response values violate vocabulary, range or task consistency constraints."""


class ModelExecutionError(RuntimeError):
    """Model execution failed; the underlying exception is preserved as cause."""
