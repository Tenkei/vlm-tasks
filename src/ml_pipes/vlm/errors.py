"""Errors independent of any model execution framework."""

__all__ = ["InvalidModelResponseError", "ModelExecutionError"]


class InvalidModelResponseError(ValueError):
    """The model response failed parsing or the task's output contract."""


class ModelExecutionError(RuntimeError):
    """Model execution failed; the underlying exception is preserved as cause."""
