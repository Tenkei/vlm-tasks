"""Classify validation failures without parsing exception-message strings."""
from pydantic import ValidationError

from .errors import (
    InvalidModelResponseError, ResponseConstraintError, ResponseSchemaError,
    ResponseSyntaxError,
)


def validation_error(exc: ValidationError) -> InvalidModelResponseError:
    errors = exc.errors(include_url=False)
    if any(error["type"] == "json_invalid" for error in errors):
        error_type = ResponseSyntaxError
    elif all(
        error["type"] in {
            "greater_than", "greater_than_equal", "less_than", "less_than_equal",
            "finite_number", "string_too_short", "string_too_long", "value_error",
        }
        # All task enums contain strings; a non-string is a type error.
        or (error["type"] == "literal_error" and isinstance(error.get("input"), str))
        for error in errors
    ):
        error_type = ResponseConstraintError
    else:
        # Missing/extra fields, wrong types and list lengths take precedence
        # over constraint errors. The message and cause retain every violation.
        error_type = ResponseSchemaError
    return error_type(str(exc))


def response_error(exc: Exception) -> InvalidModelResponseError:
    """Recover the terminal validation failure from an SDK retry wrapper.

    Follow explicit causes only: implicit context can contain a previous attempt
    and must not determine the category of the final response.
    """
    cause = exc.__cause__
    seen = {id(exc)}
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        for error_type in (ResponseSyntaxError, ResponseSchemaError, ResponseConstraintError):
            if isinstance(cause, error_type):
                # Copy rather than re-raise an ancestor and create a cause cycle.
                return error_type(str(cause))
        if isinstance(cause, ValidationError):
            return validation_error(cause)
        if isinstance(cause, InvalidModelResponseError):
            return InvalidModelResponseError(str(cause))
        cause = cause.__cause__
    # Empty output, unexpected tool calls and other unclassified SDK failures
    # need not be syntax errors. Preserve the common base without guessing.
    return InvalidModelResponseError(str(exc))
