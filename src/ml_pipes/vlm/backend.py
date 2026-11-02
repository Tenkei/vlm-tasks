"""The only module that integrates the PydanticAI execution framework."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic_ai import Agent, BinaryContent, ModelRetry, NativeOutput, PromptedOutput, StructuredDict, ToolOutput
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models import Model

from ._images import encode_image
from .errors import InvalidModelResponseError, ModelExecutionError
from .types import ModelRequest, ModelResponse

__all__ = ["PydanticAIBackend"]


class PydanticAIBackend:
    def __init__(
        self,
        model: Model,
        *,
        output_mode: Literal["prompted", "native", "tool"] = "prompted",
        retries: int = 0,
    ) -> None:
        if not isinstance(model, Model):
            raise TypeError("PydanticAIBackend requires an initialized PydanticAI Model")
        if output_mode not in {"prompted", "native", "tool"}:
            raise ValueError("output_mode must be prompted, native, or tool")
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError("retries must be a nonnegative integer")
        self.model = model
        self.output_mode = output_mode
        self.retries = retries

    def __call__(self, request: ModelRequest) -> ModelResponse:
        image = BinaryContent(data=encode_image(request.image), media_type="image/png")
        output_type = StructuredDict(dict(request.response_spec.schema))
        marker = {
            "prompted": PromptedOutput,
            "native": NativeOutput,
            "tool": ToolOutput,
        }[self.output_mode]
        agent = Agent(
            self.model,
            output_type=marker(output_type),
            retries=self.retries,
        )

        @agent.output_validator
        def validate(data: dict[str, object]) -> Mapping[str, object]:
            try:
                return request.response_spec.validate(data)
            except InvalidModelResponseError as exc:
                raise ModelRetry(str(exc)) from exc

        try:
            result = agent.run_sync([request.instruction, image])
        except UnexpectedModelBehavior as exc:
            raise InvalidModelResponseError(str(exc)) from exc
        except Exception as exc:
            raise ModelExecutionError(str(exc)) from exc
        return ModelResponse(data=result.output)
