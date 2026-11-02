"""Example-local PydanticAI adapter for loaded Transformers vision-language models.

Targets single-device, decoder-only image/text models whose processor supports
multimodal apply_chat_template, such as Qwen3-VL Instruct. No model loading here.
"""
from __future__ import annotations

from io import BytesIO
from threading import Lock
from typing import Any

from pydantic_ai.messages import (
    BinaryContent, ModelMessage, ModelRequest, ModelResponse, RetryFeedbackPart,
    SystemPromptPart, TextContent, TextPart, UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.usage import RequestUsage


def chat_messages(messages: list[ModelMessage], instructions: str | None) -> list[dict]:
    """Translate the supported PydanticAI messages, including correction history."""
    from PIL import Image

    chat: list[dict] = []
    if instructions:
        # Includes PydanticAI's prompted-output schema; do not drop this layer.
        chat.append({"role": "system", "content": [{"type": "text", "text": instructions}]})
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, SystemPromptPart):
                    chat.append({"role": "system", "content": [{"type": "text", "text": part.content}]})
                elif isinstance(part, UserPromptPart):
                    content = []
                    items = [part.content] if isinstance(part.content, str) else part.content
                    for item in items:
                        if isinstance(item, (str, TextContent)):
                            content.append({"type": "text", "text": item if isinstance(item, str) else item.content})
                        elif isinstance(item, BinaryContent) and item.is_image:
                            with Image.open(BytesIO(item.data)) as image:
                                content.append({"type": "image", "image": image.convert("RGB")})
                        else:
                            raise ValueError(f"Unsupported Transformers input: {type(item).__name__}")
                    chat.append({"role": "user", "content": content})
                elif isinstance(part, RetryFeedbackPart):
                    chat.append({"role": "user", "content": [{"type": "text", "text": part.model_response()}]})
                else:
                    raise ValueError(f"Unsupported Transformers request part: {type(part).__name__}")
        elif isinstance(message, ModelResponse):
            if any(not isinstance(part, TextPart) for part in message.parts):
                raise ValueError("The Transformers adapter supports text responses only")
            chat.append({"role": "assistant", "content": [
                {"type": "text", "text": "\n".join(part.content for part in message.parts)}
            ]})
        else:
            raise ValueError(f"Unsupported Transformers message: {type(message).__name__}")
    return chat


class TransformersModel(FunctionModel):
    """PydanticAI Model wrapping caller-owned weights and a multimodal processor.

    FunctionModel runs the synchronous callback in a worker thread. Generation
    is serialized because the same loaded objects may serve several tasks.
    Prompted output only; no native JSON constraints, tools, streaming or videos.
    """

    def __init__(
        self, model: Any, processor: Any, *, model_name: str,
        max_new_tokens: int = 4096,
    ) -> None:
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if getattr(model.config, "is_encoder_decoder", False):
            raise ValueError("This adapter supports decoder-only models")
        self.weights = model.eval()
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self._generation_lock = Lock()
        super().__init__(
            self._generate,
            model_name=model_name,
            profile=ModelProfile(
                supports_tools=False,
                supports_json_schema_output=False,
                supports_json_object_output=False,
                supports_image_input=True,
            ),
        )

    def _generate(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        import torch

        settings = info.model_settings or {}
        unsupported = set(settings) - {"max_tokens", "temperature"}
        if unsupported:
            raise ValueError(f"Unsupported Transformers settings: {sorted(unsupported)}")
        if settings.get("temperature", 0) not in (0, 0.0):
            raise ValueError("This example uses greedy generation; temperature must be zero")
        limit = settings.get("max_tokens", self.max_new_tokens)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("max_tokens must be a positive integer")
        if info.output_tools or info.function_tools:
            raise ValueError("The Transformers adapter does not support tools")

        chat = chat_messages(messages, info.instructions)
        with self._generation_lock, torch.inference_mode():
            inputs = self.processor.apply_chat_template(
                chat, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(self.weights.device)
            input_tokens = inputs["input_ids"].shape[-1]
            generated = self.weights.generate(
                **inputs, max_new_tokens=limit, do_sample=False,
            )
            # Decoder-only generate() includes the prompt; return only new tokens.
            completion = generated[:, input_tokens:]
            text = self.processor.batch_decode(
                completion, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            output_tokens = completion.shape[-1]
        return ModelResponse(
            parts=[TextPart(text)],
            usage=RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        )
