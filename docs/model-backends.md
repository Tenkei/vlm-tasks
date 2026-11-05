# Connect a model backend

Tasks call a backend to run the model and validate its response. A
`PydanticAIBackend` can be shared by tasks; its requests do not share conversation
history.

## Choose a provider

Use `PydanticAIBackend` with an initialized PydanticAI model:

- [Hugging Face Transformers](huggingface-transformers.md): run loaded weights locally.
- [Amazon Bedrock](amazon-bedrock.md): connect with an AWS profile.
- [vLLM](../README.md#vllm-and-other-providers): use a running vision-model server.
- [Other PydanticAI providers](https://ai.pydantic.dev/models/overview/): configure
  a provider that supports image input.

```python
from ml_pipes.vlm import PydanticAIBackend

backend = PydanticAIBackend(model)
```

Here, `model` is the initialized model from your provider setup. Wrap SDK clients
with the corresponding PydanticAI provider; passing a raw SDK client, provider
string, or Agent directly to `PydanticAIBackend` is unsupported. For loaded
Transformers weights, use `TransformersModel` as shown in its guide.

## Output modes and retries

| Mode | How the model receives the requested structure |
|---|---|
| `prompted` (default) | Schema instructions in the prompt |
| `native` | The provider's native structured-output support |
| `tool` | Tool-call output with the requested schema |

Native and tool modes require matching model/API support. The Transformers
adapter supports prompted output only.

`retries` defaults to zero. A positive value allows correction attempts after an
invalid response, preserving the image, prior response, and validation feedback
within that request. SDK transport retries are configured separately on the
provider client. Validation adds no further correction retry policy.

Calls are synchronous; run them outside an active asyncio event loop.
`PydanticAIBackend` encodes each image as lossless PNG without resizing it.
For accepted image types, see the [task guide](tasks.md#images).

## Custom backends

Implement the callable `ModelBackend` protocol to integrate another runtime.
Each call receives a `ModelRequest` with the decoded image, task instruction,
and `ResponseSpec`. The spec contains the JSON Schema and a validation callable.
Return a `ModelResponse` containing the structured output mapping.

The task validates the returned data before converting it into its public result.
Backends can also use `response_spec.validate(data)` to validate responses and
provide correction feedback. The backend handles the shared request/response
contract; it does not need separate logic for each task.

See the [backend API](reference.md#model-integration) for types and signatures,
and [errors](reference.md#errors) for execution and response failures.
