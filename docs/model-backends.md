# Connect a model backend

Tasks call a backend to run the model and validate its response. A
`PydanticAIBackend` can be shared by tasks; its requests do not share conversation
history.

## Choose a provider

Choose where the model runs and how to connect to it:

- [Hugging Face Transformers](huggingface-transformers.md): run loaded weights locally.
- [Amazon Bedrock](amazon-bedrock.md): connect with an AWS profile.
- [vLLM](#vllm): use a running vision-model server.
- [LiteLLM](#litellm): route requests through a LiteLLM proxy.
- [Hosted models](#hosted-models): connect through PydanticAI, for example to OpenAI.
- [Other PydanticAI providers](https://ai.pydantic.dev/models/overview/): configure
  a provider that supports image input.

## Choose a model

With your provider selected, pick a vision-language model for your task and a
PydanticAI `Model` adapter compatible with that API route. See
[PydanticAI's provider setup guides](https://ai.pydantic.dev/models/overview/#provider-directory)
for compatible combinations. For example, connect to Qwen3-VL on vLLM:

```python
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.vllm import VLLMProvider

from ml_pipes.vlm import PydanticAIBackend

model = OpenAIChatModel(
    "Qwen/Qwen3-VL-2B-Instruct",  # Replace with your server's model name.
    provider=VLLMProvider(base_url="http://localhost:8000/v1"),
)
backend = PydanticAIBackend(model)
```

The provider configures the endpoint and authentication. The model name passed
to the adapter selects the actual model or proxy alias. Here, `OpenAIChatModel`
speaks vLLM's OpenAI-compatible API.

Check image-input and structured-output capabilities for your chosen model and
route, then use [validation](validation.md) to measure its quality on your task.

Wrap the initialized model in `PydanticAIBackend`:

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
Retries can help correct invalid responses, but usually do not resolve
output-token-limit failures: each attempt still has the configured token cap.

See [response errors](#response-errors) for validation categories and handling
after correction attempts are exhausted.

Calls are synchronous; run them outside an active asyncio event loop.
`PydanticAIBackend` encodes each image as lossless PNG without resizing it.
For accepted image types, see the [task guide](tasks.md#images).

## Response errors

Response failures fall into three categories, all inheriting from
`InvalidModelResponseError` (`ValueError`). Import the exceptions from
`ml_pipes.vlm`; the [API reference](reference.md#errors) lists all error types.

| Exception | Category | Examples |
|---|---|---|
| `ResponseSyntaxError` | JSON syntax | Malformed JSON, missing delimiters |
| `ResponseSchemaError` | Fields, structure and types | Missing or extra fields, array instead of scalar/object, numeric strings, wrong list length |
| `ResponseConstraintError` | Value and consistency constraints | Unsupported string label/status, coordinates outside [0, 1], non-finite coordinates, degenerate/reversed boxes, duplicate selected labels, blank text, status/value mismatch |

An unsupported string enum value is a constraint error; a non-string supplied
for that enum is a schema/type error. If validation finds both schema and
constraint violations, `ResponseSchemaError` takes precedence and the message
retains all reported violations. Error categorization uses validator error
codes, without inferring intended values or repairing the response. A valid
but incorrect prediction is scored normally and does not trigger correction.

`PydanticAIBackend` raises the specific error after its configured correction
attempts are exhausted, preserving the final validation details in its message
and the original exception chain in `__cause__`. The category describes the
final failed attempt. Unexpected output without a known parsing/validation
cause, such as an empty response, uses `InvalidModelResponseError` without
guessing a category. Task schemas apply the same structure and constraint
categories to custom backends.

Catch specific exceptions to handle categories separately:

```python
from ml_pipes.vlm import (
    InvalidModelResponseError, ModelExecutionError,
    ResponseConstraintError, ResponseSchemaError, ResponseSyntaxError,
)

try:
    result = task(image)
except ResponseSyntaxError as exc:
    print("JSON syntax:", exc)
except ResponseSchemaError as exc:
    print("Response structure:", exc)
except ResponseConstraintError as exc:
    print("Response constraints:", exc)
except InvalidModelResponseError as exc:
    print("Other invalid response:", exc)
except ModelExecutionError as exc:
    print("Execution failed:", exc)
```

Catch only `InvalidModelResponseError` if all response categories need the
same handling. Execution failures raise the separate `ModelExecutionError`
(`RuntimeError`), with the original exception retained in `__cause__`;
response correction does not handle transport or authentication failures.
SDK transport retries are configured separately on the provider client.

## Hosted models

For OpenAI, install the provider dependency and set your API key:

```bash
python -m pip install 'pydantic-ai-slim[openai]==2.61.0'
export OPENAI_API_KEY="your-api-key"
```

```python
from pydantic_ai.models.openai import OpenAIChatModel
from ml_pipes.vlm import PydanticAIBackend

backend = PydanticAIBackend(OpenAIChatModel("gpt-4o-mini"))
```

Pass `backend` to a [task](tasks.md). For other hosted models, use the matching
[PydanticAI provider](https://ai.pydantic.dev/models/overview/).

## vLLM

Connect to a vision-language model already running on a vLLM server. Install
the client dependencies:

```bash
python -m pip install 'pydantic-ai-slim[openai]==2.61.0'
```

```python
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.vllm import VLLMProvider

from ml_pipes.vlm import PydanticAIBackend

backend = PydanticAIBackend(
  OpenAIChatModel(
    "Qwen/Qwen3-VL-2B-Instruct",  # Replace with your server's model name.
    provider=VLLMProvider(base_url="http://localhost:8000/v1"),
  )
)
```

Replace the URL with your server's address, then pass `backend` to a
[task](tasks.md).

## LiteLLM

Connect to a vision-language model through a running LiteLLM proxy. Install
the client dependencies and set your proxy API key:

```bash
python -m pip install 'pydantic-ai-slim[openai]==2.61.0'
export LITELLM_API_KEY="your-proxy-api-key"
```

```python
import os

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.litellm import LiteLLMProvider

from ml_pipes.vlm import PydanticAIBackend

backend = PydanticAIBackend(
  OpenAIChatModel(
    "qwen3-vl",  # Configure this proxy alias to route to Qwen3-VL.
    provider=LiteLLMProvider(
        api_base="http://localhost:4000/v1",
        api_key=os.environ["LITELLM_API_KEY"],
    ),
  )
)
```

Replace the URL and model alias with your proxy's settings. The selected route
must support image input. Pass `backend` to a [task](tasks.md).

## Custom backends

Implement the callable `ModelBackend` protocol to integrate another runtime.
Each call receives a `ModelRequest` with the decoded image, task instruction,
and `ResponseSpec`. The spec contains the JSON Schema and a validation callable.
Return a `ModelResponse` containing the structured output mapping.
Optionally pass `usage=Usage(...)` to include token counts and costs in task
results; import `Usage` from `ml_pipes.vlm`.

The task validates the returned data before converting it into its public result.
Backends can also use `response_spec.validate(data)` to validate responses and
provide correction feedback. The backend handles the shared request/response
contract; it does not need separate logic for each task.

See the [backend API](reference.md#model-integration) for types and signatures,
and [response errors](#response-errors) for execution and response failures.
