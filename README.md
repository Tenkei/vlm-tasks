# ml-pipes-vlm

Task operators for vision-language models, using PydanticAI for model
communication and structured output execution.

**MVP:** Classify, ClassifyMultiLabel, Detect, Ground, and PredictAttributes.
Requires Python 3.11+. Import operators from `ml_pipes.vlm`.

## Installation

From a standalone repository checkout:

    python -m pip install -e .

Install the provider SDKs you use separately, for example:

    python -m pip install 'pydantic-ai-slim[bedrock]==2.61.0'

The base VLM package depends on pydantic-ai-slim, without every provider SDK,
and reuses vision's ImagePayload. It does not load weights or start servers.
The initial standalone version is 0.1.2rc1. The framework dependencies are
`ml-pipes-core>=0.1.3,<0.2.0` and `ml-pipes-vision>=0.1.3,<0.2.0`.
Provider authentication, clients and model loading remain application concerns.

## Usage

Application code supplies an initialized PydanticAI Model:

    from ml_pipes.core import Pipeline
    from ml_pipes.vision import Decode, LoadFile
    from ml_pipes.vlm import Classify, PydanticAIBackend
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

    # existing_client: the application's configured boto3 bedrock-runtime client
    # model_id: a vision-capable model available to that client
    model = BedrockConverseModel(
        model_id,
        provider=BedrockProvider(bedrock_client=existing_client),
    )
    backend = PydanticAIBackend(model)
    pipeline = Pipeline([
        LoadFile(),
        Decode(),
        Classify(
            backend,
            labels={"clean": "No visible dirt", "dirty": "Visible dirt"},
            context="Assess the cleanliness of the visible windows.",
        ),
    ])
    result = pipeline("windows.jpg")
    print(result.labels)

All task configuration is constructor-only in the MVP. Every operator accepts
one ImagePayload and returns its dedicated result. Backends can be shared;
calls have independent histories.

For a local OpenAI-compatible inference server, initialize a PydanticAI model
in application code:

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.vllm import VLLMProvider

    model = OpenAIChatModel(
        "Qwen/Qwen3-VL-4B-Instruct",
        provider=VLLMProvider(base_url="http://localhost:8000/v1"),
    )
    backend = PydanticAIBackend(model)

The server must already be running. API/model support for images and output
modes is required. A checkpoint name alone is not evidence of task accuracy.

## Output modes and failures

PydanticAIBackend defaults to prompted structured output and zero correction
retries. Set output_mode="native" or "tool" for a model/API that supports it.
Set retries to a nonnegative integer to allow bounded correction attempts.
Provider SDK transport retries remain controlled by the configured SDK client.

Inputs must be uint8 HWC RGB/BGR or uint8 HW GRAY images. Geometry is returned
as floating-point xyxy pixel coordinates relative to the input image.
Scores are unset: generated answers do not establish calibrated confidence.

InvalidModelResponseError reports invalid output after the correction budget.
ModelExecutionError wraps failures during execution; its cause is preserved.
Invalid task configuration and images raise ValueError before model execution.
The MVP execution path is synchronous; run it outside an active asyncio loop.

## Demo and references

Run the credential-free demo from the repository root:

    python -m examples.vlm.run_demo

The demo uses a PydanticAI FunctionModel with synthetic predictions. It tests
the execution path, not a model's visual accuracy.

For real images using your AWS login, see the runnable
[Bedrock image example](examples/vlm/README.md).

- [Operator and type reference](docs/reference.md)
- [Decisions, alternatives, and roadmap](docs/design.md)

## Development

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

The suite includes the local Transformers adapter tests, using fake weights
and a processor. Tests and the synthetic demo require no credentials, model
downloads or live provider calls. Provider SDKs and local inference runtimes
remain optional.

- [Contributing](.github/CONTRIBUTING.md)
- [Release process](RELEASING.md)
- [Extraction provenance](docs/extraction.md)
