# vlm-tasks

Repository: `vlm-tasks`. Distribution: `ml-pipes-vlm`. Import: `ml_pipes.vlm`.

Task operators for vision-language models, using PydanticAI for model
communication and structured output execution.

**MVP:** Classify, ClassifyMultiLabel, Detect, Ground, and Attributes.
Requires Python 3.11+. Import operators from `ml_pipes.vlm`.

## Installation

From the repository root:

    python -m pip install -e .

Install the provider SDKs you use separately, for example:

    python -m pip install 'pydantic-ai-slim[bedrock]==2.61.0'

The base VLM package depends on pydantic-ai-slim, without every provider SDK,
and reuses vision's ImagePayload. It does not load weights or start servers.
Framework dependencies are
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

Operators can also be called directly, without constructing a Pipeline:

```python
operator = Classify(backend, ["clean", "dirty"], context="Assess the windows.")
result = operator(image)  # image is an ImagePayload
```

Operators accept an ImagePayload and return their dedicated result. Ground also
accepts `task((image, target))` to supply or override its optional constructor target;
other settings are configured at construction. Backends can be shared; calls
have independent histories.

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

## Validation

Evaluate a configured task on local labeled data:

```python
metrics = operator.val(data="/path/to/dataset", max_samples=10)
print(metrics.results_dict)
```

See the [validation guide](docs/validation.md) for dataset setup, configuration
and metrics, and the [examples](examples/README.md#dataset-validation) for CLI commands.
