# VLM task examples

Run VLM tasks on an image using Amazon Bedrock or local Hugging Face
Transformers weights. Both runners use the same task configuration and execute
all five tasks by default, with an independent model request for each task.

| Example | Model execution | Requirements |
|---|---|---|
| [Amazon Bedrock](#amazon-bedrock) — [run_bedrock.py](run_bedrock.py) | Bedrock Converse API | Provider SDK, AWS credentials and model access |
| [Hugging Face Transformers](#hugging-face-transformers) — [run_transformers.py](run_transformers.py) | Weights loaded in the local Python process | PyTorch, Transformers, Pillow and a compatible model |

To check task execution with synthetic responses, run the
[sanity check](#sanity-check).

## Setup

Use Python 3.11+ and run commands from the repository root:

```bash
python -m pip install -e .
```

Install the additional dependencies for the example you choose below.

## Amazon Bedrock

[run_bedrock.py](run_bedrock.py) uses a vision-capable model exposed through
Bedrock Converse. The application creates a boto3 client and passes it to
PydanticAI.

### Dependencies and authentication

```bash
python -m pip install 'pydantic-ai-slim[bedrock]==2.61.0'
```

For an AWS IAM Identity Center profile, log in with:

```bash
aws sso login --profile YOUR_PROFILE
```

For a profile configured with AWS console login (`login_session`), use:

```bash
python -m pip install 'botocore[crt]'
aws login --profile YOUR_PROFILE
```

The CRT extra lets boto3 use console login credentials. To log in through a
URL on another device, use `aws login --profile YOUR_PROFILE --remote` and
enter the authorization code in the waiting CLI.

Other configured AWS authentication methods work through boto3's normal
credential chain. Omit `--profile` to use that chain, including `AWS_PROFILE`.

### Run

Choose a vision-capable Bedrock Converse model or inference profile available
to your account and region. For example, where available:

```bash
python -m examples.run_bedrock \
  --input /path/to/image.jpg \
  --profile YOUR_PROFILE \
  --region us-east-1 \
  --model us.amazon.nova-pro-v1:0 \
  --output /tmp/vlm-bedrock-results.json
```

Model access and region availability must match your account. The account
needs permission to invoke the chosen model. If you omit `--region`, boto3
uses the environment or AWS profile configuration; the report then records
`requested_region` as null.

Bedrock defaults to prompted output and temperature 0. Set
`--output-mode native` or `--output-mode tool` only when the model/API supports
that mode. `--retries` controls structured-output correction attempts; boto3
retains its normal transport retry policy.

## Hugging Face Transformers

[run_transformers.py](run_transformers.py) loads model weights and a processor
locally, then wraps them in `ml_pipes.vlm.TransformersModel` and
`PydanticAIBackend`. The default checkpoint is `Qwen/Qwen3-VL-2B-Instruct`.

### Dependencies

Install a PyTorch and torchvision build suitable for your machine, together
with Transformers and Pillow:

```bash
python -m pip install 'torch>=2.5' torchvision 'transformers>=4.57,<5' 'Pillow>=10'
```

The first run downloads model weights and the processor. For gated or private
checkpoints, authenticate with the Hugging Face CLI or set `HF_TOKEN`.

### Run

```bash
python -m examples.run_transformers \
  --input /path/to/image.jpg \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --device auto \
  --output /tmp/vlm-transformers-results.json
```

`--device auto` selects CUDA if available, then Apple MPS, then CPU. Use
`--device cpu`, `--device mps` or `--device cuda:0` to choose explicitly.
`--dtype` accepts `auto`, `float32`, `float16` or `bfloat16`.

Use `--model /path/to/checkpoint` for local weights and `--local-files-only`
to restrict loading to cached or local files. `--revision` selects the model
and processor revision. The adapter supports prompted output with greedy
generation and loads the model once for all selected tasks.

See the [Hugging Face Transformers guide](../docs/guides/huggingface-transformers.md)
for adapter details, supported models and reuse of already-loaded objects.

## Shared task configuration

[image_tasks.json](image_tasks.json) contains the default settings for both
runners:

| Task | Default question/target |
|---|---|
| classify | Is the overall scene indoor, outdoor, or unclear? |
| multilabel | Which of cat, dog, person, sofa, and remote control are present? |
| detect | Locate all instances of those object categories |
| ground | Locate the leftmost cat, or report absence/ambiguity |
| attributes | Predict scene lighting and setting |

Copy the JSON file, edit the labels, descriptions, contexts, target and attribute
choices for your image, and pass `--config /path/to/tasks.json`. Keys inside
each task are passed as that operator's constructor settings. Only selected
tasks need configuration entries.

Both runners accept these options:

| Option | Purpose |
|---|---|
| `--tasks classify` or `--tasks classify detect ground` | Select one or more tasks; all five run by default |
| `--config /path/to/tasks.json` | Supply custom task settings |
| `--retries N` | Allow structured-output correction attempts; default 0 |
| `--max-tokens N` | Limit generated tokens per request; default 4096 |
| `--output /path/to/results.json` | Save the JSON report as well as printing it |
| `--dry-run` | Decode the image and validate tasks without model execution |

### Validate without model execution

Dry runs need only the base package. Bedrock skips provider SDK imports and
credential discovery; Transformers skips runtime imports and model loading.
Bedrock still requires a model ID:

```bash
python -m examples.run_bedrock \
  --input /path/to/image.jpg --model us.amazon.nova-pro-v1:0 --dry-run

python -m examples.run_transformers \
  --input /path/to/image.jpg --dry-run
```

## Results

Both runners print JSON to stdout and optionally save it with `--output`.
Reports include image size, model settings, task configuration, predictions
and elapsed time per task. Progress goes to stderr. The Transformers report
also records model loading time, device, dtype and error cause chains.

Detection and grounding coordinates are floating-point xyxy pixels in the
input image. Scores remain unset. Valid output structure does not establish
visual accuracy or benchmark performance.

If one task fails, its error is recorded and the other tasks still run. Exit
codes are 0 when all tasks succeed, 1 for task failures and 2 for setup failures.

## Sanity check

[run_sanity_check.py](run_sanity_check.py) exercises all five tasks with synthetic responses
from a PydanticAI FunctionModel:

```bash
python -m examples.run_sanity_check
```

It requires only the base package, with no credentials or model downloads.
It checks task execution and result handling using synthetic model responses.
