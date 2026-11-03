# VLM task examples

Run VLM tasks on individual images or evaluate classification, detection and
grounding against a labeled dataset. The single-image runners use Amazon Bedrock or
local Hugging Face Transformers weights and execute all five tasks by default,
with an independent model request for each task.

| Example | Model execution | Requirements |
|---|---|---|
| [Amazon Bedrock](#amazon-bedrock) — [run_bedrock.py](run_bedrock.py) | Bedrock Converse API | Provider SDK, AWS credentials and model access |
| [Hugging Face Transformers](#hugging-face-transformers) — [run_transformers.py](run_transformers.py) | Weights loaded in the local Python process | PyTorch, Transformers, Pillow and a compatible model |
| [Dataset validation](#dataset-validation) — [run_validation.py](run_validation.py) | Bedrock or local Transformers, one call per image or grounding expression | Backend dependencies; default datasets download automatically; validation extra for detection YAML |

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

## Dataset validation

[run_validation.py](run_validation.py) evaluates classification, detection and
box or point grounding with Bedrock (default) or local Transformers weights.
Install the backend dependencies described above. Default datasets download
automatically; see the [validation guide](../docs/validation.md) for defaults
and custom local data.

Classification:

```bash
python -m examples.run_validation \
  --task classify --max-samples 100 \
  --model us.amazon.nova-pro-v1:0 --profile YOUR_PROFILE --region us-east-1
```

Detection:

```bash
python -m examples.run_validation \
  --task detect --max-samples 100 \
  --model us.amazon.nova-pro-v1:0 --profile YOUR_PROFILE --region us-east-1
```

Grounding with local Qwen3-VL:

```bash
python -m examples.run_validation \
  --backend transformers --task ground \
  --max-samples 10 \
  --model Qwen/Qwen3-VL-2B-Instruct --device mps --dtype float16 \
  --retries 0 --max-tokens 4096 --output /tmp/grounding10.json
```

### Runner options and configuration

This example script wraps the Python validation API. Its CLI flags, configuration
defaults, JSON reports, and exit codes are specific to the example. See the
[validation guide](../docs/validation.md) for datasets, metrics, architecture,
and custom data formats.

| Option | Purpose and default |
|---|---|
| `--task` | Required: `classify`, `detect`, or `ground` |
| `--backend`, `--model` | Bedrock by default, requiring a model ID; Transformers defaults to `Qwen/Qwen3-VL-2B-Instruct` |
| `--data` | Registered dataset name or local path; omitted uses the task default |
| `--split`, `--max-samples` | `val` and no limit; limits images or grounding expressions |
| `--cache-dir` | Overrides `VLM_TASKS_DATASETS_DIR` and `~/.cache/vlm-tasks/datasets` |
| `--prepare-only` | Prepare a registered dataset without loading a model or running inference |
| `--config` | JSON task constructor settings; defaults described below |
| `--class-map` | JSON file mapping dataset names to task labels; classification/detection only |
| `--match-iou` | Detection matching threshold; default `0.5` |
| `--images`, `--dataset` | Local grounding image directory and annotation convention; dataset defaults to `refcocog` |
| `--output-mode`, `--retries`, `--max-tokens` | Defaults: `prompted`, `0` correction attempts, `4096` generated tokens per request |
| `--output`, `--no-verbose` | Save the JSON report; disable progress and aggregate score logging on stderr |

Bedrock authentication flags (`--profile`, `--region`) and Transformers loading
flags (`--device`, `--dtype`, `--revision`, `--local-files-only`) work as described
in their sections above. Transformers accepts only prompted output and uses
greedy generation. Backend/transport retry policies are separate from correction
retries. Use `python -m examples.run_validation --help` for all CLI options.

Without `--config`, registered classification/detection datasets use all dataset
labels; local classification/detection data uses [image_tasks.json](image_tasks.json),
so supply matching settings when needed. Grounding defaults to empty task settings
and gets targets from dataset expressions. Explicit configs retain their labels
and context. For point grounding, pass `--config /path/to/point.json` containing:

```json
{"ground": {"geometry": "point", "context": "Resolve the whole object."}}
```

Prepare a default dataset independently:

```bash
python -m examples.run_validation --task ground --prepare-only --max-samples 100
```

The runner prepares named data before model loading. Initial runs download any
uncached dataset and model files. Local paths bypass dataset downloads; local
grounding requires `--images`, with `--dataset` selecting the annotation naming
convention. Automatic grounding preparation supports only `refcocog`.

### Validation reports

JSON goes to stdout and optionally `--output`; progress goes to stderr. Reports
include aggregate/per-category metrics, failures, dataset counts and provenance,
grounding statuses, task/model settings, output mode, retries, and timings.
Transformers also records device, dtype, requested revision, and local-file mode.

`preparation_seconds` measures initial dataset preparation; `load_seconds`
measures backend/model setup. `elapsed_seconds` includes the validator's own
dataset setup/cache checks and iteration. `speed_ms_per_sample` separates sample
loading, inference, and metric updates; its `load` entry is not model loading.

Exit codes are **0** for a completed run without recorded inference failures,
**1** for a completed run with such failures, and **2** when setup or validation
cannot complete. Accuracy has no automatic pass/fail threshold.

## Shared task configuration

[image_tasks.json](image_tasks.json) contains the default single-image task
settings. Dataset runner defaults are described in
[Runner options and configuration](#runner-options-and-configuration).
The single-image defaults are:

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

The two single-image runners accept these options:

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

Detection and box grounding use floating-point xyxy pixels in the input image;
point grounding uses floating-point x/y pixels. Scores remain unset. Valid output structure does not establish
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
