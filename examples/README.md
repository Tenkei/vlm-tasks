# Run the examples

Use Python 3.11+ and run commands from the repository root; see the [project README](../README.md) for an introduction.

## Setup

Install the package:

```bash
python -m pip install -e .
```

## Task Configuration

Copy [image_tasks.json](image_tasks.json) and edit the single-image task settings:

```bash
cp examples/image_tasks.json /tmp/vlm-tasks.json
```

Or pass your own `--config /path/to/tasks.json`. Settings match the
[task constructor options](../docs/reference.md#operators); for example,
`Ground` accepts `geometry="point"` for direct point output.
`Describe` accepts optional `context` for focus and detail. `Answer` needs a
`question` in the configuration because these runners pass an image to each task.

## Hugging Face Transformers

Run tasks with local Qwen3-VL weights (`Qwen/Qwen3-VL-2B-Instruct` by default);
see the [Transformers guide](../docs/huggingface-transformers.md) for backend setup.

```bash
python -m pip install \
  'torch>=2.5' \
  torchvision \
  'transformers>=4.57,<5' \
  'Pillow>=10'

python -m examples.run_transformers \
  --input /path/to/image.jpg \
  --config /tmp/vlm-tasks.json \
  --output /tmp/vlm-transformers-results.json
```

- Use `--device mps` or `--device cuda:0` to select a GPU. The default `auto`
  selects CUDA if available, then Apple MPS, then CPU.
- `--dtype` accepts `auto`, `float32`, `float16` or `bfloat16`; choose a dtype
  supported by the device. The default `auto` uses the checkpoint's dtype.
- To load a local checkpoint, pass its directory with `--model /path/to/checkpoint`.

## Amazon Bedrock

Follow the [Bedrock setup guide](../docs/amazon-bedrock.md) to install dependencies, sign in, and choose a vision model available in your account and region.

```bash
python -m examples.run_bedrock \
  --region ap-northeast-1 \
  --profile YOUR_PROFILE \
  --model qwen.qwen3-vl-235b-a22b \
  --input /path/to/image.jpg \
  --config /tmp/vlm-tasks.json \
  --output /tmp/vlm-bedrock-results.json
```

## Task validation

Install the validation extra for detection YAML support, plus your backend dependencies above;
see the [validation guide](../docs/validation.md) for details.

```bash
python -m pip install -e '.[validation]'
```

### Built-in datasets

Evaluate attribute prediction on OVAD with automatic dataset preparation:

```bash
python -m examples.run_validation \
  --backend transformers \
  --task attributes \
  --max-samples 10 \
  --output /tmp/vlm-attributes-validation.json
```

Use `--task classify`, `--task detect`, `--task ground`, `--task answer`, or `--task describe` for
the other [built-in validations](../docs/validation.md#available-validations).

Evaluate Answer on VQA v2 with questions supplied by the dataset:

```bash
python -m examples.run_validation \
  --backend transformers \
  --task answer \
  --max-samples 100 \
  --output /tmp/vlm-answer-validation.json
```

Answer defaults to the labeled `val` split; `--max-samples` counts questions.
Reports contain `vqa_accuracy` (0–1), `per_type` breakdowns, response statuses,
and normalization provenance. For local data, provide `--data` with the official
questions/annotations directory and `--images` with the COCO 2014 image directory.
Use a configuration such as `{"answer": {"context": "Use visible evidence."}}`
with `--config` to supply task context; dataset questions override any default.

Start with `--max-samples 10` to check runtime, then try 100 samples.

Evaluate Describe on COCO Captions 2017 (requires Java on `PATH`):

```bash
python -m examples.run_validation \
  --backend transformers \
  --task describe \
  --max-samples 100 \
  --output /tmp/vlm-describe-validation.json
```

`--max-samples` counts images. Reports contain `cider`, `bleu_1`–`bleu_4`,
`rouge_l`, and one corpus `summary`. Use `--tokenizer-version` to select `3.4.1`
(default) or `4.6.0`. For local data, provide
`--data /datasets/captions_val2017.json` and `--images /datasets/val2017`.
Task context can be supplied with a config such as
`{"describe": {"context": "Describe the visible scene in one sentence."}}`.
See [Describe validation](../docs/validation.md#describe-coco-captions) for scoring
and comparison guidance.

### Your own data

Evaluate local attribute annotations and images; see the [custom dataset guide](../docs/validation.md#using-your-own-dataset) for supported formats.

```bash
python -m examples.run_validation \
  --backend transformers \
  --task attributes \
  --data /path/to/annotations.json \
  --images /path/to/images \
  --max-samples 10 \
  --output /tmp/vlm-custom-validation.json
```

### Validate with Bedrock

Use your AWS profile and model from the [Bedrock setup guide](../docs/amazon-bedrock.md):

```bash
python -m examples.run_validation \
  --backend bedrock \
  --region ap-northeast-1 \
  --profile YOUR_PROFILE \
  --model qwen.qwen3-vl-235b-a22b \
  --task attributes \
  --max-samples 10 \
  --output /tmp/vlm-bedrock-validation.json
```

### Validate with Transformers

Evaluate a task with local model weights. For example, run attribute validation
on Apple MPS with float16:

```bash
python -m examples.run_validation \
  --backend transformers \
  --task attributes \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --device mps \
  --dtype float16 \
  --max-samples 10 \
  --output /tmp/vlm-transformers-validation.json
```

The runner accepts the same `--model`, `--device`, `--dtype` and `--max-tokens`
options as `run_transformers`; see [task validation](../docs/validation.md) for details.

## Common options

- Single-image runners use `--tasks` to select one or more tasks; all available tasks run by default.
- Validation uses `--task` to select one task.
- Runners print JSON; `--output /path/to/results.json` also saves it.
- Exit codes are `0` for success, `1` for model failures, and `2` for setup or validation errors.

List all command-line options:

```bash
python -m examples.run_transformers --help
python -m examples.run_bedrock --help
python -m examples.run_validation --help
```
