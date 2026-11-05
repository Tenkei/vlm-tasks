# Run the examples

Use Python 3.11+ and run commands from the repository root; see the [project README](../README.md) for an introduction.

## Setup

Install the package:

```bash
python -m pip install -e .
```

## Task Configuration

Copy [image_tasks.json](image_tasks.json) and edit the single-image task settings; see the [task reference](../docs/reference.md#operators) for available fields.

```bash
cp examples/image_tasks.json /tmp/vlm-tasks.json
```

Pass the copy with `--config`; add `--tasks attributes` to run only attribute prediction.

## Hugging Face Transformers

Run all five tasks with local Qwen3-VL weights; see the [Transformers guide](../docs/huggingface-transformers.md) for model and device options.

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

Install the validation extra for detection YAML support, plus your backend dependencies above; see the [validation guide](../docs/validation.md) for details.

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

Use `--task classify`, `--task detect`, or `--task ground` for the other [built-in validations](../docs/validation.md#available-validations).

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

## Output and options

Runners print JSON; `--output /path/to/results.json` also saves it, and the [validation guide](../docs/validation.md#results-and-reproducibility) explains the metrics and timings.

Exit codes are `0` for success, `1` for model failures, and `2` for setup or validation errors; see the [validation runner](run_validation.py) for report fields.

List all command-line options:

```bash
python -m examples.run_transformers --help
python -m examples.run_bedrock --help
python -m examples.run_validation --help
```
