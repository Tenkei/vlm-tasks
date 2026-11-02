# VLM image tasks with Amazon Bedrock

For directly loaded Hugging Face weights with the same task operators and
PydanticAI backend, see the [local Transformers example](transformers.md).

Run one or all five MVP tasks on an image using a vision-capable model exposed
through Bedrock Converse. Each task receives the same decoded image and has an
independent model request. All five tasks run by default.

## Setup

From the repository root, in a Python 3.11+ environment:

```bash
python -m pip install -e .
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
credential chain. Omit `--profile` to use that chain, including AWS_PROFILE.
The application creates the boto3 client and passes it to PydanticAI.

## Run

Choose a vision-capable Bedrock Converse model or inference profile available
to your account and region. For example, where available:

```bash
python -m examples.vlm.run_bedrock \
  --input /path/to/image.jpg \
  --profile YOUR_PROFILE \
  --region us-east-1 \
  --model us.amazon.nova-pro-v1:0 \
  --output /tmp/vlm-image-results.json
```

The model ID above is an example; access and region availability must match
your account. The account needs permission to invoke the chosen model.
If you omit `--region`, boto3 uses the environment or AWS profile configuration.
The report records `requested_region` as null in that case.

For one task, add `--tasks classify`; for several, use
`--tasks classify detect ground`. Add `--dry-run` to decode the image and
validate the selected task pipelines without discovering AWS credentials or
making model requests. The `--model` flag is still required for a dry run.

## Task configuration

[image_tasks.json](image_tasks.json) contains the default task settings:

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

The example defaults to prompted output, zero correction retries, temperature
0 and at most 4096 output tokens per request. Set `--output-mode native` or
`--output-mode tool` only when the chosen model/API supports that mode.
`--retries N` controls correction attempts; boto3 retains its normal transport
retry policy. Models must support the configured inference settings.

## Results

JSON is printed to stdout and optionally saved with `--output`. It includes
the image size, model/configuration, predictions and elapsed time per task.
Progress goes to stderr. Detection and grounding coordinates are floating
point xyxy pixels in the original image. Scores remain unset.

If one task fails, its error is recorded and the other tasks still run; the
process exits with status 1. Invalid local setup exits with status 2. A
successful run exits with status 0. This example supports visual inspection
of predictions; it does not measure benchmark accuracy without annotations.

For the credential-free synthetic demo, run `python -m examples.vlm.run_demo`.
