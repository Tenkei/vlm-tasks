# Use Amazon Bedrock with VLM tasks

Connect VLM tasks to a vision model hosted on Amazon Bedrock using your AWS
profile.

## Install

From the repository root, install the package and Bedrock dependencies:

```bash
python -m pip install -e .
python -m pip install 'pydantic-ai-slim[bedrock]==2.61.0'
```

## Sign in to AWS

Install [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html),
then sign in through your browser using [AWS login](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sign-in.html).
This credential provider also requires `botocore[crt]`:

```bash
python -m pip install 'botocore[crt]'
aws login \
  --profile YOUR_PROFILE
```

For an existing IAM Identity Center (SSO) profile, use this command instead;
see the [SSO setup guide](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sso.html)
to configure a profile:

```bash
aws sso login \
  --profile YOUR_PROFILE
```

Repeat the appropriate login command when your session expires.

## Connect the backend and run a task

Choose a model that supports image input and is available to your account in
the selected region. This example uses Qwen3 VL in Tokyo (`ap-northeast-1`):

```python
import boto3
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider

from ml_pipes.vlm import Attributes, PydanticAIBackend

session = boto3.Session(
    profile_name="YOUR_PROFILE",
    region_name="ap-northeast-1",
)
model = BedrockConverseModel(
    "qwen.qwen3-vl-235b-a22b",
    provider=BedrockProvider(bedrock_client=session.client("bedrock-runtime")),
)
backend = PydanticAIBackend(model)

task = Attributes(backend, attributes={"setting": ["indoor", "outdoor"]})
result = task("/path/to/image.jpg")
print(result.attributes)  # Example: {'setting': 'indoor'}
```

Replace the profile and image path with your own values. The same `backend`
works with any [available task](../../README.md#available-tasks).

For command-line examples and dataset validation, see
[Run the examples](../../examples/README.md#amazon-bedrock). For other credential
methods, see the [Boto3 credentials guide](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html).
