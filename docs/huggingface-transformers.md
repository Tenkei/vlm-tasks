# Use Hugging Face Transformers with VLM tasks

Connect local Hugging Face image/text models to VLM tasks using
`TransformersModel` and `PydanticAIBackend`.

## How it connects

```text
ImagePayload → task → PydanticAIBackend → TransformersModel → loaded weights
             ← result ← validation/parsing ← generated text ←
```

[TransformersModel](reference.md#model-integration) is a reusable adapter
shipped with `ml_pipes.vlm`, using PydanticAI's public FunctionModel hook.
It accepts already-loaded weights and a processor. It translates text, image
bytes and correction history into the processor's chat format, applies the chat
template, runs greedy generation and returns only generated tokens.
PydanticAI's schema instructions reach the
model through the system message. PydanticAI and the task validator parse and
validate the result, and the task converts coordinates to input-image pixels.

The adapter is shared by all tasks. Loading, device selection and Hugging Face
authentication remain application concerns.

## Setup

Use Python 3.11+ and install a PyTorch build suitable for your machine. From the
repository root:

```bash
python -m pip install -e .
python -m pip install \
  'torch>=2.5' \
  torchvision \
  'transformers>=4.57,<5' \
  'Pillow>=10'
```

The adapter targets the package's pinned `pydantic-ai-slim==2.61.0` API.
Transformers 4.57+ includes Qwen3-VL support. Qwen's processor also requires
torchvision for its video processor, even when the task supplies only an image.
Install a torchvision build compatible with your PyTorch build.

Transformers downloads model and processor files as needed for remote checkpoints.
For gated/private checkpoints, authenticate with the Hugging Face CLI or set
`HF_TOKEN` in your environment.
Inference runs in the local Python process. Memory requirements depend on
checkpoint, dtype and image token count.

## Create a backend in Python

After installing the dependencies above, load a model and wrap it in a backend:

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

from ml_pipes.vlm import PydanticAIBackend, TransformersModel

model_id = "Qwen/Qwen3-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
weights = AutoModelForImageTextToText.from_pretrained(model_id)
backend = PydanticAIBackend(
    TransformersModel(weights, processor, model_name=model_id),
)
```

If the weights and processor are already loaded, pass those objects directly to
`TransformersModel`. For GPU execution, move the weights to your chosen device
before wrapping them. Pass `backend` to a [task](tasks.md).
