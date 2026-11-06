# Use Hugging Face Transformers with VLM tasks

Run the five existing VLM tasks with local Hugging Face weights, using
[run_transformers.py](../examples/run_transformers.py). The default checkpoint is
`Qwen/Qwen3-VL-2B-Instruct`; `--model` can select another compatible checkpoint
or a local weights directory.

## How it connects

```text
ImagePayload → task → PydanticAIBackend → TransformersModel → loaded weights
             ← result ← validation/parsing ← generated text ←
```

[TransformersModel](reference.md#model-integration) is a reusable adapter
shipped with `ml_pipes.vlm`, using PydanticAI's public FunctionModel hook.
It accepts already-loaded weights
and a processor. It translates text, image bytes and correction history into
the processor's chat format, applies the chat template, runs greedy generation
and returns only generated tokens. PydanticAI's schema instructions reach the
model through the system message. PydanticAI and the task validator parse and
validate the result, and the task converts coordinates to input-image pixels.

The adapter is shared by all tasks. Loading, device selection and Hugging Face
authentication remain application concerns. The example uses the existing
Classify, ClassifyMultiLabel, Detect, Ground and Attributes operators.

## Setup

Use Python 3.11+ and install a PyTorch build suitable for your machine. From the
repository root, for example:

```bash
python -m pip install -e .
python -m pip install 'torch>=2.5' torchvision 'transformers>=4.57,<5' 'Pillow>=10'
```

The adapter targets the package's pinned `pydantic-ai-slim==2.61.0` API.
Transformers 4.57+ includes Qwen3-VL support. Qwen's processor also requires
torchvision for its video processor, even when the task supplies only an image.
Install a torchvision build compatible with your PyTorch build.
The adapter has focused checks with fake weights and a processor through
the real PydanticAI execution path.
It was also executed on the cats image with Qwen3-VL-2B-Instruct, using
PyTorch 2.14.1, torchvision 0.29.1 and Transformers 4.57.6 on Apple MPS in
float16. All five tasks reached generation: four responses validated, and
multi-label classification was rejected for duplicate labels. Detection missed
objects and grounding selected the wrong remote, so this verifies execution
and error handling rather than task accuracy. This live run also confirmed
that every role's content must use multimodal content lists, including system
instructions, assistant history and correction messages.
The dependency range above describes the intended runtime, not a tested matrix
of every PyTorch/Transformers/device combination.

The first real run downloads the model and processor through Transformers.
For gated/private checkpoints, authenticate with the Hugging Face CLI or set
`HF_TOKEN` in your environment. Tokens are not CLI arguments or report fields.
Inference runs in the local Python process; it does not call a hosted model or
require a model server. Memory requirements depend on checkpoint, dtype and
image token count; there is no quantization, sharding or CPU offload here.

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

The first run downloads the model weights. Pass `backend` to any
[task](tasks.md); move weights to your chosen device before wrapping them for
GPU execution.

## Run

Validate the image and task configuration first, without importing PyTorch or
Transformers or downloading/loading weights:

```bash
python -m examples.run_transformers \
  --input experiments/inputs/coco_000000039769.jpg \
  --dry-run
```

Run all five tasks:

```bash
python -m examples.run_transformers \
  --input experiments/inputs/coco_000000039769.jpg \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --device auto \
  --output .example_assets/qwen3_vl_local_results.json
```

Run these commands from the repository root. Replace the image path with any
local image. The cats image is included in repository checkouts under
`experiments/inputs/`; source distributions exclude that directory.
`auto` selects CUDA if available, then Apple MPS, then CPU. It loads the entire
model onto that single device and does not retry on a different device after a
failure. For CPU execution, an explicit float32 run is available:

```bash
python -m examples.run_transformers \
  --input /path/to/image.jpg --device cpu --dtype float32 \
  --tasks classify ground --max-tokens 1024
```

Use `--device mps` or `--device cuda:0` explicitly as appropriate. `--dtype`
accepts auto, float32, float16 or bfloat16; choose a dtype supported by the
device. `auto` retains the checkpoint's dtype policy. CPU execution can be slow.

Use `--local-files-only` after downloading to restrict model/processor loading
to local files. You can also pass a directory with `--model /path/to/checkpoint`.
`--revision` applies to both weights and processor; use a commit ID when fixing
an experiment's model version.

## Task validation

Use `examples.run_validation` with `--backend transformers` to evaluate local
classification, detection or grounding datasets with this adapter. For example:

```bash
python -m examples.run_validation \
  --backend transformers --task ground \
  --data /path/to/mdetr_annotations --images /path/to/coco/train2014 \
  --dataset refcocog --split val --max-samples 10 \
  --model Qwen/Qwen3-VL-2B-Instruct --device mps --dtype float16 \
  --output /tmp/grounding10.json
```

The runner accepts the same device, dtype, revision, cached-file and generation
options. Grounding defaults to empty task settings and uses each dataset
expression as its target. See [task validation](validation.md)
for annotation preparation, metrics and report details. Begin with 10 expressions
to check timing, then evaluate 100 for a preliminary result.

## Task configuration and results

Use the same [image_tasks.json](../examples/image_tasks.json) as the Bedrock example, or pass
your own `--config /path/to/tasks.json`. `--tasks` selects one or more tasks;
all five run by default. Configurations are passed to the current operator
constructors. Ground accepts `geometry="point"` for direct point output.
The box_scale extension remains a design proposal.

The model is loaded once and reused across tasks. Each task has a fresh request
history. A correction retry preserves that task's image, previous response and
validation feedback. Generation uses `do_sample=False`, `--max-tokens` (default
4096) and `--retries` (default 0). This adapter supports prompted structured
output only; it does not constrain model decoding to valid JSON. There is no
native JSON mode, tool calling, streaming or batch inference.

JSON is printed and optionally saved with `--output`. It includes model ID,
requested revision, actual device/dtype, loading time separately from task
time, settings, task configurations, predictions and errors with their cause
chains. Tasks continue after an individual failure. Exit codes are 0 for all
tasks succeeding, 1 for task failures and 2 for setup failures.

The adapter targets decoder-only image/text models with a multimodal
`processor.apply_chat_template(..., tokenize=True, return_dict=True,
return_tensors="pt")` that returns a BatchFeature, and `generate()` returning
prompt plus generated token IDs. Qwen3-VL Instruct follows this convention;
other processors/checkpoints may need their own runtime bridge. No custom
checkpoint Python code is enabled with trust_remote_code.

Passing validation establishes output structure, not visual accuracy. For
example, an empty detection collection is valid and a correctly shaped box
can still point to the wrong object.

## Reuse already-loaded objects

```python
from ml_pipes.core import Pipeline
from ml_pipes.vlm import Ground, PydanticAIBackend, TransformersModel

# weights and processor were loaded and placed on one device by your application.
model = TransformersModel(
    weights, processor, model_name="Qwen/Qwen3-VL-2B-Instruct",
    max_new_tokens=1024,
)
backend = PydanticAIBackend(model, retries=1)
pipeline = Pipeline([Ground(backend)])
pipeline.validate(strict=True)
result = pipeline((image_payload, "the leftmost cat"))
```

## Focused verification

With the VLM dependencies, Pillow and pytest installed, run:

```bash
python -m pytest tests/test_transformers_model.py -q
```

Tests use fake weights and a processor, without requiring PyTorch or model
downloads. They cover all five tasks, schema delivery, RGB image conversion,
prompt-token removal, isolated histories, correction feedback, invalid-output
rejection, generation error propagation, unsupported output modes and the CLI.
