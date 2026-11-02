# ml-pipes-vlm reference

Public module: ml_pipes.vlm. Python 3.11+.
See the [README](../README.md) for setup and the
[design decisions](design.md) for alternatives and future tasks.

## Operators

All operators accept one vision.ImagePayload and return a dedicated dataclass.
Settings are constructor-only. A model is a ModelBackend, normally a
PydanticAIBackend wrapping an initialized PydanticAI Model.

| Constructor | Input | Output | Contract |
|---|---|---|---|
| Classify(model, labels, *, context=None) | `ml_pipes.vision.ImagePayload` | ClassificationResult | Exactly one allowed label |
| ClassifyMultiLabel(model, labels, *, context=None) | `ml_pipes.vision.ImagePayload` | ClassificationResult | Zero or more independently applicable labels |
| Detect(model, labels, *, context=None) | `ml_pipes.vision.ImagePayload` | DetectionResult | All matching instances; empty detections are valid |
| Ground(model, target, *, context=None) | `ml_pipes.vision.ImagePayload` | GroundingResult | Resolve one target; found, not_found or ambiguous |
| PredictAttributes(model, attributes, *, context=None) | `ml_pipes.vision.ImagePayload` | AttributePredictionResult | One allowed value for each requested attribute |

Labels are a sequence of distinct nonempty names or a mapping from names to
descriptions. A bare string is not a vocabulary. Attribute definitions map
nonempty attribute names to such vocabularies. Context defines subject,
property, and any aggregation rule. Ground's target is a nonempty description.

Accepted extensions are documented under
[box configuration](design.md#box-configuration) for Detect and Ground, and
[point grounding](design.md#point-grounding) for Ground's planned geometry="point"
option. Both are pending implementation; the constructors and result types on
this page describe the current API. See the illustrated comparisons for
[Detect versus Ground](design.md#visual-example-detect-versus-ground) and
[geometry and coordinate scale](design.md#visual-example-effect-of-geometry-and-coordinate-scale).

## Final values

| Type | Fields |
|---|---|
| BoundingBox | x1, y1, x2, y2: float; xyxy input-image pixels |
| ClassificationResult | labels: tuple[str, ...]; scores: optional name-to-score mapping |
| Detection | label: str; box: BoundingBox; score: optional float |
| DetectionResult | detections: tuple[Detection, ...]; image_size: (width, height) |
| GroundingResult | status: found/not_found/ambiguous; box: optional BoundingBox; image_size |
| AttributePredictionResult | attributes: mapping[str, str] |

The MVP leaves all scores unset. The classification operators share a result
type but enforce different selection cardinalities. Ground has a box exactly
when its status is found. Detection boxes may overlap. All geometry is finite,
ordered and within the input image; normalized model boxes are converted to
pixel coordinates without clipping or heuristic scale detection.

## Model integration

| Type | Contract |
|---|---|
| ModelBackend | Callable taking ModelRequest and returning ModelResponse |
| ModelRequest | image, instruction, response_spec |
| ResponseSpec | schema: JSON Schema mapping; validate: structured-data validation callable |
| ModelResponse | data: structured-data mapping |
| PydanticAIBackend(model, *, output_mode="prompted", retries=0) | Only concrete backend shipped in the MVP |

The request/response envelope is shared by all tasks; adapters do not need to
recognize task names. Task preparation supplies its instruction, dynamic schema
and validator. The backend validates structured output before task conversion.
The task validates the returned data again at its own boundary, including for
other ModelBackend implementations.

PydanticAIBackend requires an initialized pydantic_ai.models.Model. It does not
accept provider strings, Agents, raw SDK clients or loaded Transformers objects.
Use the corresponding PydanticAI provider to wrap an SDK client. Backends can be
shared between operators; requests do not share conversation history.
For caller-owned Transformers weights and a processor, wrap them in
`TransformersModel` and pass that model to `PydanticAIBackend`.

`output_mode` accepts `prompted`, `native`, or `tool` and defaults to `prompted`.
Native/tool require matching model/API capabilities. `retries` must be a
nonnegative integer and defaults to zero. It counts correction attempts after
the first prediction. Provider SDK transport retries remain controlled by the
configured SDK client.

Supported images are nonempty uint8 HWC RGB/BGR (three channels) and uint8 HW
GRAY. Input data is encoded as lossless PNG; no resizing is performed here.
Normalized floats, CHW arrays, alpha images and batches are outside the MVP.
Execution is synchronous and is not supported inside an active asyncio loop.

### TransformersModel

```python
from ml_pipes.vlm import TransformersModel, PydanticAIBackend

model = TransformersModel(weights, processor, model_name="my-model", max_new_tokens=4096)
backend = PydanticAIBackend(model)
```

`TransformersModel(model, processor, *, model_name, max_new_tokens=4096)` is a
PydanticAI Model adapter for already-loaded, single-device, decoder-only
image/text models such as Qwen3-VL Instruct. It does not load or move weights.
The caller supplies a compatible multimodal processor and chooses the device
and dtype. `max_new_tokens` must be a positive integer.

The adapter translates image messages, schema instructions and correction
history into multimodal chat, runs greedy generation, and returns only the
generated tokens. It serializes generation on shared weights. Only prompted
output is supported; native output, tools, streaming and video are unsupported.

PyTorch and Pillow are needed when generating responses and are imported on
use. They remain optional dependencies, as does Transformers for loading
weights and processors. Importing `ml_pipes.vlm` does not import those runtimes.
See the [Hugging Face Transformers guide](guides/huggingface-transformers.md)
for setup and a runnable example.

## Errors

- ValueError: invalid task configuration or image, raised before model execution.
- InvalidModelResponseError: malformed or task-invalid output after allowed
  correction attempts.
- ModelExecutionError: execution failure; the original exception is its cause.

No model accuracy, exhaustive recall or calibration guarantee follows from
passing schema validation. Live model/provider coverage requires evaluation.
