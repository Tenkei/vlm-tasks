# ml-pipes-vlm reference

Public module: ml_pipes.vlm. Python 3.11+.
See the [README](../README.md) for setup and the
[design decisions](design.md) for alternatives and future tasks.

## Operators

All operators accept one vision.ImagePayload and return a dedicated dataclass.
Settings are constructor-only. A model is a ModelBackend, normally a
PydanticAIBackend wrapping an initialized PydanticAI Model.

| Constructor | Output | Contract |
|---|---|---|
| Classify(model, labels, *, context=None) | ClassificationResult | Exactly one allowed label |
| ClassifyMultiLabel(model, labels, *, context=None) | ClassificationResult | Zero or more independently applicable labels |
| Detect(model, labels, *, context=None) | DetectionResult | All matching instances; empty detections are valid |
| Ground(model, target, *, context=None) | GroundingResult | Resolve one target; found, not_found or ambiguous |
| PredictAttributes(model, attributes, *, context=None) | AttributePredictionResult | One allowed value for each requested attribute |

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

output_mode accepts prompted, native, or tool. Native/tool require matching
model/API capabilities. retries counts correction attempts after the first
prediction, not provider SDK transport retries.

Supported images are nonempty uint8 HWC RGB/BGR (three channels) and uint8 HW
GRAY. Input data is encoded as lossless PNG; no resizing is performed here.
Normalized floats, CHW arrays, alpha images and batches are outside the MVP.
Execution is synchronous and is not supported inside an active asyncio loop.

## Errors

- ValueError: invalid task configuration or image.
- InvalidModelResponseError: malformed or task-invalid output after allowed
  correction attempts.
- ModelExecutionError: execution failure; the original exception is its cause.

No model accuracy, exhaustive recall or calibration guarantee follows from
passing schema validation. Live model/provider coverage requires evaluation.
