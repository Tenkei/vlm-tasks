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

## Dataset validation

`Classify.val()` and `Detect.val()` evaluate the configured task against a
labeled local dataset. Labels, descriptions, context, backend settings and
retries come from the task. Each selected image makes an independent task call.

```python
from ml_pipes.vlm import Detect

task = Detect(backend, labels={"pedestrian": "A visible person"})
metrics = task.val(
    data="dataset.yaml",
    class_map={"person": "pedestrian"},
    max_samples=100,
)
print(metrics.results_dict)
print(metrics.summary())
```

The method returns a metric object and stores it on `task.metrics`. Each run
has fresh counts and failures. The other task operators require an explicit
custom validator.

### Settings

| Argument | Default | Meaning |
|---|---|---|
| `data` | Required unless `dataloader` is supplied | Classification dataset root or detection YAML path |
| `split` | `"val"` | Exact split to read; there is no fallback to another split |
| `class_map` | `None` | Dataset class name → task label; also selects the evaluated subset |
| `max_samples` | `None` | Positive maximum number of evaluated images in deterministic order |
| `match_iou` | `0.5` | Detection-only matching IoU, in `(0, 1]` |
| `verbose` | `True` | Print progress and aggregate metrics to stderr |
| `validator` | Selected by task | Custom validator class, instantiated with the other arguments |
| `dataloader` | `None` | Prebuilt loader, supplied instead of `data` |

By default every task label must exist by the same name in the dataset;
dataset classes beyond the task vocabulary are excluded from scoring. An
explicit `class_map` defines aliases and the subset being evaluated. All keys
must exist in the dataset and all values must be configured task labels.
Several dataset classes can map to one task label.

Classification evaluates images whose ground-truth class is selected. A task
prediction outside the evaluated subset still counts as incorrect. Detection
retains every image, filters ground-truth annotations and predictions to the
evaluated classes, and counts unmatched selected predictions as false positives.

### Dataset formats

Classification uses directories such as:

```text
dataset/
  val/
    cat/image1.jpg
    dog/image2.jpg
```

Pass `data="dataset"`. Class names come from immediate subdirectory names;
images can be nested within those directories. Files are evaluated in sorted
class and path order. Supported extensions are BMP, JPEG, PNG, TIFF and WebP.

Detection uses a local YOLO dataset YAML, for example:

```yaml
path: ./dataset
val: images/val
names:
  0: person
  1: dog
```

`path` is relative to the YAML directory and defaults to that directory.
The split can name an image directory, a TXT image list, a single image, or a
list of these sources. Split paths are relative to `path`; paths inside a TXT
list are relative to that list's directory. Images are deduplicated by resolved
path and sorted. Only the requested split and class names are required.

The final `images` directory component is replaced with `labels` and the image
suffix with `.txt`; without an `images` component, labels are beside the image.
Each annotation row contains `class_id x_center y_center width height` with
normalized coordinates. Boxes must have positive area and stay within the
image, allowing `1e-6` normalized rounding error at image edges. Annotation
coordinates are preserved. Class IDs must index `names`, supplied as a list or consecutive integer
mapping starting at zero. Polygon, pose and other annotation formats are
unsupported.

Missing and empty detection label files represent negative images.
`metrics.dataset_counts` reports `missing_labels`, `empty_labels` and
`labeled_images` across the full dataset before applying the sample limit.
Invalid annotations and unreadable images stop validation with an error.
Data is read locally; dataset downloads, caches and run directories are not
created. Images retain their original dimensions. Ground-truth boxes are
converted into original-image pixel xyxy to match task results.

Detection YAML requires the optional validation extra:

```bash
python -m pip install 'ml-pipes-vlm[validation]'
```

Classification directories and metric calculation use base dependencies.
Neither validation path requires Ultralytics, PyTorch or torchvision.

### Metrics and failures

`ClassificationMetrics` exposes `top1`, `confusion_matrix`, and per-class
precision, recall, F1 and support through `summary()`. Matrix rows follow
`names` (evaluated labels); columns follow `prediction_names` (all task labels),
with a final column for failed executions.

`DetectionMetrics` exposes pooled `precision`, `recall`, `f1` and per-class
`tp`, `fp`, `fn` arrays in `names` order. `summary()` returns per-class counts,
support and ratios. Matching is class-aware and one-to-one at `match_iou`;
candidate pairs are considered in descending IoU order, with ties ordered by
target index then prediction index. AP and mAP require confidence-ranked
predictions, which the current tasks do not provide. Undefined ratios are zero.

Both metric objects expose:

- `results_dict`: aggregate metrics, evaluated sample count, failure count and
  failure rate; detection also includes the matching threshold and TP/FP/FN.
- `failures`: records containing `im_file`, exception `type`, and `message`.
- `seen`: number of evaluated images, including execution failures.
- `speed`: average `load`, `inference` and `metrics` milliseconds per evaluated
  image. Load time includes iteration and decoding.
- `elapsed_seconds`: total elapsed time, including setup.

`ModelExecutionError` and `InvalidModelResponseError` are recorded and the run
continues. Classification failures count as incorrect. Detection failures
contribute no predictions, leaving their targets as false negatives. Failures
on negative images remain visible in the failure count and rate. Other errors
stop the run. No extra retry policy is added by validation.

Results measure dataset performance and execution reliability. They do not
assign an automatic trust threshold or pass/fail decision.

### Custom validators and data sources

Import components from `ml_pipes.vlm.validation`:

```python
from ml_pipes.vlm import Classify
from ml_pipes.vlm.validation import ClassificationValidator

class CustomValidator(ClassificationValidator):
    def build_dataset(self):
        return my_dataset

task = Classify(backend, labels=["cat", "dog"])
metrics = task.val(validator=CustomValidator, data="my-source")
```

`BaseValidator` owns the loop. Extension methods are `build_dataset()`,
`get_dataloader(dataset)`, `init_metrics()`, `update_metrics(prediction, sample)`
and `finalize_metrics()`. Call the inherited `finalize_metrics()` when overriding
it to retain timings. Task-specific validators consume structured task results
directly.

The supplied `DataLoader` iterates an indexed dataset, one image at a time.
A prebuilt loader must expose `.dataset`; datasets expose a `.names` mapping
from integer class IDs to strings. Samples are dictionaries with `im_file`,
`img` (an `ImagePayload`) and `cls` (an integer class-ID array). Classification
has one target ID; detection additionally has `bboxes` as an Nx4 NumPy array in
original-image pixel xyxy. `ClassificationDataset` and `YOLODataset` also expose
metadata through `.samples`, used by the default classification loader to
select indices before decoding. For a different dataset layout, override
`get_dataloader()` as needed or pass your own loader through `dataloader=`.

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
