# ml-pipes-vlm reference

Public module: ml_pipes.vlm. Python 3.11+.
See the [README](../README.md) for setup and the
[design decisions](design.md) for alternatives and future tasks.

## Operators

Operators accept a vision.ImagePayload and return a dedicated dataclass. Ground
also accepts an `(image, target)` tuple; other settings are constructor-only.
A model is a ModelBackend, normally a PydanticAIBackend wrapping an initialized
PydanticAI Model.

| Constructor | Input | Output | Contract |
|---|---|---|---|
| Classify(model, labels, *, context=None) | `ml_pipes.vision.ImagePayload` | ClassificationResult | Exactly one allowed label |
| ClassifyMultiLabel(model, labels, *, context=None) | `ml_pipes.vision.ImagePayload` | ClassificationResult | Zero or more independently applicable labels |
| Detect(model, labels, *, context=None) | `ml_pipes.vision.ImagePayload` | DetectionResult | All matching instances; empty detections are valid |
| Ground(model, target=None, *, context=None, geometry="box") | ImagePayload or tuple[ImagePayload, str] | GroundingResult | Resolve one target as a box or point; found, not_found or ambiguous |
| Attributes(model, attributes, *, context=None) | `ml_pipes.vision.ImagePayload` | AttributePredictionResult | One allowed value for each requested attribute |

Labels are a sequence of distinct nonempty names or a mapping from names to
descriptions. A bare string is not a vocabulary. Attribute definitions map
nonempty attribute names to such vocabularies. Context defines subject,
property, and any aggregation rule. Ground's target is a nonempty description.

Ground's call target overrides its constructor default for that request without
changing the stored target or instruction:

```python
fixed = Ground(backend, target="the leftmost cat")
fixed(image)
fixed((image, "the person on the right"))

dynamic = Ground(backend)
dynamic((image, "the leftmost cat"))
```

An image-only call uses the constructor default. A tuple requires a nonempty
string target, including when a constructor default exists. Missing or invalid
targets raise ValueError before inference.

Ground has one call parameter, so Pipeline passes the tuple as one value. Without
a constructor target, the pipeline requires `tuple[ImagePayload, str]`. With a
default, it accepts either an ImagePayload or that tuple:

```python
from ml_pipes.core import Pipeline

pipeline = Pipeline([fixed])
pipeline.validate(strict=True)
result = pipeline(image)  # use the constructor target
result = pipeline((image, "the person on the right"))
```

`Ground(..., geometry="point")` requests one interior point, preferably near the
visible body center (torso for people). The model returns normalized x/y in
[0, 1]; the public `result.point` contains floating-point input-image pixels.
For a found response, only the selected location is present. For `not_found`
and `ambiguous`, both `box` and `point` are null. Point schemas contain only
`status` and `point`; unsupported geometry values raise ValueError.

The [box_scale extension](design.md#box-configuration) remains planned.
See the illustrated comparisons for
[Detect versus Ground](design.md#visual-example-detect-versus-ground) and
[geometry and coordinate scale](design.md#visual-example-effect-of-geometry-and-coordinate-scale).

## Dataset validation

`Classify.val()`, `Detect.val()`, and `Ground.val()` evaluate the configured task against a
labeled local dataset. Labels, descriptions, context, backend settings and
retries come from the task. Each sample makes an independent task call: one image
for classification/detection, or one image-expression pair for grounding.

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
| `data` | Task default unless `dataloader` is supplied | Registered name (`cifar10`, `coco128`, `refcocog`) or classification root, detection YAML, or MDETR annotation directory |
| `images` | Required for local grounding data | COCO train2014 image directory; grounding only |
| `dataset` | `"refcocog"` | Grounding dataset: `refcocog`, `refcoco`, or `refcoco+` |
| `split` | `"val"` | Exact split to read; there is no fallback to another split |
| `class_map` | `None` | Dataset class name → task label; unsupported for grounding |
| `max_samples` | `None` | Positive maximum number of samples; expressions for grounding |
| `match_iou` | `0.5` | Detection-only matching IoU, in `(0, 1]` |
| `verbose` | `True` | Print progress and aggregate metrics to stderr |
| `validator` | Selected by task | Custom validator class, instantiated with `task=self` and the other arguments |
| `dataloader` | `None` | Prebuilt loader, supplied instead of `data` |
| `cache_dir` | `VLM_TASKS_DATASETS_DIR` or `~/.cache/vlm-tasks/datasets` | Cache for named dataset preparation |

Omitting `data` prepares CIFAR-10 for classification, COCO128 for detection, or
RefCOCOg UMD for grounding. `prepare_dataset(name, *, split="val",
max_samples=None, cache_dir=None, classes=None)` can prepare the same data
separately and returns a `PreparedDataset` with `dataset`, `split`, `data`,
`images`, `names` and `metadata`. `classes` optionally selects CIFAR-10 class
names before the sample limit; validators derive this from task labels or
class-map keys. See [dataset preparation](validation.md#default-datasets-and-downloads)
for source, cache, split and download behavior. Local dataset reader classes
continue to require prepared files.

For classification and detection, every task label must exist by the same name in the dataset;
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

Classification directories, grounding JSON, and metric calculation use base
dependencies. Validation requires neither Ultralytics, PyTorch, torchvision,
Hugging Face datasets nor pycocotools.

#### Grounding: RefCOCO family

Prepare local MDETR annotations and COCO train2014 images using the
[MDETR guide](https://github.com/ashkamath/mdetr/blob/main/.github/refexp.md).
The [annotation archive](https://zenodo.org/records/4729015) contains data for
several tasks. Point `data` to the directory containing the requested
`finetune_{dataset}_{split}.json`, and `images` to the directory containing its
COCO image files:

```python
from ml_pipes.vlm import Ground

task = Ground(backend)
metrics = task.val(
    data="/datasets/mdetr_annotations",
    images="/datasets/coco/train2014",
    dataset="refcocog", split="val", max_samples=100,
)
print(metrics.results_dict)
print(metrics.status_counts)
```

| Dataset | Expected split protocol | Supported splits |
|---|---|---|
| `refcocog` (default) | UMD | train, val, test |
| `refcoco` | UNC | train, val, testA, testB |
| `refcoco+` | UNC | train, val, testA, testB |

MDETR's published converter generates train/val files; supply separately prepared
test annotations when requesting those splits. Missing files raise an error.
Local filenames cannot establish split provenance: `metrics.dataset_metadata`
records paths and the **expected** split protocol, rather than certifying it.

Each JSON `images` record is one expression, read from `caption`; its query `id`
joins the annotation's `image_id`. Each query requires exactly one target with
`category_id` and pixel `bbox: [x, y, width, height]`. Boxes become pixel xyxy
without clipping. Category IDs may be sparse. JSON query order and repeated image
filenames are preserved, so `max_samples` counts expressions, not unique images.
Duplicate IDs, orphan or missing/multiple annotations, unknown categories,
invalid expressions, and invalid boxes raise errors. Decoded image dimensions
must match the JSON dimensions.

`metrics.dataset_counts` contains full-split `expressions` and `unique_images`;
`results_dict` counts only evaluated samples and unique images. Categories are
reporting groups, not a task vocabulary; `class_map` is unsupported. Each dataset
expression overrides any constructor target. Explicit task context is preserved;
use neutral context for standard benchmark comparisons.

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

`GroundingMetrics` exposes `accuracy_iou_0.5`, `accuracy_iou_0.75`,
`accuracy_iou_0.9`, `mean_iou`, and evaluated `unique_images` through
`results_dict`. Accuracy is the fraction of expressions whose found box reaches
the threshold, using ordinary continuous-coordinate IoU and inclusive `>=`.
Every expression contributes to accuracy and mean-IoU denominators. Valid
`not_found` and `ambiguous` responses and execution failures contribute zero IoU.
`status_counts` counts valid responses separately from failures. `summary()`
reports each category's support, accuracies and mean IoU; empty categories score
zero. Full-split counts remain separate from evaluated counts.

With `Ground(..., geometry="point")`, the same validator selects point metrics.
`results_dict` and `summary()` expose `point_in_box_accuracy`: the fraction of
all expressions whose predicted point lies inside the target's pixel box,
including its edges. Missing/ambiguous responses and execution failures count
as misses. Points are predicted directly, rather than derived from boxes.
No IoU metrics are reported in point mode. Box containment is a coarse proxy
for target-mask containment and does not establish that a point is on the
object itself. It cannot be directly compared with box accuracy at IoU >= 0.5.

The RefCOCO family assigns one target per expression. These scores measure box
localization, not rejection or ambiguity quality. They use MDETR's annotation
format, not its generalized-IoU scoring. Report prompt context, model settings,
coordinate configuration, output mode and retries when comparing runs.

All metric objects expose:

- `results_dict`: aggregate metrics, evaluated sample count, failure count and
  failure rate; detection also includes the matching threshold and TP/FP/FN.
- `failures`: records containing `im_file`, exception `type`, and `message`.
- `seen`: number of evaluated samples, including execution failures.
- `speed`: average `load`, `inference` and `metrics` milliseconds per evaluated
  sample. Load time includes iteration and decoding.
- `elapsed_seconds`: total elapsed time, including setup.

`ModelExecutionError` and `InvalidModelResponseError` are recorded and the run
continues. Classification failures count as incorrect. Detection failures
contribute no predictions, leaving their targets as false negatives. Failures
on negative images remain visible in the failure count and rate. Other errors
stop the run. No extra retry policy is added by validation.
Grounding failures additionally include `sample_id` and `target` when supplied
by the loader; failed calls count toward evaluated unique images.

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

Validators bind their task at construction: `ClassificationValidator(task, ...)`.
Call `validator()` to run evaluation; repeated calls evaluate the same task with
fresh metrics. `.val()` constructs the validator with `task=self` and invokes it
without arguments. Custom constructors must accept `task` and pass it to the
base constructor when subclassing a built-in validator.

`BaseValidator` owns the loop. Extension methods are `build_dataset()`,
`get_dataloader(dataset)`, `init_metrics()`, `predict_sample(sample)`, `update_metrics(prediction, sample)`
and `finalize_metrics()`. Call the inherited `finalize_metrics()` when overriding
it to retain timings. Task-specific validators consume structured task results
directly. `predict_sample()` normally calls the task with the image; grounding
passes `(sample["img"], sample["target"])` as one argument.

The supplied `DataLoader` iterates an indexed dataset, one image at a time.
`ClassificationDataLoader(dataset, class_ids=...)` selects classification samples
from `.samples` metadata before decoding. The base validator iterates its loader
directly; prebuilt loaders must yield the selected samples themselves. In
particular, `class_map` aligns labels but does not filter an injected loader.
A prebuilt loader must expose `.dataset`; datasets expose a `.names` mapping
from integer class IDs to strings. Samples are dictionaries with `im_file`,
`img` (an `ImagePayload`) and `cls` (an integer class-ID array). Classification
has one target ID; detection additionally has `bboxes` as an Nx4 NumPy array in
original-image pixel xyxy. `ClassificationDataset` and `YOLODataset` also expose
metadata through `.samples`, used by the default classification loader to
select indices before decoding. For a different dataset layout, override
`get_dataloader()` as needed or pass your own loader through `dataloader=`.
Grounding loaders expose `.dataset.names` for category summaries, and yield one
`cls` ID, one `(1, 4)` pixel-xyxy `bboxes` array, `target`, and optionally
`sample_id`, alongside `img` and `im_file`. `RefCOCODataset` also exposes `.samples`,
`.counts`, and `.metadata`. Supplying `dataloader=` needs no `images` argument.

## Final values

| Type | Fields |
|---|---|
| BoundingBox | x1, y1, x2, y2: float; xyxy input-image pixels |
| Point | x, y: float; input-image pixels |
| ClassificationResult | labels: tuple[str, ...]; scores: optional name-to-score mapping |
| Detection | label: str; box: BoundingBox; score: optional float |
| DetectionResult | detections: tuple[Detection, ...]; image_size: (width, height) |
| GroundingResult | status: found/not_found/ambiguous; box: optional BoundingBox; image_size; geometry: box/point (default box); point: optional Point (default None) |
| AttributePredictionResult | attributes: mapping[str, str] |

The MVP leaves all scores unset. The classification operators share a result
type but enforce different selection cardinalities. Ground has exactly the
selected location when its status is found. Detection boxes may overlap. All geometry is finite,
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
