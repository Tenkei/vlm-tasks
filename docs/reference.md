# API reference

Public imports: `ml_pipes.vlm` and `ml_pipes.vlm.validation`. Python 3.11+.

Usage guides: [tasks](tasks.md), [model backends](model-backends.md),
and [validation](validation.md). For installation, see the [README](../README.md).

## Operators

Import from `ml_pipes.vlm`. `model` must be a callable `ModelBackend`, normally
`PydanticAIBackend`.

| Constructor | Call input | Return type |
|---|---|---|
| `Classify(model, labels=None, *, context=None)` | `ImageInput` or `tuple[ImageInput, LabelChoices]` | `ClassificationResult` with exactly one allowed label |
| `ClassifyMultiLabel(model, labels, *, context=None)` | `ImageInput` | `ClassificationResult` with zero or more distinct allowed labels |
| `Detect(model, labels=None, *, context=None)` | `ImageInput` or `tuple[ImageInput, LabelChoices]` | `DetectionResult`; empty detections are valid |
| `Ground(model, target=None, *, context=None, geometry="box")` | `ImageInput` or `tuple[ImageInput, str]` | `GroundingResult` |
| `Attributes(model, attributes=None, *, context=None)` | `ImageInput` or `tuple[ImageInput, AttributeChoices]` | `AttributePredictionResult` with exactly one allowed value per attribute |
| `Describe(model, *, context=None)` | `ImageInput` | `DescriptionResult` with one factual natural-language description |
| `Answer(model, question=None, *, context=None)` | `ImageInput` or `tuple[ImageInput, str]` | `AnswerResult` with an answer or an unanswerable outcome |

Ground's `geometry` constructor setting accepts `"box"` (default) or `"point"`.

## Task inputs

`ImageInput`, `LabelChoices`, and `AttributeChoices` below describe accepted
types; they are not exports from `ml_pipes.vlm`.

### Images

`ImageInput` accepts one of:

| Type | Contract |
|---|---|
| `str` or `pathlib.Path` | Path to a local image file |
| `PIL.Image.Image` | Nonempty image; converted to RGB |
| `numpy.ndarray` | Nonempty HWC uint8 RGB array with three channels |
| `ml_pipes.vision.ImagePayload` | Nonempty uint8 array: HWC RGB/BGR with three channels, or HW GRAY |

See [image loading](tasks.md#images) for usage.

### Labels

`labels` uses `LabelChoices`:

| Type | Contract |
|---|---|
| `Sequence[str]` | Nonempty sequence of distinct, nonempty label names; a bare string is invalid |
| `Mapping[str, str]` | Nonempty mapping of nonempty label names to descriptions; descriptions may be empty |

Used by `Classify`, `ClassifyMultiLabel`, and `Detect`.

### Attributes

`attributes` uses `AttributeChoices`: a nonempty `Mapping[str, LabelChoices]`
with nonempty attribute names. Supply it at construction or per call as
`(image, attributes)`; a per-call mapping replaces all constructor choices for
that request. Stored choices and context are unchanged.

`Attributes(backend).val()` can discover choices from its dataset.

### Prompts

| Parameter | Task | Contract |
|---|---|---|
| `target` | `Ground` | Nonempty `str`; required at construction or per call as `(image, target)` |
| `question` | `Answer` | Nonempty `str`; required at construction or per call as `(image, question)` |
| `context` | All tasks | Nonempty `str` or `None` (default); constructor setting |

Per-call targets and questions replace their defaults for that request without
changing stored configuration. Missing or invalid values fail before inference.
Target, question, and attribute overrides preserve context. Describe uses
context to guide the subject, focus, language, and level of detail.

See the [task guide](tasks.md) for configuration examples.

## Result types

Import from `ml_pipes.vlm`. All result types are frozen dataclasses.

| Type | Fields |
|---|---|
| `BoundingBox` | `x1`, `y1`, `x2`, `y2`: `float`, pixel `xyxy` |
| `Point` | `x`, `y`: `float`, pixels |
| `ClassificationResult` | `labels: tuple[str, ...]`; `scores: Mapping[str, float]` or `None` (default) |
| `Detection` | `label: str`; `box: BoundingBox`; `score: float` or `None` (default) |
| `DetectionResult` | `detections: tuple[Detection, ...]`; `image_size: tuple[int, int]` |
| `GroundingResult` | `status`: `"found"`, `"not_found"`, or `"ambiguous"`; `box: BoundingBox` or `None`; `image_size: tuple[int, int]`; `geometry`: `"box"` (default) or `"point"`; `point: Point` or `None` (default) |
| `AttributePredictionResult` | `attributes: Mapping[str, str]` |
| `DescriptionResult` | `description: str`; nonempty, non-whitespace text |
| `AnswerResult` | `status`: `"answered"` or `"unanswerable"`; `answer: str` or `None` in case of `unanswerable` |

Task results use original-image pixels and `image_size=(width, height)`.
Found grounding results contain only the selected geometry; other statuses have
`box=None` and `point=None`. Current tasks leave scores unset.

## Model integration

Import from `ml_pipes.vlm`.

| API | Contract |
|---|---|
| `ModelBackend` | Protocol: `__call__(request: ModelRequest) -> ModelResponse` |
| `PydanticAIBackend(model, *, output_mode="prompted", retries=0)` | `model` must be an initialized `pydantic_ai.models.Model` |
| `TransformersModel(model, processor, *, model_name, max_new_tokens=4096)` | PydanticAI `Model` adapter for loaded decoder-only image/text weights and a compatible processor; prompted output only |
| `backend(request: ModelRequest)` | Returns `ModelResponse` |

| Parameter | Allowed values |
|---|---|
| `output_mode` | `"prompted"`, `"native"`, `"tool"`; requires matching model/provider support |
| `retries` | Nonnegative integer; structured-output correction attempts |
| `model_name` | Required model name for `TransformersModel` |
| `max_new_tokens` | Positive integer for `TransformersModel`; defaults to `4096` |

`TransformersModel` exposes `weights`, `processor`, and `max_new_tokens`.

The communication types are frozen dataclasses:

| Type | Fields |
|---|---|
| `ModelRequest` | `image: ImagePayload`; `instruction: str`; `response_spec: ResponseSpec` |
| `ResponseSpec` | `schema: Mapping[str, object]`; `validate: Callable[[object], Mapping[str, object]]` |
| `ModelResponse` | `data: Mapping[str, object]` |

See [model backends](model-backends.md) for provider setup, output modes,
retries, and custom backend integration. The
[Transformers guide](huggingface-transformers.md) covers loading, device
selection, dependencies, and supported model conventions.

## Task validation

```text
task.val(dataset=None, *, validator=None, **kwargs)
```

Returns metrics and assigns `task.metrics`. Built-in task validations use these
datasets and metric types:

| Task | Validator | Metrics | Default dataset |
|---|---|---|---|
| `Classify` | `ClassificationValidator` | `ClassificationMetrics` | `cifar10` |
| `Detect` | `DetectionValidator` | `DetectionMetrics` | `coco128` |
| `Ground` | `GroundingValidator` | `GroundingMetrics` | `refcocog` |
| `Attributes` | `AttributesValidator` | `AttributesMetrics` | `ovad` |
| `Answer` | `AnswerValidator` | `AnswerMetrics` | `vqav2` |
| `Describe` | `DescribeValidator` | `DescribeMetrics` | `coco_captions2017` |
| `ClassifyMultiLabel` | Explicit custom validator required | Custom | None |

Validation arguments:

| Argument | Default | Contract |
|---|---|---|
| `dataset` | `None` | Loaded `ValidationDataset`; omitted to load the task default |
| `max_samples` | `None` | Positive integer or unlimited; images, grounding expressions, attribute objects, or questions |
| `match_iou` | `0.5` | Detection only; finite value in `(0, 1]` |
| `tokenizer_version` | `"3.4.1"` | Describe only: Stanford PTB `"3.4.1"` or `"4.6.0"`; requires Java |
| `verbose` | `True` | Progress and aggregate metrics on stderr |
| `cache_dir` | `None` | Dataset/tokenizer cache override; otherwise uses Hugging Face cache settings |

`validator` is a class instantiated with `task=self`, `dataset=dataset`, and
forwarded keywords, then called without arguments.

See the [validation guide](validation.md) for dataset formats, sample selection,
scoring, failure handling, and examples.

### Dataset protocol

Import `ValidationDataset` and the sample TypedDicts from `ml_pipes.vlm.validation`:
`ClassificationSample`, `DetectionSample` (with `DetectionObjects`),
`GroundingSample`, `AttributesSample`, `AnswerSample`, and `DescribeSample`.

```python
class ValidationDataset(Protocol):
    @property
    def features(self) -> Mapping[str, Any] | None: ...

    def __iter__(self) -> Iterator[Mapping[str, Any]]: ...
```

See [sample schemas](validation.md#sample-schemas) and
[custom validation](validation.md#custom-loaders-and-validators).

### Dataset loaders

Import from `ml_pipes.vlm.validation`. Return type: Hugging Face `Dataset`.

```text
load_cifar10(*, split="test", classes=None, max_samples=None, cache_dir=None)
load_coco128(*, split="val", max_samples=None, cache_dir=None)
load_refcocog(*, split="val", max_samples=None, cache_dir=None)
load_ovad(annotations=None, images=None, *, split="val", groups=None,
          max_samples=None, cache_dir=None)
load_vqav2(annotations=None, images=None, *, split="val",
           max_samples=None, cache_dir=None)
load_coco_captions2017(annotations=None, images=None, *, split="val",
                       max_samples=None, cache_dir=None)
load_classification(root, *, split="val", class_ids=None, max_samples=None, cache_dir=None)
load_yolo(data, *, split="val", max_samples=None, cache_dir=None)
load_refcoco(annotations, images, *, dataset="refcocog", split="val",
             max_samples=None, cache_dir=None)
```

See [dataset loading](validation.md#default-datasets-and-downloads) and
[local benchmark formats](validation.md#local-benchmark-formats).

### Validator classes

Import from `ml_pipes.vlm.validation`.

```text
BaseValidator(task, dataset, *, max_samples=None, verbose=True, cache_dir=None)
```

`DetectionValidator` adds `match_iou=0.5`;
`DescribeValidator` adds `tokenizer_version="3.4.1"`.
`validator()` returns fresh metrics and stores them as `validator.metrics`.

| Extension hook | Contract |
|---|---|
| `validate_sample(sample)` | Check required columns and target values; returns `None` |
| `predict_sample(sample)` | Return a task result from the dataset row |
| `update_metrics(prediction, sample)` | Score the prediction; `None` represents a failed model call |

### Metrics

Import from `ml_pipes.vlm.validation`.

| Constructor | Task-specific members |
|---|---|
| `ClassificationMetrics(names)` | `top1`, `confusion_matrix`; result key `accuracy_top1` |
| `DetectionMetrics(names, match_iou=0.5)` | `precision`, `recall`, `f1`, `tp`, `fp`, `fn`, `match_iou` |
| `GroundingMetrics(names, *, geometry="box")` | `unique_images`, `status_counts`, `support`; box result keys `accuracy_iou_0.5`, `accuracy_iou_0.75`, `accuracy_iou_0.9`, `mean_iou`; point result key `point_in_box_accuracy` |
| `AttributesMetrics(attributes)` | `accuracy`, `mean_attribute_accuracy`, `macro_f1`, `attribute_decisions`, `unique_images`, `attributes`, `confusion_matrices` |
| `AnswerMetrics()` | `vqa_accuracy` (0–1), `unique_images`, `status_counts`; `names` contains answer-type reporting groups |
| `DescribeMetrics(*, tokenizer_version="3.4.1", cache_dir=None)` | `cider`, `bleu_1`–`bleu_4`, `rouge_l`, `unique_images`, `reference_captions`; `names=()` |

| Shared member | Value |
|---|---|
| `names` | Ordered tuple of class/group names |
| `results_dict` | Aggregate scores plus `samples`, `failures`, `failure_rate` |
| `summary()` | List of per-class, per-attribute, or per-type result dictionaries; Describe has one corpus row |
| `seen`, `failure_rate` | Evaluated sample count and failed-call fraction |
| `failures` | Records with `im_file`, `type`, `message`, and optional `sample_id`/`target` |
| `dataset_counts` | Dataset and evaluated counts where provided by the validator |
| `dataset_metadata` | Dataset/preparation metadata when available |
| `speed` | Mean milliseconds per sample: `load`, `inference`, `metrics` |
| `elapsed_seconds` | Total validation duration in seconds |

See [metric definitions](validation.md#metric-definitions-and-failures) and
[results](validation.md#results).

## Errors

| Exception | Meaning |
|---|---|
| `FileNotFoundError` | Missing image or dataset file |
| `TypeError` | Invalid backend/model type |
| `ValueError` | Invalid task configuration, image, or dataset input |
| `InvalidModelResponseError` (`ValueError`) | Malformed or task-invalid model output after allowed correction attempts |
| `ModelExecutionError` (`RuntimeError`) | Model execution failure; original exception retained as `__cause__` |
