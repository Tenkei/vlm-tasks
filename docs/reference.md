# API reference

Public imports: `ml_pipes.vlm` and `ml_pipes.vlm.validation`. Python 3.11+.

Usage guides: [tasks](tasks.md), [model backends](model-backends.md),
and [validation](validation.md). For installation, see the [README](../README.md).

## Task inputs

`ImageInput`, `LabelChoices`, and `AttributeChoices` below describe accepted
types; they are not exports from `ml_pipes.vlm`.

### Images

`ImageInput` accepts one of:

| Type | Contract |
|---|---|
| `str` or `pathlib.Path` | Path to a local image file |
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
task.val(validator=None, **kwargs)
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
| `data` | Task default when no loader is supplied | Registered dataset name or local dataset path; mutually exclusive with `dataloader` |
| `dataloader` | `None` | Loader exposing `.dataset`; yields task sample dictionaries |
| `split` | `"val"` | Requested dataset split |
| `max_samples` | `None` | Positive integer or unlimited; images, grounding expressions, attribute objects, or questions |
| `images` | `None` | Required image directory for local grounding/OVAD/VQA v2/COCO captions data without a custom loader |
| `dataset` | `"refcocog"` | Grounding only: `"refcocog"`, `"refcoco"`, `"refcoco+"` |
| `match_iou` | `0.5` | Detection only; finite value in `(0, 1]` |
| `tokenizer_version` | `"3.4.1"` | Describe only: Stanford PTB `"3.4.1"` or `"4.6.0"`; requires Java |
| `verbose` | `True` | Progress and aggregate metrics on stderr |
| `cache_dir` | `None` | Dataset/tokenizer cache; falls back to `VLM_TASKS_DATASETS_DIR`, then `~/.cache/vlm-tasks/datasets` |

`validator` is a class instantiated with `task=self` and the forwarded keywords,
then called without arguments.

See the [validation guide](validation.md) for dataset formats, sample selection,
scoring, failure handling, and examples.

### Dataset preparation

Import from `ml_pipes.vlm.validation`.

```text
prepare_dataset(dataset, *, split="val", max_samples=None, cache_dir=None,
                classes=None, groups=None) -> PreparedDataset
```

`dataset`: registered name. `classes`: optional tuple of CIFAR-10 class names.
`groups`: optional tuple of OVAD group names. Selection precedes `max_samples`.

| API | Fields or contents |
|---|---|
| `PreparedDataset` | `dataset: str`; `split: str`; `data: Path`; `images: Path` or `None`; `names: tuple[str, ...]`; `metadata: dict`; `attributes: dict[str, dict[str, str]]` or `None` (default) |
| `DatasetInfo` | `task: str`; `splits: tuple[str, ...]`; `names: tuple[str, ...] = ()` |
| `DATASETS` | Registered name to `DatasetInfo`: `cifar10`, `coco128`, `refcocog`, `ovad`, `vqav2`, `coco_captions2017` |
| `DEFAULT_DATASETS` | Task name to default registered dataset name |

See [downloads and caching](validation.md#default-datasets-and-downloads).

### Metrics

Import from `ml_pipes.vlm.validation`.

| Constructor | Task-specific members |
|---|---|
| `ClassificationMetrics(names, prediction_names=None)` | `top1`, `confusion_matrix`, `prediction_names`; result key `accuracy_top1` |
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

### Validator classes

Import from `ml_pipes.vlm.validation`.

```text
BaseValidator(task, data=None, *, dataloader=None, split="val",
              max_samples=None, verbose=True, cache_dir=None)
```

Concrete validators accept the same base arguments. `DetectionValidator` adds
`match_iou=0.5`; `GroundingValidator` adds `images=None, dataset="refcocog"`;
`AttributesValidator` and `AnswerValidator` add `images=None`;
`DescribeValidator` adds `images=None, tokenizer_version="3.4.1"`.
`validator()` returns fresh metrics and stores them as `validator.metrics`.

See [custom validation](validation.md#custom-loaders-and-validators) for a
subclassing example.

### Datasets and loaders

Public imports from `ml_pipes.vlm.validation`:

| Dataset or loader | Contract |
|---|---|
| `ClassificationDataset(root, split="val", class_ids=None)` | Indexed class-directory dataset; optional nonempty iterable of class IDs; preserves original IDs and the full `.names` mapping |
| `YOLODataset(data, split="val")` | Indexed detection dataset from a local YAML |
| `RefCOCODataset(annotations, images, *, dataset="refcocog", split="val")` | Indexed MDETR-format grounding expressions |
| `OVADDataset(annotations, images, *, split="val", groups=None, max_samples=None)` | Indexed OVAD object crops; `.attributes` exposes choices |
| `VQAv2Dataset(annotations, images, *, split="val", max_samples=None)` | Official VQA v2 questions/annotations directory and COCO 2014 images; labeled `train`/`val` only; limits count questions |
| `COCOCaptionsDataset(annotations, images, *, split="val", max_samples=None)` | Official captions JSON file and image directory; labeled `train`/`val`; one sample per image with all references; limits follow source image order |
| `DataLoader(dataset, indices=None)` | Iterator over all or selected dataset indices |

See [custom loaders and validators](validation.md#custom-loaders-and-validators)
for an extension example.

## Errors

| Exception | Meaning |
|---|---|
| `FileNotFoundError` | Missing image or dataset file |
| `TypeError` | Invalid backend/model type |
| `ValueError` | Invalid task configuration, image, or dataset input |
| `InvalidModelResponseError` (`ValueError`) | Malformed or task-invalid model output after allowed correction attempts |
| `ModelExecutionError` (`RuntimeError`) | Model execution failure; original exception retained as `__cause__` |
