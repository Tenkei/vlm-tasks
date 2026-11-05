# Validation

Validation compares a configured task's predictions with labeled examples. It
measures task quality, failures, and execution time without training the model.
This page records the design decisions, summarizes the available validations,
and explains how to configure a run.

## Architecture decision record

### Context

A useful evaluation must exercise the same task semantics as application use.
Model accuracy, structured-output reliability, and execution failures are
different concerns; a successful response alone does not establish correctness.
Dataset formats should not determine task behavior or require separate scoring
logic for every data source.

### Decisions

| Decision | Alternatives considered | Rationale and tradeoffs |
|---|---|---|
| Evaluate the configured task through its public results | Call the model directly or score raw responses | Measures instructions, parsing, and task behavior together; results describe the complete configuration rather than the model alone |
| Separate data loading, annotation adaptation, and evaluation | Put dataset-specific parsing and scoring in one runner | Reuses task metrics across datasets; requires consistent labels, geometry, and sample meaning |
| Provide task defaults with replaceable validators | Require a benchmark framework or a custom runner for every evaluation | Makes reference runs easy while allowing extensions; defaults cannot represent every domain |
| Target Imagenette for the classification default | Keep CIFAR-10 as the long-term default | Higher-resolution natural images; larger downloads and an unresolved resolution choice |
| Prepare named datasets separately from inference | Bundle data or make downloading inseparable from execution | Allows caching and preparation before model use; subset preparation may still require large source downloads |
| Include failed predictions in scoring and report them separately | Drop failures or stop on every model error | Avoids inflating scores by evaluating only successful calls; aggregate quality includes execution reliability |
| Use task-specific metrics and explicit evaluation scopes | Report one universal score | Preserves the distinction between classification, detection, localization, and attributes; scores across tasks are not interchangeable |

A validator coordinates three responsibilities:

| Responsibility | Purpose |
|---|---|
| Data loading | Supply the selected images or crops and their annotations |
| Annotation adaptation | Align source labels and geometry with the task's evaluation contract |
| Evaluation | Compare public task results with targets and accumulate metrics |

**Consequences:** keep the task configuration fixed across samples, apart from
dataset-supplied inputs such as grounding expressions. In particular, missing
attribute annotations must not change the choices presented to the model.
Separate correction retries from transport retries, and record both when
comparing runs. Small deterministic subsets are useful for execution checks;
claims about quality require representative data and a stated protocol.

## Implementation notes

### Available validations

Sizes and counts below describe the source dataset or named split before class
selection or sample limits. Download sizes are compressed archives in decimal
MB/GB, excluding model weights and expanded cache files.

| Task | Default dataset | Dataset scope | Splits | Metrics | Download behavior | Archive size |
|---|---|---|---|---|---|---|
| `Classify` | **CIFAR-10** (`cifar10`) | 60,000 images at 32×32; 10 classes; 50,000 train / 10,000 test | train, val, test; **val aliases test** | Top-1 accuracy; per-class precision/recall/F1; confusion matrix | Full archive; prepare only the selected subset | ≈170 MB |
| `Detect` | **COCO128** (`coco128`) | 128 images; 941 boxes; 71 represented classes in an 80-class vocabulary | train, val; **same images in both** | Precision/recall/F1 and TP/FP/FN at a selected IoU threshold | Full archive; prepare selected images | 6.98 MB |
| `Ground`, box | **RefCOCOg UMD** (`refcocog`) | **val:** 1,300 unique images; 4,896 expressions; 80-category reporting vocabulary | train, val, test | Accuracy at IoU ≥0.5 / 0.75 / 0.9; mean IoU | Stream shared annotations; download images referenced by selected expressions | 1.08 GB shared archive; val JSON 2.27 MB; images additional |
| `Ground`, point | **RefCOCOg UMD** (`refcocog`) | Same expressions and target boxes as box mode | train, val, test | Point-in-box accuracy | Shares box-mode data and cache | Same as box mode |
| `Attributes` | **OVAD** (`ovad`) | 14,300 source objects; 14,072 eligible objects in 2,000 images; 22 categorical groups | test, val; **val aliases the same benchmark**; no train | Categorical accuracy; mean attribute accuracy; macro F1; per-group reports | Download annotations and images referenced by selected objects | 1.13 MB annotations; images additional |

`ClassifyMultiLabel` requires an explicit custom validator. Local
RefCOCO/RefCOCO+ data can reuse grounding validation; see
[local data](#using-your-own-dataset).

**Interpretation limits:** CIFAR-10 contains small images, and COCO128 is a smoke
dataset drawn from COCO training data. Neither an ordered subset nor a smoke
dataset establishes general model performance. RefCOCO-family targets do not
measure annotated ambiguity, rejection quality, or multiple-target resolution.
The OVAD protocol here adapts independent attribute annotations to categorical
decisions; it is distinct from official OVAD attribute mAP.

Dataset sources: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html),
[COCO128](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip),
[MDETR RefCOCO annotations](https://zenodo.org/records/4729015), and
[OVAD](https://lmb.informatik.uni-freiburg.de/resources/datasets/ovad.zip).
The classification design target is [Imagenette](https://github.com/fastai/imagenette);
the available automatic default in the table remains CIFAR-10.

### Metric definitions and failures

| Task | Scoring rule |
|---|---|
| Classification | Top-1 accuracy over selected examples, with per-class precision, recall, F1, and a confusion matrix |
| Detection | Match same-class boxes one-to-one in descending IoU order at an inclusive `match_iou` threshold; pool TP/FP/FN across classes for aggregate precision/recall/F1 |
| Grounding, box | Continuous-coordinate IoU in original-image pixels; report mean IoU and accuracy at inclusive thresholds of 0.5, 0.75, and 0.9 |
| Grounding, point | A hit when the point lies inside the target box, including its edges; a coarse proxy for object containment, distinct from box-IoU accuracy |
| Attributes | Score only known targets; report accuracy across decisions, mean accuracy across supported groups, and macro F1 averaged within then across supported groups |

Detection does not report AP/mAP because predictions have no confidence ranking.
Attribute groups without target support do not contribute to group averages;
values with neither target nor scored prediction support do not contribute to
macro F1. Unknown targets are unscored, rather than an invented “none” category.

`ModelExecutionError` and `InvalidModelResponseError` are recorded and evaluation
continues. Failed classification calls are incorrect; failed detection calls
produce no boxes; failed grounding calls score zero; failed attribute calls are
incorrect for every known target. Failures remain in sample counts and relevant
score denominators. Data, setup, and programming errors stop the run.

Valid grounding absence and ambiguity responses also score zero against a
positive target, but remain separate from execution failures in
`status_counts`. Each referring expression is evaluated independently, even
when several expressions use the same image.

## Run validation

### Setup

Install validation dependencies from the repository root:

```bash
python -m pip install -e '.[validation]'
```

This includes PyYAML for detection dataset configuration. Initialize a
[model backend](model-backends.md) using, for example, the
[Bedrock guide](amazon-bedrock.md) or
[Transformers guide](huggingface-transformers.md). The examples below assume
`backend` is ready.

### Calling validation

Call `.val()` on a configured task:

```python
from ml_pipes.vlm import Ground

task = Ground(backend)
metrics = task.val(split="val", max_samples=100)

print(metrics.results_dict)
print(metrics.summary())
print(metrics.failures)
```

Omitting `data` selects the task's default dataset and prepares it automatically.
Each call starts fresh metrics, returns them, and stores them as `task.metrics`.
Start with `max_samples=10` to check execution and timing. Use
`max_samples=None`, the default, to evaluate the complete selected split.
Limits count images for classification/detection, expressions for grounding,
and eligible objects for attributes. Selection is deterministic, not a random
sample.

Configure labels, descriptions, context, and geometry on the task. Configure
model settings, output mode, and correction retries on the
[backend](model-backends.md#output-modes-and-retries). Validation adds no
correction retry policy.

| Validation setting | Purpose |
|---|---|
| `data` | Registered name or local path; omitted for the task default |
| `split` | Dataset split; defaults to `"val"` |
| `max_samples` | Positive sample limit, or `None` for the full selection |
| `class_map` | Dataset class names to task labels; classification/detection only |
| `match_iou` | Detection matching threshold; defaults to `0.5` |
| `cache_dir` | Override the named-dataset cache |
| `images`, `dataset` | Local annotation configuration; see [local data](#using-your-own-dataset) |
| `verbose=False` | Disable progress and aggregate output on stderr |

For the complete argument contracts, see the
[API reference](reference.md#task-validation).

### Class selection and aliases

Use `class_map` to select dataset classes and give them task-facing names:

```python
from ml_pipes.vlm import Classify

task = Classify(
    backend,
    labels={"feline": "A cat", "canine": "A dog"},
)
metrics = task.val(
    data="cifar10",
    class_map={"cat": "feline", "dog": "canine"},
    max_samples=100,
)
```

Without a mapping, task labels must match dataset names. With a mapping, keys
are dataset classes and values are configured task labels; several classes can
map to one label.

Classification selects examples from the mapped classes before applying the
sample limit. Detection retains negative images and scores only selected
classes; unmatched selected predictions are false positives. A classification
prediction outside the evaluated subset still counts as incorrect. Grounding
and attributes do not accept `class_map`.

### Detection matching

```python
from ml_pipes.vlm import Detect

task = Detect(backend, labels=["person", "car"])
metrics = task.val(data="coco128", match_iou=0.75, max_samples=100)
```

A higher `match_iou` requires closer box alignment. Keep the threshold and class
selection fixed when comparing models.

### Grounding geometry

Choose geometry on the task; each dataset expression supplies its target:

```python
task = Ground(backend, geometry="point", context="Locate the described object.")
metrics = task.val(data="refcocog", split="val", max_samples=100)
print(metrics.results_dict["point_in_box_accuracy"])
```

Use `geometry="box"` for IoU-based evaluation. Constructor context remains in
effect, while dataset expressions override a constructor target. Keep context
neutral for benchmark comparisons. Categories are reporting groups, rather
than a grounding label vocabulary.

### Attributes: OVAD object crops

Omit constructor attributes to use the dataset's complete categorical schema:

```python
from ml_pipes.vlm import Attributes

task = Attributes(backend)
metrics = task.val(data="ovad", max_samples=100)

print(metrics.results_dict)
print(metrics.summary())
print(metrics.confusion_matrices["material"])
```

Each eligible object produces one crop and one call. All selected attribute
fields are requested on every crop; only known targets contribute to scoring.
The task's stored configuration is unchanged.

To evaluate fewer groups, prepare their choices and configure the task:

```python
from ml_pipes.vlm.validation import prepare_dataset

prepared = prepare_dataset("ovad", groups=("material",), max_samples=100)
task = Attributes(backend, attributes=prepared.attributes)
metrics = task.val(
    data=prepared.data,
    images=prepared.images,
    max_samples=100,
)
```

Keep the complete dataset vocabulary within each selected group; descriptions
and choice order may change. The categorical adaptation excludes overlapping
color groups and objects without usable targets or sufficient annotated area.
Selecting fewer groups can change which objects are eligible, so report the
selection alongside the score.

### Default datasets and downloads

Select a registered name explicitly, or omit `data` for the default. To prepare
data before model execution:

```python
from ml_pipes.vlm.validation import prepare_dataset

prepared = prepare_dataset(
    "refcocog",
    split="val",
    max_samples=100,
    cache_dir="/datasets/vlm-cache",
)
print(prepared.data, prepared.images)
print(prepared.metadata)
```

Pass the returned paths as `data` and, where required, `images` to use prepared
data without automatic downloading.

Cache precedence is `cache_dir`, then `VLM_TASKS_DATASETS_DIR`, then
`~/.cache/vlm-tasks/datasets`. Matching preparations reuse cached files.
A small sample limit does not guarantee a small download: CIFAR-10 and COCO128
require their full archives, and RefCOCOg streams a shared annotation archive
before fetching selected images. RefCOCOg uses COCO train2014 images; OVAD uses
COCO val2017 images.

### Using your own dataset

Follow the corresponding default dataset's format to reuse its reader and
metrics. Local paths bypass automatic downloading.

| Task | Format to follow | Local configuration |
|---|---|---|
| Classification | Class directories, organized as `<root>/<split>/<class>/image.jpg` | `data="/datasets/animals"`, `split="val"` |
| Detection | COCO128-style YOLO detection YAML, images, and normalized box labels | `data="/datasets/objects/data.yaml"` |
| Grounding | MDETR RefCOCO annotations named `finetune_{dataset}_{split}.json`, plus images | `data="/datasets/annotations"`, `images="/datasets/images"`, `dataset="refcocog"` |
| Attributes | OVAD annotation JSON and its referenced images | `data="/datasets/ovad2000.json"`, `images="/datasets/val2017"` |

Local grounding also accepts `dataset="refcoco"` or `"refcoco+"` with the
corresponding split files; see the
[MDETR preparation guide](https://github.com/ashkamath/mdetr/blob/main/.github/refexp.md).
Use `Path("cifar10")` or `"./cifar10"` when a local directory's name matches a
registered key. If your annotations or evaluation semantics differ, extend an
existing validator rather than forcing them into an incompatible format.

### Custom loaders and validators

Extend a concrete validator when the task and scoring rules match; use
`BaseValidator` for a different evaluation contract. For example, given a
prepared classification dataset `my_dataset`:

```python
from ml_pipes.vlm import Classify
from ml_pipes.vlm.validation import ClassificationValidator

class CustomValidator(ClassificationValidator):
    def build_dataset(self):
        return my_dataset

task = Classify(backend, labels=["cat", "dog"])
metrics = task.val(validator=CustomValidator, data="my-source")
```

The supplied dataset must follow the corresponding dataset interface and
already select the intended samples. `class_map` aligns labels but does not
filter a custom dataset or injected loader. For an existing loader, pass
`dataloader=my_loader` without `data`; it must expose `.dataset`, with
`.names` or `.attributes` as appropriate. Reuse the built-in dataset sample
format; dataset class docstrings describe its fields.

| Hook | Role |
|---|---|
| `build_dataset()` | Return the dataset |
| `prepare_named_dataset(task)` | Prepare a registered source, or return `None` for local data |
| `get_dataloader(dataset)` | Return a loader; defaults to `DataLoader(dataset)` |
| `init_metrics()` | Assign `self.metrics` |
| `predict_sample(sample)` | Return a public task result |
| `update_metrics(prediction, sample)` | Accumulate a result, or `None` for a failed call |
| `finalize_metrics()` | Finalize the report and timings |

Validators bind their task at construction and run when called without
arguments. Custom constructors must accept `task` and forward it to the base
constructor. When overriding `finalize_metrics()`, call the inherited method
to retain timings. See the [validator reference](reference.md#validator-classes)
for constructor contracts.

### Results and reproducibility

| Result | Use |
|---|---|
| `metrics.results_dict` | Aggregate scores, evaluated samples, failures, and failure rate |
| `metrics.summary()` | Per-class, per-category, or per-attribute scores and support |
| `metrics.failures` | Failed calls with image paths, error messages, and sample identifiers where available |
| `metrics.dataset_counts`, `metrics.dataset_metadata` | Dataset selection and preparation context when available |
| `metrics.speed` | Mean milliseconds per evaluated sample for loading, inference, and metric calculation |
| `metrics.elapsed_seconds` | Total validation time, including dataset preparation and cache checks |

Use the metrics object's label metadata when interpreting confusion matrices.
Grounding also reports response statuses and unique images; attributes report
known decisions and per-group confusion matrices. Source, prepared, and
evaluated counts describe different scopes and should remain distinct in
reports.

Loading time means sample loading, not model initialization. Backend setup
before `.val()` is outside the reported timings; inference includes configured
backend correction attempts.

Keep dataset version, split, sample/class/group selection, geometry, context,
model revision, generation settings, output mode, and retries fixed when
comparing runs. Report the metric definition alongside the score. Validation
returns measurements without an automatic accuracy pass/fail threshold.

For command-line execution and saved JSON reports, see the
[validation example](../examples/README.md#task-validation).
