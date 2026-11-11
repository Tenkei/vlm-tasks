# Validation

Validation compares a configured task's predictions with labeled examples. It
measures task quality, failures, and execution time without training the model.
This page records the design decisions, summarizes the available validations,
and explains how to configure a run.

### Context

A useful evaluation must exercise the same task semantics as application use.
Model accuracy, structured-output reliability, and execution failures are
different concerns; a successful response alone does not establish correctness.
Dataset formats should not determine task behavior or require separate scoring
logic for every data source.

### Decisions

| Decision                                                     | Alternatives considered                                               | Rationale and tradeoffs                                                                                                                |
|--------------------------------------------------------------|-----------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| Evaluate the configured task through its public results      | Call the model directly or score raw responses                        | Measures instructions, parsing, and task behavior together; results describe the complete configuration rather than the model alone    |
| Separate data loading, annotation adaptation, and evaluation | Put dataset-specific parsing and scoring in one runner                | Reuses task metrics across datasets; requires consistent labels, geometry, and sample meaning                                          |
| Provide task defaults with replaceable validators            | Require a benchmark framework or a custom runner for every evaluation | Makes reference runs easy while allowing extensions; defaults cannot represent every domain                                            |
| Prepare datasets separately from inference                   | Bundle data or make downloading inseparable from execution            | Allows caching and preparation before model use; subset preparation may still require large source downloads                           |
| Include failed predictions in scoring and report them        | Drop failures or stop on every model error                            | Avoids inflating scores by evaluating only successful calls; aggregate quality includes execution reliability                          |
| Use task-specific metrics and explicit evaluation scopes     | Report one universal score                                            | Preserves the distinction between classification, detection, localization, and attributes; scores across tasks are not interchangeable |
| Calculate caption metrics locally with a pinned PTB tokenizer | Unmaintained COCO Python packages; TorchMetrics as a future replacement | Avoids a metric framework dependency while preserving COCO scoring; requires Java, and tokenizer changes affect scores |
| Target Imagenette for the classification default             | Keep CIFAR-10 as the long-term default                                | Higher-resolution natural images; larger downloads and an unresolved resolution choice                                                 |

Dataset loading and validation have separate responsibilities:

| Responsibility        | Purpose                                                              |
|-----------------------|----------------------------------------------------------------------|
| Dataset loader        | Prepare images or crops and annotations as HF-style sample dictionaries |
| Dataset schema        | Define target columns, geometry, and complete categorical vocabularies |
| Evaluation            | Compare public task results with targets and accumulate metrics      |

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
| `Answer` | **VQA v2** (`vqav2`) | **val:** 214,354 questions on 40,504 COCO val2014 images; ten human answers per question | train, val; labeled splits only | VQA consensus accuracy; answer-type and question-type reports | Download both split annotation archives and images referenced by selected questions | val: 14.01 MB for questions and annotations; images additional |
| `Describe` | **COCO Captions 2017** (`coco_captions2017`) | **val:** 5,000 images, 25,014 captions; **train:** 118,287 images, 591,753 captions | train, val; labeled splits only | CIDEr, corpus BLEU-1–4, ROUGE-L | Shared annotation archive; only selected COCO 2017 images | 252.91 MB annotations; images and tokenizer additional |

`ClassifyMultiLabel` requires an explicit custom
validator. Local RefCOCO/RefCOCO+ data can reuse grounding validation; see
[local data](#using-your-own-dataset).

**Interpretation limits:** CIFAR-10 contains small images, and COCO128 is a smoke
dataset drawn from COCO training data. Neither an ordered subset nor a smoke
dataset establishes general model performance. RefCOCO-family targets do not
measure annotated ambiguity, rejection quality, or multiple-target resolution.
The OVAD protocol here adapts independent attribute annotations to categorical
decisions; it is distinct from official OVAD attribute mAP.

Dataset sources: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html),
[COCO128](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip),
[MDETR RefCOCO annotations](https://zenodo.org/records/4729015),
[OVAD](https://lmb.informatik.uni-freiburg.de/resources/datasets/ovad.zip),
[VQA v2](https://visualqa.org/download.html), and
[COCO Captions](https://cocodataset.org/#download).
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
| Answer | Normalize the prediction and all ten references; average leave-one-annotator-out consensus per question, then average across evaluated questions |
| Describe | PTB-tokenize candidates and references; compute corpus BLEU-1–4, mean COCO clipped CIDEr, and mean multi-reference ROUGE-L |

Attribute groups without target support do not contribute to group averages;
values with neither target nor scored prediction support do not contribute to
macro F1. Unknown targets are unscored, rather than an invented “none” category.

`ModelExecutionError` and `InvalidModelResponseError` are recorded and evaluation
continues. Failed classification calls are incorrect; failed detection calls
produce no boxes; failed grounding calls score zero; failed attribute calls are
incorrect for every known target; failed answer calls score zero. Failed samples
and their targets remain in the evaluation and relevant score denominators.
Data, setup, and programming errors stop the run.

Grounding `not_found` and `ambiguous` responses score zero against a positive
target. Each referring expression is evaluated independently, even when several
expressions use the same image.

## Run validation

### Setup

Install validation dependencies from the repository root:

```bash
python -m pip install -e '.[validation]'
```

This includes Hugging Face Datasets, Pillow, and PyYAML for the built-in loaders. `Describe.val()` also
requires Java on `PATH`; its tokenizer is downloaded on first use and cached
for later runs. Dataset preparation alone does not require Java.

Initialize a [model backend](model-backends.md) using, for example, the
[Bedrock guide](amazon-bedrock.md) or
[Transformers guide](huggingface-transformers.md). The examples below assume
`backend` is ready.

### Calling validation

Call `.val()` on a configured task:

```python
from ml_pipes.vlm import Ground

task = Ground(backend)
metrics = task.val(max_samples=100)

print(metrics.results_dict)
print(metrics.summary())
print(metrics.failures)
```

Each call returns fresh metrics and stores them as `task.metrics`.

| Common setting | Purpose |
|---|---|
| `dataset` | Loaded dataset; omitted to load the task default evaluation data |
| `max_samples` | Positive limit in dataset order, or `None` (default) for the full selection |
| `cache_dir` | Override the cache directory |
| `verbose=False` | Disable progress and aggregate output on stderr |

For the complete argument contracts, see the
[API reference](reference.md#task-validation).

### Classify: CIFAR-10

```python
from ml_pipes.vlm import Classify

task = Classify(backend)
metrics = task.val(
    max_samples=100,
)
```

Dataset labels replace constructor labels for classification
validation.

### Detect: COCO128

| Validation setting | Purpose |
|---|---|
| `match_iou` | Inclusive box-matching IoU threshold; defaults to `0.5` |

```python
from ml_pipes.vlm import Detect

task = Detect(backend)
metrics = task.val(match_iou=0.75, max_samples=100)
print(metrics.precision, metrics.recall, metrics.f1)
```

Dataset labels replace constructor labels for detection
validation.

Detection does not report AP/mAP because predictions have no confidence ranking.

### Ground: RefCOCOg

The task's geometry selects point-in-box or box-IoU scoring:

```python
task = Ground(backend, geometry="point", context="Locate the described object.")
metrics = task.val(max_samples=100)
print(metrics.results_dict["point_in_box_accuracy"])
```

Dataset expressions replace the constructor target. Dataset categories group
the reported scores.

### Attributes: OVAD

Omit constructor attributes to use the dataset's complete categorical schema:

```python
from ml_pipes.vlm import Attributes

task = Attributes(backend)
metrics = task.val(max_samples=100)

print(metrics.confusion_matrices["material"])
```

Each sample is an object crop. All crops use the same selected attribute choices,
including groups whose target is unknown for that object.

To evaluate fewer groups, select them while loading the dataset:

```python
from ml_pipes.vlm.validation import load_ovad

dataset = load_ovad(groups=("material",), max_samples=100)
metrics = Attributes(backend).val(dataset=dataset)
```

Each selected group requires its complete dataset vocabulary. The categorical
adaptation excludes `color` and `clothes color`, objects with annotated area
below 50 pixels², and objects without a known target in the selected groups.
Group selection can therefore change sample eligibility before the sample limit.

### Answer: VQA v2

Each question is a sample, including questions that share an image:

```python
from ml_pipes.vlm import Answer

task = Answer(backend)
metrics = task.val(max_samples=100)
print(metrics.results_dict["vqa_accuracy"])
print(metrics.summary())
```

Dataset questions replace the constructor question.

The scorer follows [lmms-eval's VQA v2 protocol](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/c7eaef373f4b1bd8a6b5f3e85c3391962c30068c/lmms_eval/tasks/vqav2/utils.py).
Predictions and references are normalized for whitespace, punctuation, case,
number words, articles, and contractions. Decimal points and apostrophes are
preserved; numeric commas are removed. Matching uses normalized strings, without
answer extraction or synonym matching: `Two!` matches `2`, while
`The answer is blue` does not match `blue`.

For each of the ten reference positions, hold out that vote and score
`min(matching_remaining_votes / 3, 1)`; average the ten scores. Duplicates are
retained. Exactly 0, 1, 2, 3, or at least 4 matching votes produce scores of
0, 0.3, 0.6, 0.9, or 1. Overall `vqa_accuracy` is the mean across questions on a
0–1 scale. An `unanswerable` response scores zero; negative answers such as `no`
and `zero` are scored normally.

`summary()` reports support and accuracy by the three annotated answer types
(`yes/no`, `number`, `other`) and by observed question types. Answer types with
no evaluated questions have zero support and a `None` score.

This always-normalized protocol differs from the legacy official evaluator,
which conditionally normalizes based on reference disagreement. VQA v2 does
not measure answerability quality.

### Describe: COCO Captions

| Validation setting | Purpose |
|---|---|
| `tokenizer_version` | Pinned PTB tokenizer: `"3.4.1"` (default), or `"4.6.0"` |

```python
from ml_pipes.vlm import Describe

task = Describe(backend, context="Describe the visible scene in one sentence.")
metrics = task.val(max_samples=100, tokenizer_version="3.4.1")
print(metrics.cider, metrics.bleu_4, metrics.rouge_l)
print(metrics.summary())
```

Each image is a sample, scored against all its reference captions, including
duplicates. Any positive number of references is supported. In the corpus
summary, `support` counts images and `reference_captions` counts captions.

Captions use COCO's lowercase Stanford PTB tokenization and punctuation filter,
preserving articles and number words.

| Result key | Definition and scale |
|---|---|
| `cider` | Mean COCO clipped TF-IDF similarity over 1–4 grams, with a Gaussian length penalty (sigma 6), multiplied by 10 |
| `bleu_1`–`bleu_4` | Corpus n-gram precision with a brevity penalty; closest reference length, shorter on ties; 0–1 scale |
| `rouge_l` | Multi-reference F-score based on the longest common subsequence, beta 1.2, averaged over images; 0–1 scale |

CIDEr's document frequencies use the references of **all selected images**,
including failed calls. Changing the selection changes its inverse document
frequencies (IDF) and the resulting score. A one-image run has zero IDF and
therefore `cider=0`. Use a fixed selection of multiple images for comparisons.
Long descriptions can also score poorly against short references because of
CIDEr's length penalty.

### Default datasets and downloads

Load a dataset before model execution to inspect or select its samples:

```python
from ml_pipes.vlm.validation import load_refcocog

dataset = load_refcocog(max_samples=100, cache_dir="/datasets/vlm-cache")
print(dataset.features)
metrics = Ground(backend).val(dataset=dataset)
```

Built-in loaders return Hugging Face `Dataset` objects. CIFAR-10 defaults to
`test`; the other loaders default to their evaluation `val` data. Choose another
split in the loader. `task.val()` and validators accept already selected data.

Use a loader's `max_samples` to limit preparation and image downloads where
supported. `task.val(dataset=dataset, max_samples=...)` limits evaluation of the
loaded dataset.

Cache precedence is `cache_dir`, then `VLM_TASKS_DATASETS_DIR`, then
`~/.cache/vlm-tasks/datasets`. Download requirements, including full archives
needed for small selections, are listed in [available validations](#available-validations).

### Results

| Result | Meaning |
|---|---|
| `metrics.results_dict` | Aggregate scores, evaluated samples, failures, and failure rate |
| `metrics.summary()` | Per-class, per-category, per-attribute, or per-type scores and support; Describe returns one corpus row |
| `metrics.failures` | Failed calls with image paths, error messages, and sample identifiers where available |
| `metrics.dataset_counts` | Source (before selection), prepared (selected dataset), and evaluated (this run) counts, where available |
| `metrics.dataset_metadata` | Dataset selection and preparation context when available |
| `metrics.speed` | Mean milliseconds per evaluated sample for loading, inference (including correction attempts), and metric calculation |
| `metrics.elapsed_seconds` | Total `.val()` time, including dataset preparation and cache checks; excludes prior backend setup |

## Using your own dataset

Pass one loaded split with the task's sample schema. Use Hugging Face's column
and formatting methods to adapt the dataset before validation:

```python
from datasets import load_dataset
from ml_pipes.vlm import Classify

dataset = load_dataset("uoft-cs/cifar10", split="test")
dataset = dataset.rename_column("img", "image")
dataset = dataset.with_format("numpy", columns=["image"], output_all_columns=True)
metrics = Classify(backend).val(dataset=dataset.select(range(100)))
```

Use `cast_column()` and `map()` for feature metadata and annotation conversions.
Format only `image` as NumPy to preserve annotation lists and scalar IDs.
For grayscale or alpha images, convert to RGB with a dataset transform first.
Streaming `IterableDataset` objects work too when their rows follow the same
schema. `DatasetDict` contains splits; pass one of its datasets.

### Sample schemas

| Task | Required columns | Sample unit |
|---|---|---|
| Classify | `image`, integer `label` | One image |
| Detect | `image`, `objects={"bbox": list[list[float]], "category": list[int]}` | One image, including images with no objects |
| Ground | `image`, `text`, `bbox`, integer `label` | One referring expression and target box |
| Attributes | `image`, `attributes` mapping groups to integer label IDs or `None` | One subject crop with at least one known target in the selected groups |
| Answer | `image`, string `question`, `answers: list[str]`, `answer_type`, `question_type` | One question with all ten reference answers |
| Describe | `image`, `captions: list[str]` | One image with all reference captions |

`image` is a nonempty HWC uint8 RGB NumPy array. Boxes are lists of
`[x1, y1, x2, y2]` coordinates in pixels of that image, with finite values and
positive area. Detection's box and category lists have equal lengths; both
may be empty. Attributes images contain the selected subject crop.

VQA `answers` contains exactly ten strings, preserving duplicate votes;
`answer_type` is `yes/no`, `number`, or `other`, and `question_type` is a
string. `captions` contains one or more nonempty reference strings.

Convert source annotations with `map()` before calling `.val()`. For example,
extract VQA answer records without discarding votes:

```python
dataset = dataset.map(
    lambda row: {"answers": [vote["answer"] for vote in row["answers"]]}
)
```

`image_id` can group repeated images in Ground, Attributes, and Answer, and
identify images for Describe. The sample types declare it on those tasks.
Additional reporting columns are optional: `sample_id` identifies a row and
`im_file` supplies its source path. Without them, reports use generated
identifiers. Rename source IDs such as VQA's `question_id` to `sample_id` when
adapting the dataset.

Categorical columns require HF-style `ClassLabel` features with the complete
vocabulary: `features["label"]` for classification/grounding,
`features["objects"]["category"] = List(ClassLabel(...))` for detection, and one
`ClassLabel` per group in `features["attributes"]` for attributes. IDs are
zero-based indices into those vocabularies. Sample selection retains these choices.
Dataset labels replace constructor labels during classification and detection
validation; stored task configuration and context are preserved.

### Local benchmark formats

These loaders adapt existing local annotations into the same sample schemas:

| Loader | Local data |
|---|---|
| `load_classification(root, split="val")` | `<root>/<split>/<class>/image.jpg` directories |
| `load_yolo(data, split="val")` | YOLO detection YAML and TXT annotations |
| `load_refcoco(annotations, images, dataset="refcocog", split="val")` | MDETR `finetune_{dataset}_{split}.json`; also supports `refcoco` and `refcoco+` |
| `load_ovad(annotations, images)` | OVAD JSON and source images; selects and crops objects |
| `load_vqav2(annotations, images, split="val")` | Official question/annotation JSON directory and COCO 2014 images |
| `load_coco_captions2017(annotations, images)` | Official COCO captions JSON file and source images |

## Custom loaders and validators

Custom datasets implement the public `ValidationDataset` protocol: iterate over
sample mappings and expose a `features` mapping. `features=None` is sufficient
for Answer and Describe. Inheritance is optional; validators use duck typing.

Direct validators consume the same loaded datasets:

```python
from ml_pipes.vlm.validation import ClassificationValidator

metrics = ClassificationValidator(task, dataset, verbose=False)()
```

Validators pass the dataset's sample mappings directly to their hooks. They
check the schema, call the task, and score its results. Incompatible rows stop
the run before inference for that row. Column renaming, image formatting,
cropping, and annotation conversion belong in loaders or HF transforms.

Extend a concrete validator to change prediction or scoring rules, or use
`BaseValidator` for another evaluation contract. Custom validator classes receive
`task`, `dataset`, and the validation options through
`task.val(dataset=dataset, validator=MyValidator)`.

## Next read

- [Validation example](../examples/README.md#task-validation): command-line execution and saved JSON reports.
