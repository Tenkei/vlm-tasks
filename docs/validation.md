# Validation

`Classify.val()`, `Detect.val()`, and `Ground.val()` evaluate configured tasks on
local labeled data. Each sample makes an independent model call: one image for
classification/detection, or one image-expression pair for grounding. Task
labels, descriptions and context come from the operator; model configuration,
output mode and retries come from its backend.

The [validation runner](../examples/run_validation.py) supports Bedrock
(default) and local weights with `--backend transformers`. Each backend reuses
the corresponding single-image runner's setup. See the
[example commands](../examples/README.md#dataset-validation) and
[API reference](reference.md#dataset-validation) for usage and custom validators.

## Dependencies and local data

Install the base package and the dependencies for your backend using the
[Bedrock example](../examples/README.md#amazon-bedrock) or
[Transformers guide](guides/huggingface-transformers.md). Detection YAML also
requires the validation extra:

```bash
python -m pip install -e '.[validation]'
```

Grounding JSON uses the base dataset dependencies. The runner reads local data
and does not download datasets.

### Classification and detection

Classification expects a root containing, for example, `val/cat/` and `val/dog/`
images. Detection expects a YOLO dataset YAML and its image/label files.
Configure labels to match the property or objects your dataset annotates. For
the example commands, save this as `/path/to/tasks.json`:

```json
{
  "classify": {
    "labels": ["cat", "dog"],
    "context": "Classify the animal shown in the image."
  },
  "detect": {
    "labels": {"pedestrian": "A visible person"},
    "context": "Locate every visible person."
  }
}
```

If the detection dataset class is `person`, save
`{"person": "pedestrian"}` as `/path/to/class-map.json` and pass it with
`--class-map` to align it with the task above. This option selects and aliases
dataset classes. `--match-iou` sets the detection matching threshold and applies
only to detection.

Classification reports top-1 accuracy; detection reports precision, recall and
F1 from class-aware box matching. See the [metric reference](reference.md#metrics-and-failures)
for definitions and failure handling.

## Grounding validation

Prepare local annotations using the
[MDETR guide](https://github.com/ashkamath/mdetr/blob/main/.github/refexp.md) and
[annotation archive](https://zenodo.org/records/4729015), alongside COCO
train2014 images. Supply the annotation directory with `--data` and the image
directory with `--images`. The annotation directory must contain the exact file
`finetune_{dataset}_{split}.json`, such as `finetune_refcocog_val.json`.

| Dataset | Expected split protocol | Supported splits |
|---|---|---|
| `refcocog` (default) | UMD | train, val, test |
| `refcoco` | UNC | train, val, testA, testB |
| `refcoco+` | UNC | train, val, testA, testB |

MDETR's published converter generates train/val annotations; requested test
files must be supplied separately. Select the dataset with `--dataset`.

`Ground.val()` uses the shared validation loop, with one independent call per
expression. `RefCOCODataset` reads local MDETR RefCOCO-family JSON and COCO
train2014 images, defaulting to RefCOCOg UMD. RefCOCO and RefCOCO+ use UNC splits.
Converted image records represent expressions; repeated filenames stay separate
and `max_samples` limits expressions. Sparse category IDs are reporting groups,
without a task vocabulary or class mapping.

The runner uses empty grounding task settings by default. Each dataset
expression supplies the target, overriding any constructor target. To set
context explicitly, pass a JSON file containing, for example,
`{"ground": {"context": "Resolve the whole object."}}` with `--config`.
For point grounding, use `{"ground": {"geometry": "point"}}`. Points are
predicted directly in normalized coordinates and returned in input-image
pixels. `--class-map` is unsupported for grounding.

Use ordinary continuous-coordinate box IoU, inclusive accuracy thresholds of
0.5, 0.75 and 0.9, and mean IoU across every evaluated expression. Valid
not_found/ambiguous responses and execution failures contribute zero IoU.
Status counts distinguish valid outcomes from failures; both evaluated and
full-split expression/unique-image counts are reported. The annotation format
comes from MDETR; its generalized-IoU evaluator is not the scoring reference.
Local paths and the expected split protocol are recorded without certifying
the provenance of arbitrary local annotation files.

Point mode uses the same expressions and reports `point_in_box_accuracy`:
the fraction whose predicted point falls inside the annotated target box,
including edges. Non-found statuses and failed calls count as misses. This is
a coarse proxy for target-mask containment, and cannot be directly compared
with box IoU accuracy. Point mode reports no IoU scores.

The validator measures single-target localization. RefCOCO-family
annotations assign one target per expression, so rejection quality and
annotated ambiguity require additional datasets. Mask-based point scoring and
multiple-target evaluation are future extensions. See the [reference](reference.md#grounding-refcoco-family)
for data preparation, annotation validation, metrics, and custom loaders.

## Sample limits and reproducibility

`--split` defaults to `val`. `--max-samples` limits evaluation in deterministic
dataset order: images for classification/detection, expressions for grounding.
Start with 10 expressions to check inference and timing, then use 100 for a
preliminary accuracy estimate. An ordered subset does not establish full-split
performance. Dataset counts describe all records in the supplied local
manifest; evaluated expression and unique-image counts are reported separately.

Classification and detection use [image_tasks.json](../examples/image_tasks.json)
by default; pass `--config` to match your annotations. Grounding reads a `ground`
entry only when an explicit config is supplied. Preserve the same dataset,
context, backend and generation settings when comparing runs, and report the
metric definition alongside each result.

`--retries` controls structured-output correction attempts and defaults to zero.
`--max-tokens` limits generated tokens per request. Each evaluated sample can
incur provider usage, including configured retries. Bedrock supports prompted,
native or tool output when the model/API supports it. Transformers supports
only prompted output and uses greedy generation.

For local Transformers runs, use `--device auto` for automatic selection or
select `mps`, `cuda` or `cpu` for your hardware, with a compatible `--dtype`.
Pin `--revision` to a model commit for reproducibility; `--local-files-only`
requires cached weights and processor files. See the
[Transformers guide](guides/huggingface-transformers.md) for runtime details.

## Reports and exit codes

The runner prints JSON containing aggregate metrics, per-class results, failure
records, dataset counts, paths/expected split protocol, grounding status counts,
task/model configuration, output mode, retries and timings. `--output` also
saves the report to a file. Transformers reports the actual device and dtype,
requested revision and separate model loading time.

Progress goes to stderr; `--no-verbose` disables it. Exit code 0 means validation
completed without recorded execution failures, 1 means task execution failures
were recorded, and 2 means setup or validation could not complete. Accuracy has
no automatic pass/fail threshold.
