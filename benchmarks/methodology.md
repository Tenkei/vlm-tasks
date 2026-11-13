# Benchmark methodology

This protocol compares configured VLM task pipelines using **100 samples per
task**. Results live in the [dedicated comparison table](README.md).
Python code lives in [`src/`](src/) and model configurations in
[`configs/`](configs/), alongside the results and this method. The code calls
the library's public `.val()` method and does not implement alternative task
scoring. Run the module commands below from the repository root.

## Layout and naming

```text
benchmarks/
├── configs/                   # Model configurations
├── src/                       # Preparation, evaluation and report code
├── runs/                      # Recorded evaluations, grouped by run ID
├── reports/
│   ├── metrics.csv
│   ├── report-manifest.json
│   └── charts/
│       ├── model-profiles/    # One radar profile per model
│       └── task-comparisons/  # One bar chart per task
├── experiments/token-budget/  # Paired token-budget experiments
├── failed-runs/               # Startup and infrastructure failures
├── methodology.md
└── README.md                  # Generated model and task overview
```

Python modules use `snake_case` and describe their action or responsibility.
Model configs use `<model>-<backend>.json`, including model size where relevant.
Report folders match the overview's **Model profiles** and **Task comparisons**
sections. Recorded run IDs, task IDs and original metadata retain their identity
when directories move. Prepared datasets are cached in
`.cache/prepared-datasets-100/`; the suffix identifies the configured sample count.

## Fixed protocol

The first model is `Qwen/Qwen3-VL-2B-Instruct` from Hugging Face, revision
`89644892e4d85e24eaac8bacfd4f463576704203`. Its configuration is
[`qwen3-vl-2b-transformers.json`](configs/qwen3-vl-2b-transformers.json). Run locally with Transformers, Apple
MPS, float16, greedy decoding (`do_sample=False`), seed 42, a maximum of 4,096
new tokens per request, prompted JSON output, and zero correction retries.
The generation configuration, processor settings, chat template, and output
instructions are saved in each run's `model.json`. Calls have independent
conversation history. The model is loaded once and tasks run sequentially in
configuration order. There is no separate warmup; the first call is included.
The baseline host is an Apple M1 Pro with 32 GiB unified memory, running macOS
15.5. Exact Python and package versions are saved with the run.

Select the first 100 eligible rows in each default loader's source order, with
no shuffle or prediction-based filtering. The seed controls model setup; it is
not a sampling seed because selection is an ordered prefix. Both grounding
modes use exactly the same expressions. This is 700 evaluated task samples,
with repeated images possible within and across tasks.

| Task | Dataset and selection unit | Split | Aggregate validation metrics |
|---|---|---|---|
| Classify | CIFAR-10, images; pinned dataset revision `0b2714987fa478483af9968de7c934580d0bb9a2` | `val` aliases test | Top-1 accuracy; per-class precision/recall/F1 in JSON |
| Detect | COCO128, images; all 80 labels | `val` uses the same training images as `train` | Pooled precision/recall/F1 at inclusive IoU 0.5; TP/FP/FN |
| Ground, box | RefCOCOg UMD, referring expressions | val | Accuracy at IoU ≥0.5/0.75/0.9; mean IoU |
| Ground, point | Same RefCOCOg expressions | val | Point inside target box, including edges |
| Attributes | OVAD, eligible object crops; all 22 categorical groups | `val` aliases benchmark | Accuracy across known decisions; mean group accuracy; macro F1 |
| Answer | VQA v2, questions with all ten answer votes | val | Mean normalized leave-one-annotator-out consensus accuracy |
| Describe | COCO Captions 2017, images with all reference captions | val | CIDEr, corpus BLEU-1/2/3/4, ROUGE-L |

Task settings use library defaults and no extra context. Dataset vocabularies
provide classification/detection labels and attribute choices. Grounding
receives each expression, and Answer receives each question. Caption scoring
uses Stanford PTBTokenizer 3.4.1 with its pinned SHA-256; CIDEr document
frequencies are computed over the selected 100 images. Detailed metric and
dataset adaptation rules are in [validation](../docs/validation.md).

`ClassifyMultiLabel` has no built-in validation dataset and is excluded. Planned
tasks such as ReadText are excluded.

## Failures and usage

Inference and parsing failures remain in all applicable denominators; validation
uses its existing failure policy. Grounding abstentions score zero against
positive targets; VQA abstentions score zero. Failed captions use empty text
with references retained. Detection failures produce no boxes, so they add false
negatives for targets but no false positives. Data/setup/programming errors stop
the run and do not produce a completed task report.

The runner observes the existing Transformers adapter **before** PydanticAI
parsing. Every returned generation is saved, including invalid JSON and
schema-invalid responses. Token totals therefore include generated responses
that fail validation. Input counts come from processed `input_ids` and include
schema, instructions, and expanded visual tokens. Output counts include all
generated token IDs, including generated special tokens. These tokenizer counts
are specific to each model; raw token counts are not interchangeable across
model families.

Each report records attempted, measured, and unmeasured requests. A generation
error that prevents the adapter from returning usage is marked unmeasured;
partial token consumption cannot be recovered. With retries fixed to zero,
there is one attempted model call per evaluated sample. Local monetary cost is
`null`, since hardware and electricity costs are not measured.

The comparison table reports mean input and output tokens **per evaluated sample**,
rounded to the nearest whole token. Estimated cost is **USD per 1,000 samples**:
each task's total estimated cost divided by its evaluated sample count (N),
including failed samples, then multiplied by 1,000. This is a projection using
observed usage and pricing; it does not assume batching or batch discounts.
Failure and token-cap rates also use N; latency is a per-sample mean.
N is shown separately. Task reports retain usage totals, and the CSV includes
totals, precise per-sample values and cost per 1,000 samples. This keeps
usage comparisons independent of sample count, although a different sample mix
can change the averages. Unmeasured requests are included in N; incomplete usage
or cost coverage means the corresponding averages are partial.

Table and chart labels omit unnecessary trailing zeros; stored values retain
their original precision.

The **Failure rate (at cap)** column shows the overall failure rate followed
by the rate of responses at the output token cap in parentheses, for example
`21% (13%)`. Both rates use N; cap hits can include valid responses and are
not necessarily a subset of failures. The cap rate flags
possible truncation; inspect raw text to distinguish an incomplete response
from one finishing exactly at the limit. A fixed, generous budget makes
quality and usage comparable while bounding repetitive generation. It can
still truncate valid long answers, so cap hits must accompany accuracy scores.
A different budget is a separate benchmark configuration and run.

Inference latency is the validator's mean task-call time, including image
encoding, preprocessing, generation and parsing. Dataset loading and metric
calculation have separate timings. Dataset preparation and shared model load
time are recorded separately. These are sequential local timings; batching,
concurrency, cold downloads, peak memory and energy are not measured.

## Model profile charts

The [radar charts](README.md#model-profiles) show one profile per
model/run, with identical axis order and a fixed radial range of 0–100. Every
plotted value is its recorded metric multiplied by 100; higher is better.
This model view begins with a four-column overview grid. Each thumbnail links
to a collapsed model section with the full radar chart and a table of that
model's task metrics and usage. Download links appear in the grid only.
Radial tick numbers, the scale subtitle and the sample-count subtitle are hidden.
Task titles are bold, with metric labels and values in regular weight.
There is no normalization to the best observed model or to sample count.
Failures retain their original scoring penalties.

| Axis | Recorded metric |
|---|---|
| Classification | Top-1 accuracy |
| Detection | F1 at IoU 0.5 |
| Grounding (Box) | Accuracy at IoU 0.5 |
| Grounding (Point) | Point-in-box accuracy |
| Attributes | Accuracy across known attribute decisions |
| VQA | VQA accuracy |
| Captioning | ROUGE-L |

ROUGE-L is bounded by 0–1, which allows a fixed scale. CIDEr has no fixed
0–1 bound and remains in the table. Each axis summarizes a different task;
equal radial values do not imply equal difficulty, and polygon area is not
an overall model score. The table retains the other validation metrics and
per-sample tokens and latency, and cost per 1,000 samples.

Generate PNG and editable SVG files using saved reports only:

```bash
uv pip install --python .venv/bin/python 'matplotlib==3.11.2'
.venv/bin/python -m benchmarks.src.generate_charts
```

The chart generator reads task reports, validates that all seven metrics are present
and bounded, writes one PNG and SVG per model/run and per task, and
refreshes the README's four-column model profile grid and task sections. It performs no inference. `reports/charts/manifest.json`
records metric choices, exact scores, sample counts, Matplotlib/NumPy versions,
and source, report and figure hashes. Use those package versions to reproduce
the rendering. The report generator embeds charts only when their report hashes
match the current comparison; regenerate the charts after adding or changing runs.

### Task comparison charts

The [task comparisons](README.md#task-comparisons) use one
horizontal bar chart per task, with the same main metrics listed above.
Models keep the same order and colors as their radar profiles. Bar lengths
use the fixed 0–100 range (recorded metric × 100), with numeric axis ticks
hidden. Direct value labels show percentages, except caption ROUGE-L, which
shows its native 0–1 value. Titles are bold and sample-count subtitles are
omitted. No scores are normalized against the observed models.

The chart manifest records each task's metric, original scores, displayed
model labels and filenames. A models-by-tasks matrix near the top of the
comparison README shows the main metrics with the best observed values in
bold. Its column and row labels link to the corresponding model and task details.
Task comparisons then begin with a four-column thumbnail grid presenting the
task view: all models compared on each task. Chart thumbnails link to the
matching details; PNG/SVG downloads appear in the overview grids only.
Each task is collapsed by default and pairs its chart with the full validation
and usage table. Grounding nests collapsible Box and Point variations using
the same chart-then-table layout. Fragment anchors sit inside the disclosures
so browsers can reveal the linked details. The tables retain additional validation metrics, failures, tokens per
sample, cost per 1,000 samples and mean latency; charts compare task quality only.
Shared column definitions and interpretation notes appear in the README's
final [Table field guide](README.md#table-field-guide), with
bullets named after the table headers and related fields grouped together.

## Reproduce

From the repository root, create a Python 3.12 environment and install the
project and compatible local model dependencies:

```bash
python -m pip install -e '.[validation,test]'
python -m pip install 'torch==2.15.0' 'torchvision==0.29.2' 'transformers==4.57.6' 'datasets==4.8.5'
```

The recorded `environment.json` and `requirements.txt` contain the actual
installed versions for the baseline. Use a suitable PyTorch/torchvision pair
for other hardware and record that difference. Java must be on `PATH` for
caption scoring. See the [Transformers setup](../docs/huggingface-transformers.md).

Prepare data independently of inference. This downloads source annotations,
the selected images, and the caption tokenizer into the benchmark cache:

```bash
python -m benchmarks.src.prepare_datasets --config benchmarks/configs/qwen3-vl-2b-transformers.json \
  --output benchmarks/.cache/prepared-datasets-100
python -m benchmarks.src.run_transformers --config benchmarks/configs/qwen3-vl-2b-transformers.json \
  --prepared benchmarks/.cache/prepared-datasets-100 \
  --output benchmarks/runs/qwen3-vl-2b-reproduction
python -m benchmarks.src.generate_report
```

Both preparation and inference require a new output directory to prevent
overwriting an existing run. `--offline` on the inference command requires
already cached weights and processor files. If interrupted, complete remaining
tasks with `--tasks TASK ...` in a new output directory; keep artifacts from
both invocations. Existing results are never silently reused as fresh inference.

For cached local source files, pass `--sources path/to/sources.json` to
preparation. The JSON maps dataset names to keyword arguments of the public
loaders. For `coco128`, `{"data": "path/to/data.yaml"}` uses the local YOLO
loader; `refcocog` accepts an `annotations` directory containing
`finetune_refcocog_val.json` and an `images` directory. OVAD and COCO captions
accept an annotations file plus images; VQA accepts an annotations directory
plus images. Other datasets use automatic downloads. The first run reuses
cached assets from earlier experiments, with these overrides recorded in its
manifest. They affect source acquisition; image/target hashes identify the
actual evaluated data.

Preparation saves portable HF datasets with embedded image bytes. Each
manifest records selected IDs, all targets, RGB image sizes/hashes, feature
vocabularies, source overrides and a content fingerprint per dataset. Inference
recomputes these fingerprints before loading weights. Reproduce the baseline
sample identities by comparing those fingerprints with its saved manifest;
abort a comparison if they differ. Grounding sample IDs are expression IDs;
classification IDs are row offsets in the pinned test split.

To add a model, copy the config with a different model ID and immutable
revision, and use the **same prepared directory**, prompts, output budget,
retry policy and scoring settings. Set device and dtype explicitly, and report
any differences. This runner targets decoder-only models compatible with the
existing Transformers adapter. Bedrock has a separate runner below. Do not
average scores across unrelated task metrics.

Task definitions, validation metric labels and chart axes are shared in
[`task_registry.py`](src/task_registry.py). Both runners and experiment scoring use the same
validation dispatch, frozen-data checks and response recording in
[`evaluation.py`](src/evaluation.py). The command modules handle model setup and
artifact writing. Reporting reads saved results and makes no model calls.

### Larger Qwen3 on AWS Bedrock

The hosted comparison uses `qwen.qwen3-vl-235b-a22b` in `ap-northeast-1`, with
the existing `benchmark` AWS profile. Configuration is in
[`qwen3-vl-235b-a22b-bedrock.json`](configs/qwen3-vl-235b-a22b-bedrock.json), and execution code is
in [`run_bedrock.py`](src/run_bedrock.py).

Install the optional dependencies and authenticate through the browser:

```bash
uv pip install --python .venv/bin/python \
  'pydantic-ai-slim[bedrock]==2.61.0' 'botocore[crt]'
aws login --profile benchmark
.venv/bin/python -m benchmarks.src.run_bedrock \
  --config benchmarks/configs/qwen3-vl-235b-a22b-bedrock.json \
  --prepared benchmarks/.cache/prepared-datasets-100 \
  --output benchmarks/runs/qwen3-vl-235b-bedrock-reproduction
.venv/bin/python -m benchmarks.src.generate_report
```

This uses the same frozen 100 rows for each of the seven task configurations,
feature vocabularies, task instructions, scoring, 4,096-token budget and zero
correction retries. Requests are sequential with temperature 0. The prompted
JSON template is copied from the local adapter, and provider schema rewriting
is disabled for these text-only schema instructions to preserve the schema
content. An offline SDK test checks the transmitted system instructions,
schema and inference settings.

The runner captures provider responses before agent parsing, including invalid
responses, input/output/cache/reasoning token counts, stop reasons, request IDs
and cost estimates. SDK requests have one total attempt, a ten-second connect
timeout and a 600-second read timeout. Infrastructure errors stop the run with
the request journal retained, rather than turning permission or connection
problems into an entire task's accuracy failures.

USD estimates use the recorded `genai-prices` package and each model response's
usage, including responses that fail parsing. They are separate from provider
invoices, which the Converse API does not return. Reports record cost coverage
and unmeasured usage. The comparison table orders models within each task and
includes estimated USD per 1,000 samples, whole tokens per sample and mean latency.

### Gemma 3 27B and Kimi K3.1 on AWS Bedrock

The additional runs use the same frozen data, task prompts, schemas,
4,096-token budget and scoring settings as the Qwen baseline:

| Model | Config | Invocation route |
|---|---|---|
| Gemma 3 27B | [`gemma-3-27b-bedrock.json`](configs/gemma-3-27b-bedrock.json) | `google.gemma-3-27b-it`, on demand, US East (N. Virginia) |
| Kimi K3.1 | [`kimi-k3.1-bedrock.json`](configs/kimi-k3.1-bedrock.json) | `global.moonshotai.kimi-k3.1`, global inference profile from Tokyo |

```bash
.venv/bin/python -m benchmarks.src.run_bedrock \
  --config benchmarks/configs/gemma-3-27b-bedrock.json \
  --prepared benchmarks/.cache/prepared-datasets-100 \
  --output benchmarks/runs/gemma3-27b-bedrock-reproduction
.venv/bin/python -m benchmarks.src.run_bedrock \
  --config benchmarks/configs/kimi-k3.1-bedrock.json \
  --prepared benchmarks/.cache/prepared-datasets-100 \
  --output benchmarks/runs/kimi-k3.1-bedrock-reproduction
.venv/bin/python -m benchmarks.src.generate_charts
```

The runner verifies that an inference profile is active and routes exclusively
to the configured foundation model. Reports retain that model ID; `model.json`
also records the invoked profile ID and its resolved routing metadata. Kimi's
global routing may serve requests outside Tokyo, which affects latency.
Gemma's first Tokyo request timed out after 600 seconds without a returned
generation; its benchmark therefore uses US East. The timeout attempt is
preserved in `benchmarks/failed-runs/` with unknown usage and contributes no
validation scores. Latency includes regional network differences. Independent
cloud model runs may overlap; each run sends only one request at a time, with
no batching or per-model concurrency.

Both models' caption tasks initially stopped before inference because `datasets`
4.8.5 checks remote tokenizer metadata even when the pinned JAR is cached,
and the processes had HF offline mode enabled. Their first six task reports were
retained. Captioning was continued with offline mode disabled and the same
checksum-verified tokenizer, data and model settings:

```bash
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 .venv/bin/python -m benchmarks.src.run_bedrock \
  --config benchmarks/configs/gemma-3-27b-bedrock.json \
  --prepared benchmarks/.cache/prepared-datasets-100 \
  --output benchmarks/runs/gemma3-27b-bedrock-20261112 --resume
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 .venv/bin/python -m benchmarks.src.run_bedrock \
  --config benchmarks/configs/kimi-k3.1-bedrock.json \
  --prepared benchmarks/.cache/prepared-datasets-100 \
  --output benchmarks/runs/kimi-k3.1-bedrock-20261112 --resume
```

`--resume` verifies the saved config, prepared manifest, completed sample
indices and model settings. It skips complete tasks and refuses to repeat a
task with a nonempty interrupted request journal. Original metadata remains
unchanged; `continuations/<timestamp>/` records the continuation's environment,
source hashes, model metadata and selected tasks. New task reports reference
their execution environment. The tokenizer setup interruption adds no model
calls or scored failures.

Kimi K3.1 reasons by default at the provider's `max` effort. These runs leave
model-specific reasoning defaults unchanged; raw reasoning text and total
output usage are recorded. Bedrock may not report a separate reasoning-token
counter even when reasoning text is present. Reasoning counts toward the common
output allowance. Do not interpret this comparison as
matching internal reasoning effort across models. Gemma uses temperature 0;
Kimi rejects the temperature field, so its config sets `null` to omit it and
uses provider-controlled sampling. The rejected startup request is retained in
`benchmarks/failed-runs/` and contributes no validation scores.

The pinned pricing package has no Bedrock K3.1 entry. Its config therefore
records the official [AWS model-card rates](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-moonshot-ai-kimi-k3-1-highspeed.html)
for Global CRIS, Standard tier: USD 4 input, 20 output, 0.20 cached input and
5 cache-write per million tokens, retrieved on 2026-11-12. Explicit config rates
take precedence over the package. Cache reads/writes are priced separately from
uncached input; reasoning tokens are included in total output and not charged
twice. Raw generation records identify the pricing source, and the config and
model metadata preserve the rate snapshot. Costs remain estimates, not invoices.

Bedrock manages the checkpoint behind its model ID, image processing, hardware,
dtype and sampling seed. These are not exposed as immutable HF revisions or
locally selectable settings. The model metadata, endpoint, time, package
versions and source hashes are saved with the run. Hosted timing includes
network and service time. This compares complete task configurations across
model sizes and backends; matching samples and prompts does not make their
hardware, tokenization or provider preprocessing identical.

### Double the budget for capped responses

After the baseline completes, rerun every sample whose response used the full
4,096-token allowance with an 8,192-token allowance:

```bash
python -m benchmarks.src.run_token_budget \
  --baseline benchmarks/runs/qwen3-vl-2b-20261112 \
  --prepared benchmarks/.cache/prepared-datasets-100 \
  --output benchmarks/experiments/token-budget/qwen3-vl-2b-20261112-double-tokens --offline
```

The experiment verifies the dataset fingerprints, keeps the same model revision,
processor, prompts, seed, dtype and retry policy, and selects only responses
at the original cap. It records selected sample identities in `selection.json`,
copies their exact RGB images into `images/<task>/`,
new requests/responses in task JSONL files, and paired outcomes in
`comparison.json`. Baseline predictions are replayed through public validation
without inference to compute baseline scores on that **same selected subset**.
For caption subsets, both conditions recompute CIDEr IDF on that subset.
Compare completion, scores, tokens, latency and whether the new cap is reached.
This is a conditional analysis of capped samples; its scores do not replace
the 100-sample comparison. It runs after the baseline to avoid concurrent GPU
use affecting timings.

If an experiment is stopped, report completed records without loading model
weights or running further inference:

```bash
python -m benchmarks.src.report_token_budget \
  --experiment benchmarks/experiments/token-budget/qwen3-vl-2b-20261112-double-tokens \
  --baseline benchmarks/runs/qwen3-vl-2b-20261112 \
  --interrupted-requests 1
```

This writes a partial table and `status.json`, distinguishing completed,
interrupted and unstarted samples. Counts and token totals in that table cover
completed cases only; interrupted generation usage is unknown. The
`--interrupted-requests` value records how many active requests were stopped.
`--baseline` supplies the current run location when its originally recorded
path predates a directory move. Paths inside original run and experiment
artifacts are retained as evidence of their execution environment.

## Artifacts

| Artifact | Contents |
|---|---|
| `benchmarks/configs/qwen3-vl-2b-transformers.json` | Fixed baseline configuration |
| `benchmarks/configs/qwen3-vl-235b-a22b-bedrock.json`, `benchmarks/src/run_bedrock.py` | Hosted Qwen3 configuration, execution and usage collection |
| `benchmarks/configs/gemma-3-27b-bedrock.json`, `benchmarks/configs/kimi-k3.1-bedrock.json` | Additional hosted model configurations, routing and pricing settings |
| `benchmarks/src/prepare_datasets.py` | Dataset preparation and sample freezing |
| `benchmarks/src/task_registry.py` | Shared task definitions, datasets, metric labels and chart axes |
| `benchmarks/src/evaluation.py` | Shared validation, frozen-data verification, response recording and replay |
| `benchmarks/src/artifacts.py` | JSON artifacts, sample identities, content hashes and source provenance |
| `benchmarks/src/run_transformers.py` | Local model setup, inference and task reports |
| `benchmarks/src/generate_report.py` | Comparison table and wide CSV generation |
| `benchmarks/src/generate_charts.py` | Model radar profiles and task comparison charts from saved reports |
| `benchmarks/src/run_token_budget.py` | Paired 2×-budget reruns of capped samples |
| `benchmarks/src/report_token_budget.py` | Report completed token-budget records after stopping inference |
| `runs/<run>/manifest.json` | Dataset identities, targets and image hashes |
| `runs/<run>/environment.json`, `requirements.txt` | Python/platform/package versions, git revision and source hashes |
| `runs/<run>/model.json` | Actual model/processor/generation configuration |
| `runs/<run>/pricing-source.html`, `pricing-source.json` | Official rate source snapshot and hash when config rates override the pricing package |
| `runs/<run>/<task>.jsonl` | Ordered requests, raw generated text, usage, predictions and errors; join to manifest by index |
| `runs/<run>/<task>.json` | Full validation metrics, breakdowns, failures, timings and usage totals |
| `runs/<run>/completed.json` | All configured tasks completed, including retained tasks after a resume |
| `runs/<run>/continuations/<timestamp>/` | Metadata and source hashes for an explicitly resumed run |
| `reports/metrics.csv`, `benchmarks/README.md` | Generated comparison views |
| `reports/report-manifest.json` | Report generator and input task-report hashes for the generated comparison |
| `reports/charts/model-profiles/<run>.png`, `*.svg` | One radar profile per model/run |
| `reports/charts/task-comparisons/<task>.png`, `*.svg` | One bar chart comparing models per task |
| `reports/charts/manifest.json` | Chart metric choices, values, versions and provenance hashes |
| `benchmarks/failed-runs/<attempt>/` | Startup or infrastructure failures retained outside scored benchmark results |
| `benchmarks/experiments/token-budget/<experiment>/` | Paired token-budget samples, responses, metrics and completion status |

Keep the source revision and files matching the recorded SHA-256 hashes.
Sources changed after inference are preserved in the runs' `source_snapshots/`
or the matching continuation's snapshots, retaining the files identified by
each environment's hashes. Generated tables and charts record their current
reporting source hashes separately, including the shared task definitions.
Benchmark code, configs and this method are included in source distributions;
downloaded data, run archives and generated charts remain repository artifacts.
The seed and greedy decoding do not promise bitwise-identical results across
hardware, dtype, kernels or dependency changes.

## Interpretation limits

These ordered prefixes provide a small repeatable baseline. They are not random
or representative estimates of full benchmark performance. Several VQA
questions, attributes or referring expressions share images. CIFAR-10 images
are only 32×32, COCO128 contains training images, and OVAD categorical scoring
differs from official attribute mAP. Point containment does not measure box
quality. Caption scores depend on tokenization and selected-reference IDF
scope. Compare models within the same task and dataset fingerprint, and retain
failure counts and measured-usage coverage alongside quality scores.
