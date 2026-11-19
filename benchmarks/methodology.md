# Benchmark methodology

This protocol compares configured VLM task pipelines on frozen validation
samples. Results live in the [benchmark report](README.md); its
[Table field guide](README.md#table-field-guide) explains metric labels,
reported columns, units and normalization.
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
│   ├── model-comparisons/      # On-demand local reports (gitignored)
│   └── charts/
│       ├── model-profiles/    # One radar profile per model
│       └── task-comparisons/  # One bar chart per task
├── methodology.md
└── README.md                  # Generated model and task overview
```

One-off investigations and token-budget outputs live in the repository-root
`experiments/` directory, separately from published benchmark runs.

Python modules use `snake_case` and describe their action or responsibility.
Model configs use `<model>-<backend>.json`, including model size where relevant;
variants append a setting such as `-retries-<limit>` or `-native`.
Report folders match the overview's **Model profiles** and **Task comparisons**
sections. Recorded run IDs, task IDs and original metadata retain their identity
when directories move. Prepared datasets are cached in
`.cache/prepared-datasets-100/`; the suffix identifies the configured sample count.

## Benchmark parameters

Each run uses a JSON configuration from [`configs/`](configs/). The sample
count, output budget, output mode and retry allowance are configurable; the values below
describe the published runs rather than requirements of the protocol.

| Parameter | Behavior | Published configurations |
|---|---|---|
| `model` and backend settings | Select the model and execution environment; record revisions, hardware or provider routing where available | Local Transformers and AWS Bedrock; see each config |
| `tasks` | Select task variants to evaluate | Seven variants across six vision tasks |
| `samples_per_task` | Positive sample count, frozen during preparation; the run must match its prepared manifest | 100 per variant |
| `max_new_tokens` | Maximum generated tokens per request, including each correction attempt | 4,096 |
| `output_mode` (Bedrock) | `prompted` supplies schema instructions in the prompt; `native` requests provider JSON Schema output. Defaults to `prompted` | Prompted baselines and a separate native Qwen3-VL-235B run |
| `retries` | Nonnegative maximum correction attempts after the initial request; defaults to `0` when omitted | `0`, plus a separate Qwen3-VL-235B run with `2` |
| Generation settings | Control decoding where the backend supports it; record differences such as provider-controlled sampling | Local greedy decoding with seed 42; hosted settings vary by model |
| `thinking` (Bedrock, optional) | Pass PydanticAI's reasoning-effort setting to the provider; reasoning counts toward `max_new_tokens` | Claude Fable 5.2: `high` adaptive effort |
| `match_iou`, `tokenizer_version` | Fix detection matching and caption tokenization for scoring | `0.5` and Stanford PTBTokenizer `3.4.1` |

The Transformers runner uses prompted JSON output. The Bedrock runner accepts
prompted or native output; see [Native JSON Schema runs](#native-json-schema-runs).
The library's output modes are described in
[output modes and retries](../docs/model-backends.md#output-modes-and-retries).
Actual generation configuration, processor or provider settings, output
instructions and retry allowance are saved in each run's `model.json`.

For a model comparison, keep the prepared data, task instructions, response
schemas, scoring, token budget and retry allowance the same, and record any
differences. To study a parameter such as retries or token budget, change that
parameter in a separate config and save a separate run. Output-mode comparisons
also change how the schema reaches the model and which constraints the provider
enforces. Retry and token-budget comparisons evaluate different inference budgets
as well as task quality.

A fixed output cap bounds repetitive generation, but can truncate valid long
answers. Increasing it is a separate experiment; see
[token-budget reproduction](#double-the-budget-for-capped-responses).
Correction retries keep the same cap and usually do not resolve cap failures;
see [Correction retry runs](#correction-retry-runs) for when they are useful.

## Fixed protocol

Select the first `samples_per_task` eligible rows in each default loader's
source order, with no shuffle or prediction-based filtering. The local seed
controls model setup; selection is an ordered prefix and has no sampling seed.
Both grounding modes use exactly the same expressions. The published runs
use 100 samples for each of seven variants, or 700 evaluated task samples per
run, with repeated images possible within and across tasks.

| Task | Dataset and selection unit | Split |
|---|---|---|
| Classify | CIFAR-10, images; pinned dataset revision `0b2714987fa478483af9968de7c934580d0bb9a2` | `val` aliases test |
| Detect | COCO128, images; all 80 labels | `val` uses the same training images as `train` |
| Ground, box | RefCOCOg UMD, referring expressions | val |
| Ground, point | Same RefCOCOg expressions | val |
| Attributes | OVAD, eligible object crops; all 22 categorical groups | `val` aliases benchmark |
| Answer | VQA v2, questions with all ten answer votes | val |
| Describe | COCO Captions 2017, images with all reference captions | val |

Task settings use library defaults and no extra context. Dataset vocabularies
provide classification/detection labels and attribute choices. Grounding
receives each expression, and Answer receives each question. Caption scoring
uses Stanford PTBTokenizer 3.4.1 with its pinned SHA-256; CIDEr document
frequencies are computed over the selected images. Detailed metric and
dataset adaptation rules are in [validation](../docs/validation.md).

`ClassifyMultiLabel` has no built-in validation dataset and is excluded. Planned
tasks such as ReadText are excluded.

Each sample starts a fresh conversation. The local model is loaded once, and
each run executes its configured tasks and samples sequentially. There is no
separate warmup; the first inference call is included in timing.

## Failures and usage

Model response failures remain in all applicable denominators; validation
uses its existing failure policy after any correction attempts are exhausted.
Grounding abstentions score zero against positive targets; VQA abstentions
score zero. Failed captions use empty text with references retained.
Detection failures produce no boxes, so they add false negatives for targets
but no false positives. Data/setup/programming errors stop the run and do
not produce a completed task report. The Bedrock runner also stops on
infrastructure failures, retaining its journal for investigation.

Both runners record model responses **before** PydanticAI parsing. Every
returned generation is saved, including invalid responses. The local runner
measures input from processed `input_ids` and output from generated token IDs,
including generated special tokens. The Bedrock runner records the provider's
usage counters, stop reasons and request IDs.

Each report records attempted, measured and unmeasured requests. A generation
error that prevents the adapter from returning usage is marked unmeasured;
partial token consumption cannot be recovered. Every initial and correction
attempt is counted, including attempts that fail. Local monetary cost is
`null`, since hardware and electricity costs are not measured.

Usage, estimated cost and inference duration include all correction attempts
and the conversation history sent with each one. Failed samples and samples
with unmeasured requests remain in the evaluation denominator; incomplete usage
or cost coverage makes the corresponding aggregates partial. Task reports
retain usage totals and the CSV retains precise values. The
[Table field guide](README.md#table-field-guide) defines the report's metric
labels, denominators, cap counting, units and formatting.

The validator times the whole task call, including image encoding,
preprocessing, generation, correction attempts and parsing. Dataset loading
and metric calculation have separate timings. Dataset preparation and shared
model load time are recorded separately. Hosted timing includes network and
service time. Batching, concurrency, cold downloads, peak memory and energy
are not measured.

## Native JSON Schema runs

Set `output_mode` to `"native"` in a Bedrock config to request provider-enforced
JSON Schema output through `outputConfig.textFormat`. Omitting it preserves
the prompted baseline. Native mode requires support from the exact model,
API route and installed adapter; a provider model card or SDK capability flag
alone may be inaccurate. Live image-input probes confirmed support for
`qwen.qwen3-vl-235b-a22b` in `ap-northeast-1` before this benchmark run.

Native runs use PydanticAI's provider schema transformer rather than the
prompted baseline's disabled transformer. It adapts the task schema to the
provider's supported subset; for example, coordinate bounds may become field
descriptions instead of enforced numerical constraints. The original task
validators still enforce all bounds, types, vocabulary and consistency rules.
Native output does not establish that a prediction is correct, and invalid
responses retain the normal scoring penalty. The same configured output cap
still applies, so a response can also be truncated.

Record this as a separate run because the output contract is delivered
differently. The runner saves the selected output mode and transformer in
`model.json`, the original task schema in every sample journal entry, and the
transmitted native `outputConfig` alongside each returned generation. The
report labels these profiles `(JSON Schema)`; retry allowance is an independent
parameter and defaults to zero. There is no dedicated warmup; measured request
latency includes any provider-side schema compilation incurred during the run.

The [Qwen3-VL-235B native config](configs/qwen3-vl-235b-a22b-bedrock-native.json)
uses the same frozen 100 samples per task variant, task instructions, original
schemas, scoring, 4,096-token cap and temperature zero as the prompted baseline,
with native output enabled and no correction retries. Hosted first-attempt
responses can vary between runs, even at temperature zero.

After preparing the data and signing in as described under
[Larger Qwen3 on AWS Bedrock](#larger-qwen3-on-aws-bedrock), reproduce it with:

```bash
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 .venv/bin/python -m benchmarks.src.run_bedrock \
  --config benchmarks/configs/qwen3-vl-235b-a22b-bedrock-native.json \
  --prepared benchmarks/.cache/prepared-datasets-100 \
  --output benchmarks/runs/qwen3-vl-235b-bedrock-native-reproduction
.venv/bin/python -m benchmarks.src.generate_charts
```

## Correction retry runs

Set `retries` in any Transformers or Bedrock benchmark config to allow
PydanticAI response correction. It is a nonnegative integer and defaults to
`0`: one initial request, with no correction attempts. A value of `2` permits
up to two additional attempts, or three model requests per sample. Retries can
help with invalid responses, but usually do not resolve output-token-limit
failures because **every attempt retains the same output cap**.

### How correction works

When parsing or response validation fails, the backend sends validation
feedback to the model in the same sample's conversation. The original image,
instructions and response schema remain available alongside previous outputs
and feedback. The first response that passes validation is scored and ends
the sample's correction loop; unused retries make no requests. A new sample
starts with a fresh conversation. The benchmark applies no additional JSON
repair, coordinate inference or schema relaxation.

If all attempts fail, the sample follows the ordinary validation failure
policy. The terminal exception describes the final failed attempt, as
documented in the [response error guide](../docs/model-backends.md#response-errors). The journal
retains every returned generation and the attempted-request count, including
failed attempts. Usage and timing cover the whole conversation; see the
[Table field guide](README.md#table-field-guide) for reporting definitions.
The [backend guide](../docs/model-backends.md#output-modes-and-retries)
describes the underlying retry API.

### When to use retries

Retries are most useful for occasional format or constraint violations that
the model can correct from specific validation feedback. Inspect saved failed
responses before choosing an allowance. The
[response error guide](../docs/model-backends.md#response-errors) explains the categories; their
implications for retries are:

- **`ResponseSyntaxError`**: a model can regenerate malformed JSON. Check for
  truncation or repetitive generation first, since those usually persist at
  the same token cap.
- **`ResponseSchemaError`**: feedback can help a model follow the required
  fields, nesting, types and item counts. The next response must satisfy the
  original schema; the benchmark does not reinterpret arrays as coordinates
  or infer missing fields.
- **`ResponseConstraintError`**: feedback can identify violations such as
  unsupported labels, out-of-range coordinates or inconsistent status/value
  combinations. A corrected response still needs to be scored for task
  quality; satisfying constraints does not establish that a prediction is
  correct.

A valid but incorrect label, answer or location does not trigger correction.
Valid abstentions also remain ordinary scored predictions. Ground-truth
targets and validation scores are never supplied as correction feedback.

For persistent truncation or token loops, investigate prompts, model behavior
or a separate token-budget configuration. A local Qwen3-VL-2B retry trial
showed little improvement while increasing token usage and latency; most
capped responses remained invalid after correction attempts.

Response correction is separate from SDK transport retries. Authentication,
connection and other execution failures are not response-validation errors;
the Bedrock benchmark stops on infrastructure failures and retains its
journal. SDK attempts are configured independently and fixed to one in the
published Bedrock configs.

Store a retry configuration as a separate full run using the same frozen
samples, first-attempt prompts, schemas and scoring. Changing the retry
allowance changes the inference budget. Hosted first-attempt outputs can
vary even at temperature zero, so differences from an earlier baseline may
also include initial-response variation. Inspect attempts within each saved
sample to see what correction changed.

### Example: Qwen3-VL-235B on Bedrock

The [two-retry configuration](configs/qwen3-vl-235b-a22b-bedrock-retries-2.json)
illustrates this parameter. It keeps the baseline's 100 frozen samples per
task, model settings, first-attempt prompts, nested response schemas, scoring
and 4,096-token limit per request, while setting `retries` to `2`. All seven
task variants are evaluated in a fresh run stored separately from the
zero-retry baseline.

See the [published overview table](https://github.com/Tenkei/vlm-tasks/blob/main/benchmarks/README.md)
and the dedicated [Qwen3-VL-235B (2 retries) profile](https://github.com/Tenkei/vlm-tasks/blob/main/benchmarks/README.md#qwen3-vl-235b-2-retries)
for results, and the [reproduction command](#reproduce-the-retry-example)
below to run the same configuration.

## Model profile charts

The [radar charts](README.md#model-profiles) show one profile per model/run.
Each axis uses the task's main metric from the report's overview table.
The [Table field guide](README.md#table-field-guide) explains these metrics.
Axes have identical order and a fixed radial range of 0–100, using recorded
scores multiplied by 100. There is no normalization to the best observed
model or to sample count, and failures retain their scoring penalties.

The chart metrics have a natural 0–1 range. Captioning uses bounded ROUGE-L;
CIDEr, which is not bounded to 0–1, remains in the tables. Each axis summarizes
a different task: equal radial values do not imply equal difficulty, and
polygon area is not an overall model score. Charts compare task quality;
use the tables to assess quality alongside failures and resource usage.

Model columns, profile grids, task bars and detail tables show the newest
completed runs first. Ordering uses `completed.json`'s recorded `completed_at`,
falling back to the latest task completion for unfinished or older runs.
Undated runs appear last; ties use the run ID. File modification times and
chart regeneration do not change the order. On-demand pair comparisons keep
the model order supplied in the command.

Colors come from the shared ten-color palette in
[`src/chart_style.py`](src/chart_style.py). The first eight bytes of a model ID's
SHA-256 checksum, interpreted as an unsigned big-endian integer modulo the
palette length, select its slot. Assignments are independent of run date,
sample count and retry allowance. Changing palette length or order changes
assignments. Profiles, task bars and pair comparisons use the same base
colors; if both comparison curves share a color, the second uses the next
palette entry, wrapping at the end. Manifests record resolved colors and
source hashes.

Generate PNG and editable SVG files using saved reports only:

```bash
uv pip install --python .venv/bin/python 'matplotlib==3.11.2'
.venv/bin/python -m benchmarks.src.generate_charts
```

The chart generator validates that all seven main metrics are present and
bounded, writes one profile per model/run and one comparison per task, and
refreshes the report. It performs no inference. `reports/charts/manifest.json`
records metric choices, exact scores, sample counts, Matplotlib/NumPy versions,
and source, report, run-completion metadata and figure hashes. Use those package versions to reproduce
the rendering. The report generator embeds charts only when their report
hashes and run order match the current results; regenerate charts after adding
or changing runs or their completion metadata. To refresh only tables and the field guide, run
`python -m benchmarks.src.generate_report`.

### Compare two models locally

Generate a dedicated report for two models by passing the names shown in the
benchmark charts. Run from the repository root, using the same Matplotlib setup
as above:

```bash
.venv/bin/python -m benchmarks.src.generate_comparison "Qwen3-VL-2B" "Qwen3-VL-235B"
```

The command reads saved results and writes only the selected pair's report to
`benchmarks/reports/model-comparisons/<run>--vs--<run>/`. Open its `README.md`
for the introduction and run configurations, followed by the overlaid radar
chart and task summary table. Expand **Breakdown** for full metrics and usage;
its fields follow the main report's
[Table field guide](README.md#table-field-guide). PNG and SVG images and a
provenance manifest are saved alongside it. These local outputs
are gitignored; the main benchmark README and charts are unchanged, and no
model inference is performed.

Use `--output /path/to/comparison` to choose another output directory, or
`--runs /path/to/runs` to read another set of recorded runs. Repeating a command
refreshes that pair's report. List the available names and run IDs with:

```bash
.venv/bin/python -m benchmarks.src.generate_comparison --list-models
```

Names are case-insensitive. Full model identifiers and individual profile chart
titles are also accepted. If a model has multiple recorded runs, pass an exact
run ID from the list to choose one. Unknown, ambiguous and duplicate selections
are rejected before report generation.

Overlays use the individual profiles' axis order, fixed scale, recorded scores
and base colors, without normalization between models. When both colors match,
the second curve uses the next palette entry. Both lines are solid and a
legend identifies each model. Polygon area is not an overall score. The table
uses the same formatting and usage accounting as the main report, with links
back to the selected source results. The manifest records selected runs, original
scores, rendering versions, source and input hashes, and generated file hashes.

### Task comparison charts

The [task comparisons](README.md#task-comparisons) show one horizontal bar
chart per task variant, using the same main metrics, model order and colors
as the model profiles. Bars use the same fixed 0–100 range (recorded metric
multiplied by 100); no scores are normalized against the observed models.

The chart manifest records each task's metric, original scores, displayed
model labels and filenames. The report pairs charts with full tables and
links both views through its overview. Shared field definitions live in the
[Table field guide](README.md#table-field-guide).

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

### Local Qwen3 baseline

The first published model is `Qwen/Qwen3-VL-2B-Instruct` from Hugging Face,
revision `89644892e4d85e24eaac8bacfd4f463576704203`. Its configuration is
[`qwen3-vl-2b-transformers.json`](configs/qwen3-vl-2b-transformers.json):
Transformers, Apple MPS, float16, greedy decoding (`do_sample=False`), seed 42,
4,096 new tokens per request and zero correction retries. The baseline host
is an Apple M1 Pro with 32 GiB unified memory running macOS 15.5. Exact
Python and package versions are saved with the run.

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
overwriting an existing run. For an interrupted run, see
[offline loading and resuming](#offline-loading-and-resuming). Existing
results are never silently reused as fresh inference.

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

### Offline loading and resuming

`--offline` on `run_transformers` restricts checkpoint and processor loading
to cached files. Caption scoring also uses a checksum-verified Stanford
tokenizer. The pinned datasets library checks its remote metadata even when
the JAR is cached, so `HF_HUB_OFFLINE` and `HF_DATASETS_OFFLINE` must remain
unset or zero for that step. Local model loading can still use `--offline`.

Both runners support `--resume` at a task boundary: rerun the original
inference command with the same config, prepared data and output directory,
adding `--resume`. They verify the saved configuration, dataset manifest,
completed sample indices and model settings. Complete tasks are retained;
nonempty journals for unfinished tasks prevent automatic resumption, so a
partly evaluated task is never silently repeated.

Original metadata is retained. `continuations/<timestamp>/` records the
continuation's environment, source hashes, model metadata and selected tasks,
and new task reports reference that environment. Scorer setup interruptions
before inference add no model requests or scored failures. The local runner
also supports `--tasks TASK ...` for a selected-task run in a fresh output
directory; retain artifacts from both invocations if using this alternative.

### Combine consecutive runs

[`combine_runs.py`](src/combine_runs.py) joins completed consecutive subsets
from local or hosted backends. It verifies that the parts cover the full
frozen suite in order, with matching images, targets, features, backend,
model and processor settings, routing where applicable, and scoring sources.
Only sample counts may differ between the run configs.

Saved responses are replayed through the public validators to recompute
metrics across the full suite, including failures. Scores are not averaged
across parts; caption IDF and corpus metrics are recomputed for the full
subset. No model requests are made. Usage totals and available estimated
costs are added, unavailable local costs remain unmeasured, and partial
estimates retain their incomplete coverage. Inference latency uses the
sample-weighted original live measurements, excluding replay time.

The output includes copies of the source runs under `segments/` and records
their hashes and index ranges in `assembly.json`. Pass source directories
in sample order and use a new output directory. The
[Fable pilot continuation](#continuing-the-pilot-without-repeating-requests)
below gives a concrete example; the same command accepts local runs.

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
and unmeasured usage; display conventions are defined in the
[Table field guide](README.md#table-field-guide).

### Reproduce the retry example

To reproduce the [Qwen3-VL-235B retry example](#example-qwen3-vl-235b-on-bedrock),
use the same environment, prepared datasets and AWS login as above, with its
separate config and a fresh output directory:

```bash
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 .venv/bin/python -m benchmarks.src.run_bedrock \
  --config benchmarks/configs/qwen3-vl-235b-a22b-bedrock-retries-2.json \
  --prepared benchmarks/.cache/prepared-datasets-100 \
  --output benchmarks/runs/qwen3-vl-235b-bedrock-retries-2-reproduction
.venv/bin/python -m benchmarks.src.generate_charts
```

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
generation; its benchmark therefore uses US East. Latency includes regional
network differences. Independent
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

These commands follow the shared [resume procedure](#offline-loading-and-resuming).
The tokenizer setup interruptions added no model calls or scored failures.

Kimi K3.1 reasons by default at the provider's `max` effort. These runs leave
model-specific reasoning defaults unchanged; raw reasoning text and total
output usage are recorded. Bedrock may not report a separate reasoning-token
counter even when reasoning text is present. Reasoning counts toward the common
output allowance. Do not interpret this comparison as
matching internal reasoning effort across models. Gemma uses temperature 0;
Kimi rejects the temperature field, so its config sets `null` to omit it and
uses provider-controlled sampling.

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

### Claude Fable 5.2 on AWS Bedrock

[`claude-fable-5.2-bedrock.json`](configs/claude-fable-5.2-bedrock.json)
uses 100 samples per variant, prompted JSON, zero correction retries and the
same 4,096-token output budget and scoring rules. The initial
[`pilot config`](configs/claude-fable-5.2-bedrock-pilot.json) uses the first
10 rows per variant; its standalone results remain under `experiments/`.

Fable is invoked through `global.anthropic.claude-fable-5-2` from Tokyo.
Temperature is omitted as required by the model. Adaptive thinking is always
enabled, with effort explicitly set to the documented `high` default;
reasoning shares the output allowance. The config records Global Standard
pricing, including reasoning in total output without charging it twice.

The [AWS model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-fable-5-2.html)
requires data retention mode `aws_review`. This is an account-wide opt-in;
the benchmark runner does not change it automatically. Confirm access before
inference. This run uses prompted schema instructions, matching the original
Qwen, Gemma and Kimi baselines.

For a fresh reproduction, use the normal runner on the full frozen dataset:

```bash
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 .venv/bin/python -m benchmarks.src.run_bedrock \
  --config benchmarks/configs/claude-fable-5.2-bedrock.json \
  --prepared benchmarks/.cache/prepared-datasets-100 \
  --output benchmarks/runs/claude-fable-5.2-bedrock-reproduction
.venv/bin/python -m benchmarks.src.generate_charts
```

#### Continuing the pilot without repeating requests

The recorded run retains the initial 10 responses per variant and evaluates
only rows 11–100 in the continuation: 70 existing requests plus 630 new ones.
Both parts use the normal Bedrock runner, with prepared dataset subsets and
configs that differ only in sample count. This is separate from `--resume`,
which continues between completed tasks and requires the original sample count.

The [combining procedure](#combine-consecutive-runs) recomputes all metrics
on the full 100 samples per variant and preserves both runs' usage and live
latency. To reproduce the offline assembly from the saved segment copies,
use a new output directory:

```bash
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 .venv/bin/python -m benchmarks.src.combine_runs \
  --segments benchmarks/runs/claude-fable-5.2-bedrock-20261119/segments/000 \
             benchmarks/runs/claude-fable-5.2-bedrock-20261119/segments/001 \
  --prepared benchmarks/.cache/prepared-datasets-100 \
  --output experiments/claude-fable-5.2-offline-reproduction/runs/claude-fable-5.2
```

### Double the budget for capped responses

After a local Transformers baseline completes, rerun samples that reached its
output cap with twice the configured allowance. Save the experiment under
the repository-root `experiments/` directory, using a fresh output path:

```bash
python -m benchmarks.src.run_token_budget \
  --baseline benchmarks/runs/qwen3-vl-2b-20261112 \
  --prepared benchmarks/.cache/prepared-datasets-100 \
  --output experiments/token-budget/reproduction --offline
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
  --experiment experiments/token-budget/reproduction \
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
| `benchmarks/configs/qwen3-vl-235b-a22b-bedrock-retries-2.json` | Example of a separate correction-retry configuration |
| `benchmarks/configs/gemma-3-27b-bedrock.json`, `benchmarks/configs/kimi-k3.1-bedrock.json` | Additional hosted model configurations, routing and pricing settings |
| `benchmarks/configs/claude-fable-5.2-bedrock.json` | Fable model configuration, required reasoning settings and recorded pricing |
| `benchmarks/src/prepare_datasets.py` | Dataset preparation and sample freezing |
| `benchmarks/src/task_registry.py` | Shared task definitions, datasets, metric labels and chart axes |
| `benchmarks/src/evaluation.py` | Shared validation, frozen-data verification, response recording and replay |
| `benchmarks/src/artifacts.py` | JSON artifacts, sample identities, content hashes and source provenance |
| `benchmarks/src/run_transformers.py` | Local model setup, inference and task reports |
| `benchmarks/src/combine_runs.py` | Verify consecutive local or hosted subsets and rescore their saved responses without inference |
| `benchmarks/src/usage.py` | Reconcile saved request usage and aggregate totals while preserving unavailable and partial costs |
| `benchmarks/src/generate_report.py` | Comparison table and wide CSV generation |
| `benchmarks/src/generate_charts.py` | Model radar profiles and task comparison charts from saved reports |
| `benchmarks/src/chart_style.py` | Shared palette and deterministic model color selection |
| `benchmarks/src/generate_comparison.py` | On-demand local report for two selected model runs |
| `benchmarks/src/run_token_budget.py` | Paired 2×-budget reruns of capped samples |
| `benchmarks/src/report_token_budget.py` | Report completed token-budget records after stopping inference |
| `runs/<run>/config.json` | Requested task, model, sample-count, token-budget and retry settings; omitted retries default to zero |
| `runs/<run>/manifest.json` | Dataset identities, targets and image hashes |
| `runs/<run>/environment.json`, `requirements.txt` | Python/platform/package versions, git revision and source hashes |
| `runs/<run>/model.json` | Actual model/processor/generation configuration |
| `runs/<run>/pricing-source.html`, `pricing-source.json` | Official rate source snapshot and hash when config rates override the pricing package |
| `runs/<run>/<task>.jsonl` | Ordered requests, raw generated text, usage, predictions and errors; join to manifest by index |
| `runs/<run>/<task>.json` | Full validation metrics, breakdowns, failures, timings and usage totals |
| `runs/<run>/completed.json` | All configured tasks completed, including retained tasks after a resume |
| `runs/<run>/continuations/<timestamp>/` | Metadata and source hashes for an explicitly resumed run |
| `runs/<run>/assembly.json`, `segments/` | Source index ranges, checksums and exact original runs when consecutive subsets are combined |
| `reports/metrics.csv`, `benchmarks/README.md` | Generated comparison views |
| `reports/report-manifest.json` | Report generator and input task-report hashes for the generated comparison |
| `reports/charts/model-profiles/<run>.png`, `*.svg` | One radar profile per model/run |
| `reports/model-comparisons/<run>--vs--<run>/` | Local README, overlay PNG/SVG and provenance for one selected pair (gitignored) |
| `reports/charts/task-comparisons/<task>.png`, `*.svg` | One bar chart comparing models per task |
| `reports/charts/manifest.json` | Chart metric choices, values, versions and provenance hashes |
| `experiments/token-budget/<experiment>/` | Local paired token-budget samples, responses, metrics and completion status |

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
