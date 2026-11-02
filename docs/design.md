# VLM design decisions

Status: five-operator MVP with accepted geometry extensions pending implementation.
Recorded 2026-10-25; updated 2026-10-26.

## Goal

Expose model capabilities as explicit tasks with meaningful input and output
boundaries. Users choose a task and a model connection; the task prepares the
request and interprets the prediction. Structured output reliability and visual
accuracy are separate evaluation concerns.

The initial research included vision-language-programs and the supplied VLM
research report. This document records the resulting design rather than
treating the report as an API specification. In particular, boxes and instance
masks may overlap, VAW means Visual Attributes in the Wild, and ordinal ratings
are distinct from continuous scores.

## Task catalog

Only the first five rows are implemented. The remaining contracts describe
the intended direction, not exported placeholders or finalized signatures.

| Task | User intent and inputs | Result |
|---|---|---|
| Classify | Image, label choices, subject/property context | Exactly one label |
| ClassifyMultiLabel | Image, independently applicable label choices, context | Zero or more labels |
| Detect | Image, labels and context; exhaustive instance search | Collection of labeled boxes, possibly empty |
| Ground | Image, one target and context; resolve one match | One location (box; point mode planned), not found, or ambiguity |
| PredictAttributes | Image, named categorical attributes and their choices | One allowed value per attribute |
| Verify | Image and statement | Typed verification decision |
| Score | Image and an assessment criterion with a defined continuous scale | Value in that scale; not a confidence probability |
| Count | Image and target | Nonnegative count |
| ReadText | Image, optional reading constraints | Text and applicable layout information |
| AnswerQuestion | Image and question | Free-text answer |
| Caption | Image; predefined caption instruction | Factual caption without a required question |
| Embed | Image or text through a suitable representation model | Vector with modality and embedding-space identity |
| Compare | Compatible embeddings | Similarity; initially cosine |
| Rank | Image, candidates and criterion | All candidates in order, with explicit tie semantics |

Segmentation, semantic segmentation, pose/keypoints, oriented boxes, physical
measurement, ordinal ratings, document field extraction, scene graphs, and
video are deferred.

## Classification and attributes

Classify and ClassifyMultiLabel are separate operators, following the classical
distinction between an exclusive decision and independent label applicability.
A previous proposal merged them using a multi_label flag; that proposal is
superseded. Top-k alternatives are not multi-label predictions. Classical
softmax and independent sigmoid heads motivate this distinction but are not
required execution mechanisms for VLMs.

Labels can be names or name-to-description mappings. Names such as clean/dirty
do not identify the subject: context supplies that meaning, for example
"Assess cleanliness of the visible windows." Context should also specify an
aggregation rule where several subjects appear. An image-level task does not
implicitly become per-instance classification.

PredictAttributes is named after attribute prediction/recognition. "Feature
extraction" commonly means numerical representations, so ExtractFeatures would
be misleading for categorical attributes. Each attribute has one categorical
value in the MVP; binary attributes use explicit choices such as yes/no.

## Detection and grounding

The distinction is requested cardinality:

- Detect requests all matching instances of supplied labels.
- Ground requests resolution of one target.

How a target is expressed does not define the task. Both support caller-defined
vocabularies, descriptions, and context. Finding only one actual instance does
not turn exhaustive detection into grounding. Several plausible grounding
matches require an ambiguous result unless context resolves the choice.
Ground does not have a selection="all" mode.

For a busy scene, "person wearing a black backpack" can describe several
instances. Requesting every such person is Detect, with a descriptive label:

```python
Detect(
    backend,
    labels={"person_with_black_backpack": "A person wearing a black backpack"},
    context=(
        "Find every matching person. Return a box around the whole person, "
        "not just the backpack. Include partially visible people when the "
        "backpack criterion is visually supported."
    ),
)
```

Each matching person receives a separate box with the same label. The result
can be empty, contain one instance, or contain several. Descriptions may include
attributes and relationships; labels need not be bare object categories.
Localizing backpacks themselves would require a different description and box
scope. A dark shirt or a backpack of indeterminate color does not establish
that a person meets the black-backpack criterion.

Ground with the same description requests one person and should report
ambiguous if several people satisfy it. To resolve one, supply a distinguishing
spatial or relational description, or an explicit selection rule such as
"the leftmost person wearing a black backpack." Do not silently select the
first match or enclose several people in one box. Returning ambiguity is a
valid task outcome, separate from malformed output or execution failure.

### Visual example: Detect versus Ground

The following experiment used the same **"a person wearing a black backpack"**
criterion for both tasks on a 1920×1080 image. Detect requested every matching
person; Ground requested one. Both requested boxes around whole people.
Rows are Qwen3 VL 235B, Claude Sonnet 4.6 and Nemotron Nano 12B v2 VL;
columns are Detect (left) and Ground (right).

![Detect versus Ground for people wearing black backpacks](assets/backpack-detect-vs-ground.jpg)

| Model | Coordinate scale | Detect: all matches | Ground: one match |
|---|---|---|---|
| Qwen | normalized | Accepted empty collection | Selected the central backpack wearer |
| Sonnet | normalized | Rejected: normalized x coordinates exceeded 1 | Rejected: prose before JSON, although the trailing object declared ambiguous |
| Nemotron | 1000 | Rejected: malformed JSON | Selected a different person lower in the image |

These are actual model responses, not idealized examples of successful
exhaustive detection. An empty collection is structurally valid but does not
establish absence; rejected outputs have no accepted boxes to draw. The
comparison illustrates the different requested cardinalities and observed
behavior. It does not establish either task's accuracy. Each condition had one
independent call, temperature 0, max_tokens 4096, prompted structured output and
zero correction retries. The later output-mode experiment below explicitly
allowed choosing any matching person.

YOLO-World fits Detect because it predicts matching instances. SAM-style models
need capability-specific treatment: point/box-prompted masks, automatic mask
proposals, and text concept segmentation are different capabilities. They
should map to the corresponding task contract when implemented; model family
names are not sufficient capability declarations.

The current MVP requests normalized xyxy geometry and converts it to input-image
pixels. The accepted box configuration extension below adds explicit coordinate
formats for Detect and Ground. Overlapping instances are valid.

### Box configuration

Decision: accepted design; implementation pending. The existing constructors
currently use normalized model coordinates. Their planned signatures are:

```python
Detect(model, labels, *, context=None, box_scale="normalized")
Ground(model, target, *, context=None, box_scale="normalized")
```

The planned Ground signature is extended with a geometry selector in
[point grounding](#point-grounding). The signature above describes its box path.

box_scale belongs to the operator's box configuration. Users select the
model-facing coordinate convention based on evaluation of their chosen model.
The same setting and semantics apply to both operators. The default is
"normalized".

Let W and H be the width and height of the ImagePayload submitted to the task.
All modes use xyxy boxes with numeric coordinates; fractional values are valid
even in pixel mode. Coordinate limits describe image boundaries, so a box may
end at W or H.

| box_scale | Model-facing bounds | Conversion to public pixel coordinates |
|---|---|---|
| "normalized" | x and y in [0, 1] | x_pixel = x * W; y_pixel = y * H |
| "1000" | x and y in [0, 1000] | x_pixel = x * W / 1000; y_pixel = y * H / 1000 |
| "pixels" | x in [0, W]; y in [0, H] | Use coordinates directly |

For "1000", each axis spans the full image independently: 1000 identifies
the right boundary for x and the bottom boundary for y. For "pixels", bounds
are determined from each call's image dimensions.

#### Task constraints and validation

The task and its configuration define the expected model response. The selected
box_scale must control both the generated coordinate instruction and the response
schema bounds. The task supplies the corresponding validator through
ResponseSpec; the backend executes that specification. The task also validates
returned data at its own boundary before conversion.

- Detect validates allowed class names, the detection collection, box fields,
  configured coordinate bounds and geometry. An empty collection is valid.
- Ground validates its status and box consistency, plus the same configured
  coordinate bounds and geometry when a box is present. A box is required
  exactly when status is found.
- Every box requires finite numeric coordinates, x1 < x2 and y1 < y2. All
  coordinates must lie within the bounds of the selected mode.
- Unsupported box_scale values are invalid task configuration. Malformed or
  task-invalid model responses follow the existing InvalidModelResponseError
  and configured correction-retry behavior.

Conversion follows the configured scale exactly. Out-of-range coordinates are
rejected; validation does not guess a scale, clip boxes, swap coordinates or
repair malformed output. Values within the bounds can still describe the wrong
location or reflect unintended units, so visual accuracy requires evaluation
after conversion.

#### Public results and model integration

DetectionResult and GroundingResult in box mode expose floating-point xyxy
boxes in input-image pixels, with image_size = (W, H). The model-facing scale
does not alter the public box representation or the task's cardinality semantics.
The planned point mode adds a distinct point representation described below.

The backend remains shared across tasks and receives the task-generated
instruction, schema and validator. Coordinate semantics and conversion belong
to the task; adapters do not select a scale or implement task-specific geometry
parsers.

The three-model experiment on the cat image motivates making this configuration
explicit: "1000" improved Nemotron's accepted predictions and matching, while
the other models did not show the same benefit. Qwen's malformed responses also
showed that coordinate scale and output-format reliability are separate concerns.
These observations support evaluating the setting per model; they do not define
an automatic model-to-scale mapping.

#### Grounding scale experiment: central remote

Recorded 2026-10-26. We requested **"the remote control near the middle of the
image, immediately to the left of the right-hand cat"**, with context
"Resolve the described remote as a single whole object."

The original image was 640×480, with no cropping or resizing. Each of three
models received one independent request per scale (nine calls), using Bedrock
in ap-northeast-1, prompted structured output, temperature 0, max_tokens 4096,
and zero correction retries. Target, context and non-coordinate instructions
were held constant; coordinate wording and schema bounds followed the selected
mode. An example-local bridge exercised the planned configuration; the public
box_scale constructor option remains pending implementation.

The reference was the existing YOLO prediction for the intended remote:
pixel xyxy (334.117, 76.514, 370.768, 187.507). Reference coordinates were used
only for evaluation. The table reports box IoU; the matching threshold was 0.5.

| Model | normalized [0, 1] | [0, 1000] | pixels |
|---|---|---|---|
| Qwen3 VL 235B (`qwen.qwen3-vl-235b-a22b`) | 0.910 | 0.894 | Invalid JSON |
| Claude Sonnet 4.6 (`jp.anthropic.claude-sonnet-4-6`) | 0.287 | 0.121 | 0.414 |
| Nemotron Nano 12B v2 VL (`nvidia.nemotron-nano-12b-v2`) | 0.511 | 0.608 | 0.000 |

Four conditions met the threshold. All eight accepted responses reported found;
none reported not_found or ambiguous. Qwen's pixel response failed JSON parsing.
Nemotron's pixel response passed structure, geometry and range validation but
placed its box elsewhere: (522, 157, 562, 382). Those values resemble a
1000-scale box yet also fit the image's pixel bounds. Validation cannot prove
that the model used the requested units, and the task must not infer different
units from an accepted response.

These observations support explicit scale selection and evaluation per model
and task. Qwen worked best here with normalized coordinates; Nemotron improved
with 1000-scale coordinates. Sonnet remained below the matching threshold in
every mode. This single image and one call per condition measure agreement
with a YOLO prediction, not ground-truth accuracy or repeatability.

The [full protocol and raw results](https://github.com/trained-by-humans/vlm-tasks/blob/main/experiments/archive/vlm_remote_grounding/README.md)
include prompts, per-model responses, box coordinates and summaries. Use the
[examples](../examples/README.md) for new runs.

### Point grounding

Decision: accept a point output option for Ground; implementation pending.
The proposed constructor combines the existing box configuration with an
independent choice of output geometry:

```python
Ground(
    model, target, *, context=None,
    geometry="box", box_scale="normalized",
)
```

- geometry="box" is the default and requests one bounding box. box_scale
  selects normalized, 1000 or pixel coordinates as specified above.
- geometry="point" requests one point inside the selected target, preferably
  near the center of its visible body. For a person, prefer the visible torso.
  It identifies a location without estimating the object's extent.
- The initial point mode uses normalized model coordinates x and y in [0, 1],
  relative to the full image. The origin is the top-left corner; x increases
  rightward and y downward. Return public floating-point input-image pixels:
  x_pixel = x * W and y_pixel = y * H.
- box_scale applies only to box geometry. Point mode uses its fixed normalized
  convention; reject a nondefault box_scale with geometry="point" so a caller
  does not mistakenly expect scaled point coordinates. Additional point scales
  can be evaluated separately before being added to the interface.
- The task predicts the point from the image and target description. No input
  point hint, segmentation pass or inferred bounding box is part of this mode.
  Detect retains its collection-of-boxes contract.

#### Response contract and validation

The planned public GroundingResult carries status, geometry, box, point and
image_size. A Point value has numeric x and y in input-image pixels. For found,
exactly the selected geometry is present: box for box mode, point for point
mode. For not_found or ambiguous, both are null. All public types remain plain
framework-independent values. This extends the current box-only result; these
new fields and Point are not yet implemented or exported.

The model-facing point schema requests status and point (an object with x and
y, or null), without box fields. A point is required exactly when status is
found. Coordinates must be finite and within [0, 1]; reject extra fields,
invalid status/point combinations and out-of-range values. The task owns the
instruction, schema, validation and pixel conversion; the shared backend is
unchanged. Unsupported geometry values fail task configuration. Correction
budgets and invalid-response errors follow the existing task policy.

Range validation does not prove that a point is inside the right object.
Evaluate point location against target masks or point annotations where
available, separately from box IoU. A point does not provide a box or mask.

### Visual example: effect of geometry and coordinate scale

This experiment repeated grounding of **"a person wearing a black backpack"**
with all three models and all four output modes. Each request explicitly
allowed selecting any one matching person. All twelve were fresh calls with
the same criterion and selection rule; box modes requested the whole person,
and point mode requested an interior point, preferably near the torso center.

Rows are Qwen, Sonnet and Nemotron. Columns are normalized box, 1000-scale box,
pixel box and normalized point. Yellow rectangles are accepted box predictions;
red markers are predicted points. No prediction is drawn for a rejected output.

![Grounding with three box scales and a predicted point](assets/backpack-grounding-modes.jpg)

| Model | Normalized box | 1000-scale box | Pixel box | Normalized point |
|---|---|---|---|---|
| Qwen | Closely encloses central backpack wearer | Closely encloses same person | Rejected: list under x1 instead of four numeric box fields | Inside the same person's backpack/torso |
| Sonnet | Shifted upward; incomplete | Mostly floor | Misplaced | Near upper-left head boundary; misses torso preference |
| Nemotron | Encloses lower person; black backpack uncertain | Narrower box on same person | Misplaced | On floor just left of central person's head |

All three point responses passed validation; eleven of twelve responses were
accepted overall. Qwen's point was a useful localization, while the other
points were near the head boundary or outside the person. Point output is a
useful alternative with a simpler response, and scale affected box placement
and format compliance. This one call per condition does not show that points
are universally more accurate. There are no ground-truth backpack annotations
or quantitative accuracy claims for this scene.

Settings were Bedrock in ap-northeast-1, prompted output, temperature 0,
max_tokens 4096 and zero correction retries, on the full 1920×1080 image without
task-side cropping or resizing. Model IDs match the central-remote experiment.
The experiment used local task preparation and schemas without changing the
public package. Exact prompts, coordinates and responses are recorded in the
[Detect versus Ground archive](https://github.com/trained-by-humans/vlm-tasks/blob/main/experiments/archive/vlm_backpack_people/README.md)
and [grounding modes archive](https://github.com/trained-by-humans/vlm-tasks/blob/main/experiments/archive/vlm_backpack_grounding_modes/README.md).
The two comparison figures are also available under `docs/assets/`.
See the [experiment index](https://github.com/trained-by-humans/vlm-tasks/blob/main/experiments/README.md) for all recorded runs.

## Public and model-facing boundaries

    ImagePayload
        -> task preparation (context, choices, instruction, response schema)
        -> ModelRequest
        -> ModelBackend
        -> ModelResponse
        -> task validation and conversion
        -> dedicated result dataclass

Tasks share a communication envelope. Request content and output schemas vary
by task; adapters do not dispatch on task names or own task-specific parsers.
ResponseSpec carries a JSON Schema and a validator; ModelResponse contains
plain structured data. These integration types contain no PydanticAI types.

Final results are dedicated dataclasses, not TensorRegistry values. Registries
serve mutable tensor processing; these predictions are finalized outcomes.
Optional scores must have a documented meaning and source. The MVP does not
request self-reported confidence or invent probabilities.

The initial wish for nullable constructor settings and per-call overrides is
deferred. The current Pipeline requires fixed positional signatures. The MVP
therefore uses constructor-only configuration and one ImagePayload argument,
without changing core dispatch. All operators retain the same public shape
if the execution framework changes.

## Model communication alternatives

| Route | Examples | Integration considerations |
|---|---|---|
| Direct local Python runtime | Transformers, vLLM engine, SGLang engine, MLX-VLM | Requires a runtime bridge around already loaded objects |
| Local inference server | vLLM, SGLang, llama.cpp, Ollama, LM Studio | Often exposes OpenAI-compatible chat; native APIs also exist |
| Hosted inference | HF Inference Providers, Together AI, OpenRouter | Model catalog, image support and structured-output support vary |
| Native remote API | OpenAI Responses/Chat, Anthropic Messages, Gemini, Bedrock Converse | Distinct API formats and authentication |

These are eligible routes, not a compatibility certification or a ranking of
deployment popularity. The MVP uses the routes supported by a supplied
PydanticAI Model. For directly loaded Transformers weights, the package provides
`TransformersModel`, a PydanticAI adapter wrapping caller-owned weights and a
multimodal processor. Model loading, device placement and authentication remain
application concerns. See the [Transformers guide](guides/huggingface-transformers.md).

A communication implementation usually corresponds to an API/runtime; each
instance selects a particular model. Qwen and Gemma can share an integration
when their runtime provides the same API. A shared generation interface does
not establish accuracy or useful aligned embeddings. Ordinary instruction
checkpoint hidden states are not automatically equivalent to trained retrieval
embeddings such as Qwen3-VL-Embedding.

## PydanticAI, Pydantic, BAML, and LiteLLM

| Choice | Useful contribution | Decision |
|---|---|---|
| Pydantic | Schema generation and data validation | Used privately for internal wire schemas |
| PydanticAI | Provider integration, image messages, output modes and correction execution | MVP execution engine, behind PydanticAIBackend |
| BAML | Typed task functions, dynamic types, schema-guided parsing and modular request/parse APIs | Design reference; no mandatory DSL or generated clients |
| LiteLLM | Common calling interface across providers and proxies | Eligible future integration; no additional MVP dependency |

PydanticAI's direct API supplies requests without the Agent's full structured
parsing/retry behavior. The MVP uses Agent internally for structured execution.
Its StructuredDict helper forwards a JSON Schema but does not itself validate
all constraints: the response validator must do so.

Borrowed practices: one response specification as the source of truth; native,
tool and prompted output paths; independently testable preparation/validation;
dynamic vocabularies; explicit correction budgets; and provider-independent
results. Formatting success does not prove that the right object was found or
that exhaustive detection was achieved.

The BAML modular request/parse and TypeBuilder references below document the
v0 generated-client system. The documentation identifies v1 as a separate
language in public beta; adopting it would require a separate version decision.

## MVP dependency and ownership decisions

- Use an explicit PydanticAIBackend wrapping an initialized PydanticAI Model.
- Keep all task-facing types independent of PydanticAI.
- Keep provider authentication, SDK clients, connection lifetime and model
  selection in application code. Already initialized Bedrock Runtime clients
  are supported through PydanticAI's BedrockProvider.
- Use pydantic-ai-slim 2.61.0, without mandatory provider SDK bundles.
- Require Python 3.11+ only for VLM; core stays independent.
- Default to prompted output and zero correction retries. Native/tool modes
  are explicit options for suitable models/APIs.
- Separate correction retries from SDK transport retries; do not hide either
  when benchmarking.
- Use fresh histories for every prediction even when sharing a backend.

Earlier proposals for direct PydanticAI Model arguments, provider-string
shortcuts, custom callable bridges, automatic loading, and per-model task
bindings are not part of this MVP. Native embeddings and tensor predictions
must not be forced through a text-generation path in a later implementation.

## Evaluation roadmap

Evaluate checkpoint + task instruction/schema + adapter/backend settings +
output mode + retry budget + dataset/protocol. Record prompt/parser and
dependency versions. Unit tests establish contracts, not model proficiency.

Starting points include standard image classification datasets; COCO for
detection; RefCOCO-style data for grounding; VAW/OVAD for attributes; KonIQ-10k
for a future quality score; ICDAR/IAM for OCR; VQAv2/GQA for QA; and
Flickr30k/COCO retrieval for embedding comparison. Grounding evaluation needs
explicit absent and ambiguous cases in addition to successful references.
Use task-appropriate accuracy, precision/recall, IoU/mAP and attribute metrics.
Calibration needs separate evidence. Dataset ingestion and live comparative
benchmarks are future work.

## Sources

Reviewed 2026-10-25:

- [Vision-language programs](https://github.com/ml-research/vision-language-programs)
- [PydanticAI models and providers](https://pydantic.dev/docs/ai/models/overview/)
- [PydanticAI structured output](https://pydantic.dev/docs/ai/core-concepts/output/)
- [PydanticAI direct requests](https://pydantic.dev/docs/ai/core-concepts/direct/)
- [PydanticAI multimodal inputs](https://pydantic.dev/docs/ai/core-concepts/input/)
- [PydanticAI Bedrock clients](https://pydantic.dev/docs/ai/models/bedrock/)
- [PydanticAI HF inference](https://pydantic.dev/docs/ai/models/huggingface/)
- [PydanticAI slim installation](https://pydantic.dev/docs/ai/overview/install/#slim-install)
- [BAML modular API](https://docs.boundaryml.com/guide/baml-advanced/modular-api)
- [BAML dynamic types](https://docs.boundaryml.com/ref/baml_client/type-builder)
- [LiteLLM vision](https://docs.litellm.ai/docs/completion/vision)
- [SGLang vision API](https://docs.sglang.io/docs/basic_usage/openai_api_vision)
- [vLLM serving](https://docs.vllm.ai/en/latest/serving/online_serving/)
- [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
- [Ollama compatibility](https://docs.ollama.com/api/openai-compatibility)
- [MLX-VLM](https://github.com/Blaizzy/mlx-vlm)
- [LM Studio compatibility](https://lmstudio.ai/docs/developer/openai-compat)
- [Qwen3-VL embeddings](https://github.com/QwenLM/Qwen3-VL-Embedding)
