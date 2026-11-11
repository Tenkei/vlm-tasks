# Configure and run tasks

Choose a task, provide its labels or target, and add context to describe what
the model should assess. Start with an initialized
[model backend](model-backends.md); see the [API reference](reference.md#operators)
for signatures and result types.

## Images

Pass a local image path directly:

```python
from ml_pipes.vlm import Classify

task = Classify(backend, labels=["indoor", "outdoor"])
result = task("image.jpg")
print(result.labels)
```

`pathlib.Path` works too. Paths are loaded and decoded to BGR images;
missing files raise `FileNotFoundError`, and undecodable files raise `ValueError`.
An existing `ml_pipes.vision.ImagePayload` is used without reloading.

Payloads must contain nonempty uint8 arrays: HWC RGB/BGR with three channels,
or HW GRAY. Convert normalized floats, CHW arrays, or alpha images before
calling a task. Each call handles one image.

## Labels, descriptions, and context

Use a list of distinct label names, or a mapping with descriptions to clarify
what each label means. Context identifies the subject, property, and any
aggregation rule:

```python
task = Classify(
    backend,
    labels={"clean": "No visible dirt", "dirty": "Visible dirt or stains"},
    context="Assess the chair, ignoring the surrounding room.",
)
```

`Classify` selects one label; `ClassifyMultiLabel` selects all applicable labels.
`Detect` finds matching object instances and returns labeled boxes. For the
design distinction between finding all instances and locating one target, see
[Detection vs grounding](design.md#detection-vs-grounding).

Supply `Classify` and `Detect` labels at construction or with each call:

```python
task = Classify(backend)
result = task(("image.jpg", ["indoor", "outdoor"]))
```

Per-call labels replace the defaults for that request without changing stored
labels or context. [Validation](validation.md) uses dataset labels.

## Grounding targets and geometry

Set a default target, then override it for individual requests:

```python
from ml_pipes.vlm import Ground

task = Ground(backend, target="the leftmost cat")
result = task("image.jpg")
result = task(("image.jpg", "the person on the right"))
```

Or omit the default and supply a target with every call:

```python
task = Ground(backend)
result = task(("image.jpg", "the leftmost cat"))
```

Tuple targets must be nonempty strings, even when a default exists. Overrides
do not change the stored target or context. Missing or invalid targets raise
`ValueError` before inference.

Use `geometry="point"` to request an interior point instead of a box:

```python
task = Ground(backend, target="the leftmost cat", geometry="point")
result = task("image.jpg")
print(result.status, result.point)
```

A found result contains only the requested geometry. `not_found` and `ambiguous`
results have neither a box nor a point. Points are requested near the visible
body center (the torso for people). See the
[geometry comparison](design.md#geometry-and-coordinates).

## Attribute choices

Map each attribute name to its allowed values or value descriptions:

```python
from ml_pipes.vlm import Attributes

task = Attributes(backend, attributes={"material": ["wood", "metal"]})
result = task("image.jpg")
print(result.attributes)

result = task(("image.jpg", {"color": {"blue": "A blue surface", "red": ""}}))
```

A per-call mapping replaces the complete attribute definition for that request;
it does not merge with the defaults. Stored choices and context stay unchanged,
including after a failed call. Empty or invalid overrides raise `ValueError`.

With `Attributes(backend)`, direct calls must supply `(image, attributes)`.
[Validation](validation.md#attributes-ovad) can discover choices
from the dataset instead.

## Descriptions and questions

Use `Describe` for a factual natural-language description. Context controls the
subject, focus, language, and level of detail; the default asks for a concise
description of salient visible content:

```python
from ml_pipes.vlm import Answer, Describe

describe = Describe(backend, context="Describe visible damage to the chair.")
result = describe("chair.jpg")
print(result.description)
```

Use `Answer` for an explicit question. Set a default question or supply one per
call as `(image, question)`, following the grounding input pattern:

```python
answer = Answer(backend, question="What material is the chair made of?")
result = answer("chair.jpg")

answer = Answer(backend)
result = answer(("street.jpg", "What is the cyclist carrying?"))
print(result.status, result.answer)
```

Questions must be nonempty strings. A missing question or invalid override
raises `ValueError` before inference. Overrides preserve stored question and
context, and independent calls have no conversation history.

An `answered` result contains nonempty answer text. An `unanswerable` result has
`answer=None`, for example when evidence is insufficient, text is unreadable, or
a referent remains ambiguous. “No” and “zero” are answers, not abstentions.
Malformed responses and execution failures raise the usual task errors.

Answer uses visual evidence, relevant general knowledge, and supplied context.
Use context to specify a stricter evidence policy or answer language when needed.
Output wording is preserved; benchmark normalization belongs in a validator.
For allowed labels, categorical attributes, or coordinates, use the corresponding
task contract. Descriptions and free-text answers do not enforce those formats.

`Describe.val()` defaults to
[COCO Captions 2017](validation.md#describe-coco-captions).
`Answer.val()` defaults to [VQA v2](validation.md#answer-vqa-v2), with questions
supplied by the dataset.

## Coordinates and results

Task results use original-image pixels, with the origin at the top left.
Boxes use `xyxy`, and `image_size` is `(width, height)`. The framework converts
normalized model coordinates without clipping or guessing another scale.
Coordinates must be finite and inside the image; boxes must have positive area.
Detection boxes may overlap.

Valid structure does not establish visual accuracy: an empty detection list or
a box on the wrong object can still be valid. Use [validation](validation.md)
to measure quality on labeled data.

## Use in ml-pipes pipelines

Tasks also work as pipeline operators:

```python
from ml_pipes.core import Pipeline

task = Ground(backend, target="the leftmost cat")
pipeline = Pipeline([task])
pipeline.validate(strict=True)
result = pipeline("image.jpg")
result = pipeline(("image.jpg", "the person on the right"))
```

The pipeline passes a tuple as one argument. Grounding without a default target
requires `(image, target)`; attributes without default choices requires
`(image, attributes)`. Answer without a default question requires
`(image, question)`. With defaults, these tasks also accept an image alone.
Describe accepts an image alone.

## Implementing a task

`_Task` in `ml_pipes.vlm.ops` is the internal shared base; its extension
interface may change. Subclasses follow this convention:

| Member | Subclass responsibility |
|---|---|
| `__init__` | Call `_configure(model, context)`, then set `_instruction` and `_spec` (`ResponseSpec`) defaults |
| `__call__` | Validate task inputs, load an `ImagePayload`, call `_predict()`, and convert its validated mapping into a result |
| `resolve_contract` | Declare input types when they depend on instance configuration |

An instruction or response specification may default to `None` when supplied
per call. Resolve it before calling `_predict(image, instruction=...,
response_spec=...)`. Keep overrides local to the call. `_predict()` appends
context and validates the model response against the selected specification.

Decorate concrete tasks with `@Operator` and annotate their input and result
types for pipeline integration. New task types use
`.val(validator=MyValidator, ...)` for evaluation; see
[custom validation](validation.md#custom-loaders-and-validators).
