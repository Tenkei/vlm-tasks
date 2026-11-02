# Contributing

## Development setup

Use Python 3.11 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python -m examples.vlm.run_demo
```

The test extra includes Pillow for the Transformers adapter tests. These use
fake weights and a processor; PyTorch, Transformers, provider credentials and
model downloads are not required. Provider SDKs and local inference runtimes
are installed separately for live examples.

## Changes

Keep changes focused. Update tests for behavior changes and keep the README,
`docs/reference.md`, `docs/design.md` and affected examples aligned with the
public `ml_pipes.vlm` API. Run tests without network access or live providers.

Documentation is Markdown, read directly in the repository. Preserve relative
links and documentation assets when moving files.

## Packaging

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

See [RELEASING.md](../RELEASING.md) for staging and publication.
