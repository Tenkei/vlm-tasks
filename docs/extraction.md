# Extraction provenance

The initial standalone tree was extracted from the local InferencePipeline
checkout at commit `7f24815aa13f16825e309b4557cae00b919943b7`.
`ml-pipes-supervision` and `ml-pipes-ultralytics` supplied the standalone
packaging and release conventions.

## File mapping

| Original path | Standalone path |
|---|---|
| `packages/vlm/src/ml_pipes/vlm/` | `src/ml_pipes/vlm/` |
| `packages/vlm/tests/` | `tests/` |
| `packages/vlm/docs/` | `docs/` |
| `packages/vlm/README.md`, `LICENSE`, `pyproject.toml` | Repository root |
| `examples/vlm/` | `examples/vlm/` |

The examples include Bedrock and directly loaded Transformers runners, the
synthetic demo, task JSON, adapter implementation, adapter tests and guides.
Both tracked design comparison images are included under `docs/assets/`.
The adapter tests remain beside the example and are collected by the default
pytest configuration.

## Adaptations

- Preserve the `ml_pipes.vlm` namespace and all existing public exports.
- Keep the initial version `0.1.2rc1` and Python 3.11 minimum.
- Replace workspace framework candidate pins with published dependencies
  `ml-pipes-core>=0.1.3,<0.2.0` and `ml-pipes-vision>=0.1.3,<0.2.0`.
- Point metadata and installation instructions at this standalone package.
- Include docs, examples and tests in the source distribution.
- Adapt CI, TestPyPI staging and PyPI promotion from the sibling repositories.

The source checkout is unchanged. Shared monorepo documentation only listed
or linked to VLM; its VLM API content is already in the extracted package
reference. Shared release-plan and workspace configuration are replaced by
standalone metadata and workflows. The shared lightweight namespace import
test is retained with a standalone import path.

Generated `.example_assets/` experiments were ignored in the source checkout
and are not included. Their recorded findings and tracked comparison images
remain in [design.md](design.md).

Documentation is plain Markdown in this iteration. VLM is an original package,
so there is no upstream API coverage report.
