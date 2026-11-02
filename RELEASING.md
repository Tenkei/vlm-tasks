# Releasing ml-pipes-vlm

This distribution is versioned independently under the shared `ml_pipes`
namespace. The initial standalone version preserves `0.1.2rc1` from the
original package. Framework dependencies use `>=0.1.3,<0.2.0`; PydanticAI stays
pinned to the implementation's tested API, `2.61.0`.

## Prepare

Update `project.version`, dependency bounds, release notes, installation
instructions and affected tests together in a release preparation PR.

CI tests Python 3.11 through 3.13, including the local Transformers adapter
with fake weights. It runs the synthetic pipelines, builds wheel and source
artifacts, checks their metadata and installs the wheel in a clean environment.
Require `Compat: Gate` and `Build and validate distribution` in branch rules.

## Stage on TestPyPI

After merging, create and push an annotated `v<project.version>` tag, such as
`v0.1.2rc1`. The `Stage` workflow:

1. Checks that the tag matches the package version and runs CI.
2. Builds and validates the distributions and records SHA-256 checksums.
3. Attaches these artifacts to a GitHub Release.
4. Downloads and verifies those assets, publishes them to TestPyPI via OIDC,
   and verifies an installation and import.

Configure the TestPyPI trusted publisher with:

```text
Owner:       trained-by-humans
Repository:  ml-pipes-vlm
Workflow:    .github/workflows/stage-release.yml
Environment: test-pypi
```

TestPyPI cannot reuse uploaded filenames. Increment the candidate version
when rebuilding a candidate that has already been uploaded.

## Promote to PyPI

After staging succeeds, dispatch `Release` from the staged tag. It rejects
branch refs, downloads the existing release assets, verifies their checksums
and publishes the same artifacts to PyPI. Configure required reviewers on the
`pypi` environment and configure the PyPI trusted publisher with:

```text
Owner:       trained-by-humans
Repository:  ml-pipes-vlm
Workflow:    .github/workflows/publish-release.yml
Environment: pypi
```

Organization administrators control release tags and deployment approval.
Protect `v*` tags and require review for changes to package metadata and release
workflows. Repository creation, trusted publishers and environment protection
must be configured on GitHub before publishing.
