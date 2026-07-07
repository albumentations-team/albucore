# PR And Publishing Architecture

This document describes the Albucore PR, benchmark, release-candidate, and publishing architecture.
Benchmarks are PR review evidence. Release-candidate and publish workflows validate packaging,
correctness, provenance, and artifact integrity; they do not run performance benchmarks.

## Core Rule

Performance regressions should be discovered during PR review. Release workflows should not discover,
reinterpret, or package benchmark results.

## Workflow Set

### 1. `ci.yml`

Purpose: prove that a PR is functionally correct.

Required on every PR:

- linting and formatting through `pre-commit`
- test matrix for supported Python versions
- `uv lock --check`
- router contract check
- support matrix check
- golden vector verification
- property tests, at least on one Linux Python version

This workflow should not publish artifacts to PyPI or create releases.

### 2. `benchmark-pr.yml`

Purpose: make performance changes visible during review.

Trigger:

- PRs that touch performance-sensitive paths:
  - `albucore/**`
  - `benchmarks/**`
  - `tests/router_contracts.py`
  - `tests/verification_constants.py`
  - `pyproject.toml`
  - `uv.lock`
  - `.github/workflows/benchmark-pr.yml`

Behavior:

- run a quick synthetic router benchmark for the PR branch
- run the same quick benchmark for the target branch
- compare PR vs base in advisory mode
- upload raw JSON and Markdown report with `if: always()`
- write the regression report to the Actions step summary

Policy:

- the check is required for report generation and script health
- regressions are advisory in PR context while thresholds and runner noise are calibrated
- PRs that intentionally change routing should include benchmark evidence in the PR description

### 3. `release-candidate.yml`

Purpose: validate the exact release commit before anything public is published.

Trigger:

- manual `workflow_dispatch`
- inputs:
  - `version`
  - `commit_sha`

Required checks:

- checkout exactly `commit_sha`
- verify `pyproject.toml` version equals the requested version
- verify `uv.lock` version equals the requested version
- verify the commit is reachable from `main`
- verify the `CI` workflow succeeded for that commit
- build wheel and sdist
- install release validation dependencies with `uv sync --frozen --extra headless --group dev`
- run router contract checks
- run CI matrix policy checks
- run golden vector verification
- run release property tests
- run `twine check dist/*`
- smoke test the built wheel in a clean venv outside the repository checkout
- export locked runtime dependencies
- generate SBOM
- generate release-candidate metadata
- generate checksums
- generate release summary

Implementation:

- `tools/validate_release_candidate.py metadata` owns package version, lockfile version, checkout
  SHA, `origin/main` ancestry, SBOM filename, and headless-runtime dependency export.
- `tools/validate_release_candidate.py ci-runs` owns the successful-CI-run requirement.
- `tools/validate_release_candidate.py candidate-metadata` writes the release-candidate provenance
  file included in the artifact bundle.

Artifacts:

- wheel
- sdist
- `SHA256SUMS.txt`
- SBOM
- release summary
- runtime requirements export
- release-candidate metadata

This workflow should not run benchmarks, publish to PyPI, or create a public GitHub Release.

### 4. `publish.yml`

Purpose: publish a previously validated release candidate.

Trigger:

- manual `workflow_dispatch`
- inputs:
  - `version`
  - `commit_sha`
  - `candidate_run_id`

Required checks:

- download artifacts from `candidate_run_id`
- verify the candidate run succeeded
- verify candidate artifacts were produced from `commit_sha`
- verify `version` matches artifact filenames and package metadata
- verify checksums match
- verify PyPI does not already have this version
- stage only the wheel and sdist into the PyPI upload directory
- publish to PyPI through trusted publishing
- verify PyPI JSON shows the new version and files
- create or publish the GitHub Release only after PyPI publish succeeds
- attach the validated artifacts to the GitHub Release

Implementation:

- `tools/verify_publish_artifacts.py prepublish` owns candidate-run provenance, artifact names,
  checksums, metadata matching, and the PyPI duplicate-version guard.
- `tools/verify_publish_artifacts.py prepare-pypi-dist` stages only publishable distribution files
  so release evidence files are not passed to the PyPI upload action.
- `tools/verify_publish_artifacts.py publication` owns post-publish PyPI JSON verification.

Publishing should not run benchmarks or rebuild release artifacts.

## Correct Publishing Sequence

Use this sequence for every normal release:

1. Open a normal PR for all feature and bug-fix changes.
2. Run CI and PR benchmark checks when touched paths require them.
3. Merge the code changes.
4. Open a version bump PR.
5. Merge the version bump after CI is green.
6. Run `release-candidate.yml` on the exact `main` commit.
7. Review generated artifacts.
8. If the candidate fails, fix the repo and start again from a new PR.
9. If the candidate passes, run `publish.yml` with the candidate run id.
10. Verify PyPI has the new version.
11. Verify the GitHub Release has all expected assets.
12. Announce the release.

Do not publish a public GitHub Release before step 9 succeeds.

## Publishing Invariants

The release system must avoid these failure modes:

1. Publishing a public GitHub Release before validation succeeds.
2. Treating a GitHub Release as proof that a package version exists on PyPI.
3. Running tools that import optional dependencies before installing the matching extras.
4. Rebuilding different artifacts at publish time than the artifacts that were validated.
5. Allowing a failed release attempt to become the baseline for the next release.

## CI Matrix Policy Check

`tools/ci_matrix.py check` enforces the intended workflow fragments and checks that release workflows
stay benchmark-free.

## Success Criteria

The publishing system is healthy when all of these are true:

- A public GitHub Release cannot exist without validated artifacts.
- PRs show performance impact before merge.
- Release candidates do not run benchmarks.
- Publishing does not rebuild unvalidated artifacts.
- Failed release-candidate runs leave enough artifacts to diagnose the failure.
- Failed publish runs do not create empty public GitHub Releases.
