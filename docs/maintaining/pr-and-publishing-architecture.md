# PR And Publishing Architecture

This document describes the Albucore PR, benchmark, release-candidate, and publishing architecture.
The design keeps performance discovery in PR/release-candidate validation and keeps publishing as an
artifact/provenance step.

The core rule is simple:

> Performance regressions should be discovered during PR review and release-candidate validation,
> not for the first time after a public GitHub Release has already been published.

Publishing should be the last step after a specific commit, benchmark result, wheel, sdist, SBOM, and
release summary have already been validated.

## Terminology

- **PyPI:** The Python Package Index where `albucore` wheels and sdists are published.
- **GitHub Release:** The public release page and tag on GitHub.
- **Release candidate:** A validated commit and version that is ready to publish, but is not yet a
  public package release.
- **Previous published release:** The latest installable, non-yanked version from PyPI. This is not
  the same thing as the latest GitHub Release, because a GitHub Release can exist even when publishing
  to PyPI failed.
- **Benchmark gate:** A comparison against a known baseline that can block merge or release.
- **Benchmark evidence:** JSON and Markdown artifacts that explain benchmark results, including
  accepted regressions.

## Answer: PRs, Publishing, Or Both?

Run benchmarks in both places, but not with the same purpose.

PR benchmarks are for early discovery. They should tell reviewers that a code change affects hot
paths before the change lands on `main`.

Release-candidate benchmarks are the official release gate. They compare the exact release commit
against the previous published PyPI release and produce durable release evidence.

Publishing should not be where new performance information appears. Publishing may verify that the
release-candidate benchmark artifact exists and belongs to the exact commit being published, but it
should not be the first place a regression can block. If a benchmark runs during publishing, it should
be a final consistency check with the report uploaded even on failure.

There is one acceptable shortcut: if the PR or post-merge workflow already produced a successful
benchmark gate for the exact release commit, exact dependency lock, exact benchmark script version,
and exact previous PyPI baseline, the release-candidate workflow can reuse that artifact instead of
rerunning the benchmark. In that case the release-candidate workflow must verify artifact provenance
and checksums. It must not blindly trust "some benchmark passed on a PR" as release evidence.

The publish workflow should not run performance benchmarks. Publishing should consume validated
release-candidate artifacts.

Workflow split:

| Stage | Benchmark Role | Blocking? |
| --- | --- | --- |
| Normal PR | Quick signal for reviewers | Required report generation; advisory benchmark interpretation |
| Performance-sensitive PR | Quick comparison plus targeted benchmarks | Required report generation; maintainer review for regressions |
| Main / scheduled | Release-mode benchmark evidence against selected PyPI baseline | Fails the performance workflow on unaccepted blocking regressions |
| Release candidate | Official benchmark gate for the exact release SHA | Blocks release unless regression is accepted |
| Publish | Verify candidate artifacts and provenance | Blocks only on packaging/provenance/integrity mismatch |

## Publishing Invariants

The release system must avoid these failure modes:

1. Publishing a public GitHub Release before validation succeeds.
2. Treating a GitHub Release as proof that a package version exists on PyPI.
3. Running tools that import optional dependencies before installing the matching extras.
4. Failing a benchmark gate without uploading the generated regression report.
5. Discovering performance regressions only after a version tag and public GitHub Release already
   exist.
6. Rebuilding different artifacts at publish time than the artifacts that were validated.
7. Allowing a failed release attempt to become the baseline for the next release.

## Workflow Set

The system has separate workflows with narrow responsibilities.

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
  - `.github/workflows/performance.yml`
  - `.github/workflows/release-candidate.yml`

Behavior:

- run a quick synthetic router benchmark for the PR branch
- run the same quick benchmark for the merge base or target branch
- compare PR vs base
- upload raw JSON and Markdown report with `if: always()`
- print the top regressions directly to the Actions log before exiting

Policy:

- the check must be required only for report generation and script health
- regressions are advisory in PR context while thresholds and runner noise are being calibrated
- PRs that intentionally change routing must include benchmark evidence in the PR description
- keep noisy one-cell regressions as review-required instead of automatic failure

### 3. `performance.yml`

Purpose: catch drift and runner-sensitive regressions outside individual PRs.

Trigger:

- scheduled weekly or daily
- manual dispatch with a baseline version input

Behavior:

- resolve the previous published PyPI release from PyPI metadata
- run the fuller benchmark grid on `main`
- run the same grid against the previous published PyPI release
- upload JSON, Markdown, and environment metadata
- fail on unaccepted release-blocking regressions
- produce `release-benchmark-evidence` that `release-candidate.yml` can reuse

This workflow should not create releases or publish to PyPI.

### 4. `release-candidate.yml`

Purpose: validate the exact release commit before anything public is published.

Trigger:

- manual `workflow_dispatch`
- inputs:
  - `version`
  - `commit_sha`
  - optional `accepted_regressions` JSON file path
  - optional successful `benchmark_run_id` from `performance.yml`

Required checks:

- checkout exactly `commit_sha`
- verify `pyproject.toml` version equals the requested version
- verify `uv.lock` version equals the requested version
- verify the commit is reachable from `main`
- verify the `CI` workflow succeeded for that commit
- install release validation dependencies with `uv sync --frozen --extra headless --group dev`
- run router contract checks
- run golden vector verification
- run release property tests
- run `twine check dist/*`
- smoke test the built wheel in a clean venv outside the repository checkout
- run memory smoke
- resolve previous published PyPI release from PyPI, not GitHub Releases
- run release benchmark comparison against that previous published PyPI release
- or verify reusable benchmark evidence from a successful `performance.yml` run for the exact commit,
  version, and previous PyPI baseline
- generate release summary
- generate SBOM
- generate checksums

Implementation:

- `tools/validate_release_candidate.py metadata` owns package version, lockfile version, checkout
  SHA, `origin/main` ancestry, SBOM filename, and headless-runtime dependency export.
- `tools/validate_release_candidate.py ci-runs` owns the successful-CI-run requirement.
- `tools/validate_release_candidate.py benchmark-evidence` owns reusable Performance workflow
  evidence provenance and copies verified artifacts into `dist/`.
- `tools/validate_release_candidate.py candidate-metadata` writes the release-candidate provenance
  file included in the artifact bundle.

Artifacts:

- wheel
- sdist
- `SHA256SUMS.txt`
- SBOM
- release summary
- current benchmark JSON
- baseline benchmark JSON
- benchmark regression report
- memory smoke JSON
- reusable benchmark metadata when benchmark evidence is reused
- environment metadata

Important behavior:

- upload artifacts with `if: always()` for diagnostic reports
- fail if benchmark artifacts cannot be generated
- fail if there are unaccepted blocking regressions
- do not create a public GitHub Release

This is the stage where performance can block release.

### 5. `publish.yml`

Purpose: publish a previously validated release candidate.

Trigger:

- manual `workflow_dispatch`
- inputs:
  - `version`
  - `candidate_run_id`
  - `commit_sha`

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

This workflow should not be doing first-time benchmark discovery. It can verify that a benchmark
report exists and that it is for the same SHA.

Trusted publishing should be configured for this workflow and the `pypi` environment. A release
event trigger is not required for trusted publishing if PyPI is configured for the workflow that
performs the publish.

## Correct Publishing Sequence

Use this sequence for every normal release:

1. Open a normal PR for all feature and bug-fix changes.
2. Run CI and PR benchmark checks.
3. Merge the code changes.
4. Open a version bump PR.
5. Merge the version bump after CI is green.
6. Run `release-candidate.yml` on the exact `main` commit.
7. Review generated artifacts and benchmark report.
8. If the candidate fails, fix the repo and start again from a new PR.
9. If the candidate passes, run `publish.yml` with the candidate run id.
10. Verify PyPI has the new version.
11. Verify the GitHub Release has all expected assets.
12. Announce the release.

Do not publish a public GitHub Release before step 9 succeeds.

## Benchmark Policy Details

### Baseline Selection

Always resolve the release benchmark baseline from PyPI:

- fetch `https://pypi.org/pypi/albucore/json`
- select the highest final numeric version lower than the candidate version
- require at least one non-yanked installable file

Never use the latest GitHub Release as the benchmark baseline. A failed release attempt can have a
GitHub Release and tag without any PyPI files.

### PR Benchmark Baselines

For PRs, compare against the PR target branch or merge base. Do not compare every PR against the
latest PyPI release, because that makes each PR inherit unrelated drift since the last release.

For release candidates, compare against the previous published PyPI release because that is the user
visible upgrade path.

### Thresholds

Recommended initial thresholds:

- report any slowdown on release-blocking operations
- mark `review` above 5%
- mark `blocking` above 10%
- require repeated evidence for known noisy cells before blocking normal PRs
- allow release blocking only in the release-candidate workflow, not as a surprise during publish

The regression checker should print a short summary to stdout:

- total regressions
- blocking count
- top 10 blocking rows
- path to the full Markdown report

It should also write the full report to disk for artifact upload.

### Accepted Regressions

Some regressions are intentional, for example when a correctness fix trades speed for correctness.
Use an explicit accepted-regression record instead of silently weakening thresholds.

Recommended format:

`docs/maintaining/accepted-performance-regressions/<version>.json`

Each accepted regression object should include:

- operation
- layout
- shape
- dtype
- baseline version
- candidate version
- measured slowdown
- reason
- approving maintainer
- expiration or follow-up issue

The regression checker reads this file when the release owner passes
`--accepted-regressions <file>`.

Accepted regressions must appear in the release summary.

## PR Rules

Every PR should answer these questions:

- Does this change affect runtime behavior?
- Does this change affect benchmarked routes?
- Does this change affect packaging or release workflows?
- Does this change affect dependency resolution?
- Does this change require a benchmark artifact?

For performance-sensitive PRs, require:

- quick router benchmark report
- targeted benchmark report when routing changes
- explanation for any regression above review threshold
- updated docs when routing policy changes

For version bump PRs, require:

- only version and lockfile changes unless there is an explicit release fix
- green CI
- no public GitHub Release yet
- release-candidate workflow run after merge

## Publishing Rules

Publishing must be boring. The publish workflow should fail only for concrete release mechanics:

- missing candidate artifacts
- checksum mismatch
- version mismatch
- PyPI already has the version
- trusted publishing configuration failure
- PyPI upload failure
- post-publish PyPI verification failure

Publishing should not fail because a new benchmark report was interpreted for the first time.

## Required Script Behavior

### `tools/check_benchmark_regressions.py`

- always write the Markdown report before returning non-zero
- also print the top blocking regressions to stdout
- include `--print-summary` or make summary printing default
- support accepted-regression input
- distinguish:
  - `report`
  - `review`
  - `blocking`
  - `accepted`
- return non-zero only for unaccepted blocking regressions in release mode

### Previous Release Resolver

- resolve previous published version from PyPI JSON
- ignore GitHub Releases
- ignore yanked files
- fail with a clear bootstrap message if no previous PyPI release exists
- print the selected baseline version in the workflow log

### Artifact Upload

Every workflow that produces diagnostic reports should upload them with `if: always()`.

At minimum:

- benchmark JSON
- benchmark Markdown report
- memory smoke JSON
- release summary draft
- environment metadata

### CI Matrix Policy Check

The policy checker should enforce the intended release workflow fragments, but it should not make
the workflow hard to improve.

## Success Criteria

The publishing system is healthy when all of these are true:

- A public GitHub Release cannot exist without validated artifacts.
- A version cannot be used as a benchmark baseline unless it exists on PyPI.
- PRs show performance impact before merge.
- Release candidates produce complete benchmark evidence before publish.
- Publishing does not rebuild unvalidated artifacts.
- Failed release-candidate runs leave enough artifacts to diagnose the failure.
- Failed publish runs do not create empty public GitHub Releases.
- Every accepted performance regression is documented in the release summary.
