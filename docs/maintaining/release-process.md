# Release Process

Albucore releases are published to PyPI through trusted publishing. The preferred path builds
validated release-candidate artifacts first, then publishes those exact artifacts manually. Maintainer
GitHub Release publishing is also supported: publishing release notes for a tag whose version matches
`pyproject.toml` and `uv.lock` triggers PyPI publication and attaches the generated artifacts.

## Release Ownership

- Primary release owner: Vladimir Iglovikov
- Backup release owner: Mikhail Druzhinin

## Required Workflows

- `.github/workflows/ci.yml` verifies correctness on every PR and on `main`.
- `.github/workflows/benchmark-pr.yml` produces early PR benchmark evidence for performance-sensitive
  changes.
- `.github/workflows/release-candidate.yml` validates the exact release commit and produces all
  release artifacts.
- `.github/workflows/publish.yml` publishes already validated release-candidate artifacts, or publishes
  directly when a maintainer publishes a GitHub Release.

Manual publishing is artifact-based through `publish.yml`; GitHub Release publishing validates and
builds from the release tag before uploading to PyPI.

## Release Gates

Before publishing, all of the following must be true:

1. The release commit is reachable from `origin/main`.
2. Required CI checks for the release commit are green.
3. `pyproject.toml` and `uv.lock` contain the same release version.
4. The version is not already present on PyPI.
5. For manual publishing, the release-candidate workflow succeeds for the exact commit SHA and
   version.
6. For manual publishing, the release-candidate artifact bundle contains:
   - wheel
   - source distribution
   - CycloneDX SBOM
   - `SHA256SUMS.txt`
   - release summary
   - runtime requirements export
   - release-candidate metadata
7. PyPI trusted publishing is configured for `.github/workflows/publish.yml` and the `pypi`
   environment.

## PR Requirements

Normal code PRs must pass CI. PRs touching performance-sensitive paths also run
`benchmark-pr.yml`, which compares the PR branch against the target branch. The PR benchmark is early
review evidence; release workflows do not rerun or reinterpret benchmarks.

Version bump PRs should contain only version and lockfile changes unless they are explicitly fixing
release infrastructure. Do not create a public GitHub Release for a version bump PR.

## Release Candidate Steps

1. Merge the version bump PR to `main`.
2. Copy the exact release commit SHA from `main`.
3. Run `release-candidate.yml` manually with:
   - `version`
   - `commit_sha`
4. Review the workflow summary and artifacts.
5. If the workflow fails, fix the repository in a new PR and restart from step 1.
6. If the workflow succeeds, record the release-candidate workflow run id.

The release-candidate workflow validates packaging, correctness, SBOM generation, checksums, and
release summary generation. It does not run performance benchmarks, publish to PyPI, or create a
public GitHub Release.

Release-candidate metadata and CI-run checks are validated by `tools/validate_release_candidate.py`.
These rules are unit-tested in `tests/test_verification_tools.py`; do not reimplement them as ad hoc
workflow snippets.

## Manual Publish Steps

1. Run `publish.yml` manually with:
   - `version`
   - `commit_sha`
   - `candidate_run_id`
2. The publish workflow downloads the `release-candidate-artifacts` bundle from the successful
   candidate run.
3. The publish workflow verifies:
   - the candidate run succeeded
   - the candidate workflow was `Release Candidate`
   - the candidate run SHA matches `commit_sha`
   - release-candidate metadata matches `version` and `commit_sha`
   - checksums match every referenced file
   - the expected wheel and sdist exist
   - PyPI does not already contain the version
4. The publish workflow stages only the wheel and sdist into `pypi-dist/`.
5. The publish workflow uploads `pypi-dist/` to PyPI through trusted publishing.
6. The publish workflow verifies PyPI now exposes the new version and files.
7. Publish to PyPI before creating or publishing the GitHub Release.
8. The publish workflow creates or updates the GitHub Release for the version and attaches the
   validated artifacts.

Publishing should not run performance benchmarks.
Candidate-run provenance, artifact filenames, checksum verification, duplicate-version checks, and
PyPI upload staging/post-publish PyPI verification are owned by `tools/verify_publish_artifacts.py`.

## GitHub Release Publish Steps

This path preserves the maintainer workflow where writing and publishing GitHub Release notes starts
the PyPI upload.

1. Merge the version bump PR to `main`.
2. Confirm CI is green for the release commit.
3. Create or publish the GitHub Release for the release tag.
4. The `publish.yml` release-published job checks out that tag, normalizes a leading `v` if present,
   and verifies that the tag version matches `pyproject.toml` and `uv.lock`.
5. The workflow verifies the release commit is reachable from `origin/main` and that CI succeeded for
   that commit.
6. The workflow builds and validates the wheel and sdist, generates SBOM/checksum/summary artifacts,
   verifies PyPI does not already contain the version, uploads only the wheel and sdist to PyPI, and
   verifies PyPI now exposes the files.
7. After PyPI verification succeeds, the workflow attaches the generated artifacts to the existing
   GitHub Release.

Until the `publish.yml` run succeeds, a GitHub Release is not proof that the version exists on PyPI.

## Benchmark Baselines

PR benchmarks compare against the PR target branch so reviewers see the impact of that PR alone.
Release workflows do not resolve benchmark baselines.

## Rollback And Hotfix

If PyPI publishing succeeds but post-publish verification finds a problem:

1. Stop promotion immediately.
2. Update the GitHub Release notes with the known issue.
3. Yank the release on PyPI if the package is unsafe to install.
4. Publish a new patch release from the smallest safe fix.

Do not silently republish different files under the same version.
