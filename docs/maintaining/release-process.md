# Release Process

Albucore releases are built from validated release-candidate artifacts and published to PyPI through
trusted publishing. The public GitHub Release is created or updated only after PyPI confirms the
package files exist.

## Release Ownership

- Primary release owner: Vladimir Iglovikov
- Backup release owner: Mikhail Druzhinin

## Required Workflows

- `.github/workflows/ci.yml` verifies correctness on every PR and on `main`.
- `.github/workflows/benchmark-pr.yml` produces early PR benchmark evidence for performance-sensitive
  changes.
- `.github/workflows/performance.yml` monitors scheduled or manually requested benchmark drift.
- `.github/workflows/release-candidate.yml` validates the exact release commit and produces all
  release artifacts.
- `.github/workflows/publish.yml` publishes already validated release-candidate artifacts.

All public publishing is manual and artifact-based through `publish.yml`; validation happens before
any public package release is created.

## Release Gates

Before publishing, all of the following must be true:

1. The release commit is reachable from `origin/main`.
2. Required CI checks for the release commit are green.
3. `pyproject.toml` and `uv.lock` contain the same release version.
4. The version is not already present on PyPI.
5. The release-candidate workflow succeeds for the exact commit SHA and version.
6. The release-candidate artifact bundle contains:
   - wheel
   - source distribution
   - CycloneDX SBOM
   - `SHA256SUMS.txt`
   - release summary
   - current benchmark JSON
   - previous published PyPI baseline benchmark JSON
   - benchmark regression report
   - memory smoke JSON
   - runtime requirements export
   - release-candidate metadata
   - reusable benchmark metadata when benchmark evidence came from `performance.yml`
7. Benchmark regressions are either below release-blocking thresholds or explicitly accepted with
   documented maintainer approval.
8. PyPI trusted publishing is configured for `.github/workflows/publish.yml` and the `pypi`
   environment.

## PR Requirements

Normal code PRs must pass CI. PRs touching performance-sensitive paths also run
`benchmark-pr.yml`, which compares the PR branch against the target branch. The PR benchmark is early
review evidence; it is not a substitute for release-candidate evidence unless the artifact is
provably for the exact release commit, lockfile, benchmark script version, and previous PyPI baseline.

Version bump PRs should contain only version and lockfile changes unless they are explicitly fixing
release infrastructure. Do not create a public GitHub Release for a version bump PR.

## Release Candidate Steps

1. Merge the version bump PR to `main`.
2. Copy the exact release commit SHA from `main`.
3. Run `release-candidate.yml` manually with:
   - `version`
   - `commit_sha`
   - optional `accepted_regressions` JSON file path
   - optional successful `benchmark_run_id` from `performance.yml`
4. Review the workflow summary and artifacts.
5. If the workflow fails, fix the repository in a new PR and restart from step 1.
6. If the workflow succeeds, record the release-candidate workflow run id.

The release-candidate workflow validates packaging, correctness, benchmark evidence, memory smoke,
SBOM generation, checksums, and release summary generation. If `benchmark_run_id` is provided, it
verifies that the `performance.yml` run succeeded for the exact commit, version, and previous PyPI
baseline, then reuses those benchmark artifacts instead of rerunning benchmarks. It does not publish
to PyPI and does not create a public GitHub Release.

Release-candidate metadata, CI-run checks, benchmark-evidence provenance, and reusable benchmark
metadata are validated by `tools/validate_release_candidate.py`. These rules are unit-tested in
`tests/test_verification_tools.py`; do not reimplement them as ad hoc workflow snippets.

## Publish Steps

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

Publishing should not run performance benchmarks. It consumes release-candidate benchmark evidence.
Candidate-run provenance, artifact filenames, checksum verification, duplicate-version checks, and
PyPI upload staging/post-publish PyPI verification are owned by `tools/verify_publish_artifacts.py`.

## Benchmark Baselines

Release-candidate benchmarks compare against the previous published PyPI release, not the latest
GitHub Release. The resolver must ignore GitHub Releases because a GitHub Release can exist without
installable PyPI files.

PR benchmarks compare against the PR target branch so reviewers see the impact of that PR alone.
Scheduled benchmarks compare `main` against a PyPI release baseline and are used for drift
monitoring.

## Accepted Performance Regressions

Accepted regressions must be explicit. The `accepted_regressions` file passed to
`release-candidate.yml` must identify each accepted cell by:

- operation
- layout
- shape
- dtype
- reason
- approving maintainer

Accepted regressions must be summarized in the release notes or release summary. Do not weaken global
thresholds to hide an intentional slowdown.

## Rollback And Hotfix

If PyPI publishing succeeds but post-publish verification finds a problem:

1. Stop promotion immediately.
2. Update the GitHub Release notes with the known issue.
3. Yank the release on PyPI if the package is unsafe to install.
4. Publish a new patch release from the smallest safe fix.

Do not silently republish different files under the same version.
