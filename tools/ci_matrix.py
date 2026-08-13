"""Validate support policy, package metadata, and CI matrix consistency."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
ANTIGRAVITY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "antigravity-pr-checks.yml"
BENCHMARK_PR_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "benchmark-pr.yml"
LEGAL_INTEGRITY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "legal-integrity.yml"
CLA_STATUS_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "cla-status.yml"
PUBLISH_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish.yml"
RELEASE_CANDIDATE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-candidate.yml"
SECURITY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "security.yml"
SUPPORT_POLICY = REPO_ROOT / "docs" / "maintaining" / "support-policy.md"
VALIDATE_RELEASE_CANDIDATE_TOOL = REPO_ROOT / "tools" / "validate_release_candidate.py"
VERIFY_PUBLISH_ARTIFACTS_TOOL = REPO_ROOT / "tools" / "verify_publish_artifacts.py"
LEGAL_ARTIFACT_VERIFY_COMMAND = "python tools/verify_legal_integrity.py --artifacts dist/*.whl dist/*.tar.gz"
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
CI_FOUNDATION_SHA = "2468af6e982545e2fd1d5ba4249f1b44154149fe"
CI_FOUNDATION_SETUP_ACTION = "albumentations-team/ci-foundation/actions/setup-python-uv@" + CI_FOUNDATION_SHA
CI_FOUNDATION_TORCH_ACTION = "albumentations-team/ci-foundation/actions/torch-cpu@" + CI_FOUNDATION_SHA
CI_FOUNDATION_ANTIGRAVITY_WORKFLOW = (
    "albumentations-team/ci-foundation/.github/workflows/antigravity-review.yml@" + CI_FOUNDATION_SHA
)


def _load_pyproject() -> dict[str, Any]:
    return tomllib.loads(PYPROJECT.read_text())


def _classifier_python_versions(classifiers: list[str]) -> set[str]:
    pattern = re.compile(r"^Programming Language :: Python :: (3\.\d+)$")
    versions: set[str] = set()
    for classifier in classifiers:
        match = pattern.match(classifier)
        if match is not None:
            versions.add(match.group(1))
    return versions


def _workflow_job(text: str, job_name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(job_name)}:\n.*?(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        text,
    )
    return match.group(0) if match is not None else ""


def _check_torch_cpu_source(pyproject: dict[str, Any], errors: list[str]) -> None:
    uv_config = pyproject.get("tool", {}).get("uv", {})
    sources = uv_config.get("sources", {}) if isinstance(uv_config, dict) else {}
    if not isinstance(sources, dict) or sources.get("torch") != {"index": "pytorch-cpu"}:
        errors.append("pyproject.toml must pin torch to the pytorch-cpu index")

    indexes = uv_config.get("index", []) if isinstance(uv_config, dict) else []
    if not isinstance(indexes, list) or not any(
        index
        == {
            "name": "pytorch-cpu",
            "url": PYTORCH_CPU_INDEX,
            "explicit": True,
        }
        for index in indexes
    ):
        errors.append("pyproject.toml must define the explicit pytorch-cpu index")


def _check_pyproject(errors: list[str]) -> set[str]:
    pyproject = _load_pyproject()
    project = pyproject.get("project", {})
    if not isinstance(project, dict):
        errors.append("pyproject.toml is missing [project]")
        return set()

    requires_python = project.get("requires-python")
    if requires_python != ">=3.10":
        errors.append(f"Expected requires-python >=3.10, found {requires_python!r}")

    classifiers = project.get("classifiers", [])
    if not isinstance(classifiers, list) or not all(isinstance(item, str) for item in classifiers):
        errors.append("project.classifiers must be a list of strings")
        return set()

    versions = _classifier_python_versions(classifiers)
    expected = {"3.10", "3.11", "3.12", "3.13", "3.14"}
    if versions != expected:
        errors.append(f"Expected Python classifiers {sorted(expected)}, found {sorted(versions)}")

    optional_dependencies = project.get("optional-dependencies", {})
    if not isinstance(optional_dependencies, dict) or "headless" not in optional_dependencies:
        errors.append("pyproject.toml must define project.optional-dependencies.headless")
    elif "torch" not in optional_dependencies:
        errors.append("pyproject.toml must define project.optional-dependencies.torch")

    dependencies = project.get("dependencies", [])
    if not isinstance(dependencies, list) or not all(isinstance(item, str) for item in dependencies):
        errors.append("project.dependencies must be a list of strings")
    elif any(re.match(r"^torch(?:[<>=!~;\[ ]|$)", item, flags=re.IGNORECASE) for item in dependencies):
        errors.append("project.dependencies must not require torch; use the torch extra")

    _check_torch_cpu_source(pyproject, errors)

    return versions


def _check_ci_torch_install(errors: list[str], text: str) -> None:
    errors.extend(
        f"CI {job_name} job must install Torch through the shared CPU-only action"
        for job_name in ("test", "macos-arm64-matmul", "declared-dependency-ranges")
        if CI_FOUNDATION_TORCH_ACTION not in _workflow_job(text, job_name)
    )


def _check_shared_foundation_setup(errors: list[str]) -> None:
    for workflow in (
        CI_WORKFLOW,
        BENCHMARK_PR_WORKFLOW,
        LEGAL_INTEGRITY_WORKFLOW,
        RELEASE_CANDIDATE_WORKFLOW,
        PUBLISH_WORKFLOW,
        SECURITY_WORKFLOW,
    ):
        if not workflow.exists():
            continue
        _check_file_fragments(
            errors,
            workflow,
            {"shared Python and uv setup": CI_FOUNDATION_SETUP_ACTION},
        )


def _check_ci(errors: list[str], versions: set[str]) -> None:
    text = CI_WORKFLOW.read_text()
    macos_matmul_job = _workflow_job(text, "macos-arm64-matmul")
    errors.extend(
        f"CI matrix does not mention Python {version}"
        for version in sorted(versions)
        if f'"{version}"' not in text and f"'{version}'" not in text
    )
    if "declared-dependency-ranges" not in text:
        errors.append("CI workflow is missing declared-dependency-ranges job")
    if "tools/check_router_contracts.py" not in text:
        errors.append("CI workflow does not run router contract check")
    if "python tools/classify_ci_changes.py" not in text:
        errors.append("CI workflow is missing version-only change classifier")
    if text.count("if: needs.change_scope.outputs.run_tests == 'true'") != 3:
        errors.append("CI workflow does not gate all three test jobs on change scope")
    if re.search(r"""runs-on:\s*["']?macos-latest["']?""", macos_matmul_job) is None:
        errors.append("CI workflow is missing the macOS arm64 runner")
    if (
        re.search(r"pytest\s+tests/test_matmul\.py", macos_matmul_job) is None
        or re.search(r"-W\s+error", macos_matmul_job) is None
    ):
        errors.append("CI workflow is missing warnings-as-errors matmul tests on macOS")
    if "numpy==2.2.6" not in macos_matmul_job:
        errors.append("CI workflow does not pin the affected NumPy version for the macOS regression tests")
    if (
        re.search(
            r"""(?:^|\s)(?:-r|--requirement)(?:\s+|=)["']?requirements-dev\.txt["']?(?=\s|$)""",
            macos_matmul_job,
        )
        is None
    ):
        errors.append("CI workflow does not install test dependencies for the macOS regression tests")
    if "permissions:" not in text or "contents: read" not in text:
        errors.append("CI workflow must declare minimal GITHUB_TOKEN permissions")
    _check_ci_torch_install(errors, text)
    _check_shared_foundation_setup(errors)


def _check_support_policy(errors: list[str], versions: set[str]) -> None:
    text = SUPPORT_POLICY.read_text()
    errors.extend(
        f"Support policy does not mention Python {version}" for version in sorted(versions) if version not in text
    )
    if "opencv-python-headless" not in text:
        errors.append("Support policy does not name opencv-python-headless")
    if "(H, W, 1)" not in text:
        errors.append("Support policy does not state explicit grayscale channel convention")


def _check_file_fragments(errors: list[str], path: Path, required_fragments: dict[str, str]) -> None:
    if not path.exists():
        errors.append(f"Required workflow {path.relative_to(REPO_ROOT)} is missing")
        return
    text = path.read_text()
    errors.extend(
        f"{path.relative_to(REPO_ROOT)} is missing {label}"
        for label, fragment in required_fragments.items()
        if fragment not in text
    )


def _check_file_absent_fragments(errors: list[str], path: Path, forbidden_fragments: dict[str, str]) -> None:
    if not path.exists():
        return
    text = path.read_text()
    errors.extend(
        f"{path.relative_to(REPO_ROOT)} must not contain {label}"
        for label, fragment in forbidden_fragments.items()
        if fragment in text
    )


def _check_torch_runtime_exports(errors: list[str], path: Path) -> None:
    if not path.exists():
        return
    export_commands = re.findall(r"(?m)^\s*run:\s*(uv export[^\n]+)", path.read_text())
    if any("--extra torch" not in command and "--all-extras" not in command for command in export_commands):
        errors.append(f"{path.relative_to(REPO_ROOT)} must include the torch extra in every uv export")


def _check_torch_pip_audits(errors: list[str], path: Path) -> None:
    if not path.exists():
        return
    audit_commands = re.findall(r"(?m)^\s*run:\s*(uv tool run --from pip-audit pip-audit[^\n]+)", path.read_text())
    if not audit_commands or any(f"--extra-index-url {PYTORCH_CPU_INDEX}" not in command for command in audit_commands):
        errors.append(f"{path.relative_to(REPO_ROOT)} must pass the PyTorch CPU index to every pip-audit command")


def _check_release_workflows(errors: list[str]) -> None:
    for workflow in (SECURITY_WORKFLOW, RELEASE_CANDIDATE_WORKFLOW, PUBLISH_WORKFLOW):
        _check_torch_runtime_exports(errors, workflow)
    _check_file_fragments(
        errors,
        BENCHMARK_PR_WORKFLOW,
        {
            "PR trigger": "pull_request:",
            "PR router benchmark": "benchmarks/benchmark_router_synthetic.py",
            "PR advisory regression check": "--mode advisory",
            "PR benchmark artifacts": "pr-router-benchmark-results",
        },
    )
    if BENCHMARK_PR_WORKFLOW.exists():
        benchmark_text = BENCHMARK_PR_WORKFLOW.read_text()
        base_benchmark_command = (
            'PYTHONPATH="$PWD" uv run --project "$GITHUB_WORKSPACE" python benchmarks/benchmark_router_synthetic.py'
        )
        if base_benchmark_command not in benchmark_text:
            errors.append("PR benchmark workflow must run the base source tree in the PR CPU environment")
    _check_file_fragments(
        errors,
        LEGAL_INTEGRITY_WORKFLOW,
        {
            "pull request trigger": "pull_request:",
            "manual trigger": "workflow_dispatch:",
            "minimal contents permission": "contents: read",
            "legal integrity job": "License, CLA, and package notices",
            "source-tree verifier step": "Verify source-tree legal integrity",
            "source-tree legal verifier": "python tools/verify_legal_integrity.py",
            "legal verifier tests": "tests/test_legal_integrity.py",
            "distribution build": 'uv build --out-dir "${RUNNER_TEMP}/albucore-legal-dist"',
            "artifact verifier step": "Verify distribution license and CLA exclusion",
            "distribution legal verifier": "python tools/verify_legal_integrity.py --artifacts",
            "wheel verification": "albucore-legal-dist/*.whl",
            "source distribution verification": "albucore-legal-dist/*.tar.gz",
            "distribution metadata check": 'twine check "${RUNNER_TEMP}"/albucore-legal-dist/*',
        },
    )
    _check_file_fragments(
        errors,
        RELEASE_CANDIDATE_WORKFLOW,
        {
            "manual release candidate trigger": "workflow_dispatch:",
            "exact commit input": "commit_sha:",
            "release metadata validator": "tools/validate_release_candidate.py metadata",
            "candidate CI success check": "Verify CI workflow succeeded for candidate",
            "candidate CI validator": "tools/validate_release_candidate.py ci-runs",
            "release validation Torch profile": "uv sync --frozen --extra headless --extra torch --group dev",
            "shared CPU Torch action": CI_FOUNDATION_TORCH_ACTION,
            "project-free runtime dependency export": "uv export --frozen --no-dev --no-emit-project",
            "candidate metadata writer": "tools/validate_release_candidate.py candidate-metadata",
            "legal artifact verifier": LEGAL_ARTIFACT_VERIFY_COMMAND,
            "release candidate artifact upload": "release-candidate-artifacts",
        },
    )
    _check_file_absent_fragments(
        errors,
        RELEASE_CANDIDATE_WORKFLOW,
        {
            "release benchmark runner": "benchmarks/benchmark_router_synthetic.py",
            "release benchmark regression checker": "tools/check_benchmark_regressions.py",
            "accepted benchmark regression input": "accepted_regressions:",
            "reusable benchmark run input": "benchmark_run_id:",
            "release benchmark artifacts": "router-release",
            "release memory smoke": "memory-smoke",
        },
    )
    _check_file_fragments(
        errors,
        PUBLISH_WORKFLOW,
        {
            "GitHub Release publish trigger": "release:",
            "published release trigger": "types: [published]",
            "manual publish trigger": "workflow_dispatch:",
            "candidate run input": "candidate_run_id:",
            "candidate artifact download": "release-candidate-artifacts",
            "legal artifact verifier": LEGAL_ARTIFACT_VERIFY_COMMAND,
            "prepublish verifier": "tools/verify_publish_artifacts.py prepublish",
            "direct release verifier": "tools/verify_publish_artifacts.py direct-release",
            "PyPI distribution staging": "tools/verify_publish_artifacts.py prepare-pypi-dist",
            "PyPI publication verifier": "tools/verify_publish_artifacts.py publication",
            "trusted publishing": "pypa/gh-action-pypi-publish",
            "release event publish job": "Publish from GitHub Release",
            "GitHub Release only after PyPI": "Create or update GitHub Release",
            "shared CPU Torch action": CI_FOUNDATION_TORCH_ACTION,
        },
    )
    if PUBLISH_WORKFLOW.exists() and PUBLISH_WORKFLOW.read_text().count(LEGAL_ARTIFACT_VERIFY_COMMAND) == 1:
        errors.append(
            f"{PUBLISH_WORKFLOW.relative_to(REPO_ROOT)} must run the legal artifact verifier in both publish paths",
        )
    _check_file_absent_fragments(
        errors,
        PUBLISH_WORKFLOW,
        {
            "published release benchmark artifacts": "router-release",
            "published benchmark baseline artifacts": "router-baseline",
            "published memory smoke artifacts": "memory-smoke",
        },
    )
    _check_file_fragments(
        errors,
        VALIDATE_RELEASE_CANDIDATE_TOOL,
        {
            "release metadata validator": "validate_release_environment",
            "candidate CI run validator": "validate_ci_runs",
        },
    )
    _check_file_fragments(
        errors,
        VERIFY_PUBLISH_ARTIFACTS_TOOL,
        {
            "candidate run validator": "verify_candidate_run",
            "checksum verifier": "verify_checksums",
            "distribution staging": "copy_distribution_files",
            "PyPI existing-version guard": "verify_pypi_absent",
            "PyPI publication verifier": "verify_pypi_publication",
            "direct release artifact verifier": "verify_direct_release_artifacts",
        },
    )


def _check_release_docs(errors: list[str]) -> None:
    process = (REPO_ROOT / "docs" / "maintaining" / "release-process.md").read_text()
    required_fragments = {
        "release candidate workflow": "release-candidate.yml",
        "publish workflow": "publish.yml",
        "PyPI-before-GitHub-Release ordering": "Publish to PyPI before creating or publishing the GitHub Release",
    }
    errors.extend(
        f"release-process.md is missing {label}"
        for label, fragment in required_fragments.items()
        if fragment not in process
    )


def _check_release_workflows_and_docs(errors: list[str]) -> None:
    _check_release_workflows(errors)
    _check_release_docs(errors)


def _check_release_workflow(errors: list[str]) -> None:
    _check_release_workflows_and_docs(errors)


def _check_security_workflow(errors: list[str]) -> None:
    text = SECURITY_WORKFLOW.read_text()
    if "uv export --frozen --no-dev --no-emit-project" not in text:
        errors.append("Security workflow runtime audit must omit the editable project from exported requirements")
    if "uv export --frozen --no-emit-project" not in text:
        errors.append("Security workflow dev audit must omit the editable project from exported requirements")
    _check_torch_pip_audits(errors, SECURITY_WORKFLOW)


def _check_cla_status_workflow(errors: list[str]) -> None:
    _check_file_fragments(
        errors,
        CLA_STATUS_WORKFLOW,
        {
            "pull request trigger": "pull_request:",
            "manual recovery trigger": "workflow_dispatch:",
            "read-only status permission": "statuses: read",
            "CLA reporter job": "CLA status reported",
            "hosted CLA context": "license/cla",
            "paginated status lookup": "--paginate --slurp",
            "CLA Assistant recheck URL": "https://cla-assistant.io/check/",
            "maintainer recovery procedure": "docs/maintaining/license-provenance.md",
        },
    )
    _check_file_absent_fragments(
        errors,
        CLA_STATUS_WORKFLOW,
        {
            "status write permission": "statuses: write",
            "pull-request write permission": "pull-requests: write",
        },
    )


def _check_antigravity_workflow(errors: list[str]) -> None:
    _check_file_fragments(
        errors,
        ANTIGRAVITY_WORKFLOW,
        {
            "pull_request_target trigger": "pull_request_target:",
            "same-repository guard": "github.event.pull_request.head.repo.full_name == github.repository",
            "shared trusted review workflow": CI_FOUNDATION_ANTIGRAVITY_WORKFLOW,
            "data-only policy": "policy-path: .github/ci-foundation/antigravity.toml",
            "workload identity permission": "id-token: write",
            "review publication permission": "pull-requests: write",
        },
    )
    _check_file_absent_fragments(
        errors,
        ANTIGRAVITY_WORKFLOW,
        {
            "Gemini API key": "secrets.GEMINI_API_KEY",
            "local Gemini runner": "google-github-actions/run-gemini-cli",
        },
    )


def check() -> list[str]:
    """Return support-matrix consistency errors."""
    errors: list[str] = []
    versions = _check_pyproject(errors)
    if versions:
        _check_ci(errors, versions)
        _check_support_policy(errors, versions)
        _check_release_workflow(errors)
        _check_security_workflow(errors)
        _check_cla_status_workflow(errors)
        _check_antigravity_workflow(errors)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check",), nargs="?", default="check")
    parser.parse_args()

    errors = check()
    if errors:
        sys.stderr.write("CI matrix check failed:\n")
        for error in errors:
            sys.stderr.write(f"- {error}\n")
        return 1

    sys.stdout.write("CI matrix check passed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
