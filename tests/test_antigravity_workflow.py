"""Security contracts for the shared Antigravity pull-request review workflow."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "antigravity-pr-checks.yml"
POLICY = REPO_ROOT / ".github" / "ci-foundation" / "antigravity.toml"
INSTRUCTIONS = REPO_ROOT / ".github" / "ci-foundation" / "antigravity-review.md"
FOUNDATION_SHA = "93efc801c2f22e08e40000dec2541fd1cafa5f59"


def test_antigravity_caller_uses_the_trusted_shared_workflow() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "name: Antigravity PR Checks" in workflow
    assert "on: # zizmor: ignore[dangerous-triggers]" in workflow
    assert "The PR head is never checked out or executed" in workflow
    assert "\n  pull_request_target:\n" in workflow
    assert "\n  pull_request:\n" not in workflow
    assert "branches: [main]" in workflow
    assert "types: [opened, reopened, synchronize, ready_for_review]" in workflow
    assert "github.repository == 'albumentations-team/albucore'" in workflow
    assert "github.event.pull_request.head.repo.full_name == github.repository" in workflow
    assert "github.event.pull_request.draft == false" in workflow
    assert "contents: read" in workflow
    assert "id-token: write" in workflow
    assert "pull-requests: write" in workflow
    assert (
        "albumentations-team/ci-foundation/.github/workflows/antigravity-review.yml@" + FOUNDATION_SHA
        in workflow
    )
    assert "policy-path: .github/ci-foundation/antigravity.toml" in workflow
    assert "secrets.GEMINI_API_KEY" not in workflow
    assert "run_shell_command" not in workflow


def test_antigravity_policy_and_review_instructions_are_trusted_base_files() -> None:
    policy = POLICY.read_text(encoding="utf-8")
    instructions = INSTRUCTIONS.read_text(encoding="utf-8")

    assert 'include = ["**"]' in policy
    assert 'instructions = ".github/ci-foundation/antigravity-review.md"' in policy
    assert "Read `AGENTS.md`" in instructions
    assert "untrusted input" in instructions
    assert "Do not modify files" in instructions
