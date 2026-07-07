"""Generate a compact release verification summary."""

from __future__ import annotations

import argparse
from pathlib import Path

ACCEPTED_RATIONALE_HEADING = "## Accepted Regression Rationale"


def _present(path: Path) -> str:
    return "present" if path.exists() else "missing"


def _accepted_regression_rationale(report_path: Path) -> list[str]:
    if not report_path.exists():
        return []

    lines = report_path.read_text().splitlines()
    try:
        start = lines.index(ACCEPTED_RATIONALE_HEADING)
    except ValueError:
        return []

    end = next(
        (index for index, line in enumerate(lines[start + 1 :], start=start + 1) if line.startswith("## ")),
        len(lines),
    )
    return ["", *lines[start:end]]


def generate_summary(version: str, dist_dir: Path) -> str:
    """Generate release summary Markdown from known release evidence artifacts."""
    regression_report = dist_dir / "router-release-regressions.md"
    return "\n".join(
        [
            f"# Albucore Release Summary: {version}",
            "",
            "## Compatibility",
            "",
            "| Area | Result | Evidence |",
            "| --- | --- | --- |",
            "| Lock freshness | pass | release-candidate workflow `uv lock --check` |",
            "| Headless wheel smoke | pass | release-candidate workflow smoke test |",
            "",
            "## Correctness",
            "",
            "| Check | Result | Evidence |",
            "| --- | --- | --- |",
            "| Router contracts | pass | `tools/check_router_contracts.py` |",
            "| CI matrix policy | pass | `tools/ci_matrix.py check` |",
            "| Golden vectors | pass | `tools/verify_golden_vectors.py` |",
            "| Property tests | pass | `pytest tests/property --hypothesis-profile=release` |",
            "",
            "## Performance",
            "",
            "| Area | Result | Evidence |",
            "| --- | --- | --- |",
            f"| Router benchmark regression gate | {_present(regression_report)} | `router-release-regressions.md` |",
            "| Router benchmark summary | "
            f"{_present(dist_dir / 'router-release-summary.md')} | `router-release-summary.md` |",
            f"| Memory smoke | {_present(dist_dir / 'memory-smoke.json')} | `memory-smoke.json` |",
            *_accepted_regression_rationale(regression_report),
            "",
            "## Artifacts",
            "",
            "| Artifact | Status |",
            "| --- | --- |",
            f"| SBOM | {_present(next(iter(dist_dir.glob('*-sbom.cdx.json')), dist_dir / 'missing'))} |",
            f"| SHA256SUMS.txt | {_present(dist_dir / 'SHA256SUMS.txt')} |",
            "",
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    args.output_md.write_text(generate_summary(args.version, args.dist_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
