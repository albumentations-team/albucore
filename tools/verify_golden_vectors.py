"""Verify Albucore golden-vector artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np

from tools.golden_vectors import GOLDEN_CASES, MANIFEST_PATH, OUTPUTS_PATH, array_metadata, output_key


def _case_by_name() -> dict[str, Any]:
    return {case.name: case for case in GOLDEN_CASES}


def _assert_metadata_matches(
    actual: dict[str, Any],
    expected: dict[str, Any],
    label: str,
    keys: tuple[str, ...],
) -> list[str]:
    return [
        f"{label}: metadata {key} mismatch: expected {expected[key]!r}, got {actual[key]!r}"
        for key in keys
        if actual[key] != expected[key]
    ]


def verify() -> list[str]:
    """Return golden-vector verification errors."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    case_by_name = _case_by_name()
    errors: list[str] = []

    with np.load(OUTPUTS_PATH) as expected_arrays:
        for case_entry in manifest["cases"]:
            case_name = case_entry["name"]
            case = case_by_name.get(case_name)
            if case is None:
                errors.append(f"{case_name}: no configured case")
                continue

            computed_outputs = case.compute()
            for output_name, expected_meta in case_entry["outputs"].items():
                label = f"{case_name}.{output_name}"
                key = output_key(case_name, output_name)
                if key not in expected_arrays:
                    errors.append(f"{label}: missing array key {key}")
                    continue
                if output_name not in computed_outputs:
                    errors.append(f"{label}: computed output missing")
                    continue

                expected = expected_arrays[key]
                computed = np.asarray(computed_outputs[output_name])
                errors.extend(
                    _assert_metadata_matches(
                        array_metadata(expected),
                        expected_meta,
                        f"{label}.expected",
                        ("shape", "dtype", "c_contiguous", "sha256"),
                    ),
                )
                errors.extend(
                    _assert_metadata_matches(
                        array_metadata(computed),
                        expected_meta,
                        f"{label}.computed",
                        ("shape", "dtype", "c_contiguous"),
                    ),
                )

                if np.issubdtype(expected.dtype, np.floating):
                    try:
                        np.testing.assert_allclose(computed, expected, rtol=case.rtol, atol=case.atol)
                    except AssertionError as exc:
                        errors.append(f"{label}: values differ: {exc}")
                else:
                    try:
                        np.testing.assert_array_equal(computed, expected)
                    except AssertionError as exc:
                        errors.append(f"{label}: values differ: {exc}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    errors = verify()
    if errors:
        sys.stderr.write("Golden vector verification failed:\n")
        for error in errors:
            sys.stderr.write(f"- {error}\n")
        return 1

    sys.stdout.write("Golden vector verification passed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
