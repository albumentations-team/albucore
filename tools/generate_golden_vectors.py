"""Generate Albucore golden-vector artifacts."""

from __future__ import annotations

import argparse
import json

import numpy as np

from tools.golden_vectors import GOLDEN_CASES, MANIFEST_PATH, OUTPUTS_PATH, array_metadata, output_key


def generate() -> None:
    """Generate manifest JSON and compressed NPZ outputs."""
    OUTPUTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    cases: list[dict[str, object]] = []

    for case in GOLDEN_CASES:
        outputs = case.compute()
        output_manifest: dict[str, object] = {}
        for name, array in outputs.items():
            key = output_key(case.name, name)
            arrays[key] = np.asarray(array)
            output_manifest[name] = array_metadata(arrays[key])
        cases.append(
            {
                "name": case.name,
                "rtol": case.rtol,
                "atol": case.atol,
                "outputs": output_manifest,
            },
        )

    np.savez_compressed(OUTPUTS_PATH, **arrays)
    manifest = {
        "version": 1,
        "outputs_file": "v1/outputs.npz",
        "cases": cases,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    generate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
