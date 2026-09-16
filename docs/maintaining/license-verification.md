# License verification record

On 2026-09-16, GitHub's License API identified this repository's `LICENSE` as
MIT at remote `main` commit `b6f2eded82e86c6cd7b9df1e1c8de053018528f7`.
Published version 0.2.18 is the checked release snapshot.

`tools/verify_legal_integrity.py` verifies the MIT license text, package
metadata, CLA archive, and wheel/sdist exclusion of inbound acceptance material.
The dependency review and SBOM controls are documented in
[dependency-license-review.md](dependency-license-review.md). The contribution
history boundary is recorded in [contribution-history.md](contribution-history.md).

Albucore currently has no copied third-party source, font, or asset that needs
a project-level `THIRD_PARTY_NOTICES.md`. Runtime dependencies are installed
separately from its wheel and sdist. Their licenses and binary-wheel notice
handling are recorded in
[`legal/dependency-licenses.json`](../../legal/dependency-licenses.json); add
project-level notices and package checks in the same pull request if that
boundary changes.
