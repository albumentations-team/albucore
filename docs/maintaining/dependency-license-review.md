# Dependency license review

Albucore records its reviewed runtime dependency set in
[`legal/dependency-licenses.json`](../../legal/dependency-licenses.json). It
covers the union of the base package and each declared extra resolved from
`uv.lock`. It is not a dependency set installed together. The base package uses
NumPy, NumKong, and StringZilla. `headless`, `gui`, `contrib`, and
`contrib-headless` are four alternative OpenCV profiles; a user selects one
when installing OpenCV through an Albucore extra. Torch is optional and brings
its own transitive packages. The current union has 17 reviewed possible runtime
components, including platform-specific NumPy, NetworkX, and Torch versions.
Build, test, and CI tools are excluded because they do not ship in the library
wheel or sdist.

The initial review was completed on 2026-09-16 from the locked distribution
metadata and license files, with the named PyPI release or configured PyTorch
index as the source record. The registry records the SPDX expression, reviewed
versions, and required handling for each binary wheel. NumPy, OpenCV, and Torch
can include additional binary components; their notices remain with the
separately installed wheel and must be retained by a distributor of a combined
environment.

`tools/verify_dependency_licenses.py` rejects a dependency name or resolved
version absent from the registry and writes the reviewed SPDX expressions into
the CycloneDX SBOM. The release workflow generates that SBOM from the concrete
`headless,torch` validation profile, then compares each installed distribution's
declared license metadata with the identifiers accepted in the registry. The
security workflow checks the locked runtime export. Future dependency and
license changes follow
[`LICENSE_POLICY.md`](../../LICENSE_POLICY.md).

Albucore does not currently copy runtime dependencies into its wheel or sdist.
If copied or bundled material creates a project-level notice requirement, add
the notice and extend the existing package-integrity verification in the same
pull request.
