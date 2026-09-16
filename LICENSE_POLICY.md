# Dependency and contribution license policy

This policy governs third-party material proposed for Albucore. It does not
replace the repository's [MIT License](LICENSE), the contributor rights
required by the [CLA](CLA.md), or file-specific third-party notices.

## What is reviewed

[`legal/dependency-licenses.json`](legal/dependency-licenses.json) records the
union of components used by the base package and each declared extra. It is not
one installation recipe. The base package has three dependencies: NumPy,
NumKong, and StringZilla. An installation that adds OpenCV through an Albucore
extra selects one of four alternative OpenCV wheels and can also select Torch.
The current registry therefore has 17 reviewed possible runtime components: the
base dependencies, four alternative OpenCV wheels, and Torch with its
transitive dependencies. It records
platform-specific locked versions, each component's SPDX expression, evidence
source, decision, and notice handling. Build, test, and CI tools are outside
this runtime registry because they are not part of the distributed library.

The release SBOM represents the concrete `headless,torch` validation profile;
it does not claim that all four OpenCV wheels are installed together.

The registry is a reviewed record, not a generic list of allowed or forbidden
licenses. Copied or vendored code, binary wheels, fonts, minified assets, and a
combined application environment need their own notice and redistribution
review.

## Changing dependencies or third-party material

Use a normal pull request for a new runtime dependency, a declared license
change, or copied or bundled third-party material. Include the graph change, an
update to the registry with upstream license evidence and a short reason for
the decision, and any necessary notice or packaging change. CI compares the
locked export with the registry, and the release workflow adds the reviewed SPDX
expressions to the published SBOM. Vladimir Iglovikov reviews and merges through
the usual process; no special label or approval service is used.

A version update requires a new reviewed version entry. It can keep the prior
license expression only after its verified license information and applicable
notices are confirmed unchanged. Missing, contradictory, or changed evidence
requires an updated decision before merge. Runtime dependencies are installed
separately from Albucore's wheel and sdist; retain upstream notices when
redistributing a combined environment.

## Contributions

Contributors must have the rights needed to submit a change under the CLA. The
CLA cannot grant rights in third-party material that the contributor does not
control. State the source, version, license, and required attribution for any
such material in the pull request. [CONTRIBUTING.md](CONTRIBUTING.md) gives the
submission checklist.
