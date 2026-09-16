# External services and third-party assets

This inventory covers the Albucore source repository, package release process,
and Markdown rendered from this repository.

| Component or service | Use and boundary | Data or material | Source and notes |
| --- | --- | --- | --- |
| GitHub and GitHub Actions | Source hosting, pull requests, release automation, and CI badges. | Repository and workflow data. | Workflow actions are pinned by commit. |
| PyPI and the PyTorch wheel index | Resolve dependencies and publish Albucore artifacts. | Package names, versions, hashes, and publishing metadata. | The PyTorch index is used for the documented CPU Torch wheel. |
| CLA Assistant | Individual CLA status for a pull request. | The contributor identity and acceptance statement defined by the CLA procedure. | Acceptance records remain private. |
| README badges | Rendered only by a Markdown viewer through shields.io and GitHub. | Browser requests from a documentation reader; no library runtime call. | The source URLs are in `README.md`; no badges are vendored. |
| PyTorch documentation | Linked installation selector for users choosing a Torch build. | Browser request initiated by a reader. | The Albucore library makes no request to that site. |

The scan covered `albucore/`, `docs/`, `README.md`, `pyproject.toml`, and
`.github/` for network clients and external URLs. No runtime network client,
CDN-loaded runtime code, minified JavaScript, or vendored third-party web asset
was found in the package. Runtime dependency records are maintained in
[`dependency-licenses.json`](../../legal/dependency-licenses.json).
