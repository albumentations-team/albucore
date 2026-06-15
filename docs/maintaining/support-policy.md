# Support Policy

This policy describes what Albucore verification must cover before release.

## Python

Albucore supports the Python versions declared in `pyproject.toml` classifiers and
`requires-python`.

Current policy:

- Python 3.10, 3.11, 3.12, 3.13, and 3.14
- Ubuntu CI across every supported Python version
- Windows and macOS smoke coverage on the oldest and latest supported Python versions when CI cost
  allows

Classifier, `requires-python`, CI, and release docs must stay in sync.

## Runtime Dependencies

Runtime dependencies are supported from their declared lower bounds through the latest compatible
versions allowed by `pyproject.toml`.

Required checks:

- locked/latest dependency set in normal CI;
- declared dependency-range job on the oldest supported Python version;
- scheduled latest-dependency job.

When dependency bounds change, update `uv.lock` in the same PR and verify with:

```bash
uv lock --check
```

## OpenCV

Behavior tests use `opencv-python-headless`. Albucore does not depend on OpenCV GUI APIs.

Other OpenCV package variants are packaging alternatives, not separate behavior targets for core
verification.

## Dtypes And Layouts

Public image kernels support `uint8` and `float32` images with explicit channel dimensions:

- HWC
- XHWC
- NDHWC

Grayscale images use `(H, W, 1)` or `(..., H, W, 1)`. Implicit-channel grayscale inputs are invalid.

## Support Changes

Dropping a Python version, raising dependency lower bounds, or changing dtype/layout support requires:

- a changelog entry;
- updated `pyproject.toml`;
- updated docs;
- updated CI matrix;
- updated release verification notes.
