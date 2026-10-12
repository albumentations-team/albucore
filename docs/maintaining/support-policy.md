# Support Policy

This policy describes what Albucore verification must cover before release.

## Python

Albucore verifies the Python versions declared in `pyproject.toml` classifiers.
`requires-python` sets the minimum installation version without an upper bound.

Current policy:

- Python 3.11, 3.12, 3.13, and 3.14 in CI; package installation requires Python 3.11 or newer
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

Public image kernels support `uint8`, `float16`, and `float32` image data with explicit channel dimensions. `float64`
is not a supported image input dtype. Statistics may return wider result dtypes; the two specialized LUT routers
remain `uint8`-only.

- NumPy: `HWC`, `NHWC`, `DHWC`, `NDHWC`
- Torch CPU: `CHW`, `NCHW`, `CDHW`, `NCDHW`; pass explicit `layout=` where channel-axis semantics apply

Grayscale images use `(H, W, 1)` / `(..., H, W, 1)` for NumPy or a channel-first Tensor with `C=1`.
Implicit-channel grayscale inputs are invalid.

## Support Changes

Dropping a Python version, raising dependency lower bounds, or changing dtype/layout support requires:

- a changelog entry;
- updated `pyproject.toml`;
- updated docs;
- updated CI matrix;
- updated release verification notes.
