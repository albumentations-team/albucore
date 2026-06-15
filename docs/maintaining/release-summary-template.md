# Albucore Release Summary Template

Version: `<version>`

Commit: `<sha>`

Release URL: `<url>`

## Compatibility

| Area | Result | Evidence |
| --- | --- | --- |
| Python matrix |  |  |
| OS smoke |  |  |
| Locked dependencies |  |  |
| Lower-bound dependencies |  |  |
| Headless OpenCV wheel smoke |  |  |

## Correctness

| Check | Result | Evidence |
| --- | --- | --- |
| Unit tests |  |  |
| Router contract check |  |  |
| Golden vectors |  |  |
| Property tests |  |  |

## Performance

Baseline version: `<previous-version>`

| Area | Result | Evidence |
| --- | --- | --- |
| Router benchmark comparison |  |  |
| Release-blocking hot paths |  |  |
| Memory smoke |  |  |

## Security And Artifacts

| Check | Result | Evidence |
| --- | --- | --- |
| Runtime dependency audit |  |  |
| GitHub Actions audit |  |  |
| SBOM |  |  |
| SHA256 checksums |  |  |
| PyPI provenance |  |  |

## Known Limitations

- Behavior tests use `opencv-python-headless`.
- Benchmark results from hosted runners are noisy; blocking decisions use documented thresholds and
  maintainer review.
