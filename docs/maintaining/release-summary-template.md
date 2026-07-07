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
- Performance evidence is reviewed in PR benchmark checks, not in release workflows.
