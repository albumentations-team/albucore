# Shared uint8 LUT output route

Run date: 2026-10-29. The public `apply_uint8_lut` route was measured with a separate, preallocated C-contiguous output. The baseline copies the input to that same output with `np.copyto`, then calls StringZilla `translate(..., inplace=True)`. The new route calls `translate(source, lut, out=out)` directly, removing the full-image copy. Allocating, exact-alias, and strided-output paths are unchanged. StringZilla added caller-owned `out` support in [PR #353](https://github.com/ashvardanian/StringZilla/pull/353), released in 5.3.2.

The grid covered 12 uint8 HWC cases: 128×160, 240×320, 480×640, and 768×1024, each with C=1/3/9. Each comparison used the same non-trivial 256-entry LUT, five warmups, and 51 interleaved paired samples with randomized order. Every output matched exactly and every input remained unchanged.

| Platform | CPU | Python / NumPy / StringZilla / NumKong | Median paired old/new | Cases >5% faster | Regressions >5% | Per-case range |
|---|---|---|---:|---:|---:|---:|
| macOS ARM64 | Apple M4 Max | 3.11.11 / 2.4.6 / 5.3.2 / 7.8.6 | 1.166× | 11/12 | 0/12 | 1.020–1.192× |
| Linux x86-64 | C4, Intel Xeon Platinum 8581C | 3.11.18 / 2.4.6 / 5.3.2 / 7.8.6 | 1.579× | 11/12 | 0/12 | 1.037–1.864× |

The ratio is the median of the 51 paired old/new timing ratios, so values above 1 favor the new route. Thread environment variables were set to one; StringZilla's translate operation is the only timed kernel. Reproduce the grid on the current host with:

```bash
uv run python benchmarks/benchmark_lut_stringzilla_out.py \
  --output /tmp/lut-stringzilla-out.json
```
