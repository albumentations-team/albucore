# FP16 cast guard storage

The baseline is the exponent-mask guard from Albucore `3c876924` / published 0.3.3. The candidate replaces its
full-image mask with integer reductions over views. All other numerical routing and public ownership rules are shared.

Each platform has 108 paired cells: four non-square H/W sizes, C=1/3/9, positive/negative/mixed finite values,
and direct guard / allocating conversion / caller-provided output. Outputs match bitwise and inputs remain unchanged.
Python 3.11.15, NumPy 2.4.6, OpenCV 5.0.0.93, NumKong 7.8.7, StringZilla 5.3.2 and CPU Torch 2.15.0 use one native thread.

| Platform | Route | Median new/old time | Maximum new/old time |
| --- | --- | ---: | ---: |
| Apple M4 Max | Guard | 0.727 | 0.871 |
| Apple M4 Max | Allocating cast | 1.000 | 1.012 |
| Apple M4 Max | Explicit output | 1.000 | 1.032 |
| Intel Xeon 2.80 GHz | Guard | 0.556 | 0.806 |
| Intel Xeon 2.80 GHz | Allocating cast | 0.759 | 0.906 |
| Intel Xeon 2.80 GHz | Explicit output | 0.698 | 0.847 |

The public x86 route executes the guard; ARM retains the same NumPy cast path. No measured cell regresses over 5%.
The largest guard-only traced temporary falls from 13.5 MiB to less than 2 KiB. With a resident 48 MiB output,
fresh Linux peak RSS falls from 342.45 to 318.72 MiB. ARM public allocation behavior is unchanged; its two single-process
RSS observations vary by less than 0.4 MiB. [memory.csv](memory.csv) records fresh-process peak and steady values.

Run from the Core checkout with the Python 3.11 acceptance environment:

```bash
python benchmarks/benchmark_float_cast_guard.py --output guard-pairs.json
python benchmarks/benchmark_float_cast_guard.py --memory-worker out --variant baseline --output rss-before.json
python benchmarks/benchmark_float_cast_guard.py --memory-worker out --variant candidate --output rss-after.json
```

The script asserts its imported Core path, verifies native NEON on ARM, preserves every completed row atomically,
and records package versions, capabilities, thread settings and the exact candidate cast-source hash.
[macos.csv](macos.csv) and [linux.csv](linux.csv) retain every cell; [metadata.json](metadata.json) identifies the profiles.

## Unit-scale byte conversion

Contiguous `uint8 → float32` with `max_value=1` now casts directly instead of building a normalization LUT.
The target dtype, C-order allocation and explicit-output contract stay unchanged; other scales/layouts retain their routes.
The 24 paired cells per platform cover the same four non-square sizes and C=1/3/9, allocating and explicit-output calls.
All outputs match the full byte-domain reference bitwise.

| Platform | Median new/old time | Maximum new/old time |
| --- | ---: | ---: |
| Apple M4 Max | 0.989 | 1.008 |
| Intel Xeon 2.80 GHz | 0.406 | 0.661 |

```bash
python benchmarks/benchmark_float_cast_guard.py --raw-unit-scale --output raw-pairs.json
```

[raw-macos.csv](raw-macos.csv) and [raw-linux.csv](raw-linux.csv) retain every cell. The candidate converter source hash
is recorded alongside the complete runtime profile in metadata.json.
