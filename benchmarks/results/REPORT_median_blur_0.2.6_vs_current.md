# Median blur 0.2.6 vs current

This report records the public `median_blur` matrix required by issue #119. The baseline is the published
`albucore==0.2.6` wheel (`08eb651`); current is the PR runtime implementation at `9bf521c`, rebased onto `3d3bdf9`.
The channel-contract and report-only review changes made after `9bf521c` do not alter the timed runtime path.

## Method

- Date: 2026-07-31.
- Python 3.12.13, NumPy 2.5.1, OpenCV 5.0.0, one OpenCV thread.
- Linux 7.0.0-28-generic x86-64; the host was lightly loaded before measurement.
- Full matrix: four canonical non-square sizes, 1/3/9 channels, uint8/float32, and kernels 3/5/7.
- Baseline and current each used 3 warmups and 11 timed samples per cell.
- The baseline command ran from an isolated temporary directory. This prevents the checkout from shadowing the
  published wheel on `sys.path`; `albucore.__file__` resolved inside the uv cache's site-packages directory.
- Ratio is baseline/current, so values above 1 mean current is faster.

The independent-process sweep is intentionally reported in full below. Because process order and host-frequency
changes can make unchanged sub-millisecond cells look slower, every unchanged route was also measured with an
alternating, same-process paired probe: all 36 uint8 cells plus all 12 float32 kernel-7 cells, with 5 warmups and 31
samples per candidate. The paired probe recreates the 0.2.6 decorated router and alternates legacy/current order on
every sample.

## Full 72-cell matrix

| Shape | dtype | k3 baseline→current (old/new) | k5 baseline→current (old/new) | k7 baseline→current (old/new) |
|---|---|---:|---:|---:|
| 128x160x1 | float32 | 0.0313→0.0161 (1.95x) | 0.0698→0.0764 (0.91x) | 0.4651→0.6553 (0.71x) |
| 128x160x1 | uint8 | 0.0131→0.0126 (1.04x) | 0.0477→0.0794 (0.60x) | 0.4346→0.6066 (0.72x) |
| 128x160x3 | float32 | 0.0618→0.0297 (2.08x) | 0.1792→0.2023 (0.89x) | 1.2787→1.3105 (0.98x) |
| 128x160x3 | uint8 | 0.0150→0.0228 (0.66x) | 0.1326→0.2188 (0.61x) | 1.2192→1.2463 (0.98x) |
| 128x160x9 | float32 | 0.6354→0.0827 (7.68x) | 0.9904→0.5989 (1.65x) | 4.6636→4.8861 (0.95x) |
| 128x160x9 | uint8 | 0.0501→0.0409 (1.23x) | 0.4999→0.4023 (1.24x) | 4.0572→4.1174 (0.99x) |
| 240x320x1 | float32 | 0.0751→0.0381 (1.97x) | 0.1601→0.2331 (0.69x) | 1.7213→1.6951 (1.02x) |
| 240x320x1 | uint8 | 0.0162→0.0155 (1.04x) | 0.1008→0.1019 (0.99x) | 1.6337→1.6184 (1.01x) |
| 240x320x3 | float32 | 0.8173→0.1233 (6.63x) | 1.0593→0.6988 (1.52x) | 5.3582→5.6118 (0.95x) |
| 240x320x3 | uint8 | 0.0374→0.0369 (1.01x) | 0.2932→0.2951 (0.99x) | 4.6732→5.2727 (0.89x) |
| 240x320x9 | float32 | 2.6119→0.2878 (9.08x) | 3.3941→2.0641 (1.64x) | 17.7390→20.1111 (0.88x) |
| 240x320x9 | uint8 | 0.1028→0.1003 (1.03x) | 0.8773→0.8749 (1.00x) | 15.1671→15.3272 (0.99x) |
| 480x640x1 | float32 | 0.3600→0.1396 (2.58x) | 0.4527→0.9145 (0.49x) | 6.7162→6.7433 (1.00x) |
| 480x640x1 | uint8 | 0.0399→0.0473 (0.84x) | 0.2541→0.3938 (0.65x) | 6.4798→6.9971 (0.93x) |
| 480x640x3 | float32 | 3.9692→0.3778 (10.51x) | 4.1806→2.6041 (1.61x) | 23.7382→22.8223 (1.04x) |
| 480x640x3 | uint8 | 0.1357→0.1053 (1.29x) | 0.9528→0.7579 (1.26x) | 19.3671→19.0190 (1.02x) |
| 480x640x9 | float32 | 6.5734→1.1330 (5.80x) | 8.8873→7.9604 (1.12x) | 67.5470→69.6647 (0.97x) |
| 480x640x9 | uint8 | 0.3254→0.3070 (1.06x) | 2.2682→2.2533 (1.01x) | 61.2404→61.4955 (1.00x) |
| 768x1024x1 | float32 | 0.5358→0.3346 (1.60x) | 0.9674→2.2578 (0.43x) | 17.5209→17.7949 (0.98x) |
| 768x1024x1 | uint8 | 0.0918→0.0857 (1.07x) | 0.5236→0.6463 (0.81x) | 16.7932→16.9744 (0.99x) |
| 768x1024x3 | float32 | 2.5337→0.9956 (2.54x) | 4.0787→6.7284 (0.61x) | 51.2028→51.7842 (0.99x) |
| 768x1024x3 | uint8 | 0.2534→0.2316 (1.09x) | 1.5392→1.5501 (0.99x) | 50.7441→50.0801 (1.01x) |
| 768x1024x9 | float32 | 16.6014→3.1056 (5.35x) | 19.3965→19.6199 (0.99x) | 192.2647→177.6863 (1.08x) |
| 768x1024x9 | uint8 | 0.7283→0.6885 (1.06x) | 4.6381→4.5952 (1.01x) | 162.5060→159.7557 (1.02x) |

Float32 kernel 3 improved by 1.6-10.5x. Kernel 5 is intentionally shape-dependent: the native OpenCV path is
slower for some one- and three-channel images but faster for high-channel cases. The route is required to preserve
precision and follows OpenCV's supported float32 aperture contract rather than a performance threshold.

## Unchanged-route paired confirmation

The independent runs initially showed more than 5% slowdown in the following unchanged cells. Alternating paired
measurement removed process-order noise in every case:

| Shape | dtype | kernel | independent old/new | paired legacy→current ms | paired old/new | paired regression |
|---|---|---:|---:|---:|---:|---:|
| 128x160x1 | float32 | 7 | 0.71x | 0.4612→0.4628 | 0.996x | +0.35% |
| 128x160x1 | uint8 | 5 | 0.60x | 0.0786→0.0783 | 1.005x | -0.47% |
| 128x160x1 | uint8 | 7 | 0.72x | 0.5905→0.5901 | 1.001x | -0.06% |
| 128x160x3 | uint8 | 3 | 0.66x | 0.0150→0.0149 | 1.009x | -0.93% |
| 128x160x3 | uint8 | 5 | 0.61x | 0.1336→0.1335 | 1.001x | -0.12% |
| 240x320x3 | uint8 | 7 | 0.89x | 4.5778→4.5779 | 1.000x | +0.00% |
| 240x320x9 | float32 | 7 | 0.88x | 17.6150→17.6596 | 0.997x | +0.25% |
| 480x640x1 | uint8 | 3 | 0.84x | 0.0400→0.0398 | 1.005x | -0.45% |
| 480x640x1 | uint8 | 5 | 0.65x | 0.2560→0.2527 | 1.013x | -1.28% |
| 480x640x1 | uint8 | 7 | 0.93x | 6.3982→6.3929 | 1.001x | -0.08% |
| 768x1024x1 | uint8 | 5 | 0.81x | 0.5158→0.5161 | 0.999x | +0.06% |

All 48 unchanged-route cells were included in the paired rerun, not only the initially flagged rows above. There were
zero confirmed regressions above 5%; the worst paired slowdown was 0.35%.

## Acceptance result

- Every required matrix cell is reported.
- Native float32 kernel-3/5 performance reflects the precision-preserving implementation.
- The unchanged uint8 and float32 kernel-7 routes have no confirmed regression greater than 5%.
- No routing threshold or implementation change was made from noisy independent-process measurements.
