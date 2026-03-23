Albucore NumKong benchmark
  repeats=41, warmup=12
  OpenCV (cv2): available

====================================================================================================
DETAILED RESULTS (median ms; lower is better)
====================================================================================================

## addWeighted / blend (uint8)
----------------------------------------------------------------------------------------------------
  128x128x1 pixels (16384 elems raveled)  numkong_blend=0.0013ms  numpy=0.0240ms  opencv_addWeighted=0.0029ms
      → WINNER: numkong_blend
      → numpy is 18.59x slower than numkong_blend
      → note: albucore add_weighted uses NumKong blend on this path.
  128x128x3 pixels (49152 elems raveled)  numkong_blend=0.0034ms  numpy=0.0654ms  opencv_addWeighted=0.0078ms
      → WINNER: numkong_blend
      → numpy is 19.37x slower than numkong_blend
      → note: albucore add_weighted uses NumKong blend on this path.
  256x256x3 pixels (196608 elems raveled)  numkong_blend=0.0125ms  numpy=0.3130ms  opencv_addWeighted=0.0305ms
      → WINNER: numkong_blend
      → numpy is 24.95x slower than numkong_blend
      → note: albucore add_weighted uses NumKong blend on this path.
  512x512x3 pixels (786432 elems raveled)  numkong_blend=0.0491ms  numpy=1.7249ms  opencv_addWeighted=0.1167ms
      → WINNER: numkong_blend
      → numpy is 35.11x slower than numkong_blend
      → note: albucore add_weighted uses NumKong blend on this path.

## addWeighted / blend (float32)
----------------------------------------------------------------------------------------------------
  256x256x3  numkong_blend=0.0206ms  numpy=0.0434ms  opencv_addWeighted=0.0343ms
      → WINNER: numkong_blend
      → numpy is 2.10x slower than numkong_blend
  512x512x3  numkong_blend=0.2197ms  numpy=0.4405ms  opencv_addWeighted=0.2625ms
      → WINNER: numkong_blend
      → numpy is 2.00x slower than numkong_blend

## pairwise_distances_sq (cdist)
----------------------------------------------------------------------------------------------------
  n=5, m=5, d=2 → n*m=25; albucore: NumKong (n*m < 1000)  numkong_cdist=0.0004ms  numpy_formula=0.0037ms
      → WINNER: numkong_cdist
      → numpy_formula is 9.78x slower than numkong_cdist
      → note: Threshold 1000 matches albucore.functions.pairwise_distances_squared.
  n=10, m=10, d=2 → n*m=100; albucore: NumKong (n*m < 1000)  numkong_cdist=0.0008ms  numpy_formula=0.0038ms
      → WINNER: numkong_cdist
      → numpy_formula is 4.79x slower than numkong_cdist
      → note: Threshold 1000 matches albucore.functions.pairwise_distances_squared.
  n=20, m=20, d=2 → n*m=400; albucore: NumKong (n*m < 1000)  numkong_cdist=0.0023ms  numpy_formula=0.0043ms
      → WINNER: numkong_cdist
      → numpy_formula is 1.84x slower than numkong_cdist
      → note: Threshold 1000 matches albucore.functions.pairwise_distances_squared.
  n=32, m=32, d=2 → n*m=1024; albucore: NumPy (n*m >= 1000)  numkong_cdist=0.0055ms  numpy_formula=0.0052ms
      → WINNER: numpy_formula
      → numkong_cdist is 1.06x slower than numpy_formula
      → note: Threshold 1000 matches albucore.functions.pairwise_distances_squared.
  n=50, m=50, d=2 → n*m=2500; albucore: NumPy (n*m >= 1000)  numkong_cdist=0.0129ms  numpy_formula=0.0081ms
      → WINNER: numpy_formula
      → numkong_cdist is 1.59x slower than numpy_formula
      → note: Threshold 1000 matches albucore.functions.pairwise_distances_squared.
  n=100, m=10, d=2 → n*m=1000; albucore: NumPy (n*m >= 1000)  numkong_cdist=0.0058ms  numpy_formula=0.0065ms
      → WINNER: numkong_cdist
      → numpy_formula is 1.12x slower than numkong_cdist
      → note: Threshold 1000 matches albucore.functions.pairwise_distances_squared.
  n=100, m=100, d=2 → n*m=10000; albucore: NumPy (n*m >= 1000)  numkong_cdist=0.0502ms  numpy_formula=0.0705ms
      → WINNER: numkong_cdist
      → numpy_formula is 1.40x slower than numkong_cdist
      → note: Threshold 1000 matches albucore.functions.pairwise_distances_squared.

## Global sum (uint8 ravel)
----------------------------------------------------------------------------------------------------
  256x256x3 → 196608 elems  Tensor.sum=0.0093ms  np.sum=0.0265ms
      → WINNER: Tensor.sum
      → np.sum is 2.85x slower than Tensor.sum

## sum + sumsq (uint8 — like stats for var)
----------------------------------------------------------------------------------------------------
  256x256x3 → 196608 elems  nk.moments=0.0123ms  numpy_uint64_2pass=0.2141ms
      → WINNER: nk.moments
      → numpy_uint64_2pass is 17.42x slower than nk.moments
      → note: albucore uses nk.moments for global uint8 mean/std in some LUT paths.

## Global sum (uint8 ravel)
----------------------------------------------------------------------------------------------------
  512x512x3 → 786432 elems  Tensor.sum=0.0365ms  np.sum=0.1039ms
      → WINNER: Tensor.sum
      → np.sum is 2.85x slower than Tensor.sum

## sum + sumsq (uint8 — like stats for var)
----------------------------------------------------------------------------------------------------
  512x512x3 → 786432 elems  nk.moments=0.0464ms  numpy_uint64_2pass=0.9890ms
      → WINNER: nk.moments
      → numpy_uint64_2pass is 21.31x slower than nk.moments
      → note: albucore uses nk.moments for global uint8 mean/std in some LUT paths.

## Reduction shape (float32)
----------------------------------------------------------------------------------------------------
  (512, 512, 3): Tensor.sum() over all vs np.mean(axis=(0,1))  Tensor.sum_whole=1.0624ms  np.mean_H_W=1.2865ms
      → WINNER: Tensor.sum_all
      → np.mean_H_W is 1.21x slower than Tensor.sum_whole
      → note: Not identical reductions; shows cost of ‘one big sum’ vs ‘per-channel spatial mean’.

## min + max (float32 contiguous 1D)
----------------------------------------------------------------------------------------------------
  65536 elements  Tensor.minmax_1pass=0.0164ms  np_min_plus_max=0.0063ms
      → WINNER: np.min+np.max
      → Tensor.minmax_1pass is 2.59x slower than np_min_plus_max
      → note: NumPy often wins on large contiguous float32 (optimized separately); verify on your CPU.
  262144 elements  Tensor.minmax_1pass=0.0651ms  np_min_plus_max=0.0208ms
      → WINNER: np.min+np.max
      → Tensor.minmax_1pass is 3.14x slower than np_min_plus_max
      → note: NumPy often wins on large contiguous float32 (optimized separately); verify on your CPU.
  786432 elements  Tensor.minmax_1pass=0.1948ms  np_min_plus_max=0.0602ms
      → WINNER: np.min+np.max
      → Tensor.minmax_1pass is 3.24x slower than np_min_plus_max
      → note: NumPy often wins on large contiguous float32 (optimized separately); verify on your CPU.
  2097152 elements  Tensor.minmax_1pass=0.5191ms  np_min_plus_max=0.1585ms
      → WINNER: np.min+np.max
      → Tensor.minmax_1pass is 3.27x slower than np_min_plus_max
      → note: NumPy often wins on large contiguous float32 (optimized separately); verify on your CPU.

## scale (float32 1D)
----------------------------------------------------------------------------------------------------
  196608 elems  nk.scale=0.0138ms  numpy_a*x+b=0.0275ms
      → WINNER: nk.scale
      → numpy_a*x+b is 1.99x slower than nk.scale

## fma-style (float32 1D)
----------------------------------------------------------------------------------------------------
  196608 elems  nk.fma=0.0205ms  numpy_x*y+x=0.0364ms
      → WINNER: nk.fma
      → numpy_x*y+x is 1.77x slower than nk.fma

## scale (float32 1D)
----------------------------------------------------------------------------------------------------
  786432 elems  nk.scale=0.1942ms  numpy_a*x+b=0.4003ms
      → WINNER: nk.scale
      → numpy_a*x+b is 2.06x slower than nk.scale

## fma-style (float32 1D)
----------------------------------------------------------------------------------------------------
  786432 elems  nk.fma=0.2161ms  numpy_x*y+x=0.2741ms
      → WINNER: nk.fma
      → numpy_x*y+x is 1.27x slower than nk.fma

====================================================================================================
SUMMARY — what is faster when (on this machine)
====================================================================================================
  • [addWeighted / blend (uint8)] 128x128x1 pixels (16384 elems raveled): **numkong_blend** (numpy is 18.59x slower than numkong_blend)
  • [addWeighted / blend (uint8)] 128x128x3 pixels (49152 elems raveled): **numkong_blend** (numpy is 19.37x slower than numkong_blend)
  • [addWeighted / blend (uint8)] 256x256x3 pixels (196608 elems raveled): **numkong_blend** (numpy is 24.95x slower than numkong_blend)
  • [addWeighted / blend (uint8)] 512x512x3 pixels (786432 elems raveled): **numkong_blend** (numpy is 35.11x slower than numkong_blend)
  • [addWeighted / blend (float32)] 256x256x3: **numkong_blend** (numpy is 2.10x slower than numkong_blend)
  • [addWeighted / blend (float32)] 512x512x3: **numkong_blend** (numpy is 2.00x slower than numkong_blend)
  • [pairwise_distances_sq (cdist)] n=5, m=5, d=2 → n*m=25; albucore: NumKong (n*m < 1000): **numkong_cdist** (numpy_formula is 9.78x slower than numkong_cdist)
  • [pairwise_distances_sq (cdist)] n=10, m=10, d=2 → n*m=100; albucore: NumKong (n*m < 1000): **numkong_cdist** (numpy_formula is 4.79x slower than numkong_cdist)
  • [pairwise_distances_sq (cdist)] n=20, m=20, d=2 → n*m=400; albucore: NumKong (n*m < 1000): **numkong_cdist** (numpy_formula is 1.84x slower than numkong_cdist)
  • [pairwise_distances_sq (cdist)] n=32, m=32, d=2 → n*m=1024; albucore: NumPy (n*m >= 1000): **numpy_formula** (numkong_cdist is 1.06x slower than numpy_formula)
  • [pairwise_distances_sq (cdist)] n=50, m=50, d=2 → n*m=2500; albucore: NumPy (n*m >= 1000): **numpy_formula** (numkong_cdist is 1.59x slower than numpy_formula)
  • [pairwise_distances_sq (cdist)] n=100, m=10, d=2 → n*m=1000; albucore: NumPy (n*m >= 1000): **numkong_cdist** (numpy_formula is 1.12x slower than numkong_cdist)
  • [pairwise_distances_sq (cdist)] n=100, m=100, d=2 → n*m=10000; albucore: NumPy (n*m >= 1000): **numkong_cdist** (numpy_formula is 1.40x slower than numkong_cdist)
  • [Global sum (uint8 ravel)] 256x256x3 → 196608 elems: **Tensor.sum** (np.sum is 2.85x slower than Tensor.sum)
  • [sum + sumsq (uint8 — like stats for var)] 256x256x3 → 196608 elems: **nk.moments** (numpy_uint64_2pass is 17.42x slower than nk.moments)
  • [Global sum (uint8 ravel)] 512x512x3 → 786432 elems: **Tensor.sum** (np.sum is 2.85x slower than Tensor.sum)
  • [sum + sumsq (uint8 — like stats for var)] 512x512x3 → 786432 elems: **nk.moments** (numpy_uint64_2pass is 21.31x slower than nk.moments)
  • [Reduction shape (float32)] (512, 512, 3): Tensor.sum() over all vs np.mean(axis=(0,1)): **Tensor.sum_all** (np.mean_H_W is 1.21x slower than Tensor.sum_whole)
  • [min + max (float32 contiguous 1D)] 65536 elements: **np.min+np.max** (Tensor.minmax_1pass is 2.59x slower than np_min_plus_max)
  • [min + max (float32 contiguous 1D)] 262144 elements: **np.min+np.max** (Tensor.minmax_1pass is 3.14x slower than np_min_plus_max)
  • [min + max (float32 contiguous 1D)] 786432 elements: **np.min+np.max** (Tensor.minmax_1pass is 3.24x slower than np_min_plus_max)
  • [min + max (float32 contiguous 1D)] 2097152 elements: **np.min+np.max** (Tensor.minmax_1pass is 3.27x slower than np_min_plus_max)
  • [scale (float32 1D)] 196608 elems: **nk.scale** (numpy_a*x+b is 1.99x slower than nk.scale)
  • [fma-style (float32 1D)] 196608 elems: **nk.fma** (numpy_x*y+x is 1.77x slower than nk.fma)
  • [scale (float32 1D)] 786432 elems: **nk.scale** (numpy_a*x+b is 2.06x slower than nk.scale)
  • [fma-style (float32 1D)] 786432 elems: **nk.fma** (numpy_x*y+x is 1.27x slower than nk.fma)


Note: NumKong 7.x Python has `Tensor.sum`, `moments`, etc.; no `Tensor.mean` here — use sum/size or moments-derived mean for benchmarks.
