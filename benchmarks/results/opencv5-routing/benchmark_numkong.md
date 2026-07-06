Albucore NumKong benchmark
  repeats=11, warmup=3
  OpenCV (cv2): available

====================================================================================================
DETAILED RESULTS (median ms; lower is better)
====================================================================================================

## addWeighted / blend (uint8)
----------------------------------------------------------------------------------------------------
  128x128x1 pixels (16384 elems raveled)  numkong_blend=0.0014ms  numpy=0.0259ms  opencv_addWeighted=0.0032ms
      → WINNER: numkong_blend
      → numpy is 18.29x slower than numkong_blend
      → note: albucore add_weighted uses NumKong blend on this path.
  128x128x3 pixels (49152 elems raveled)  numkong_blend=0.0036ms  numpy=0.0693ms  opencv_addWeighted=0.0083ms
      → WINNER: numkong_blend
      → numpy is 19.13x slower than numkong_blend
      → note: albucore add_weighted uses NumKong blend on this path.
  256x256x3 pixels (196608 elems raveled)  numkong_blend=0.0133ms  numpy=0.2580ms  opencv_addWeighted=0.0307ms
      → WINNER: numkong_blend
      → numpy is 19.47x slower than numkong_blend
      → note: albucore add_weighted uses NumKong blend on this path.
  512x512x3 pixels (786432 elems raveled)  numkong_blend=0.0505ms  numpy=1.0108ms  opencv_addWeighted=0.1206ms
      → WINNER: numkong_blend
      → numpy is 20.00x slower than numkong_blend
      → note: albucore add_weighted uses NumKong blend on this path.

## addWeighted / blend (float32)
----------------------------------------------------------------------------------------------------
  256x256x3  numkong_blend=0.0213ms  numpy=0.0445ms  opencv_addWeighted=0.0207ms
      → WINNER: opencv_addWeighted
      → numpy is 2.15x slower than opencv_addWeighted
  512x512x3  numkong_blend=0.0849ms  numpy=0.1769ms  opencv_addWeighted=0.0821ms
      → WINNER: opencv_addWeighted
      → numpy is 2.15x slower than opencv_addWeighted

## pairwise_distances_sq (cdist)
----------------------------------------------------------------------------------------------------
  n=5, m=5, d=2 → n*m=25; albucore: NumKong (n*m < 1000)  numkong_cdist=0.0005ms  numpy_formula=0.0043ms
      → WINNER: numkong_cdist
      → numpy_formula is 9.37x slower than numkong_cdist
      → note: Threshold 1000 matches albucore.functions.pairwise_distances_squared.
  n=10, m=10, d=2 → n*m=100; albucore: NumKong (n*m < 1000)  numkong_cdist=0.0010ms  numpy_formula=0.0043ms
      → WINNER: numkong_cdist
      → numpy_formula is 4.52x slower than numkong_cdist
      → note: Threshold 1000 matches albucore.functions.pairwise_distances_squared.
  n=20, m=20, d=2 → n*m=400; albucore: NumKong (n*m < 1000)  numkong_cdist=0.0037ms  numpy_formula=0.0047ms
      → WINNER: numkong_cdist
      → numpy_formula is 1.26x slower than numkong_cdist
      → note: Threshold 1000 matches albucore.functions.pairwise_distances_squared.
  n=32, m=32, d=2 → n*m=1024; albucore: NumPy (n*m >= 1000)  numkong_cdist=0.0071ms  numpy_formula=0.0057ms
      → WINNER: numpy_formula
      → numkong_cdist is 1.25x slower than numpy_formula
      → note: Threshold 1000 matches albucore.functions.pairwise_distances_squared.
  n=50, m=50, d=2 → n*m=2500; albucore: NumPy (n*m >= 1000)  numkong_cdist=0.0170ms  numpy_formula=0.0102ms
      → WINNER: numpy_formula
      → numkong_cdist is 1.66x slower than numpy_formula
      → note: Threshold 1000 matches albucore.functions.pairwise_distances_squared.
  n=100, m=10, d=2 → n*m=1000; albucore: NumPy (n*m >= 1000)  numkong_cdist=0.0073ms  numpy_formula=0.0071ms
      → WINNER: numpy_formula
      → numkong_cdist is 1.03x slower than numpy_formula
      → note: Threshold 1000 matches albucore.functions.pairwise_distances_squared.
  n=100, m=100, d=2 → n*m=10000; albucore: NumPy (n*m >= 1000)  numkong_cdist=0.0683ms  numpy_formula=0.0179ms
      → WINNER: numpy_formula
      → numkong_cdist is 3.82x slower than numpy_formula
      → note: Threshold 1000 matches albucore.functions.pairwise_distances_squared.

## Global sum (uint8 ravel)
----------------------------------------------------------------------------------------------------
  256x256x3 → 196608 elems  Tensor.sum=0.0095ms  np.sum=0.0283ms
      → WINNER: Tensor.sum
      → np.sum is 2.98x slower than Tensor.sum

## sum + sumsq (uint8 — like stats for var)
----------------------------------------------------------------------------------------------------
  256x256x3 → 196608 elems  nk.moments=0.0131ms  numpy_uint64_2pass=0.1078ms
      → WINNER: nk.moments
      → numpy_uint64_2pass is 8.21x slower than nk.moments
      → note: albucore uses nk.moments for global uint8 mean/std in some LUT paths.

## Global sum (uint8 ravel)
----------------------------------------------------------------------------------------------------
  512x512x3 → 786432 elems  Tensor.sum=0.0373ms  np.sum=0.1072ms
      → WINNER: Tensor.sum
      → np.sum is 2.87x slower than Tensor.sum

## sum + sumsq (uint8 — like stats for var)
----------------------------------------------------------------------------------------------------
  512x512x3 → 786432 elems  nk.moments=0.0478ms  numpy_uint64_2pass=0.4249ms
      → WINNER: nk.moments
      → numpy_uint64_2pass is 8.89x slower than nk.moments
      → note: albucore uses nk.moments for global uint8 mean/std in some LUT paths.

## Reduction shape (float32)
----------------------------------------------------------------------------------------------------
  (512, 512, 3): Tensor.sum() over all vs np.mean(axis=(0,1))  Tensor.sum_whole=0.3524ms  np.mean_H_W=1.2702ms
      → WINNER: Tensor.sum_all
      → np.mean_H_W is 3.60x slower than Tensor.sum_whole
      → note: Not identical reductions; shows cost of ‘one big sum’ vs ‘per-channel spatial mean’.

## min + max (float32 contiguous 1D)
----------------------------------------------------------------------------------------------------
  65536 elements  Tensor.minmax_1pass=0.0166ms  np_min_plus_max=0.0066ms
      → WINNER: np.min+np.max
      → Tensor.minmax_1pass is 2.53x slower than np_min_plus_max
      → note: NumPy often wins on large contiguous float32 (optimized separately); verify on your CPU.
  262144 elements  Tensor.minmax_1pass=0.0658ms  np_min_plus_max=0.0217ms
      → WINNER: np.min+np.max
      → Tensor.minmax_1pass is 3.04x slower than np_min_plus_max
      → note: NumPy often wins on large contiguous float32 (optimized separately); verify on your CPU.
  786432 elements  Tensor.minmax_1pass=0.2040ms  np_min_plus_max=0.0660ms
      → WINNER: np.min+np.max
      → Tensor.minmax_1pass is 3.09x slower than np_min_plus_max
      → note: NumPy often wins on large contiguous float32 (optimized separately); verify on your CPU.
  2097152 elements  Tensor.minmax_1pass=0.5226ms  np_min_plus_max=0.1708ms
      → WINNER: np.min+np.max
      → Tensor.minmax_1pass is 3.06x slower than np_min_plus_max
      → note: NumPy often wins on large contiguous float32 (optimized separately); verify on your CPU.

## scale (float32 1D)
----------------------------------------------------------------------------------------------------
  196608 elems  nk.scale=0.0140ms  numpy_a*x+b=0.0280ms
      → WINNER: nk.scale
      → numpy_a*x+b is 2.01x slower than nk.scale

## fma-style (float32 1D)
----------------------------------------------------------------------------------------------------
  196608 elems  nk.fma=0.0217ms  numpy_x*y+x=0.0377ms
      → WINNER: nk.fma
      → numpy_x*y+x is 1.74x slower than nk.fma

## scale (float32 1D)
----------------------------------------------------------------------------------------------------
  786432 elems  nk.scale=0.0592ms  numpy_a*x+b=0.0990ms
      → WINNER: nk.scale
      → numpy_a*x+b is 1.67x slower than nk.scale

## fma-style (float32 1D)
----------------------------------------------------------------------------------------------------
  786432 elems  nk.fma=0.0778ms  numpy_x*y+x=0.1360ms
      → WINNER: nk.fma
      → numpy_x*y+x is 1.75x slower than nk.fma

====================================================================================================
SUMMARY — what is faster when (on this machine)
====================================================================================================
  • [addWeighted / blend (uint8)] 128x128x1 pixels (16384 elems raveled): **numkong_blend** (numpy is 18.29x slower than numkong_blend)
  • [addWeighted / blend (uint8)] 128x128x3 pixels (49152 elems raveled): **numkong_blend** (numpy is 19.13x slower than numkong_blend)
  • [addWeighted / blend (uint8)] 256x256x3 pixels (196608 elems raveled): **numkong_blend** (numpy is 19.47x slower than numkong_blend)
  • [addWeighted / blend (uint8)] 512x512x3 pixels (786432 elems raveled): **numkong_blend** (numpy is 20.00x slower than numkong_blend)
  • [addWeighted / blend (float32)] 256x256x3: **opencv_addWeighted** (numpy is 2.15x slower than opencv_addWeighted)
  • [addWeighted / blend (float32)] 512x512x3: **opencv_addWeighted** (numpy is 2.15x slower than opencv_addWeighted)
  • [pairwise_distances_sq (cdist)] n=5, m=5, d=2 → n*m=25; albucore: NumKong (n*m < 1000): **numkong_cdist** (numpy_formula is 9.37x slower than numkong_cdist)
  • [pairwise_distances_sq (cdist)] n=10, m=10, d=2 → n*m=100; albucore: NumKong (n*m < 1000): **numkong_cdist** (numpy_formula is 4.52x slower than numkong_cdist)
  • [pairwise_distances_sq (cdist)] n=20, m=20, d=2 → n*m=400; albucore: NumKong (n*m < 1000): **numkong_cdist** (numpy_formula is 1.26x slower than numkong_cdist)
  • [pairwise_distances_sq (cdist)] n=32, m=32, d=2 → n*m=1024; albucore: NumPy (n*m >= 1000): **numpy_formula** (numkong_cdist is 1.25x slower than numpy_formula)
  • [pairwise_distances_sq (cdist)] n=50, m=50, d=2 → n*m=2500; albucore: NumPy (n*m >= 1000): **numpy_formula** (numkong_cdist is 1.66x slower than numpy_formula)
  • [pairwise_distances_sq (cdist)] n=100, m=10, d=2 → n*m=1000; albucore: NumPy (n*m >= 1000): **numpy_formula** (numkong_cdist is 1.03x slower than numpy_formula)
  • [pairwise_distances_sq (cdist)] n=100, m=100, d=2 → n*m=10000; albucore: NumPy (n*m >= 1000): **numpy_formula** (numkong_cdist is 3.82x slower than numpy_formula)
  • [Global sum (uint8 ravel)] 256x256x3 → 196608 elems: **Tensor.sum** (np.sum is 2.98x slower than Tensor.sum)
  • [sum + sumsq (uint8 — like stats for var)] 256x256x3 → 196608 elems: **nk.moments** (numpy_uint64_2pass is 8.21x slower than nk.moments)
  • [Global sum (uint8 ravel)] 512x512x3 → 786432 elems: **Tensor.sum** (np.sum is 2.87x slower than Tensor.sum)
  • [sum + sumsq (uint8 — like stats for var)] 512x512x3 → 786432 elems: **nk.moments** (numpy_uint64_2pass is 8.89x slower than nk.moments)
  • [Reduction shape (float32)] (512, 512, 3): Tensor.sum() over all vs np.mean(axis=(0,1)): **Tensor.sum_all** (np.mean_H_W is 3.60x slower than Tensor.sum_whole)
  • [min + max (float32 contiguous 1D)] 65536 elements: **np.min+np.max** (Tensor.minmax_1pass is 2.53x slower than np_min_plus_max)
  • [min + max (float32 contiguous 1D)] 262144 elements: **np.min+np.max** (Tensor.minmax_1pass is 3.04x slower than np_min_plus_max)
  • [min + max (float32 contiguous 1D)] 786432 elements: **np.min+np.max** (Tensor.minmax_1pass is 3.09x slower than np_min_plus_max)
  • [min + max (float32 contiguous 1D)] 2097152 elements: **np.min+np.max** (Tensor.minmax_1pass is 3.06x slower than np_min_plus_max)
  • [scale (float32 1D)] 196608 elems: **nk.scale** (numpy_a*x+b is 2.01x slower than nk.scale)
  • [fma-style (float32 1D)] 196608 elems: **nk.fma** (numpy_x*y+x is 1.74x slower than nk.fma)
  • [scale (float32 1D)] 786432 elems: **nk.scale** (numpy_a*x+b is 1.67x slower than nk.scale)
  • [fma-style (float32 1D)] 786432 elems: **nk.fma** (numpy_x*y+x is 1.75x slower than nk.fma)


Note: NumKong 7.x Python has `Tensor.sum`, `moments`, etc.; no `Tensor.mean` here — use sum/size or moments-derived mean for benchmarks.
