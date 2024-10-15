# LUT Benchmark

This benchmark compares the performance of OpenCV's `cv2.LUT` function against a custom `sz_lut` function implemented using StringZilla.

## Overview

The benchmark tests the performance of both LUT implementations across various image sizes and channel counts. It provides detailed timing information, speedup calculations, and CPU information to give a comprehensive view of the performance characteristics.

## Requirements

- Python 3.9+
- OpenCV (cv2)
- NumPy
- Pandas
- tqdm
- psutil
- StringZilla

You can install the required packages using:

```python
pip install -r requirements.txt
```


## Running the Benchmark

1. Clone this repository:
   ```nash
   git clone https://github.com/albumentations-team/albucore.git
   cd lut-benchmark
   ```

2. Run the benchmark script:
   ```bash
   python -m benchmark.benchmark_lut --runs 100
   ```

   The `--runs` argument specifies how many times each test should be repeated for more accurate results. Default is 100.

## Benchmark Details

The benchmark performs the following tests:

- Image sizes: 100x100, 500x500, 1000x1000, 2000x2000
- Channel counts: 1 (grayscale), 3 (RGB), 4 (RGBA)
- Each test is repeated the specified number of times to get statistically significant results

For each combination of image size and channel count, the benchmark:

1. Generates a random input image and LUT
2. Times the execution of both `cv2.LUT` and `sz_lut`
3. Calculates the mean execution time and standard deviation
4. Computes the speedup of `sz_lut` compared to `cv2.LUT`

## Output

The benchmark produces the following output:

1. A table of results showing:
   - Image size
   - Number of channels
   - Total number of pixels
   - Execution time for `cv2.LUT` (mean ± std dev)
   - Execution time for `sz_lut` (mean ± std dev)
   - Speedup (mean ± error)

2. Average speedup across all tests
3. Best case speedup scenario (image size and channel count)
4. Worst case speedup scenario (image size and channel count)
5. CPU information (name, frequency, core count)

The results are also saved to a CSV file named `lut_benchmark_results.csv` for further analysis.

## Interpreting Results

- A speedup > 1 indicates that `sz_lut` is faster than `cv2.LUT`
- The error margins provide an indication of the variability in the measurements
- The best and worst case scenarios can help identify where `sz_lut` performs particularly well or poorly
- CPU information is provided to give context to the benchmark results, as performance may vary across different hardware

## Notes

- The benchmark sets various environment variables to limit thread usage and ensure consistent results
- OpenCV's threading and OpenCL usage are disabled for fair comparison
- Random seed is not fixed, so results may vary slightly between runs

## Contributing

If you have suggestions for improving this benchmark or find any issues, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
