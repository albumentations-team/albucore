# Albucore Benchmark

This benchmark suite evaluates the performance of various image processing operations implemented in Albucore, comparing them against other popular libraries such as OpenCV, NumPy, and TorchVision.

## Overview

Albucore is a high-performance image processing library designed for efficiency and ease of use. This benchmark aims to quantify its performance across a range of common image operations, providing insights into its strengths and areas for potential optimization.

## Requirements

- Python 3.9+
- Albucore
- NumPy
- OpenCV (cv2)
- PyTorch and TorchVision
- Pandas (for results processing)
- tqdm (for progress bars)

You can install the required packages using:

```bash
pip install -r requirements.txt
```

## Running the Benchmark

Run the benchmark script from the root of the repo:

```bash
python -m benchmark.albucore_benchmark.benchmark -d /path/to/image/directory -n 2000 -c 3 -t uint8 -r 5 -m
```

Options:

- `-d, --data-dir`: Path to a directory with sample images (required)
- `-n, --num_images`: Number of images to use in the benchmark (default: 2000)
- `-c, --num_channels`: Number of channels in the images (default: 3)
- `-t, --img_type`: Image data type ('float32' or 'uint8', default: 'uint8')
- `-r, --runs`: Number of runs for each benchmark (default: 5)
- `--show-std`: Show standard deviation for benchmark runs
- `-p, --print-package-versions`: Print versions of packages used
- `-m, --markdown`: Print benchmarking results as a markdown table
- `-b, --benchmark`: Specific benchmark class to run (if not provided, all benchmarks will be run)

## Benchmark Details

The benchmark suite includes the following operations:

1. Multiply (Constant, Vector, Array)
2. Add (Constant, Vector, Array)
3. Normalize (various methods)
4. Power
5. Add Weighted
6. Multiply Add
7. Type Conversions (To/From Float)
8. Flips (Horizontal, Vertical)

Each operation is tested across different libraries (Albucore, OpenCV, NumPy, TorchVision) when applicable.

## Interpreting Results

The benchmark outputs results in the following formats:

- Markdown tables (if `-m` option is used) saved in the `results` directory
- Console output with summary statistics

Key metrics to look for:

- Images processed per second for each operation and library
- Relative performance of Albucore compared to other libraries
- Performance across different image types (uint8 vs float32)

## Notes

- The benchmark sets various environment variables to limit thread usage and ensure consistent results
- OpenCV's threading and OpenCL usage are disabled for fair comparison
- Random seed is set for reproducibility, but may be reset between benchmark runs

## Contributing

If you have suggestions for improving this benchmark suite or find any issues, please open an issue or submit a pull request in the main repository.

## License

This project is licensed under the MIT License.
