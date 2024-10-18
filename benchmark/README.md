# Image Processing Benchmarks

This directory contains various benchmarks for image processing operations, focusing on performance comparisons between different libraries and implementations.

## Overview

Our benchmark suite currently includes:

1. Albucore Benchmark: Evaluates the performance of Albucore against other popular image processing libraries.
2. StringZilla LUT vs cv2.LUT Benchmark: Compares the performance of StringZilla's LUT implementation against OpenCV's cv2.LUT function.

These benchmarks are designed to provide insights into the performance characteristics of different image processing approaches across various operations and image types.

## Benchmark Suites

### 1. Albucore Benchmark

Located in: `./albucore_benchmark/`

This suite compares Albucore's performance against OpenCV, NumPy, and TorchVision for a wide range of image processing operations.

[More details](./albucore_benchmark/README.md)

### 2. StringZilla LUT vs cv2.LUT Benchmark

Located in: `./stringzilla_vs_cv2_lut/`

This benchmark focuses specifically on comparing the performance of Look-Up Table (LUT) operations between StringZilla and OpenCV.

[More details](./stringzilla_vs_cv2_lut/README.md)

## General Requirements

- Python 3.9+
- NumPy
- OpenCV (cv2)
- Pandas
- tqdm

Additional requirements specific to each benchmark are listed in their respective README files.

## Running the Benchmarks

Each benchmark suite has its own script and specific running instructions. Please refer to the individual README files in each benchmark directory for detailed instructions.

## Interpreting Results

While each benchmark produces its own specific outputs, generally you can expect:

- CSV files with detailed results
- Markdown tables summarizing the results
- Console output with summary statistics

Key metrics usually include:

- Execution time for operations
- Images processed per second
- Relative performance (speedup) between different implementations

## Contributing

We welcome contributions to improve these benchmarks or add new ones. If you have suggestions or find any issues, please open an issue or submit a pull request.

When contributing:

1. Ensure your code adheres to the project's coding standards.
2. Update or add appropriate documentation, including README files.
3. Add or update tests as necessary.
4. Verify that all existing tests pass.

## License

[Specify your license here]

## Contact

[Your contact information or contribution guidelines]
