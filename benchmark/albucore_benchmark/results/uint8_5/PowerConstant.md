# Benchmark Results: PowerConstant

Number of images: 500

## CPU Information

- CPU: Apple M1 Pro
- Frequency: Current: 3228.00 MHz, Min: 600.00 MHz, Max: 3228.00 MHz
- Physical cores: 10
- Total cores: 10

## Package Versions

| Python                                | albucore   | opencv-python-headless   | numpy   | torchvision   |
|:--------------------------------------|:-----------|:-------------------------|:--------|:--------------|
| 3.9.20 (main, Oct  3 2024, 02:24:59)  | 0.0.20     | 4.10.0.84                | 2.0.2   | 0.19.1        |
| [Clang 14.0.6 ]                       |            |                          |         |               |

## Performance (images/second)

|               | albucore         | lut             | opencv        | numpy         | simsimd   |
|:--------------|:-----------------|:----------------|:--------------|:--------------|:----------|
| PowerConstant | 3241.29 ± 143.27 | 2741.72 ± 96.40 | 184.17 ± 3.33 | 117.40 ± 7.79 | N/A       |