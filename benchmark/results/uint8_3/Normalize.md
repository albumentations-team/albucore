# Benchmark Results: Normalize

Number of images: 100

## CPU Information

- CPU: Apple M1 Pro
- Frequency: Current: 3228.00 MHz, Min: 600.00 MHz, Max: 3228.00 MHz
- Physical cores: 10
- Total cores: 10

## Package Versions

| Python                                | albucore   | opencv-python-headless   | numpy   | torchvision   |
|:--------------------------------------|:-----------|:-------------------------|:--------|:--------------|
| 3.9.20 (main, Oct  3 2024, 02:24:59)  | 0.0.17     | 4.10.0.84                | 2.0.2   | 0.19.1        |
| [Clang 14.0.6 ]                       |            |                          |         |               |

## Performance (images/second)

Raw data:
                 albucore            lut        opencv          numpy    torchvision
Normalize  248.03 ± 10.94  261.17 ± 5.82  96.01 ± 0.23  120.34 ± 0.29  436.90 ± 3.48

|           | albucore       | lut           | opencv       | numpy         | torchvision   |
|:----------|:---------------|:--------------|:-------------|:--------------|:--------------|
| Normalize | 248.03 ± 10.94 | 261.17 ± 5.82 | 96.01 ± 0.23 | 120.34 ± 0.29 | 436.90 ± 3.48 |
