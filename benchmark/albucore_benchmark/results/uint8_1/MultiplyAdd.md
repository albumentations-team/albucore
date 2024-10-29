# Benchmark Results: MultiplyAdd

Number of images: 10

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
                     albucore                lut          opencv           numpy      torchvision
MultiplyAdd  9280.80 ± 130.97  13171.47 ± 163.34  565.93 ± 39.34  975.94 ± 50.21  2942.87 ± 65.14

|             | albucore         | lut               | opencv         | numpy          | torchvision     |
|:------------|:-----------------|:------------------|:---------------|:---------------|:----------------|
| MultiplyAdd | 9280.80 ± 130.97 | 13171.47 ± 163.34 | 565.93 ± 39.34 | 975.94 ± 50.21 | 2942.87 ± 65.14 |
