# Benchmark Results: MultiplyConstant

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
                          albucore                lut           opencv            numpy      torchvision
MultiplyConstant  7222.91 ± 703.90  10563.73 ± 386.46  1139.02 ± 11.01  1431.15 ± 28.97  3248.38 ± 46.54

|                  | albucore         | lut               | opencv          | numpy           | torchvision     |
|:-----------------|:-----------------|:------------------|:----------------|:----------------|:----------------|
| MultiplyConstant | 7222.91 ± 703.90 | 10563.73 ± 386.46 | 1139.02 ± 11.01 | 1431.15 ± 28.97 | 3248.38 ± 46.54 |
