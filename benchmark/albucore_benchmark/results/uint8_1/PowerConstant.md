# Benchmark Results: PowerConstant

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
                       albucore                lut          opencv           numpy     torchvision
PowerConstant  8649.01 ± 993.03  12182.63 ± 339.55  900.60 ± 55.19  589.17 ± 32.33  398.42 ± 27.04

|               | albucore         | lut               | opencv         | numpy          | torchvision    |
|:--------------|:-----------------|:------------------|:---------------|:---------------|:---------------|
| PowerConstant | 8649.01 ± 993.03 | 12182.63 ± 339.55 | 900.60 ± 55.19 | 589.17 ± 32.33 | 398.42 ± 27.04 |
