# Benchmark Results: NormalizePerImagePerChannel

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
                                   albucore            lut         opencv           numpy      torchvision
NormalizePerImagePerChannel  540.99 ± 11.17  545.36 ± 7.59  652.92 ± 6.41  772.75 ± 24.10  969.31 ± 201.96

|                             | albucore       | lut           | opencv        | numpy          | torchvision     |
|:----------------------------|:---------------|:--------------|:--------------|:---------------|:----------------|
| NormalizePerImagePerChannel | 540.99 ± 11.17 | 545.36 ± 7.59 | 652.92 ± 6.41 | 772.75 ± 24.10 | 969.31 ± 201.96 |
