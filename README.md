# albucore

Benchmark results for 1000 images of float32 type with (256, 256, 1):
|                  | albucore<br><small>0.0.1</small> | opencv<br><small>4.9.0.80</small> | numpy<br><small>1.24.4</small> |
| ---------------- | -------------------------------- | --------------------------------- | ------------------------------ |
| MultiplyConstant | 12925 ± 1237                     | 10963 ± 1053                      | **14040 ± 2063**               |
| MultiplyVector   | 3832 ± 512                       | **10824 ± 1005**                  | 8986 ± 511                     |

Benchmark results for 1000 images of uint8 type with (256, 256, 1):
|                  | albucore<br><small>0.0.1</small> | opencv<br><small>4.9.0.80</small> | numpy<br><small>1.24.4</small> |
| ---------------- | -------------------------------- | --------------------------------- | ------------------------------ |
| MultiplyConstant | **24131 ± 1129**                 | 11622 ± 175                       | 6969 ± 643                     |
| MultiplyVector   | **24279 ± 908**                  | 11756 ± 152                       | 6936 ± 408                     |

Benchmark results for 1000 images of float32 type with (256, 256, 3):
|                  | albucore<br><small>0.0.1</small> | opencv<br><small>4.9.0.80</small> | numpy<br><small>1.24.4</small> |
| ---------------- | -------------------------------- | --------------------------------- | ------------------------------ |
| MultiplyConstant | 5313 ± 323                       | 3274 ± 778                        | **5804 ± 288**                 |
| MultiplyVector   | 1037 ± 107                       | **3794 ± 745**                    | 1306 ± 39                      |

Benchmark results for 1000 images of uint8 type with (256, 256, 3):
|                  | albucore<br><small>0.0.1</small> | opencv<br><small>4.9.0.80</small> | numpy<br><small>1.24.4</small> |
| ---------------- | -------------------------------- | --------------------------------- | ------------------------------ |
| MultiplyConstant | **5516 ± 1114**                  | 2026 ± 391                        | 1372 ± 225                     |
| MultiplyVector   | 1365 ± 395                       | **2012 ± 439**                    | 670 ± 56                       |

Benchmark results for 1000 images of float32 type with (256, 256, 7):
|                  | albucore<br><small>0.0.1</small> | opencv<br><small>4.9.0.80</small> | numpy<br><small>1.24.4</small> |
| ---------------- | -------------------------------- | --------------------------------- | ------------------------------ |
| MultiplyConstant | 231 ± 35                         | -                                 | **280 ± 85**                   |
| MultiplyVector   | **611 ± 49**                     | -                                 | 584 ± 165                      |

Benchmark results for 1000 images of uint8 type with (256, 256, 7):
|                  | albucore<br><small>0.0.1</small> | opencv<br><small>4.9.0.80</small> | numpy<br><small>1.24.4</small> |
| ---------------- | -------------------------------- | --------------------------------- | ------------------------------ |
| MultiplyConstant | 688 ± 22                         | -                                 | **1412 ± 68**                  |
| MultiplyVector   | **1807 ± 36**                    | -                                 | 739 ± 43                       |
