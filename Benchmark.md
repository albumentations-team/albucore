# Benchmark

## Benchmark results for 1000 images of float32 type with (256, 256, 1):

|                  | albucore<br><small>0.0.1</small> | opencv<br><small>4.9.0.80</small> | numpy<br><small>1.24.4</small> |
| ---------------- | -------------------------------- | --------------------------------- | ------------------------------ |
| MultiplyConstant | **12636 ± 1248**                 | 10129 ± 663                       | 10113 ± 2622                   |
| MultiplyVector   | 4535 ± 447                       | **10907 ± 1255**                  | 9260 ± 651                     |

## Benchmark results for 1000 images of uint8 type with (256, 256, 1)

|                  | albucore<br><small>0.0.1</small> | opencv<br><small>4.9.0.80</small> | numpy<br><small>1.24.4</small> |
| ---------------- | -------------------------------- | --------------------------------- | ------------------------------ |
| MultiplyConstant | **23798 ± 773**                  | 11504 ± 177                       | 6487 ± 423                     |
| MultiplyVector   | **23576 ± 1429**                 | 11299 ± 180                       | 6263 ± 272                     |

## Benchmark results for 1000 images of float32 type with (256, 256, 3)

|                  | albucore<br><small>0.0.1</small> | opencv<br><small>4.9.0.80</small> | numpy<br><small>1.24.4</small> |
| ---------------- | -------------------------------- | --------------------------------- | ------------------------------ |
| MultiplyConstant | 4783 ± 1071                      | 3835 ± 224                        | **5642 ± 255**                 |
| MultiplyVector   | 1032 ± 45                        | **3740 ± 233**                    | 1204 ± 182                     |

## Benchmark results for 1000 images of uint8 type with (256, 256, 3)

|                  | albucore<br><small>0.0.1</small> | opencv<br><small>4.9.0.80</small> | numpy<br><small>1.24.4</small> |
| ---------------- | -------------------------------- | --------------------------------- | ------------------------------ |
| MultiplyConstant | **8635 ± 1230**                  | 3578 ± 135                        | 3333 ± 263                     |
| MultiplyVector   | **3906 ± 169**                   | 3596 ± 75                         | 1289 ± 48                      |

## Benchmark results for 1000 images of float32 type with (256, 256, 7)

|                  | albucore<br><small>0.0.1</small> | opencv<br><small>4.9.0.80</small> | numpy<br><small>1.24.4</small> |
| ---------------- | -------------------------------- | --------------------------------- | ------------------------------ |
| MultiplyConstant | **1759 ± 499**                   | -                                 | 1363 ± 473                     |
| MultiplyVector   | 302 ± 184                        | -                                 | **624 ± 59**                   |

## Benchmark results for 1000 images of uint8 type with (256, 256, 7):

|                  | albucore<br><small>0.0.1</small> | opencv<br><small>4.9.0.80</small> | numpy<br><small>1.24.4</small> |
| ---------------- | -------------------------------- | --------------------------------- | ------------------------------ |
| MultiplyConstant | 330 ± 28                         | -                                 | **446 ± 42**                   |
| MultiplyVector   | 660 ± 217                        | -                                 | **751 ± 20**                   |
