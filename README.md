# albucore

Benchmark results for 1000 images of uint8 type:
|                  | albucore<br><small>0.0.1</small> | opencv<br><small>4.9.0.80</small> | numpy<br><small>1.24.4</small> |
| ---------------- | -------------------------------- | --------------------------------- | ------------------------------ |
| MultiplyConstant | **11505 ± 95**                   | 4203 ± 39                         | 11120 ± 641                    |


Benchmark results for 1000 images of float32 type:
|                  | albucore<br><small>0.0.1</small> | opencv<br><small>4.9.0.80</small> | numpy<br><small>1.24.4</small> |
| ---------------- | -------------------------------- | --------------------------------- | ------------------------------ |
| MultiplyConstant | 22274 ± 10938                    | 3401 ± 1729                       | **28286 ± 2460**               |
