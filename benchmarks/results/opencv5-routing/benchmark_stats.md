
=== uint8 ===

--- mean_std(global) ---
  HWC 240x320x3
    albucore 0.0120 ms | numpy 0.3155 ms → albucore (26.30x)
  HWC 480x640x3
    albucore 0.0442 ms | numpy 1.2407 ms → albucore (28.04x)
  HWC 480x640x9
    albucore 0.1304 ms | numpy 3.8684 ms → albucore (29.67x)
  HWC 768x1024x3
    albucore 0.1113 ms | numpy 3.2629 ms → albucore (29.31x)
  NHWC 4x240x320x3
    albucore 0.0439 ms | numpy 1.2541 ms → albucore (28.56x)
  NHWC 4x240x320x9
    albucore 0.1319 ms | numpy 3.8247 ms → albucore (28.99x)

--- mean_std(per_channel) ---
  HWC 240x320x3
    albucore 0.0150 ms | numpy 1.8898 ms → albucore (125.64x)
  HWC 480x640x3
    albucore 0.0480 ms | numpy 7.3754 ms → albucore (153.65x)
  HWC 480x640x9
    albucore 0.8677 ms | numpy 9.6896 ms → albucore (11.17x)
  HWC 768x1024x3
    albucore 0.1136 ms | numpy 18.0871 ms → albucore (159.18x)
  NHWC 4x240x320x3
    albucore 0.0481 ms | numpy 7.1817 ms → albucore (149.36x)
  NHWC 4x240x320x9
    albucore 0.8342 ms | numpy 9.7505 ms → albucore (11.69x)

--- std(global) ---
  HWC 240x320x3
    albucore 0.0121 ms | numpy 0.2336 ms → albucore (19.33x)
  HWC 480x640x3
    albucore 0.0445 ms | numpy 0.9090 ms → albucore (20.45x)
  HWC 480x640x9
    albucore 0.1303 ms | numpy 2.8515 ms → albucore (21.89x)
  HWC 768x1024x3
    albucore 0.1127 ms | numpy 2.4032 ms → albucore (21.33x)

--- std(per_channel) ---
  HWC 240x320x3
    albucore 0.0407 ms | numpy 1.3053 ms → albucore (32.10x)
  HWC 480x640x3
    albucore 0.1568 ms | numpy 5.1540 ms → albucore (32.86x)
  HWC 480x640x9
    albucore 0.8192 ms | numpy 7.2847 ms → albucore (8.89x)
  HWC 768x1024x3
    albucore 0.4014 ms | numpy 13.3485 ms → albucore (33.25x)
  NHWC 4x240x320x3
    albucore 0.0500 ms | numpy 5.2014 ms → albucore (104.12x)
  NHWC 4x240x320x9
    albucore 0.8412 ms | numpy 7.0201 ms → albucore (8.35x)

--- reduce_sum(global) ---
  HWC 240x320x3
    albucore 0.0115 ms | numpy 0.0322 ms → albucore (2.81x)
  HWC 480x640x3
    albucore 0.0445 ms | numpy 0.1248 ms → albucore (2.81x)
  HWC 480x640x9
    albucore 0.1282 ms | numpy 0.3672 ms → albucore (2.86x)
  HWC 768x1024x3
    albucore 0.1121 ms | numpy 0.3176 ms → albucore (2.83x)
  NHWC 4x240x320x3
    albucore 0.0434 ms | numpy 0.1235 ms → albucore (2.85x)
  NHWC 4x240x320x9
    albucore 0.1297 ms | numpy 0.3782 ms → albucore (2.92x)

--- reduce_sum(per_channel) ---
  HWC 240x320x3
    albucore 0.0125 ms | numpy 0.4615 ms → albucore (36.92x)
  HWC 480x640x3
    albucore 0.0450 ms | numpy 1.8808 ms → albucore (41.83x)
  HWC 480x640x9
    albucore 0.2237 ms | numpy 2.3632 ms → albucore (10.56x)
  HWC 768x1024x3
    albucore 0.1112 ms | numpy 4.7349 ms → albucore (42.58x)
  NHWC 4x240x320x3
    albucore 0.0455 ms | numpy 1.8941 ms → albucore (41.67x)
  NHWC 4x240x320x9
    albucore 0.2206 ms | numpy 2.3397 ms → albucore (10.61x)

=== float32 ===

--- mean_std(global) ---
  HWC 240x320x3
    albucore 0.3135 ms | numpy 0.3167 ms → albucore (1.01x)
  HWC 480x640x3
    albucore 1.2387 ms | numpy 1.2318 ms → numpy (1.01x)
  HWC 480x640x9
    albucore 3.8329 ms | numpy 3.8477 ms → albucore (1.00x)
  HWC 768x1024x3
    albucore 3.2378 ms | numpy 3.2618 ms → albucore (1.01x)
  NHWC 4x240x320x3
    albucore 1.2940 ms | numpy 1.2915 ms → numpy (1.00x)
  NHWC 4x240x320x9
    albucore 3.9053 ms | numpy 3.9279 ms → albucore (1.01x)

--- mean_std(per_channel) ---
  HWC 240x320x3
    albucore 0.0797 ms | numpy 1.7832 ms → albucore (22.38x)
  HWC 480x640x3
    albucore 0.3108 ms | numpy 7.0817 ms → albucore (22.78x)
  HWC 480x640x9
    albucore 3.9418 ms | numpy 9.6633 ms → albucore (2.45x)
  HWC 768x1024x3
    albucore 0.7820 ms | numpy 18.0545 ms → albucore (23.09x)
  NHWC 4x240x320x3
    albucore 0.3943 ms | numpy 7.0631 ms → albucore (17.92x)
  NHWC 4x240x320x9
    albucore 3.9578 ms | numpy 9.5653 ms → albucore (2.42x)

--- std(global) ---
  HWC 240x320x3
    albucore 0.2346 ms | numpy 0.2340 ms → numpy (1.00x)
  HWC 480x640x3
    albucore 0.9025 ms | numpy 0.8900 ms → numpy (1.01x)
  HWC 480x640x9
    albucore 2.7509 ms | numpy 2.7544 ms → albucore (1.00x)
  HWC 768x1024x3
    albucore 2.3613 ms | numpy 2.3382 ms → numpy (1.01x)

--- std(per_channel) ---
  HWC 240x320x3
    albucore 0.0772 ms | numpy 1.2891 ms → albucore (16.71x)
  HWC 480x640x3
    albucore 0.3061 ms | numpy 5.1362 ms → albucore (16.78x)
  HWC 480x640x9
    albucore 6.9597 ms | numpy 7.1325 ms → albucore (1.02x)
  HWC 768x1024x3
    albucore 0.7740 ms | numpy 13.4968 ms → albucore (17.44x)
  NHWC 4x240x320x3
    albucore 0.4062 ms | numpy 5.1217 ms → albucore (12.61x)
  NHWC 4x240x320x9
    albucore 4.0025 ms | numpy 7.1455 ms → albucore (1.79x)

--- reduce_sum(global) ---
  HWC 240x320x3
    albucore 0.0860 ms | numpy 0.0832 ms → numpy (1.03x)
  HWC 480x640x3
    albucore 0.3390 ms | numpy 0.3359 ms → numpy (1.01x)
  HWC 480x640x9
    albucore 1.0069 ms | numpy 1.0296 ms → albucore (1.02x)
  HWC 768x1024x3
    albucore 0.8855 ms | numpy 0.8650 ms → numpy (1.02x)
  NHWC 4x240x320x3
    albucore 0.3415 ms | numpy 0.3391 ms → numpy (1.01x)
  NHWC 4x240x320x9
    albucore 1.0409 ms | numpy 1.0284 ms → numpy (1.01x)

--- reduce_sum(per_channel) ---
  HWC 240x320x3
    albucore 0.1003 ms | numpy 0.4944 ms → albucore (4.93x)
  HWC 480x640x3
    albucore 0.4038 ms | numpy 1.9839 ms → albucore (4.91x)
  HWC 480x640x9
    albucore 0.2971 ms | numpy 2.6309 ms → albucore (8.86x)
  HWC 768x1024x3
    albucore 1.0417 ms | numpy 5.1368 ms → albucore (4.93x)
  NHWC 4x240x320x3
    albucore 0.3988 ms | numpy 1.9612 ms → albucore (4.92x)
  NHWC 4x240x320x9
    albucore 0.3572 ms | numpy 2.5907 ms → albucore (7.25x)
