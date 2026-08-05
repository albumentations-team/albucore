# Greenfield-план `warp_affine3d` для NumPy и Torch

Дата плана: 2026-08-03.

Связанные задачи:

- [AlbumentationsX #328 — `Affine3D`](https://github.com/albumentations-team/AlbumentationsX/issues/328);
- [AlbumentationsX #358 — true 3D augmentation tracker](https://github.com/albumentations-team/AlbumentationsX/issues/358);
- [OpenCV #29605 — native 3D volume resampling](https://github.com/opencv/opencv/issues/29605);
- [NumKong #362 — true 3D resampling](https://github.com/ashvardanian/NumKong/issues/362).

## Статус реализации на 2026-08-03

V1 реализован в Albucore как public router `warp_affine3d`: NumPy `DHWC` и CPU Torch `CDHW`, `uint8`/`float32`, one-volume-per-call, forward matrix в `(x, y, z)`, nearest/trilinear interpolation и constant/replicate borders. AlbumentationsX передаёт уже проверенные input и control data. Router не повторяет проверки CPU, `torch.strided`, `requires_grad`, matrix, size, flags или fill; он не делает `.detach()`, `.cpu()` или device transfer.

Quick CPU matrix зафиксировал `affine_grid` + `grid_sample` как единственный production kernel. Manual grid и coverage sampler остались diagnostic candidates: первый не имеет устойчивого выигрыша и меняет uint8 rounding на boundary, второй проиграл 12 из 16 nonzero-fill cells. Расширенный nine-shape sweep подтвердил scope public path, но не вводит размерный route. Полный протокол и конкретные числа находятся в [CPU benchmark report](research/warp-affine3d-cpu-benchmark.md).

Tiled output, additional border modes, OpenCV/NumKong backend и native extension не входят в текущий router. Каждый из них требует отдельного correctness и full-path performance trigger. AlbumentationsX integration остаётся downstream work после release Albucore; local duplicate resampler не предусматривается.

## Какой результат должен получить пользователь

Albucore предоставляет один публичный geometric router `warp_affine3d`. Он применяет одну заранее построенную 3D affine matrix к volume, интерполирует depth, height и width как три пространственные оси и сохраняет контейнер, dtype и число каналов.

Первая production-версия поддерживает:

- NumPy volume в layout `DHWC` с обязательной channel axis;
- CPU `torch.Tensor` в layout `CDHW`;
- `uint8` и `float32`;
- произвольное число каналов, включая `C=1` и `C>4`;
- output size в порядке `(depth, height, width)`;
- nearest-neighbor и trilinear interpolation;
- constant и replicate border modes;
- scalar и per-channel constant fill;
- eager execution с `requires_grad=False`;
- identity, rotation, scale, translation, shear и reflection, если они уже закодированы в matrix.

Torch уже является обязательной runtime dependency Albucore. Router использует прямой dispatch по контейнеру и может передавать CPU storage между NumPy и Torch без копии, когда strides и ownership это позволяют. Import time Torch не входит в задачу.

## Решение по границе Albucore и AlbumentationsX

Albucore владеет reusable resampling primitive:

- преобразует voxel-space matrix в backend coordinates;
- использует benchmark-selected Torch CPU baseline и принимает другие backend только после отдельного benchmark gate;
- сохраняет container, layout, dtype и channels;
- реализует interpolation, border и fill semantics;
- контролирует полные conversions, allocations и peak memory.

AlbumentationsX владеет transform policy:

- sampling rotation, scale, translation и shear;
- выбором или выводом center;
- построением forward affine matrix;
- отдельными `interpolation` и `mask_interpolation`;
- `fill` и `fill_mask`;
- target dispatch для `volume`, `mask3d` и keypoints;
- replay, serialization, seeded reproducibility и `Compose` validation.

Такое разделение повторяет уже принятый путь `resize3d`: AlbumentationsX передаёт prevalidated array/Tensor в Albucore и не знает, какой backend выбран внутри router.

## Границы первой версии

`warp_affine3d` работает с одним volume за вызов. Batched layouts `NDHWC`/`NCDHW` не входят в план; AlbumentationsX предоставляет только single-volume target path.

В V1 не входят:

- CUDA, MPS и другие accelerator devices;
- autograd, backward kernels и сохранение computational graph;
- `torch.compile`, `vmap`, graph capture и export contracts;
- `float16`, `bfloat16`, `float64`, `int64`, `bool` и integer dtypes кроме `uint8`;
- cubic, Lanczos и antialiased affine resampling;
- dense displacement maps: для них нужен отдельный `remap3d`;
- fit-output policy и вычисление нового output bounding volume;
- чтение spacing, orientation, DICOM или NIfTI metadata;
- построение transform matrix из углов и диапазонов;
- преобразование keypoints;
- in-place execution и `out=`.

AlbumentationsX проверяет target-specific rank, layout, dtype, CPU device, `torch.strided`, `requires_grad=False` и control data до kernel call. Albucore использует эти preconditions и не вызывает `.detach()`, `.cpu()` или скрытый device transfer.

`mask3d` использует тот же dtype contract, что и volume: только `uint8` или `float32` для NumPy и Torch. AlbumentationsX отклоняет `int64`, `bool` и любой другой dtype до вызова Albucore. Неявный cast mask в `float32` не допускается.

## Контракт публичного API

Рекомендуемая сигнатура следует 2D `warp_affine`, но использует 3D spatial order из `resize3d`:

```python
@overload
def warp_affine3d(
    volume: np.ndarray,
    matrix: np.ndarray,
    size: tuple[int, int, int],
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] | np.ndarray | None = None,
) -> np.ndarray: ...


@overload
def warp_affine3d(
    volume: torch.Tensor,
    matrix: np.ndarray,
    size: tuple[int, int, int],
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] | np.ndarray | None = None,
) -> torch.Tensor: ...
```

`matrix` остаётся NumPy control data для обоих overloads. AlbumentationsX строит её на CPU, а Albucore копирует только 12–16 matrix values в маленький float32 Tensor перед `affine_grid`. Поддержка Tensor matrix не нужна для V1 transform path и расширит device/autograd contract без пользовательского выигрыша.

Backend-specific helpers остаются internal и не входят в package `__all__`.

### Layout и dtype

| Вход | Spatial axes | Channel axis | Результат |
|---|---:|---:|---|
| NumPy `DHWC` | `(0, 1, 2)` | `3` | NumPy `D'H'W'C` |
| Torch `CDHW` | `(1, 2, 3)` | `0` | Torch `CD'H'W'` |

Grayscale NumPy volume имеет shape `(D, H, W, 1)`. AlbumentationsX временно добавляет channel axis к channel-less `mask3d`, вызывает primitive и удаляет axis после вызова.

| Свойство | NumPy | Torch |
|---|---|---|
| Input dtype | `np.uint8`, `np.float32` | `torch.uint8`, `torch.float32` |
| Output dtype | равен input dtype | равен input dtype |
| Device | CPU memory | CPU Tensor |
| Autograd | неприменим | `requires_grad=False` |
| Mutation | input не изменяется | input не изменяется |

Identity matrix при одинаковых input/output spatial shapes возвращает исходный объект. Этот aliasing fast path проверяет точную identity matrix после приведения `3×4` к homogeneous `4×4`; tolerance-based identity могла бы удалить небольшое, но реальное преобразование.

### Matrix и coordinate order

Публичная matrix использует voxel-center coordinates в порядке `(x, y, z)`:

- `x` индексирует width;
- `y` индексирует height;
- `z` индексирует depth;
- `size` остаётся в array order `(depth, height, width)`.

`matrix` имеет shape `3×4` или homogeneous `4×4`. Она задаёт forward mapping из input voxel coordinates в output voxel coordinates:

```text
[x_out, y_out, z_out, 1]ᵀ = matrix @ [x_in, y_in, z_in, 1]ᵀ
```

Этот же forward matrix AlbumentationsX применяет к 3D keypoints. Raster kernel использует inverse mapping, потому что каждый output voxel должен найти source coordinate. Albucore инвертирует matrix один раз на вызов.

Центр volume не является параметром `warp_affine3d`. AlbumentationsX включает перенос к center и обратно в готовую matrix. Стандартный voxel center для shape `(D, H, W)` равен:

```text
((W - 1) / 2, (H - 1) / 2, (D - 1) / 2)
```

AlbumentationsX проверяет shape matrix, конечность значений и обратимость до вызова. Albucore преобразует уже проверенную `3×4` matrix к homogeneous `4×4` и инвертирует её один раз; backend не должен получать NaN grid values или singular matrix.

### Преобразование voxel matrix в Torch `theta`

`torch.nn.functional.affine_grid` принимает inverse mapping из normalized output coordinates в normalized input coordinates. V1 фиксирует `align_corners=False`, чтобы unit-length axes имели определённую semantics.

Для spatial shape `(D, H, W)` homogeneous matrix из normalized `(x, y, z)` в voxel coordinates равна:

```text
V_from_N(D, H, W) =
[[W/2,   0,   0, (W-1)/2],
 [  0, H/2,   0, (H-1)/2],
 [  0,   0, D/2, (D-1)/2],
 [  0,   0,   0,       1]]
```

Обратная matrix переводит voxel coordinates в normalized coordinates:

```text
N_from_V(D, H, W) =
[[2/W,   0,   0, -(W-1)/W],
 [  0, 2/H,   0, -(H-1)/H],
 [  0,   0, 2/D, -(D-1)/D],
 [  0,   0,   0,          1]]
```

Torch `theta` вычисляется так:

```text
theta_h = N_from_V(input_shape) @ inverse(matrix) @ V_from_N(output_shape)
theta = theta_h[:3, :]
```

Инверсия и multiplication маленьких matrices выполняются в float64 для numerical stability. Перед grid construction итоговый `theta` переводится в float32. Полноразмерные volume/grid buffers остаются float32.

### Interpolation

V1 принимает общий semantic subset:

| Albucore argument | Torch mode | Назначение |
|---|---|---|
| `cv2.INTER_LINEAR` | `mode="bilinear"` на 5D input | trilinear intensity interpolation |
| `cv2.INTER_NEAREST` | `mode="nearest"` | categorical masks и labels |

`align_corners=False` передаётся и в `affine_grid`, и в `grid_sample`. AlbumentationsX проверяет interpolation flag и передаёт только этот subset.

V1 фиксирует nearest tie rule как round-to-nearest-even: source coordinates `0.5`, `1.5`, `2.5` выбирают indices `0`, `2`, `2`. Это поведение CPU `grid_sample` в Torch 2.13.0 закрепляется golden tests. Backend с другим rounding должен получить adapter или отдельный interpolation mode.

Affine downscale может alias high-frequency data. V1 не добавляет скрытый prefilter и не принимает `antialias`: общий antialiased affine contract требует отдельного фильтра, footprint и benchmark решения.

### Border и fill

Минимальный release contract:

| `border_mode` | Torch implementation | Статус V1 |
|---|---|---|
| `cv2.BORDER_CONSTANT` | zero padding плюс exact fill adapter | обязателен |
| `cv2.BORDER_REPLICATE` | `padding_mode="border"` | обязателен |
| `cv2.BORDER_REFLECT_101` | differential gate против independent oracle | включить после exact boundary tests |
| `cv2.BORDER_REFLECT` | отдельный differential gate | включить после exact boundary tests |
| `cv2.BORDER_WRAP` | coordinate folding/circular-padding candidate | отложить, если exact adapter не проходит gate |

Torch предоставляет только `zeros`, `border` и один `reflection` mode. Albucore не должен отображать оба OpenCV reflection modes в один Torch mode без доказанной parity. AlbumentationsX экспонирует только подтверждённый subset и отклоняет другой mode до kernel call.

`border_value=None` означает scalar zero. Scalar value broadcast на все channels. Tuple или one-dimensional array имеет длину `C`; AlbumentationsX проверяет этот control-data contract, а Albucore только нормализует значения к contiguous float32 buffer для kernel.

Для nonzero constant fill benchmark сравнивает минимум два algebraically correct paths:

1. `grid_sample(input - fill, zeros) + fill`;
2. `grid_sample(input, zeros) + fill * (1 - grid_sample(ones, zeros))`.

Первый path читает и записывает все input/output values дополнительно. Второй выполняет ещё один single-channel sampler. Их ranking зависит от `C`, dtype, output size и доли out-of-bounds coordinates. Router выбирает path только по full benchmark. Fill zero пропускает обе дополнительные работы.

Quick matrix выбрал первый path, `grid_sample(input - fill, zeros) + fill`, как production F0. Coverage sampler выиграл только в 4 из 16 nonzero-fill cells и проиграл на large representative shape, поэтому не добавляет runtime branch.

## Проверенные capability gaps на 2026-08-03

Локальный feasibility check выполнен на macOS arm64 с `torch==2.13.0`, `opencv==5.0.0`, `numpy==2.2.6` и `numkong==7.7.0`. Это API/numerical smoke check, не performance evidence.

| Проверка | Результат |
|---|---|
| 5D float32 CPU `affine_grid` + `grid_sample` | работает; identity exact на проверенной non-cubic shape |
| non-contiguous float32 CPU `CDHW` | принимается; output contiguous |
| nearest на half-voxel coordinates | использует round-to-nearest-even |
| CPU `uint8`, nearest | `NotImplementedError: "grid_sampler3d_cpu" not implemented for 'Byte'` |
| CPU `uint8`, trilinear | та же ошибка; нужен float32 working buffer |
| MPS float32 5D identity smoke | работает локально, но V1 не обещает MPS support |
| OpenCV true 3D affine API | отсутствует в публичном namespace 5.0.0 |
| NumKong warp/remap/affine API | отсутствует в публичном namespace 7.7.0 |

PyTorch документирует 5D input, grid order `(x, y, z)`, trilinear behavior для `mode="bilinear"` и три padding modes в [`grid_sample`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html). `affine_grid` получает `theta` shape `(1, 3, 4)` и служебную ось `N=1`; проект фиксирует `align_corners=False` для unit axes.

## Reference semantics до оптимизации

Production Torch kernel не может одновременно быть единственным correctness oracle. Tests получают маленький independent NumPy trilinear reference, который работает прямо в voxel coordinates:

1. строит output coordinates `(x_out, y_out, z_out)`;
2. применяет inverse public matrix;
3. для linear собирает восемь соседей и их trilinear weights;
4. применяет constant fill отдельно по каждой spatial axis;
5. переводит uint8 в float32 до sampling и один раз saturating-round конечный result.

Reference materializes coordinates только для маленьких test shapes. Он остаётся заведомо медленным и не входит в public router.

Обязательные exact references:

- identity;
- integer translation;
- все 90-degree rotations вокруг каждой пары spatial axes;
- reflections по x, y и z;
- `D=1`, `H=1`, `W=1`;
- in-plane transform при `D=1`, который сравнивается с существующим 2D `warp_affine` после явного согласования matrix convention;
- coordinates ровно на границе, на половине voxel и непосредственно снаружи.

Arbitrary rotations, scales и shears сравниваются с reference по tolerance. Exact cases выполняются раньше tolerance-based tests, чтобы coordinate swap или неправильная inverse matrix не скрылись за допустимой погрешностью.

## Production candidates

### T0: native Torch grid — обязательный baseline

Основной CPU Tensor path:

```text
prevalidated CPU CDHW Tensor
  → add N=1 view
  → matrix → normalized inverse theta
  → affine_grid
  → grid_sample
  → remove N view
  → dtype repair для uint8
  → CD'H'W' Tensor
```

Float32 data не копируется перед kernel. Strided input передаётся напрямую, пока benchmark не докажет пользу `.contiguous()` в связном регионе.

Kernel выполняется внутри `torch.no_grad()`. AlbumentationsX передаёт Tensor с `requires_grad=False`, поэтому kernel не строит autograd graph, но возвращает normal Tensor, пригодный как input следующего trainable layer.

### T1: manual affine grid

`affine_grid` сравнивался с manual grid construction из трёх one-dimensional normalized coordinate vectors и broadcasted affine combinations. Timed path включает все broadcasts, stacks и allocations. Quick matrix не дал manual-grid выигрыша ни в одной из 64 cells и показал uint8 boundary rounding difference. Candidate отклонён и не входит в router.

### T2: tiled output-depth grid

Полный grid имеет shape `(1, D_out, H_out, W_out, 3)` и занимает `12 × D_out × H_out × W_out` bytes в float32:

| Output spatial shape | Grid storage |
|---|---:|
| `32×256×256` | 24 MiB |
| `64×128×160` | 15 MiB |
| `48×240×320` | 42.2 MiB |

Tiled candidate строит grid по output-depth slabs, вызывает `grid_sample` для каждого slab и записывает результат в заранее выделенный output Tensor. Он должен учитывать глобальную output z-coordinate; простой вызов `affine_grid` с уменьшенным local depth изменит normalization и даст неверный result.

T2 нужен как memory route. Router выбирает его только по измеренному peak-RSS threshold; small-volume path остаётся одним kernel call.

### N0: NumPy → Torch → NumPy без full input copy

NumPy float32 path использует:

```text
DHWC np.ndarray
  → torch.from_numpy shared storage
  → permute CDHW view
  → T0/T1/T2
  → permute DHWC view
  → NumPy view
```

`torch.from_numpy` допускается, если strides неотрицательные. Read-only arrays и negative strides получают явный repair copy. Positive-stride non-contiguous inputs проверяются contract tests и передаются напрямую; отдельный `.contiguous()` candidate добавляется только при новом performance question.

Output contiguity не меняется скрыто без benchmark. Если downstream требует C-contiguous `DHWC`, стоимость final materialization входит в timed path и public contract фиксирует это решение.

### U0/U1: uint8 working paths

CPU `grid_sample` не принимает uint8 5D input. Базовый U0:

```text
uint8 input
  → one float32 conversion
  → affine sampling
  → clamp to [0, 255]
  → add 0.5 and cast to uint8
```

Этот conversion задаёт round-half-up и saturating behavior. `torch.round` не используется, потому что tie-to-even изменит integer contract.

U1 обрабатывает channels небольшими группами, повторно использует один grid и пишет каждый result chunk в preallocated uint8 output. Он уменьшает peak memory для high-channel inputs, но добавляет kernel calls. Benchmark определяет, нужен ли U1 и при каком `C`.

### Future native candidates

Когда OpenCV или NumKong выпустит matrix-based true 3D warp, benchmark добавляет:

- native NumPy `DHWC → DHWC`;
- CPU Tensor `CDHW → DHWC` view → native kernel → `CDHW` view;
- contiguous repair variants, если backend не принимает strides.

NumPy↔Torch conversion разрешён contract. Router выбирает cross-container path только по complete public timing и не использует его для autograd или accelerator devices.

SciPy `ndimage.affine_transform` допустим как development-only differential/performance reference через изолированное benchmark environment. SciPy не становится runtime dependency без отдельного dependency, wheel-size и full-path performance решения.

## Delete-first и memory audit

Implementation review обязан проверить способы удалить работу до выбора backend:

- точная identity matrix и одинаковый spatial shape возвращают input;
- matrix инвертируется один раз на call;
- shape-normalization matrices строятся из 12–16 scalar values без dense coordinate arrays;
- float32 NumPy storage передаётся в Torch через view, когда strides позволяют;
- `.contiguous()` не вызывается без benchmark region;
- `border_value=0` пропускает fill correction;
- replicate border вызывает native `padding_mode="border"` без coordinate repair;
- uint8 переводится в float32 один раз на full-volume path;
- если future U1 channel chunks появится, grid переиспользуется между chunks;
- если future tiled path появится, он пишет в preallocated output и не делает финальный `torch.cat` full copy;
- float32 linear result не clip’ается: interpolation и constant fill уже определяют numeric range;
- output dtype conversion и layout repair выполняются максимум один раз.

Общий affine warp не является separable operation: rotation и shear смешивают x, y и z. Three-pass `resize3d` packing нельзя использовать как correctness fallback.

LUT, random generation, `bincount` и grouped reductions неприменимы. In-place warp небезопасен, потому что output sampling читает source voxels в произвольном порядке. Эти результаты performance audit фиксируются в benchmark report.

## Benchmark plan

Реализованы два scripts:

- `benchmarks/benchmark_warp_affine3d.py` для NumPy `DHWC → DHWC`;
- `benchmarks/benchmark_warp_affine3d_tensor.py` для CPU Tensor `CDHW → CDHW`.

Оба используют `benchmarks/timing.py`, проверяют correctness до timing и могут записать Markdown report в указанный `--output` path.

### Candidate matrix

| Input | Candidate | Что входит во время |
|---|---|---|
| Torch float32 | T0 `affine_grid + grid_sample` | matrix normalization/inversion, theta, grid, kernel, output views |
| Torch float32 | T1 manual grid | coordinate vectors, broadcasts, stack, kernel |
| Torch float32 | T2 tiled grid — deferred | Не реализован без peak-RSS trigger |
| Torch uint8 | U0 full cast | float32 conversion, grid, kernel, clamp/round/cast |
| Torch uint8 | U1 channel chunks — deferred | Не реализован без memory/performance trigger |
| NumPy float32 | N0 zero-copy Torch bridge | wrapper, permutations, grid, kernel, NumPy output |
| NumPy strided/read-only | N1 repaired bridge | Contract-tested repair path; отдельный timing route пока не нужен |
| Constant nonzero fill | F0 shifted input | subtract, kernel, add |
| Constant nonzero fill | F1 coverage sampler | data kernel, one-channel coverage kernel, blend |
| Future backend | OpenCV/NumKong/native — deferred | Добавляется только после availability/profile trigger |
| Public | `warp_affine3d` | полный dispatch и выбранный route |

Kernel-only timing может объяснять profile, но routing принимает только public-path results.

### Shape matrix

Single-volume matrix следует canonical DHWC grid:

- `16×128×160×C`;
- `32×128×160×C`;
- `64×128×160×C`;
- `96×128×160×C`;
- `48×240×320×C`;
- channels `C ∈ {1, 3, 5, 9}`;
- dtypes `uint8` и `float32`.

Quick mode использует non-cubic shapes `5×11×13×C` и `16×128×160×C`. Full mode измеряет nine canonical contiguous NumPy shapes и contiguous/channel-last-strided Torch `CDHW` inputs, включая unit output axis. Negative-stride и read-only NumPy repair, а также positive-stride NumPy views проверяются отдельными contract tests; они не входят в native full-path timing matrix, пока для них не появится отдельный routing question.

### Transform scenarios

Current scripts time four fixed 3×4 forward-matrix scenarios:

1. downscale с небольшим translation;
2. upscale с H/W и H/D shear;
3. mixed scale/shear/translation с nonzero scalar fill;
4. unit-depth output с nearest interpolation.

Они вместе покрывают zero/nonzero fill и nearest/trilinear sampling. Identity, per-channel fill, reflections, 90-degree rotations, homogeneous 4×4 matrix и arbitrary transform correctness принадлежат contract/differential tests, а не timing matrix. Matrix generation и upstream validation остаются вне timed region; normalization, inversion и backend conversion готовой matrix входят внутрь.

### Controlled environment

Current reports сохраняют дату, platform/architecture, Albucore/NumPy/OpenCV/Torch versions, Torch/OpenCV thread count,
input/output shapes, dtype, Tensor stride class, scenario, warmups, repeats, median и MAD. Input generation, Torch import
и filesystem I/O остаются за timed region.

Перед добавлением memory route или threshold report нужно расширить изолированным peak-RSS measurement, grid/working-buffer
bytes, copy count, matrix/bounds metadata и минимум тремя независимыми runs на release reference machine. `tracemalloc`
для native Torch allocations недостаточен.

### Routing gate

Candidate принимается, если он:

- сохраняет correctness contract;
- устойчиво быстрее минимум на 5% в связном регионе shapes;
- не создаёт regression рядом с threshold;
- не повышает peak memory настолько, что canonical volume переходит в другой operational режим.

Разница в пределах 3% считается tie. При tie остаётся более простой path или path с меньшим peak memory. Performance policy отдельно блокирует slowdown больше 15% в hot-path cell или больше 10% по median router family.

Threshold нельзя вводить по одной shape или одному run. Report сохраняет rejected candidates и указывает причину проигрыша: dense grid, dtype cast, fill correction, contiguity copy, repeated kernel calls или output materialization.

## Correctness tests

Создать `tests/test_warp_affine3d.py` и `tests/property/test_warp_affine3d_properties.py`.

### Обязательная matrix

| Измерение | Значения |
|---|---|
| Container/layout | NumPy `DHWC`, Torch `CDHW` |
| Dtype | uint8, float32 |
| Channels | 1, 3, 5, 9 |
| Input shape | non-cubic, unit D/H/W |
| Output shape | same, crop-like smaller, padded larger, unit D/H/W |
| Memory layout | contiguous, positive-stride strided, negative-stride NumPy |
| Interpolation | nearest, linear |
| Border | constant zero/nonzero/per-channel, replicate; gated reflect/wrap |
| Matrix | identity, translation, scale, shear, rotations, reflection |

### Exact tests

- Identity с одинаковым spatial shape возвращает `result is input`.
- Identity с другим output size не использует alias fast path.
- Integer translations и nearest interpolation совпадают с direct indexing/reference.
- Все 90-degree rotations совпадают с `np.rot90`/axis permutations после согласования output shape.
- Reflections совпадают с `np.flip`.
- `D=1`, `H=1` и `W=1` сохраняют явные axes.
- `C=1` сохраняет channel dimension.
- Nearest mask не создаёт новых label values.
- Nearest sampling на interior half-voxel coordinates закрепляет CPU Torch round-to-nearest-even rule.
- Forward keypoint matrix и inverse raster mapping попадают в один marked voxel в downstream AlbumentationsX integration test.

### Differential tests

- Float32 arbitrary transforms сравниваются с independent NumPy oracle. Начальный target — `rtol=2e-4`, `atol=3e-5`; tolerance меняется только после анализа error distribution.
- Uint8 linear result сравнивается после final-only saturating round; ожидаемая delta относительно float32 oracle не больше одного value.
- NumPy и Tensor paths сравниваются после `DHWC ↔ CDHW` permutation.
- Direct 2D in-plane transforms при `D=1` сравниваются с Albucore `warp_affine` для общего interpolation/border subset.
- Если tiled route будет добавлен, он обязан совпадать с full-grid path по тому же contract.
- Fill adapters сравниваются на coordinates внутри, ровно на edge, на half-voxel снаружи и далеко снаружи.

### Validation tests

AlbumentationsX tests отвечают за invalid dtype, size, interpolation, border mode, matrix, fill, input container,
target-specific rank, missing explicit channel adapter, CPU/strided/autograd Tensor contract, `mask3d` dtype и
unsupported target combinations до вызова Albucore. Albucore tests используют только inputs, удовлетворяющие этому
contract, и проверяют resampling semantics, aliasing, layout repair и dtype preservation.

Property tests генерируют только маленькие single-volume inputs. Они проверяют shape, dtype, channel count, uint8 range, finite float32 output и label-set preservation для nearest. Equality tiled/full-grid добавляется только вместе с tiled route.

## Public API и ожидаемые файлы

| Файл | Изменение |
|---|---|
| `albucore/affine3d.py` | matrix normalization, grid adapters, Torch/NumPy kernels и public function implementation |
| `albucore/geometric.py` | импорт/re-export `warp_affine3d` и имя в `geometric.__all__` |
| `albucore/__init__.py` | symbol появляется через merged `geometric.__all__` |
| `docs/public-api.md` | классификация `warp_affine3d` как public geometric router |
| `tests/router_contracts.py` | dtype/layout/value/aliasing/benchmark contract |
| `tests/test_warp_affine3d.py` | contract, exact и differential tests |
| `tests/property/test_warp_affine3d_properties.py` | bounded generated cases |
| `benchmarks/benchmark_warp_affine3d.py` | NumPy candidates и public router |
| `benchmarks/benchmark_warp_affine3d_tensor.py` | Tensor, manual-grid и coverage-fill candidates |
| `docs/research/warp-affine3d-cpu-benchmark.md` | accepted/rejected routes and the evidence for deferred thresholds/memory routes |
| `benchmarks/README.md` | quick/full commands и measurement scope |

`warp_affine3d` входит в package `__all__` как geometric router. Backend helpers остаются private. Compatibility shim в `albucore.functions` не добавляется без отдельного public-API решения.

Torch уже присутствует в `pyproject.toml` и `uv.lock`. Этот план не добавляет runtime dependencies. Любое dependency change требует синхронного lock update и `uv lock --check`.

## Интеграция с AlbumentationsX `Affine3D`

После release Albucore transform layer добавляет:

1. `create_affine_transformation_matrix_3d(...)`, которая строит forward homogeneous matrix в `(x, y, z)`;
2. `affine_3d(...)`, которая добавляет/удаляет channel axis для mask и вызывает `albucore.warp_affine3d`;
3. `keypoints_affine_3d(...)`, которая применяет ту же forward matrix к XYZ coordinates;
4. `Affine3D(Transform3D)`, которая один раз samples параметры и передаёт matrix всем targets.

Volume call использует image interpolation и `fill`. Mask call использует `mask_interpolation` и `fill_mask`. Albucore не выводит interpolation из target semantics.

Downstream correctness matrix покрывает:

- `volume` и `mask3d` alignment;
- NumPy и CPU Tensor volume inputs;
- channel-less NumPy/Tensor mask adapters;
- keypoints с extra columns;
- non-cubic volume shapes и anisotropic scales;
- explicit/inferred center и geometry-only case из issue #115;
- seeded reproducibility, replay и serialization;
- unsupported targets на `Compose` construction boundary.

AlbumentationsX обновляет exact Albucore pin только после публикации release с `warp_affine3d`. Local duplicate resampler не добавляется как временный production path.

## Порядок реализации

### PR 1. Контракт и independent reference

1. Зафиксировать public signature, layouts, dtypes, forward matrix и `(x, y, z)` order.
2. Зафиксировать `align_corners=False`, interpolation subset и mandatory border subset.
3. Добавить pure NumPy oracle и exact identity/translation/90-degree tests.
4. Добавить API skeleton, overloads, matrix normalization и identity fast path.
5. Добавить upstream links и capability smoke records.

Условие завершения: один forward matrix даёт согласованные keypoint/raster coordinates, unit axes определены, а public contract не содержит layout или inverse-mapping ambiguity.

### PR 2. Benchmark harness и candidate report

1. Реализовать T0, T1, N0/N1, U0 и F0/F1 как benchmark candidates; T2/U1 оставить deferred без memory trigger.
2. Добавить correctness check перед каждым timing cell.
3. Запустить quick matrix с controlled threads.
4. Запустить full matrix и isolated peak-RSS measurements.
5. Сохранить accepted/rejected report без production thresholds.

Условие завершения: report показывает стоимость matrix setup, dense grid, sampling, fill correction, uint8 conversion, layout repair и tiling на representative single-volume inputs.

### PR 3. Production single-volume router

1. Включить минимальное число benchmark-backed routes.
2. Реализовать direct `np.ndarray`/`torch.Tensor` dispatch.
3. Добавить uint8 final rounding, border/fill adapters и tiled memory route, если gate его выбрал.
4. Экспортировать router и обновить public API/router contracts.
5. Выполнить unit/property tests, lint, type checks и quick benchmark regression.

Условие завершения: `warp_affine3d` выполняет single-volume CPU contract для NumPy/Tensor без hidden device/autograd conversions и с сохранённым performance report.

### PR 4. Downstream `Affine3D`

1. Выпустить Albucore и обновить pin AlbumentationsX.
2. Реализовать parameter sampling, center и forward matrix builder.
3. Добавить volume/mask/keypoint adapters и transform class.
4. Проверить replay, serialization, geometry-only и target construction failures.
5. Сравнить direct functional и `Compose` wall time/peak memory.

Условие завершения: AlbumentationsX делегирует raster sampling Albucore, а volume, mask и keypoints остаются геометрически согласованы.

### PR 5. Native extension только при сработавшем trigger

Отдельный RFC фиксирует language, ABI, threading, packaging и wheel matrix. Native extension не блокирует baseline release, если Torch route выполняет correctness, wall-time и peak-memory contract.

## Когда нужен собственный native kernel

Dense Torch grid отделяет coordinate generation от sampling и требует 12 bytes на output voxel. Fused matrix sampler может вычислять source coordinate внутри interpolation loop и удалить этот buffer.

Native kernel рассматривается, если full report показывает хотя бы одно:

- grid allocation вызывает OOM или неприемлемый peak RSS на canonical single-volume inputs;
- grid construction остаётся измеримым bottleneck после delete-first pass;
- tiled path снижает memory, но repeated calls делают public route неприемлемо медленным;
- uint8 casts и channel chunks доминируют wall time;
- Torch CPU route заметно проигрывает доступному fused reference, а profile указывает на kernel/grid, не на Python dispatch;
- OpenCV/NumKong по-прежнему не предоставляют нужный matrix-based primitive.

Предпочтительный first RFC candidate — C++/ATen CPU extension: Torch уже обязателен, а Tensor integration не требует промежуточного Python buffer. NumPy wrapper использует buffer protocol или zero-copy Tensor view. Минимальный native contract:

- fused inverse affine mapping без dense grid;
- nearest и trilinear interpolation;
- constant/replicate borders и подтверждённые reflection modes;
- uint8 и float32;
- arbitrary channels;
- DHWC/CDHW wrappers;
- input strides либо один benchmark-backed contiguous repair;
- final-only saturating uint8 rounding;
- controlled single-thread и parallel execution;
- no autograd registration в V1;
- prebuilt wheels для supported Python/platform matrix;
- никакой JIT compilation при import.

Native route проходит те же correctness и public-path performance gates. Язык реализации не даёт исключения из benchmark policy.

## Upstream tracking

На 2026-08-03 OpenCV #29605 и NumKong #362 открыты. Они запрашивают matrix-based true 3D warp, arbitrary channels, uint8/float32, nearest/trilinear interpolation и explicit coordinate semantics.

PyTorch предоставляет рабочий float32 baseline, но normalized coordinates требуют отдельного conversion layer. Запрос на absolute pixel coordinates отслеживается в [pytorch #36107](https://github.com/pytorch/pytorch/issues/36107). Перед созданием нового PyTorch issue для CPU uint8 `grid_sample` нужно повторить поиск и приложить standalone 5D reproduction на declared lower-bound version.

Upstream implementation не меняет public Albucore contract. Она добавляется как benchmark candidate и получает route только после differential tests и full-path measurement.

## Команды проверки

Минимальный локальный gate после implementation:

```bash
uv lock --check
uv run pytest tests/test_warp_affine3d.py -q
uv run pytest tests/property/test_warp_affine3d_properties.py -q
uv run pytest tests/test_verification_tools.py -q
uv run ruff check albucore tests/test_warp_affine3d.py benchmarks/benchmark_warp_affine3d.py
uv run mypy albucore
uv run python benchmarks/benchmark_warp_affine3d.py --quick --threads 1
uv run python benchmarks/benchmark_warp_affine3d_tensor.py --quick --threads 1
```

Перед release:

```bash
uv export --frozen
uv run pytest -q
uv run python benchmarks/benchmark_warp_affine3d.py --full --threads 1 \
  --output benchmarks/results/warp-affine3d.md
uv run python benchmarks/benchmark_warp_affine3d_tensor.py --full --threads 1 \
  --output benchmarks/results/warp-affine3d-tensor.md
```

Reference machine повторяет full runs в фиксированном многопоточном режиме. Release artifacts проверяются на supported Python/OS matrix.

## Definition of done

- `warp_affine3d` экспортируется из package как public geometric router.
- Public matrix задаёт forward input→output mapping в voxel-center `(x, y, z)` coordinates.
- Albucore переводит matrix в normalized inverse `theta` по зафиксированной формуле с `align_corners=False`.
- NumPy `DHWC` и CPU Tensor `CDHW` сохраняют container, dtype и channel count.
- Output spatial shape в точности равен `(depth, height, width)` из `size`.
- `uint8` и `float32` проходят exact/differential matrix для `C=1/3/5/9`, non-cubic shapes и unit axes.
- `mask3d` принимает только `uint8` или `float32`; `int64`, `bool` и остальные dtypes получают ошибку до вызова Albucore.
- Identity fast path возвращает input; любой реальный warp не изменяет input и возвращает отдельный result.
- Linear и nearest semantics, nearest rounding, uint8 saturation и border edges документированы и протестированы.
- Constant zero/nonzero/per-channel fill проходит independent oracle; replicate border проходит exact edge tests.
- Reflection/wrap modes либо проходят отдельные exact gates, либо получают явный unsupported error и не появляются в AlbumentationsX API.
- AlbumentationsX отклоняет non-CPU, non-strided и `requires_grad=True` Tensor до вызова; Albucore never performs hidden `.detach()`, `.cpu()` or device transfer.
- NumPy↔Torch bridge для contiguous input измеряется end to end, включая dtype cast, grid и output view; stride repair проверяется contract tests до появления отдельного route.
- Public router использует только benchmark-backed routes и ссылается на сохранённый CPU report.
- Benchmark сохраняет wall time, MAD и rejected candidates; peak RSS, grid bytes and copy counts are required before any memory route is added.
- Tiled route появляется только по memory/performance gate и совпадает с full-grid result.
- LUT, RNG, grouped reductions и in-place paths отмечены как неприменимые.
- OpenCV/NumKong/PyTorch gaps имеют актуальные upstream links или standalone issue plan.
- После Albucore release downstream `Affine3D` может использовать primitive для своих raster targets и ту же forward matrix для keypoints; это отдельная AlbumentationsX task, не legacy code в Albucore.
- Full tests, lint, type checks, lock check и release export проходят; reference-machine benchmark gate обязателен только перед новым threshold или native route.
