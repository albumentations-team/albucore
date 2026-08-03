# Greenfield-план `resize3d` для NumPy и CPU Torch

Дата плана: 2026-08-03.

Связанная задача: [albucore#134](https://github.com/albumentations-team/albucore/issues/134).

## Какой результат должен получить пользователь

Albucore предоставляет один публичный router `resize3d`. Он изменяет только пространственные оси `(depth, height, width)`, сохраняет контейнер, dtype и число каналов и возвращает запрошенный spatial shape без скрытого перехода на GPU.

Первая версия поддерживает:

- NumPy volumes в layout `DHWC` с обязательной channel axis;
- CPU `torch.Tensor` в layout `CDHW`;
- `uint8` и `float32`;
- channels `C=1`, `C=3` и произвольные `C>4`;
- eager Tensor execution; AlbumentationsX передаёт CPU Tensor с `requires_grad=False`;
- линейную 3D-интерполяцию и identity resize;
- NumPy antialiasing при уменьшении размера.

Torch становится обязательной зависимостью Albucore. Целевой процесс обучает модель и уже импортировал Torch до вызова `resize3d`. План не оптимизирует import time и использует прямой container dispatch.

## Решение по архитектуре

Публичный API содержит один router. Внутри остаются два независимых kernel path:

```text
prevalidated np.ndarray DHWC
    → benchmark-routed NumPy/OpenCV/Torch-CPU implementation
    → np.ndarray DHWC

prevalidated torch.Tensor CDHW
    → native Torch или zero-copy DHWC NumPy/OpenCV route для large all-axis linear upscale
    → torch.Tensor CDHW
```

Один router нужен AlbumentationsX: transform передаёт volume и не знает, какой backend выбран внутри Albucore. Раздельные kernels нужны для производительности и честной семантики. NumPy использует channel-last packing, а Torch принимает channel-first Tensor и выполняет true 3D interpolation.

Backend-specific helpers не входят в package `__all__`. Если после benchmark выяснится, что двум контейнерам нужны несовместимые публичные параметры, отдельные `resize3d_numpy` и `resize3d_torch` можно добавить отдельным API-решением. Первая версия не требует такого расширения.

## Статус реализации на 2026-08-03

PR 1–3 реализованы в Albucore: `torch>=2.13.0` зафиксирован как обязательная dependency, package экспортирует `resize3d`, а Tensor path использует прямой `isinstance` dispatch, `torch.inference_mode()` и benchmark-selected zero-copy bridge для large all-axis linear upscale (минимум 10,000 output elements). Контрактные и property tests покрывают валидные `DHWC`/`CDHW` inputs, uint8/float32, `C=1/3/5`, non-contiguous views, unit axes, identity и antialias failure Tensor path. AlbumentationsX проверяет layout, size, CPU и autograd до вызова primitive.

NumPy router выбирает между three-pass NumPy, OpenCV axis packing, joint H/W packing, per-slice OpenCV и полным `NumPy → Torch → NumPy` путём. Выбор основан на full-path benchmark, включая packing и dtype conversion; результаты и команды находятся в [CPU benchmark report](research/resize3d-cpu-benchmark.md). Следующий release Albucore подготовлен как `0.2.10`. После его публикации AlbumentationsX обновляет pin с `albucore==0.2.9`, затем заменяет local kernels; до публикации такая замена сломала бы обычную установку downstream проекта.

## Границы первой версии

В release scope входят только одиночные volumes. Batch layouts `NDHWC` и `NCDHW` откладываются до отдельной задачи: у них другая memory/performance matrix. Уточнённый контракт issue #134 использует NumPy `DHWC` и Torch `CDHW`.

В первую версию не входят:

- CUDA, MPS и другие accelerator devices;
- autograd, сохранение computational graph и backward kernels;
- `torch.compile`, `vmap` и graph capture;
- mixed precision, `float16`, `bfloat16`, `float64` и integer dtypes кроме `uint8`;
- in-place resize и `out=`: shape-changing operation всегда создаёт новый результат;
- transform policy, случайный выбор осей и downscale factor;
- resize масок, keypoints или других annotations;
- cubic/Lanczos 3D semantics до отдельного API и benchmark решения.

AlbumentationsX отклоняет Tensor с `requires_grad=True` и non-CPU device до kernel call. Albucore не повторяет эти caller checks и не вызывает `.detach()`; его shape-changing Tensor kernel выполняется внутри `torch.inference_mode()`.

## Контракт публичного API

Рекомендуемая сигнатура сохраняет форму из issue:

```python
@overload
def resize3d(
    volume: np.ndarray,
    size: tuple[int, int, int],
    interpolation: int = cv2.INTER_LINEAR,
    antialias: bool = False,
) -> np.ndarray: ...

@overload
def resize3d(
    volume: torch.Tensor,
    size: tuple[int, int, int],
    interpolation: int = cv2.INTER_LINEAR,
    antialias: bool = False,
) -> torch.Tensor: ...
```

`size` всегда задаётся в порядке `(depth, height, width)`. Это отличается от 2D `resize`, где OpenCV принимает `(width, height)`. AlbumentationsX проверяет layout и size до вызова; Albucore использует эти preconditions без повторной runtime validation.

### Поддерживаемые представления

| Вход | Spatial axes | Channel axis | Результат |
|---|---:|---:|---|
| NumPy `DHWC` | `(0, 1, 2)` | `3` | NumPy `D'H'W'C` |
| Torch `CDHW` | `(1, 2, 3)` | `0` | Torch `CD'H'W'` |

NumPy volume всегда имеет явную channel axis. Grayscale volume использует shape `(D, H, W, 1)`. AlbumentationsX, а не Albucore, отклоняет rank-3 NumPy input и другие invalid layouts.

### Dtype, device и autograd

| Свойство | NumPy | Torch |
|---|---|---|
| Dtypes | `np.uint8`, `np.float32` | `torch.uint8`, `torch.float32` |
| Device | CPU memory | CPU precondition проверяет AlbumentationsX |
| Autograd | неприменим | `requires_grad=False` precondition проверяет AlbumentationsX |
| Output dtype | равен input dtype | равен input dtype |
| Output device | CPU | тот же device; AlbumentationsX supplies CPU input |

Unsupported dtype получает `ValueError` с перечнем двух поддерживаемых dtypes. Tensor CPU/device и autograd contract проверяет AlbumentationsX до вызова; Albucore не содержит duplicate runtime checks.

### Preconditions и identity

AlbumentationsX выполняет shape/runtime validation до backend dispatch:

1. `size` содержит ровно три значения.
2. Каждое значение является integer, не является `bool` и больше нуля.
3. Все входные spatial dimensions больше нуля.
4. Rank и layout соответствуют таблице выше.
5. Tensor находится на CPU и имеет `requires_grad=False`.

Albucore проверяет только dtype, interpolation и Tensor antialias limitation. Invalid direct calls с неправильной shape/runtime metadata остаются вне его contract. Если input spatial shape уже равен `size`, router возвращает исходный объект. Этот fast path удаляет kernel call и allocation. Документация явно говорит, что identity result может alias input; shape-changing result не alias input.

### Interpolation и antialias

Первая версия публично принимает общий semantic subset:

| `interpolation` | NumPy | Torch CPU | Статус v1 |
|---|---|---|---|
| `cv2.INTER_LINEAR` | separable linear/OpenCV path | `mode="trilinear", align_corners=False` | обязателен |
| `cv2.INTER_NEAREST` | nearest axis resize | `mode="nearest"` после differential tests | включить, если semantics зафиксированы |
| `cv2.INTER_AREA` | внутренний downscale candidate | прямого общего контракта нет | не принимать публично в v1 |
| cubic/Lanczos/exact modes | OpenCV может обработать отдельные оси | нет общего 3D Torch mode | отклонить в v1 |

`antialias=False` оставляет выбранную interpolation без prefilter. Для NumPy с `antialias=True` каждая уменьшаемая ось использует `INTER_AREA`; оси, которые сохраняются или увеличиваются, используют публичную interpolation. Комбинация `INTER_NEAREST + antialias=True` завершается `ValueError`, потому что она задаёт конфликтующие sampling semantics.

Для Torch `antialias=True` завершается `NotImplementedError`. PyTorch 2.13.0 ограничивает antialias 4D bilinear/bicubic/Lanczos input и не принимает 5D trilinear. Ограничение уже вынесено в [pytorch#191896](https://github.com/pytorch/pytorch/issues/191896). Albucore не должен молча игнорировать аргумент.

## Проверенные capability gaps

Локальный feasibility check 2026-08-03 выполнялся на arm64 с `torch==2.13.0`, `opencv==5.0.0` и `numkong==7.7.0`. Это проверка API и numerical behavior, не performance benchmark.

| Проверка | Результат |
|---|---|
| float32 CPU `F.interpolate(..., mode="trilinear")` | работает для non-cubic output и unit-length output axes |
| non-contiguous float32 CPU Tensor | принимается; результат contiguous |
| uint8 CPU trilinear | `NotImplementedError: "compute_indices_weights_linear" not implemented for 'Byte'` |
| 5D trilinear с `antialias=True` | `ValueError`: antialias разрешён только для поддерживаемых 4D modes |
| NumKong resize/interpolate/resample API | в публичном namespace 7.7.0 подходящего API нет |
| OpenCV `resize` | 2D API; готового true 3D volumetric resize route нет |

Один differential smoke check сравнил текущий AlbumentationsX OpenCV axis-packing с Torch trilinear на `5×11×13×3 → 7×8×17×3`. Для float32 maximum absolute difference составила `8.94e-7`, mean absolute difference — `4.74e-8`. Для uint8 maximum difference составила `1`; отличались `587` из `2856` значений из-за округления после каждого OpenCV axis pass. Эти числа подтверждают feasibility и показывают, почему v1 не должна обещать побитовую cross-container parity для uint8.

## Reference semantics до оптимизации

Сначала нужен независимый benchmark/test reference: separable true trilinear на NumPy в три axis passes.

Для каждой output coordinate используется half-pixel mapping, совместимый с `align_corners=False`:

```text
source = (output + 0.5) * input_size / output_size - 0.5
left = floor(source)
right = left + 1
weight = source - floor(source)
```

Indices clamp к допустимой границе. Reference последовательно применяет линейную интерполяцию по depth, height и width. Он работает в float32 и создаёт отдельный output buffer на каждом pass. Это понятный correctness oracle и заведомо allocation-heavy performance baseline.

Для uint8 reference один раз переводит input в float32, выполняет все три passes и округляет только конечный результат. Он нужен для определения математической semantics. Production OpenCV path может округлять после каждого pass, поэтому его uint8 output сравнивается с закреплёнными backend-specific golden vectors и range invariant, а не с требованием побитовой Torch parity.

Reference implementation остаётся benchmark/test oracle. Benchmark подтвердил ещё один production region: float32 `D → 1` использует тот же three-pass path, потому что он быстрее packing на измеренных unit-depth cells.

## NumPy implementation candidates

### Candidate N0: три чистых NumPy pass — обязательный baseline

N0 применяет reference interpolation по одной spatial axis. Его преимущества: прозрачная half-pixel semantics, arbitrary channels и отсутствие OpenCV channel limits. Его известная стоимость: три полных чтения, три полных записи и float32 temporaries для uint8.

N0 нужен даже при очевидном проигрыше. Он отвечает на два вопроса: правильно ли вычисляются coordinates и сколько времени экономит каждый optimized candidate.

### Candidate N1: три OpenCV axis-packing pass — первый production candidate

Этот path переносится из текущего `Anisotropy3D`:

```text
depth:  DHWC → (H*W, D, C) → resize width D→D' → (D', H, W, C)
height: DHWC → (D*W, H, C) → resize width H→H' → (D, H', W, C)
width:  DHWC → (D*H, W, C) → resize width W→W' → (D, H, W', C)
```

Каждый call использует существующий Albucore `resize`, поэтому `INTER_AREA` и `C>4` проходят через уже поддерживаемую chunking logic. Identity axes пропускаются. Benchmark учитывает transpose/reshape copies, OpenCV output allocation и возможный финальный contiguity copy.

N1 сохраняет текущую NumPy behavior `Anisotropy3D`, что уменьшает downstream migration risk. Для uint8 он может округлять после каждого resized axis. Эта semantics фиксируется тестами и не маскируется loose tolerance.

### Candidate N2: joint H/W resize плюс depth pass

N2 пакует depth и channels в channel axis, делает один 2D H/W resize и отдельно меняет depth. Он сокращает число interpolation calls с трёх до двух, но создаёт сложный transpose/reshape и может превысить OpenCV channel limit при большом `D*C`.

Этот candidate измеряется только в явно допустимом регионе `D*C <= get_opencv_max_channels()`. За пределами региона он требует chunking. Routing threshold принимается лишь при устойчивой победе, иначе N2 удаляется, чтобы не оставлять сложную ветку без выигрыша.

### Candidate N3: NumPy → Torch CPU → NumPy

Torch является обязательной зависимостью, поэтому NumPy route может использовать native true trilinear kernel:

```text
NumPy DHWC
  → torch.from_numpy
  → permute CDHW
  → add N=1
  → F.interpolate
  → remove N
  → permute DHWC
  → NumPy view или contiguous materialization по контракту
```

Benchmark включает dtype conversion для uint8, axis permutations, non-contiguous kernel input и output materialization. Candidate включён только в связанных регионах, где полный `np.ndarray → np.ndarray` route устойчиво выигрывает. В частности, это all-axis float32 downscale, multi-channel downscale и uint8 `D → 1` для writable array без negative strides.

### Candidate N4: per-slice OpenCV loops

N4 вызывает 2D OpenCV resize для depth slices и отдельный packed depth resize. Он избегает `D*C` channel packing, но добавляет Python/OpenCV call overhead. Это обязательный candidate для больших `D*C`, где N2 недоступен, и для shapes с малым depth.

## CPU Torch implementation

Torch path принимает только `CDHW`. Внутри он добавляет batch dimension и вызывает native 5D interpolation:

```python
def _resize3d_torch_cpu(
    volume: torch.Tensor,
    size: tuple[int, int, int],
    interpolation: int,
) -> torch.Tensor:
    working = volume if volume.dtype == torch.float32 else volume.to(torch.float32)
    result = torch.nn.functional.interpolate(
        working.unsqueeze(0),
        size=size,
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)
    if volume.dtype == torch.uint8:
        result = torch.minimum(result + 0.5, result.new_tensor(255)).to(torch.uint8)
    return result
```

Фрагмент задаёт intended behavior. Production code валидирует interpolation и выполняется в `torch.inference_mode()`; AlbumentationsX уже проверил shape, size, device и autograd state.

PyTorch 2.13.0 не реализует uint8 trilinear CPU kernel. V1 переводит только Tensor data в float32, интерполирует и один раз округляет конечный output. Линейная интерполяция не создаёт значений вне convex hull input; `min(..., 255)` защищает conversion границу. Тесты всё равно проверяют полный `[0, 255]` range.

Input Tensor может быть strided. Router не делает безусловный `.contiguous()`: локальный capability check подтвердил прием non-contiguous float32 CPU input. Для linear resize, где строго увеличиваются depth, height и width и output содержит минимум 10,000 `C×D×H×W` элементов, он передаёт zero-copy `CDHW → DHWC` NumPy view в выбранный OpenCV/NumPy route и оборачивает его output обратно в Tensor view. Этот bridge дал 2.5–5.8× выигрыш на canonical measured upscale cells. Для float32 он сохраняет native Torch result в `rtol=2e-4`, `atol=3e-5`; для uint8 delta не превышает 1. Очень малые upscale, downscale, mixed resize и `D → 1` остаются native Torch: там bridge не быстрее либо расширяет uint8 difference.

## Прямой container dispatch

Процесс обучения уже импортировал Torch. Router использует прямой container dispatch:

1. `isinstance(volume, np.ndarray)` выбирает NumPy route.
2. `isinstance(volume, torch.Tensor)` выбирает Tensor route; large all-axis linear upscale проходит через zero-copy NumPy/OpenCV bridge, остальные regions используют native Torch. AlbumentationsX уже передал CPU `CDHW` Tensor с `requires_grad=False`.
3. Другой container получает `TypeError` с перечнем поддерживаемых типов.

`sys.modules`, class-name heuristics и lazy kernel imports для этого API не нужны. Прямой dispatch уменьшает число внутренних состояний и делает type narrowing очевидным для runtime и type checker.

## Delete-first и memory audit

До выбора backend нужно проверить следующие способы удалить работу:

- вернуть input на identity size;
- пропускать каждую axis, размер которой не меняется;
- не переставлять и не материализовывать channel axis между axis passes: весь NumPy path сохраняет `DHWC`;
- не вызывать `.contiguous()` перед Torch kernel без benchmark evidence;
- не переводить NumPy float32 в новый Tensor buffer, если `torch.from_numpy` может разделить CPU storage;
- для выбранного large Tensor upscale не копировать buffer на Tensor/NumPy boundaries: `permute`, `.numpy()` и `torch.from_numpy` создают views;
- переводить uint8 в float32 один раз на весь true trilinear route;
- не clip’ать float32 linear result: interpolation сохраняет input range;
- выбирать порядок separable passes только после проверки numerical contract.

Порядок axis passes влияет на intermediate volume sizes. Для float32 можно проверить стратегию «сначала уменьшаемые оси, затем увеличиваемые». Для uint8 смена порядка меняет места промежуточного округления, поэтому production order остаётся `(D, H, W)`, пока отдельное решение не обновит golden vectors.

Resize с изменением shape не имеет safe in-place route. `out=` откладывается: caller-provided storage усложнит aliasing и layout contract, а выигрыш нужно сначала доказать на повторяющихся pipelines.

LUT, grouped reductions, `bincount` и random generation к этой operation неприменимы. Это должно быть записано в benchmark report как проверенный пункт performance audit, чтобы review не искал пропущенный candidate.

## Benchmark plan

`benchmarks/benchmark_resize3d.py` использует `benchmarks/timing.py`, пишет Markdown report и измеряет candidates вместе с public router. Опция `--shape D,H,W,C` запускает изолированную canonical cell на memory-constrained машине.

### Candidate matrix

| Input container | Candidate | Что входит во время |
|---|---|---|
| NumPy | N0 pure NumPy three-pass reference | coordinates, three passes, allocations |
| NumPy | N1 OpenCV axis packing | transpose/reshape, Albucore 2D router, output repair |
| NumPy | N2 joint H/W + depth | packing, channel chunking, two interpolation stages |
| NumPy | N3 Torch full path | wrapper, permutations, dtype conversion, kernel, NumPy output |
| NumPy | N4 per-slice OpenCV | Python calls, slice outputs, depth pass |
| Torch | T0 native trilinear | dispatch, unsqueeze/squeeze, kernel |
| Torch uint8 | T1 cast + trilinear + round | two full dtype conversions и kernel |
| Torch | T2 zero-copy Tensor → NumPy → Tensor | two layout views, selected NumPy/OpenCV route, output Tensor view |

Custom C++/Rust candidate не нужен для первого measurement pass. Он появляется после профиля, если существующие libraries не выполняют функциональный или performance contract.

### Shape matrix

Матрица включает canonical DHWC volumes из benchmark policy и обязательный `C=5` из issue #134:

- `16×128×160×C`;
- `32×128×160×C`;
- `64×128×160×C`;
- `96×128×160×C`;
- `48×240×320×C`;
- channels `C ∈ {1, 3, 5, 9}`;
- dtypes `uint8` и `float32`.

Каждый representative input получает четыре resize scenarios:

1. downscale всех осей;
2. upscale всех осей;
3. mixed resize, где одна ось уменьшается, одна сохраняется и одна увеличивается;
4. anisotropic resize только одной spatial axis.

Quick matrix для локальной итерации использует `5×11×13×C` и targets `7×8×17`, `3×11×13`, `5×1×13`. Full matrix добавляет input/output unit axes, `D*C` по обе стороны OpenCV channel limit и strided views.

### Controlled environment

Report обязан сохранить:

- CPU model, OS, architecture и дату;
- Python, NumPy, OpenCV, Torch, NumKong и Albucore versions;
- Torch/OpenCV/BLAS thread counts;
- input/output shapes, dtype, channels, strides и interpolation;
- warmup count, repeats, median, MAD и минимум три независимых runs;
- peak resident memory или profiler-based native allocation estimate;
- число full-array copies/conversions для каждого candidate.

Основные runs выполняются с одним CPU thread и с одним зафиксированным многопоточным режимом reference machine. I/O и random input generation остаются за timed region.

### Routing rule

Router выбирается по полному public path. Candidate принимается, если он устойчиво быстрее минимум на 5% в связном регионе shapes и не создаёт regression рядом с boundary. Разница в пределах 3% считается tie; при tie остаётся более простой path или path с меньшим peak memory.

Дополнительно действует [performance policy](maintaining/performance-policy.md): regression больше 15% в hot-path cell или больше 10% по median router family требует отклонения, отдельного route или явного review decision.

Нельзя вводить threshold по одному шумному измерению. Report сохраняет rejected candidates и объясняет, где они проиграли: kernel time, packing copy, dtype cast, channel chunking или output materialization.

## Correctness tests

Создать `tests/test_resize3d.py` и `tests/property/test_resize3d_properties.py`.

### Обязательная matrix

| Измерение | Значения |
|---|---|
| Container/layout | NumPy `DHWC`, Torch `CDHW` |
| Dtype | uint8, float32 |
| Channels | 1, 3, 5 |
| Input spatial shape | non-cubic, unit D, unit H, unit W |
| Output spatial shape | downscale, upscale, mixed, unit D/H/W, identity |
| Memory layout | contiguous и strided/non-contiguous |
| Interpolation | linear; nearest после semantic gate |
| Antialias | NumPy false/true; Torch false и explicit failure для true |

### Test oracles

- Native Tensor regions сравниваются напрямую с `F.interpolate(..., mode="trilinear", align_corners=False)`.
- Large all-axis linear Tensor upscale сравнивается с native float32 reference в `rtol=2e-4`, `atol=3e-5`; uint8 сравнивается с delta не больше 1.
- NumPy linear path сравнивается с pure NumPy three-pass reference с документированной tolerance.
- NumPy OpenCV production path получает backend-specific golden vectors для uint8, потому что intermediate rounding отличается от true float32 trilinear.
- Identity проверяет `result is input`.
- Cross-container float32 test сравнивает `DHWC` и `CDHW` после перестановки axes с tight tolerance; он не требует побитовой equality.
- Cross-container uint8 test проверяет shape, dtype, range и ограниченную numerical difference. Он не объявляет два backend’а побитово одинаковыми.

### Failure tests

Отдельные tests проверяют kernel-level failures:

- `float64`, `int16` и другие unsupported dtypes;
- `antialias=True` для Tensor;
- unsupported interpolation flags;
- корректный NumPy dispatch при уже импортированном Torch.

AlbumentationsX test suite отвечает за invalid rank, implicit channel axis, invalid `size`, zero-length axes, non-CPU Tensor и `requires_grad=True` до вызова Albucore.

## Public API и файлы

Ожидаемые изменения:

| Файл | Изменение |
|---|---|
| `pyproject.toml` | обязательный `torch` с минимальной версией, подтверждённой wheel matrix |
| `uv.lock` | lock после dependency change |
| `albucore/geometric.py` | public router, dtype/interpolation checks, packing prevalidated arrays и `resize3d` в `__all__` |
| `albucore/torch_backend.py` или отдельный internal module | CPU Tensor kernel без public star export |
| `albucore/__init__.py` | `resize3d` появляется через `geometric.__all__` |
| `docs/public-api.md` | классификация `resize3d` как public geometric router |
| `tests/test_resize3d.py` | contract и differential tests |
| `benchmarks/benchmark_resize3d.py`, `benchmarks/benchmark_resize3d_tensor.py` | NumPy и Tensor candidate matrices, включая direct Tensor и zero-copy bridge |
| `benchmarks/README.md` | команда запуска и scope benchmark |

Реализация приняла `torch>=2.13.0`, обновлённый lock и eager `torch_backend.py`: Torch является обязательной runtime dependency, а benchmark/test switch остаётся только явным внутренним флагом.

## Upstream issues

Отсутствующий upstream functionality фиксируется отдельными issues. Эти issues не блокируют baseline release: Albucore сохраняет работающий local route и ссылку на upstream gap.

### PyTorch

Существующие issues:

- [pytorch#191896: antialias support for 5D trilinear interpolate](https://github.com/pytorch/pytorch/issues/191896).
- [pytorch#191907: CPU uint8 trilinear interpolate](https://github.com/pytorch/pytorch/issues/191907).

Issue #191907 содержит минимальный 5D reproduction, exact error на Torch 2.13.0, ожидаемое сохранение uint8 dtype и Albucore use case.

### OpenCV

Поиск 2026-08-03 не нашёл issue по `3D resize volume`, `volumetric resize` или `n-dimensional resize`. [opencv#29654](https://github.com/opencv/opencv/issues/29654) просит CPU volumetric resize для `DHWC`, linear interpolation, uint8/float32, arbitrary channels и unit-length axes; в нём приведён packing cost.

### NumKong

NumKong 7.7.0 не экспортирует resize/interpolate/resample operation. [NumKong#372](https://github.com/ashvardanian/NumKong/issues/372) уточняет scope separable 3D linear resampling и содержит canonical volume shape, dtype/channel matrix и comparison context против OpenCV packing и Torch CPU.

После создания ссылки добавляются в этот документ и albucore#134.

## Когда нужен собственный C++ или Rust kernel

Собственный extension начинается только после сохранённого benchmark report. Trigger выполняется, если хотя бы одно условие остаётся нерешённым:

- Tensor antialiasing требуется release contract, а PyTorch support отсутствует;
- NumPy/OpenCV packing создаёт неприемлемый peak memory на canonical volumes;
- лучший полный route остаётся заметным pipeline bottleneck, и profile указывает на kernel, а не на Python dispatch;
- существующие libraries не могут одновременно сохранить half-pixel semantics, uint8 rounding и arbitrary channels.

Для общего NumPy/Torch CPU kernel практичнее C++/ATen extension: Torch уже обязателен, ATen даёт Tensor integration, а NumPy wrapper может использовать buffer protocol. Rust/PyO3 остаётся допустимым вариантом для NumPy-only kernel, но Tensor integration и wheel build matrix нужно оценить до выбора языка.

Минимальный custom-kernel contract:

- separable trilinear CPU resize с half-pixel coordinates;
- uint8 и float32;
- arbitrary `C`, включая `C>4`;
- `DHWC` и `CDHW` wrappers;
- input strides либо один явно измеренный contiguous conversion;
- final-only saturating uint8 rounding;
- single-thread и controlled parallel execution;
- no autograd registration; AlbumentationsX отклоняет `requires_grad=True` до native call;
- prebuilt wheels на поддерживаемой Python/platform matrix; никакой JIT-компиляции при import.

Custom route принимается по тем же correctness и full-path performance gates. Нативный kernel не получает исключение из benchmark policy.

## Интеграция с `Anisotropy3D`

После release Albucore локальные helpers в AlbumentationsX заменяются двумя вызовами primitive:

```python
downsampled = resize3d(
    volume,
    downsample_shape,
    interpolation=cv2.INTER_LINEAR,
    antialias=antialias if isinstance(volume, np.ndarray) else False,
)
return resize3d(
    downsampled,
    source_shape,
    interpolation=cv2.INTER_LINEAR,
    antialias=False,
)
```

Tensor branch передаёт `antialias=False` явно и сохраняет текущую non-antialiased behavior до решения pytorch#191896. Albucore API при прямом `antialias=True` Tensor call по-прежнему сообщает ошибку. Transform documentation должна продолжать объяснять эту разницу.

Из AlbumentationsX удаляются `_resize_numpy_volume_axis`, `_anisotropy_3d_numpy` и `_anisotropy_3d_torch`. Sampling выбранных осей, downscale factor, replay и mask policy остаются в transform layer.

Downstream acceptance:

- существующие `Anisotropy3D` tests проходят для NumPy/Tensor, uint8/float32 и `C=1/3/5`;
- replay и mask invariance не меняются;
- direct functional benchmark показывает отсутствие regression против удалённых local helpers;
- AlbumentationsX pin указывает на Albucore release с `resize3d`.

## Порядок реализации

### PR 1. Контракт, dependency и upstream records

1. Зафиксировать NumPy `DHWC`-only contract в issue, docstring и tests.
2. Зафиксировать минимальную Torch version по Python 3.10–3.14 и platform wheel matrix.
3. Добавить обязательную dependency и обновить `uv.lock`.
4. Создать недостающие PyTorch/OpenCV/NumKong issues и добавить ссылки.
5. Добавить API skeleton, precondition documentation, kernel tests и type overloads.

Условие завершения: dependency lock воспроизводим, Torch доступен в заявленной platform/Python matrix, контракт не содержит неразрешённой layout ambiguity.

### PR 2. Reference и benchmark harness

1. Добавить N0 true three-pass reference.
2. Перенести N1 как benchmark candidate без routing threshold.
3. Добавить N2–N4 и T0/T1 candidates.
4. Запустить quick и full matrices с controlled threads.
5. Сохранить accepted/rejected report.

Условие завершения: report позволяет выбрать NumPy route по dtype/shape/channel regions и показывает стоимость всех conversions.

### PR 3. Production router и correctness matrix

1. Реализовать прямой dispatch по `np.ndarray`/`torch.Tensor` с preconditions из AlbumentationsX.
2. Включить выбранный NumPy route или минимальное число benchmark-backed routes.
3. Реализовать CPU Torch trilinear и uint8 cast/round path.
4. Добавить identity fast path, public export, docstring и API docs.
5. Выполнить unit/property tests, lint и type checks.

Условие завершения: `resize3d` выполняет весь kernel contract; AlbumentationsX проверяет device и autograd preconditions, а Albucore документированно завершает unsupported antialias и interpolation.

### PR 4. Downstream extraction

1. Выпустить Albucore.
2. Обновить AlbumentationsX dependency.
3. Заменить local `Anisotropy3D` kernels на `resize3d`.
4. Удалить duplicate code.
5. Сравнить direct functional и transform-level benchmarks.

Условие завершения: AlbumentationsX делегирует reusable resize operation в Albucore без behavioral или performance regression.

### PR 5. Native extension, только при сработавшем trigger

Отдельный RFC фиксирует язык, ABI, threading, packaging и benchmark evidence. Этот PR не блокирует базовый `resize3d`, если Python/OpenCV/Torch routes уже выполняют contract.

## Команды проверки

Минимальный локальный gate после implementation:

```bash
uv lock --check
uv run pytest tests/test_resize3d.py -q
uv run pytest tests/property/test_resize3d_properties.py -q
uv run ruff check albucore tests/test_resize3d.py benchmarks/benchmark_resize3d.py
uv run mypy albucore
uv run python benchmarks/benchmark_resize3d.py --quick
uv run python benchmarks/benchmark_resize3d_tensor.py --quick
```

Перед release:

```bash
uv export --frozen
uv run pytest -q
uv run python benchmarks/benchmark_resize3d.py --full --output benchmarks/results/resize3d.md
uv run python benchmarks/benchmark_resize3d_tensor.py --full --output benchmarks/results/resize3d-tensor.md
```

CI также проверяет wheel/sdist install на заявленной OS/Python matrix. Linux job должен зафиксировать фактические Torch/Triton/CUDA transitive packages и размер окружения.

## Синхронизация performance guide

2026-08-03 canonical `docs/performance-optimization.md` и fallback-копия `../AlbumentationsX/.codex/skills/performance-optimization/references/performance-optimization.md` синхронизированы byte-for-byte. Это сохраняет один и тот же performance workflow в обоих репозиториях.

## Definition of done

- `torch` является обязательной locked dependency на поддерживаемой platform/Python matrix.
- Router использует уже импортированный Torch и прямой `isinstance` dispatch.
- `resize3d` экспортируется из package и документирован как public router.
- AlbumentationsX проверяет `DHWC`/`CDHW`, output size, CPU и `requires_grad=False` до вызова; Albucore не повторяет эти checks.
- NumPy `DHWC` и CPU Tensor `CDHW` сохраняют container, dtype и channel count.
- Output spatial shape в точности равен `(depth, height, width)` из `size`.
- uint8 и float32 проходят correctness matrix для `C=1/3/5`, non-cubic shapes и unit-length axes.
- Linear Tensor large all-axis upscale (минимум 10,000 output elements) использует zero-copy NumPy/OpenCV bridge только в benchmark-selected region; float32 остаётся в `rtol=2e-4`, `atol=3e-5` от native Torch, uint8 — в delta не больше 1.
- AlbumentationsX отклоняет non-CPU device и `requires_grad=True` до вызова; Tensor route сообщает `antialias=True` отдельной ошибкой.
- NumPy antialiasing применяется только к уменьшаемым axes.
- Public route ссылается на сохранённый benchmark report; rejected candidates остаются в report.
- Benchmark считает packing, conversions, contiguity copies, dtype casts и output repair.
- Performance review записывает, что LUT, random generation, grouped reductions и in-place route неприменимы.
- PyTorch/OpenCV/NumKong capability gaps имеют upstream issues со standalone reproductions и benchmark context.
- После выпуска Albucore `0.2.10` `Anisotropy3D` использует primitive, а duplicate local kernels удалены. Пока downstream pin’ит `albucore==0.2.9`, это остаётся release-blocking шагом PR 4.
- Albucore и AlbumentationsX performance guides синхронизированы.
- Full tests, lint, type checks, lock check, release export и reference-machine benchmark gates проходят.
