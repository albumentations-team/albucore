---
name: albucore-public-api
description: Albucore explicit root exports, operation owners, input/output typing, and dependent checks. Use when changing exports, documenting API, or deciding what belongs in package __all__.
---

# Albucore Public API

## Export ownership

`albucore/__init__.py` directly imports public operations and declares a literal `__all__`.
Keep one entry per public name. Types live in `albucore/_types.py` and are exported from the package root.
There is no intermediate operation re-export module.

Public callers use `from albucore import <operation>`. Backend-specific functions are internal implementation
choices; tests and benchmarks import them from the owning module rather than exposing them at the package root.
Use `docs/public-api.md` for operation contracts and module ownership.

## Input/output type relationships

Public operations use concrete signatures and overloads. Preserve input-dependent container and dtype results;
normalization, conversion, and statistics follow their declared output dtype rather than a generic image policy.
Keep implementation unions separate from precise caller overloads. Do not replace required arguments with `Any`.

Write concrete NumPy dtypes directly as `NDArray[np.uint8]`, `NDArray[np.float16]` and `NDArray[np.float32]`.
Dtype annotations do not define image, video or volume rank. Keep unions and TypeVars only where they express
supported inputs or an input/output relationship.

`ImageInputT` bounds a known helper to supported NumPy image dtypes or Tensor when it preserves both dtype and
container. Keep scalar/control arguments explicit. It does not transform an arbitrary callback signature or describe
a dtype-changing operation.

Extend checked consumer cases for the affected dtype, container, required controls, layout, and explicit output.
Run mypy and Pyrefly with actual dependency stubs. Existing callback-decorator consumers must remain checked until
those decorators and their consumers are removed.

Before interpreting a typing result, verify the imported producer path and a Python target compatible with the
installed dependency stubs. Missing Torch stubs turn Tensor overloads into `Any` and can silently accept a NumPy
output with the wrong dtype. Keep the isolated hook's Torch stubs and the full-runtime consumer check; the latter also
loads the real OpenCV stubs. A passing check against an installed old Albucore package is not evidence for this tree.

`clip` infers dtype from `out`, or from the input without output. An explicit dtype still defines conversion when
output is optional and must match a supplied buffer. Without output, input is unchanged; supplied output is returned
by identity. Clip truncates uint8 targets without intensity scaling. AX validates metadata, overlap and writeability;
it must protect an original array when a preceding identity operation returns that array as its result.

`add` and `multiply` preserve dtype, use concrete NumPy/Tensor output overloads, and require a Tensor layout.
Their keyword-only `out` returns the supplied storage by identity, including writable strided views and exact
image/full-field aliases. Without it, input storage remains unchanged. When adapting AX, audit ownership:
the former `inplace=True` could allocate, so replacing it with `out=image` may introduce an input write.

`to_float` and `from_float` also accept explicit result-dtype `out`. Floating `to_float` is an identity; uint8
input is divided into float32. Floating `from_float` targets cast without scaling; uint8 targets scale, round
and saturate. Keep concrete destination-dtype overloads; generic target-dtype fallbacks accept no destination.

`power` has concrete dtype-matching `out` overloads and requires a Tensor layout. Preserve scalar/channel uint8
LUT saturation and truncation; floating inputs also accept exponent fields. A supplied output does not promise
allocation-free execution: preserve a measured temporary when a native exact-alias kernel is slower. Pass `out`
from AX only when the complete caller gains speed or memory; ownership alone does not establish that gain.

## Adding or changing a public operation

1. Implement in its owning arithmetic, conversion, normalization, statistics, LUT, elementwise, or geometric module.
2. Import the public name directly in `albucore/__init__.py` and add it once to `__all__`.
3. Add or update its contract in `tests/router_contracts.py` and a workload in the existing benchmark registry.
4. Run `python tools/check_router_contracts.py`; it checks root exports, contracts, and benchmark coverage.
5. Check dependent AX imports and typing consumers before removing or changing an export.
6. Update `docs/public-api.md` and the relevant operation documentation.

Public backend/scalar/vector/array shims are being retired under the accepted architecture plan. Do not add new
technical variants to the public root API. Keep migration progress and remaining retirements in the internal ledger.

`multiply_add` uses concrete NumPy dtype/output overloads and a required Tensor layout. Output may alias the image
or a full coefficient field; preserve reads of the original bias before writing. Its half NumPy/Torch routes round
the product before addition, while selected NumKong scalar routes retain fused rounding. A supplied output does not
promise no temporary: keep one when wide controls or aliases require it. The former inplace argument and unprefixed
NumPy/OpenCV/LUT compound entrypoints are retired.

`add_weighted` belongs to `albucore.weighted` and accepts dtype/container-matching keyword-only `out`.
Both operands have matching shape, dtype and container under AX's prevalidated contract. Output may alias either
complete input. NumPy outputs retain the first input dtype; Tensor calls require an explicit layout. Preserve the
existing backend region and its non-finite behavior rather than adding neutral-weight shortcuts. Unprefixed weighted
backend entrypoints and scalar NumKong convenience wrappers are retired; private scalar scale lives in arithmetic.

Fixed-statistics `normalize` and adaptive `normalize_per_image` share the `albucore.normalize` owner. Root imports
both directly. `normalize` accepts result-dtype `out`: half input produces half; uint8/float32 produce float32.
Exact floating input aliases are supported, and Tensor layout is required. Preserve the selected backend's operation
order, working precision and approximate identity policy even for strided outputs. Fixed and adaptive NumPy/OpenCV/LUT backend entrypoints are retired.
Adaptive normalization also accepts result-dtype output and requires Tensor layout. Preserve its distinct statistics,
epsilon and clipping rules; NumPy global modes span the whole array, while Tensor NCHW/NCDHW normalize each item.
All statistics precede alias writes. Empty inputs return empty outputs without a reduction.
