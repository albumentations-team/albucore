---
name: albucore-public-api
description: Albucore star-exported API (__all__), routers vs albucore.functions shims, dependents (e.g. Albumentations). Use when changing exports, documenting API, or deciding what belongs in package __all__.
---

# Albucore public API

## Star import

`from albucore import *` follows **`albucore/__init__.py`**: merges `__all__` from `functions`, `decorators`, `geometric`, `utils`, plus metadata.

Routers live in **`albucore/functions.py`** `__all__` (re-exports from `arithmetic`, `lut`, `normalize`, `convert`, `ops_misc`, `stats`).

## Classification doc

See **`docs/public-api.md`**: routers vs explicit **`from albucore.functions import …`** shims (`*_opencv`, `*_numpy`, NumKong helpers, LUT plumbing).

## Dependents

If **Albumentations** (or others) import a symbol, it must be either:

- On **package** `__all__` (star-import friendly), or
- Documented as **explicit** import path (`albucore.functions`).

Example: **`sz_lut`** and **`apply_uint8_lut`** are public routers on `__all__` (implemented in **`albucore/lut.py`**).

## Adding an export

1. Implement in the appropriate submodule (`arithmetic.py`, `lut.py`, `geometric.py`, …).
2. Ensure `from submodule import *` in `functions.py` exposes the name.
3. Append to **`functions.__all__`** (sorted / grouped with peers).
4. Update **`docs/public-api.md`** if classification changes.
