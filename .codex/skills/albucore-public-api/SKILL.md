---
name: albucore-public-api
description: Albucore star-exported API (__all__), routers vs albucore.functions shims, and dependents such as Albumentations. Use when changing exports, documenting API, or deciding what belongs in package __all__.
---

# Albucore Public API

## Star Import

`from albucore import *` follows `albucore/__init__.py`, which merges `__all__` from `functions`, `decorators`, `geometric`, and `utils`, plus metadata.

Routers live in `albucore/functions.py` `__all__`, re-exported from `arithmetic`, `lut`, `normalize`, `convert`, `ops_misc`, and `stats`.

## Classification Doc

Use `docs/public-api.md` for router classification and explicit `from albucore.functions import ...` shims such as `*_opencv`, `*_numpy`, NumKong helpers, and LUT plumbing.

## Dependents

If Albumentations or another downstream package imports a symbol, it must be either:

- On package `__all__` for star-import compatibility, or
- Documented with an explicit import path such as `albucore.functions`.

Example: `sz_lut` and `apply_uint8_lut` are public routers on `__all__` and are implemented in `albucore/lut.py`.

## Adding an Export

1. Implement in the appropriate submodule (`arithmetic.py`, `lut.py`, `geometric.py`, etc.).
2. Ensure `from submodule import *` in `functions.py` exposes the name.
3. Append to `functions.__all__`, sorted or grouped with peers.
4. Update `docs/public-api.md` if classification changes.
