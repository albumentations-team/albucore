# Changelog

## Unreleased

- Replace implicit mutation flags with explicit `out=` buffers across array operations. Tensor calls require `layout=`;
  calls without `out` leave input storage unchanged, though identity and view returns are allowed.
- Add float16 image support across NumPy and eager CPU Torch routers, preserving documented output dtype and
  wide-statistics policies. Specialized uint8 LUT APIs remain uint8-only.
- Write shared NumPy uint8 LUT results directly into separate contiguous outputs with StringZilla.
- Raise the NumKong and StringZilla minimums to 7.8.6 and 5.3.2, respectively; keep the Torch minimum at 2.14.1.
- Require Python 3.11 or newer. CI covers Python 3.11, 3.12, 3.13, and 3.14.
