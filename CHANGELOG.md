# Changelog

## Unreleased

- Add float16 image support across NumPy and eager CPU Torch routers, preserving documented output dtype and
  wide-statistics policies. Specialized uint8 LUT APIs remain uint8-only.
- Raise the NumKong minimum to 7.8.5 and the Torch minimum to 2.14.1.
- Require Python 3.11 or newer. CI covers Python 3.11, 3.12, 3.13, and 3.14.
