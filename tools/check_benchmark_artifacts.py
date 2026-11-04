"""Reject generated benchmark artifacts passed by pre-commit."""

from __future__ import annotations

import sys

PREFIXES = ("benchmarks/results/", "benchmark/results/", "benchmark/albucore_benchmark/results/")
FILES = {"Benchmark.md", "array_operations_benchmark_results.json", "lut_benchmark_results.csv"}


def main() -> int:
    forbidden = [name for name in sys.argv[1:] if name in FILES or name.startswith(PREFIXES)]
    if forbidden:
        sys.stderr.write("Generated benchmark artifacts must be stored outside Git:\n" + "\n".join(forbidden) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
