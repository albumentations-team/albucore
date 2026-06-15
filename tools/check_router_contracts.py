"""Validate public router contract and benchmark coverage."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from tests.router_contracts import FUNCTION_CONTRACTS, GEOMETRIC_CONTRACTS, ROUTER_CONTRACTS, RouterContract
from tests.verification_constants import ALL_CHANNEL_SPECS, ALL_DTYPE_NAMES, ALL_LAYOUT_NAMES, ALL_VALUE_KINDS

if TYPE_CHECKING:
    from collections.abc import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
FUNCTIONS_PATH = REPO_ROOT / "albucore" / "functions.py"
GEOMETRIC_PATH = REPO_ROOT / "albucore" / "geometric.py"
BENCHMARK_ROUTER_PATH = REPO_ROOT / "benchmarks" / "benchmark_router_synthetic.py"


def _parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _string_list_from_node(node: ast.AST) -> set[str]:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return set()
    return {
        element.value for element in node.elts if isinstance(element, ast.Constant) and isinstance(element.value, str)
    }


def _read_all(path: Path) -> set[str]:
    module = _parse_module(path)
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    return _string_list_from_node(node.value)
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
            and node.value is not None
        ):
            return _string_list_from_node(node.value)
    msg = f"Could not find __all__ in {path}"
    raise RuntimeError(msg)


def _registry_names_from_function(module: ast.Module, function_name: str) -> set[str]:
    for node in module.body:
        if not isinstance(node, ast.FunctionDef) or node.name != function_name:
            continue
        for statement in node.body:
            if not isinstance(statement, ast.Return):
                continue
            if not isinstance(statement.value, ast.List):
                continue
            names: set[str] = set()
            for element in statement.value.elts:
                if not isinstance(element, ast.Tuple) or not element.elts:
                    continue
                first = element.elts[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    names.add(first.value)
            return names
    msg = f"Could not find benchmark registry function {function_name}"
    raise RuntimeError(msg)


def _benchmark_names() -> set[str]:
    module = _parse_module(BENCHMARK_ROUTER_PATH)
    return _registry_names_from_function(module, "_registry_functions") | _registry_names_from_function(
        module,
        "_registry_geometric",
    )


def _format_names(names: Iterable[str]) -> str:
    return ", ".join(f"`{name}`" for name in sorted(names))


def _check_exports(errors: list[str]) -> None:
    function_exports = _read_all(FUNCTIONS_PATH)
    function_contracts = set(FUNCTION_CONTRACTS)
    missing_function_contracts = function_exports - function_contracts
    extra_function_contracts = function_contracts - function_exports

    if missing_function_contracts:
        errors.append(f"Missing function contracts: {_format_names(missing_function_contracts)}")
    if extra_function_contracts:
        errors.append(f"Function contracts for non-exported names: {_format_names(extra_function_contracts)}")

    geometric_exports = _read_all(GEOMETRIC_PATH)
    geometric_contracts = set(GEOMETRIC_CONTRACTS)
    missing_geometric_contracts = geometric_exports - geometric_contracts
    extra_geometric_contracts = geometric_contracts - geometric_exports

    if missing_geometric_contracts:
        errors.append(f"Missing geometric contracts: {_format_names(missing_geometric_contracts)}")
    if extra_geometric_contracts:
        errors.append(f"Geometric contracts for non-exported names: {_format_names(extra_geometric_contracts)}")


def _contracts_requiring_benchmarks() -> dict[str, RouterContract]:
    return {
        name: contract
        for name, contract in ROUTER_CONTRACTS.items()
        if contract.kind != "decorator" and contract.benchmark_names
    }


def _check_benchmarks(errors: list[str]) -> None:
    benchmark_names = _benchmark_names()
    missing_benchmarks: list[str] = []
    for contract in _contracts_requiring_benchmarks().values():
        missing = set(contract.benchmark_names) - benchmark_names
        if missing:
            missing_benchmarks.append(f"{contract.name}: {_format_names(missing)}")

    if missing_benchmarks:
        errors.append("Missing benchmark registry entries: " + "; ".join(missing_benchmarks))

    release_blocking_without_benchmark = [
        name
        for name, contract in ROUTER_CONTRACTS.items()
        if contract.release_blocking_performance and not contract.benchmark_names
    ]
    if release_blocking_without_benchmark:
        errors.append(
            "Release-blocking contracts without benchmark names: " + _format_names(release_blocking_without_benchmark),
        )


def _unknown_members(values: Iterable[str], allowed: Iterable[str]) -> set[str]:
    return set(values) - set(allowed)


def _check_contract_vocabulary(errors: list[str]) -> None:
    for contract in ROUTER_CONTRACTS.values():
        unknown_dtypes = _unknown_members(contract.dtypes, ALL_DTYPE_NAMES)
        if unknown_dtypes:
            errors.append(f"{contract.name}: unknown dtypes {_format_names(unknown_dtypes)}")

        unknown_layouts = _unknown_members(contract.layouts, ALL_LAYOUT_NAMES)
        if unknown_layouts:
            errors.append(f"{contract.name}: unknown layouts {_format_names(unknown_layouts)}")

        unknown_channels = _unknown_members(contract.channels, ALL_CHANNEL_SPECS)
        if unknown_channels:
            errors.append(f"{contract.name}: unknown channels {_format_names(unknown_channels)}")

        unknown_values = _unknown_members(contract.values, ALL_VALUE_KINDS)
        if unknown_values:
            errors.append(f"{contract.name}: unknown values {_format_names(unknown_values)}")


def main() -> int:
    errors: list[str] = []
    _check_exports(errors)
    _check_benchmarks(errors)
    _check_contract_vocabulary(errors)

    if errors:
        sys.stderr.write("Router contract check failed:\n")
        for error in errors:
            sys.stderr.write(f"- {error}\n")
        return 1

    sys.stdout.write(f"Router contract check passed for {len(ROUTER_CONTRACTS)} public contracts.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
