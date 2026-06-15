from __future__ import annotations

from tools.verify_golden_vectors import verify


def test_golden_vectors_match_current_outputs() -> None:
    assert verify() == []
