from __future__ import annotations

import pytest

from src.risk_engine.graph import Node


def test_derived_node_recomputes_only_when_stale() -> None:
    a = Node.source("a", 1)
    b = Node.source("b", 2)
    calls: list[tuple[int, int]] = []

    def add(x: int, y: int) -> int:
        calls.append((x, y))
        return x + y

    c = Node.derived("c", add, a, b)
    assert c.value() == 3
    assert len(calls) == 1

    assert c.value() == 3
    assert len(calls) == 1

    a.set(10)
    assert c.value() == 12
    assert len(calls) == 2


def test_unrelated_branch_does_not_recompute() -> None:
    a = Node.source("a", 1)
    b = Node.source("b", 2)
    calls_c: list[int] = []
    calls_d: list[int] = []

    def compute_c(x: int) -> int:
        calls_c.append(x)
        return x * 2

    def compute_d(y: int) -> int:
        calls_d.append(y)
        return y * 3

    c = Node.derived("c", compute_c, a)
    d = Node.derived("d", compute_d, b)

    assert c.value() == 2
    assert d.value() == 6

    a.set(5)
    assert c.value() == 10
    assert len(calls_c) == 2
    assert len(calls_d) == 1


def test_setting_a_derived_node_raises() -> None:
    a = Node.source("a", 1)
    c = Node.derived("c", lambda x: x + 1, a)
    with pytest.raises(TypeError):
        c.set(5)


def test_diamond_dependency_recomputes_sink_once_per_change() -> None:
    a = Node.source("a", 1)
    b = Node.derived("b", lambda x: x + 1, a)
    c = Node.derived("c", lambda x: x * 2, a)
    calls: list[tuple[int, int]] = []

    def combine(x: int, y: int) -> int:
        calls.append((x, y))
        return x + y

    d = Node.derived("d", combine, b, c)
    assert d.value() == 4  # (1+1) + (1*2)
    assert len(calls) == 1

    a.set(3)
    assert d.value() == 10  # (3+1) + (3*2)
    assert len(calls) == 2
