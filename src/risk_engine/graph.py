"""A minimal incremental-computation graph.

Nodes form a DAG: `source` nodes hold a value you can `set`; `derived` nodes
hold a pure function of their dependencies. Setting a source only marks its
*downstream* nodes stale — everything else in the graph keeps its cached
value. Reading a stale node recomputes it (and only it) from its already-up-
-to-date dependencies. This means a live risk dashboard can push a single
price tick and only the metrics that actually depend on that price get
recomputed, instead of recalculating the whole portfolio on every update.
"""

from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class Node(Generic[T]):
    def __init__(
        self,
        name: str,
        compute: Callable[..., T] | None,
        deps: tuple["Node[Any]", ...] = (),
    ) -> None:
        self.name = name
        self._compute = compute
        self._deps = deps
        self._value: T | None = None
        self._stale = compute is not None
        self._observers: list["Node[Any]"] = []
        self.recompute_count = 0
        for dep in deps:
            dep._observers.append(self)

    @classmethod
    def source(cls, name: str, value: T) -> "Node[T]":
        node: "Node[T]" = cls(name, compute=None)
        node._value = value
        node._stale = False
        return node

    @classmethod
    def derived(cls, name: str, fn: Callable[..., T], *deps: "Node[Any]") -> "Node[T]":
        return cls(name, compute=fn, deps=deps)

    def set(self, value: T) -> None:
        if self._compute is not None:
            raise TypeError(f"node {self.name!r} is derived; set its upstream source instead")
        self._value = value
        self._mark_observers_stale()

    def _mark_observers_stale(self) -> None:
        for obs in self._observers:
            if not obs._stale:
                obs._stale = True
                obs._mark_observers_stale()

    def value(self) -> T:
        if self._stale:
            assert self._compute is not None
            args = [dep.value() for dep in self._deps]
            self._value = self._compute(*args)
            self._stale = False
            self.recompute_count += 1
        return self._value  # type: ignore[return-value]
