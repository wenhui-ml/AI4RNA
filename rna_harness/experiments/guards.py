"""Autonomy guards for bounded agent / self-revise / background loops.

Borrowed from OpenScience (task.ts nested-spawn deny; research.txt "each iteration MUST change
something"; prompt.ts text-doom-loop detector) — reimplemented in Python to fix the failure modes we
actually hit this project: the 45-min unbounded hang, wheel-spinning search, and the reparenting
spawn fork-bomb (see memory: spawn-forkbomb-gotcha). Small, dependency-free, unit-tested.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from typing import Callable, List, Optional, Tuple


def in_main_process() -> bool:
    return mp.current_process().name == "MainProcess"


def guard_main_only(what: str = "process-pool launch") -> None:
    """Raise if called from a worker process — prevents the reparenting fork-bomb where a custom
    launcher re-spawns a pool inside a pool worker. Call at the top of any pool-spawning entrypoint."""
    if not in_main_process():
        raise RuntimeError(
            f"{what} attempted inside a worker process (fork-bomb guard). "
            "Put the pool-spawning call under `if __name__ == '__main__':`.")


class BudgetGuard:
    """Per-episode ceiling on oracle calls and/or wall-seconds. `ok()` is False once exceeded, so a
    loop `while guard.ok(): ...; guard.spend(n)` is hard-bounded (the fix for the 45-min hang).
    WARNING: with NEITHER limit set, the guard is unbounded (`ok()` always True) — `bounded` is False;
    a `while guard.ok()` loop would never stop. Always pass at least one of max_calls / wall_s."""

    def __init__(self, max_calls: Optional[int] = None, wall_s: Optional[float] = None):
        self.max_calls = max_calls
        self.wall_s = wall_s
        self.calls = 0
        self._t0 = time.time()

    @property
    def bounded(self) -> bool:
        return self.max_calls is not None or self.wall_s is not None

    def spend(self, n: int = 1) -> None:
        self.calls += n

    def elapsed(self) -> float:
        return time.time() - self._t0

    def remaining_calls(self) -> Optional[int]:
        return None if self.max_calls is None else max(0, self.max_calls - self.calls)

    def ok(self) -> bool:
        if self.max_calls is not None and self.calls >= self.max_calls:
            return False
        if self.wall_s is not None and self.elapsed() >= self.wall_s:
            return False
        return True

    def reason(self) -> str:
        if self.max_calls is not None and self.calls >= self.max_calls:
            return f"call budget spent ({self.calls}/{self.max_calls})"
        if self.wall_s is not None and self.elapsed() >= self.wall_s:
            return f"wall budget spent ({self.elapsed():.0f}/{self.wall_s}s)"
        return "within budget"


class IterationGuard:
    """Caps iterations AND requires each iteration to CHANGE the state non-trivially. `advance(state)`
    returns (should_continue, reason). Stops on (a) max_iters, or (b) `stall_patience`+1 consecutive
    identical states (a doom loop / wheel-spin). `key` maps a state to a comparable fingerprint; it
    defaults to `repr`, which snapshots the VALUE at call time — so a mutable state (dict/list) that is
    mutated in place is still compared correctly. For a custom object with identity-based repr, pass an
    explicit value-based `key` (e.g. `key=lambda s: (s.bp, s.seq)`), else in-place mutation looks
    'unchanged' and would trip a false doom-loop."""

    def __init__(self, max_iters: int, stall_patience: int = 2,
                 key: Optional[Callable[[object], object]] = None):
        self.max_iters = max_iters
        self.stall_patience = stall_patience
        self.key = key or repr
        self.i = 0
        self._recent: List[object] = []

    def advance(self, state) -> Tuple[bool, str]:
        self.i += 1
        if self.i > self.max_iters:
            return False, f"max_iters={self.max_iters} reached"
        self._recent.append(self.key(state))
        w = self.stall_patience + 1
        if len(self._recent) >= w and all(x == self._recent[-1] for x in self._recent[-w:]):
            return False, f"no state change for {self.stall_patience} iters (doom loop)"
        return True, "ok"
