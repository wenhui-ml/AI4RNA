"""Open-problem distribution for generalization evaluation.

To test whether strategies learned on Eterna100-V2 generalize to *arbitrary* RNA design
problems, we need held-out targets the agent has never seen. We generate them as random
DESIGNABLE targets: fold a random sequence with the primary oracle and take its MFE
structure as the target. The random "witness" sequence proves the target is designable (a
solution exists) but is NEVER shown to the solver -- only the target structure is.
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

from ..oracles.registry import primary_oracle
from ..structure_utils import base_pairs
from ..task import DesignTask


def random_designable_tasks(
    n_tasks: int,
    len_range: Tuple[int, int] = (30, 80),
    seed: int = 0,
    min_pairs: int = 4,
) -> List[DesignTask]:
    rng = random.Random(seed)
    orc = primary_oracle()
    out: List[DesignTask] = []
    tries = 0
    while len(out) < n_tasks and tries < n_tasks * 40:
        tries += 1
        L = rng.randint(*len_range)
        seq = "".join(rng.choice("ACGU") for _ in range(L))
        struct, _ = orc.fold_mfe(seq)
        if len(base_pairs(struct)) >= min_pairs:
            out.append(DesignTask(
                task_id=f"open_random:{len(out)}",
                target=struct,
                family="open_random",
                meta={"witness_seq": seq, "n": L},
            ))
    return out
