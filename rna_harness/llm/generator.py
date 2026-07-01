"""LLM design generator: one-shot + actuation-instrumented repair loop.

The repair loop feeds the exact folding-mismatch diagnosis back to the model and logs the
actuation chain (did the model's next sequence actually change the diagnosed positions?),
so we can measure A for a real LLM policy -- and run the random-feedback / silent-fallback
ablations on it. The reward always comes from the canonical verifier, never the model.
"""
from __future__ import annotations

import random
import re
from typing import Dict, List, Optional, Tuple

from ..actuation.diagnosis import diagnose
from ..actuation.instrument import ActuationLogger
from ..generators.base import GenerationResult, Generator
from ..metrics import bp_distance
from ..oracles.vienna import ViennaRNAOracle
from ..rl.prompts import render_repair_prompt, render_task_prompt
from ..task import DesignTask
from ..verifier import verify
from .openrouter import OpenRouterClient

_SEQ_RE = re.compile(r"[ACGUacgu]{4,}")


def parse_sequence(text: str, n: int) -> Optional[str]:
    """Extract an n-nt RNA sequence from model output (robust to extra prose)."""
    if not text:
        return None
    up = text.upper().replace("T", "U")
    # Prefer an exact-length ACGU run; else the longest ACGU run trimmed/checked.
    runs = _SEQ_RE.findall(up)
    if not runs:
        return None
    exact = [r for r in runs if len(r) == n]
    if exact:
        return exact[-1]
    longest = max(runs, key=len)
    return longest if len(longest) == n else (longest[:n] if len(longest) > n else None)


class LLMGenerator(Generator):
    name = "llm"

    def __init__(self, client: Optional[OpenRouterClient] = None, temperature: float = 0.7, max_repair: int = 6):
        self.client = client or OpenRouterClient()
        self.temperature = temperature
        self.max_repair = max_repair
        self.name = f"llm:{self.client.model.split('/')[-1]}"

    def available(self) -> bool:
        return self.client.available()

    def _propose(self, prompt: str, n: int, temperature: Optional[float] = None) -> Tuple[Optional[str], object]:
        r = self.client.chat(
            [{"role": "user", "content": prompt}],
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max(256, n * 3 + 200),
            system="You are an expert RNA inverse-design scientist. Return only the RNA sequence requested.",
        )
        return parse_sequence(r.text, n), r

    def design(self, task: DesignTask, budget: int = 6, seed: Optional[int] = None) -> GenerationResult:
        """One-shot + iterative repair with actuation logging. ``budget`` caps LLM calls."""
        n = task.n
        orc = ViennaRNAOracle(
            param_set=task.oracle.param_set, temperature=task.oracle.temperature, dangles=task.oracle.dangles
        )
        logger = ActuationLogger()
        candidates: List[str] = []
        calls = 0

        prompt = render_task_prompt(task)
        seq, _ = self._propose(prompt, n)
        calls += 1
        if not seq:
            return GenerationResult(task.task_id, "A" * n, False, n, calls, [], self.name, {"error": "parse_fail"})
        candidates.append(seq)
        struct, _ = orc.fold_mfe(seq)
        cur_d = bp_distance(struct, task.target)
        best_seq, best_d = seq, cur_d

        max_calls = min(budget, 1 + self.max_repair)
        while calls < max_calls and best_d > 0:
            diag = diagnose(task.target, struct)
            rprompt = render_repair_prompt(task, seq, diag.directive)
            seq2, _ = self._propose(rprompt, n)
            calls += 1
            if not seq2:
                break
            struct2, _ = orc.fold_mfe(seq2)
            d2 = bp_distance(struct2, task.target)
            logger.log(diag.diagnosed_position_set(), seq, seq2, cur_d, d2, read=True, executed=True)
            candidates.append(seq2)
            seq, struct, cur_d = seq2, struct2, d2  # accept (LLM proposals are trusted forward)
            if d2 < best_d:
                best_seq, best_d = seq2, d2

        res = GenerationResult(
            task_id=task.task_id,
            best_sequence=best_seq,
            solved=(best_d == 0),
            best_bp_distance=best_d,
            n_evals=calls,
            candidates=candidates,
            method=self.name,
            meta={"actuation": logger.actuation_fidelity(), "usage": self.client.usage()},
        )
        return res
