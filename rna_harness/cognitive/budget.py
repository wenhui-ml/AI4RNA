#!/usr/bin/env python3
"""One currency for the evaluation surface: USD, with token capex amortised explicitly.

runs/ceiling/metric.py put capability and efficiency on one curve S(B), but with B in oracle-CPU-seconds,
which silently assumes the only cost is CPU. That is true of every operator in the harness TODAY -- they
are frozen Python and make zero LLM calls at solve time -- but it is not true of the project, because the
operators had to be written, and it will stop being true the moment an LLM enters the per-target loop.

The two costs have different structure and must not be summed naively:

    oracle CPU   MARGINAL   paid on every instance
    tokens       FIXED      paid once per artefact, then amortised over every reuse

so the per-solve cost carries an explicit reuse count N:

    cost(N) = c_cpu + C_invent / N

N -> infinity recovers the CPU-only curve; N = 1 is a one-shot agent that re-derives its operator for
every problem. Reporting the surface at several declared N is what makes the difference visible instead
of assumed.

Measured constants, from this branch:

    CPU                        $1.389e-05 per core-second (cost.py, $0.05/core-hour)
    one solve attempt, gated    111.9 oracle-s  = $0.00155
    one panel-unanimous solve  1220.8 oracle-s  = $0.01696
    one LLM call (2k in/0.5k out, default prices)      = $0.00175

The last two lines are worth keeping in view: ONE LLM CALL COSTS ABOUT THE SAME AS ONE SOLVE ATTEMPT. So
an LLM consulted once per target, amortised over the ~8 attempts a target takes, adds about 12% -- cheap
enough to be worth doing. An LLM consulted on every attempt doubles the bill.

Why this makes the A_theta claim measurable. The project's position is that a Skill (transferable
procedure) beats declarative memory. In this accounting that is exactly the statement that the
amortisation denominator is large: N = (instances) x (problem classes the artefact transfers to). A
procedure that transfers has its token cost divided away; a per-instance action does not.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .cost import CPU_USD_PER_SEC, llm_cost_usd


@dataclass
class Spend:
    """Everything one attempt (or one aggregate of attempts) consumed."""
    cpu_s: float = 0.0                 # core-seconds: wall x cores, plus oracle seconds
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = "default"
    n_attempts: int = 1

    @property
    def cpu_usd(self) -> float:
        return self.cpu_s * CPU_USD_PER_SEC

    @property
    def token_usd(self) -> float:
        return llm_cost_usd(self.tokens_in, self.tokens_out, self.model)

    @property
    def marginal_usd(self) -> float:
        """Cost that recurs on every instance. Runtime tokens ARE marginal; invention tokens are not."""
        return self.cpu_usd + self.token_usd

    def __add__(self, o: "Spend") -> "Spend":
        return Spend(self.cpu_s + o.cpu_s, self.tokens_in + o.tokens_in,
                     self.tokens_out + o.tokens_out, self.model,
                     self.n_attempts + o.n_attempts)


@dataclass
class SystemCost:
    """A system's cost model: marginal spend per attempt, plus a one-off invention capex."""
    marginal: Spend = field(default_factory=Spend)
    invent_usd: float = 0.0            # what it cost to PRODUCE this operator (tokens, human time, ...)
    invent_note: str = ""

    def per_attempt_usd(self, n_reuse: float = float("inf")) -> float:
        m = self.marginal.marginal_usd / max(self.marginal.n_attempts, 1)
        return m + (self.invent_usd / n_reuse if n_reuse and n_reuse != float("inf") else 0.0)

    def breakeven_reuse(self, other_marginal_usd: float) -> float:
        """How many reuses before the invention capex is repaid against a cheaper alternative."""
        d = other_marginal_usd - self.marginal.marginal_usd / max(self.marginal.n_attempts, 1)
        return float("inf") if d <= 0 else self.invent_usd / d


def usd(cpu_s: float = 0.0, tokens_in: int = 0, tokens_out: int = 0, model: str = "default") -> float:
    """Convenience: total USD for a mixed CPU + token spend."""
    return cpu_s * CPU_USD_PER_SEC + llm_cost_usd(tokens_in, tokens_out, model)


# ---------------------------------------------------------------- self-check / reference table
if __name__ == "__main__":
    print(f"CPU: ${CPU_USD_PER_SEC:.3e} per core-second (${CPU_USD_PER_SEC*3600:.3f}/core-hour)\n")
    rows = [
        ("one solve attempt, gated (111.9 oracle-s)", usd(cpu_s=111.9)),
        ("one solve attempt, ungated (49.7 oracle-s)", usd(cpu_s=49.7)),
        ("one panel-unanimous solve, gated (1220.8 s)", usd(cpu_s=1220.8)),
        ("one panel-unanimous solve, ungated (918.4 s)", usd(cpu_s=918.4)),
        ("one LLM call, 2k in / 0.5k out", usd(tokens_in=2000, tokens_out=500)),
        ("one LLM call, 20k in / 4k out", usd(tokens_in=20000, tokens_out=4000)),
        ("a 60-target crossing pass, gated (6714 s)", usd(cpu_s=6714)),
    ]
    w = max(len(r[0]) for r in rows)
    for lab, v in rows:
        print(f"  {lab:<{w}}  ${v:.5f}")

    print("\namortisation of an operator-invention capex against gated CPU ($0.01696/solve):")
    print(f"  {'capex':>10} {'episodes @ $0.08':>17} {'breakeven (solves)':>20} "
          f"{'benchmark passes':>18}")
    for cap in (0.08, 0.40, 2.00, 20.00):
        be = cap / 0.01696
        print(f"  ${cap:9.2f} {cap/0.08:17.0f} {be:20.0f} {be/22:18.1f}")
    print("  (a gated 60-target pass yields about 22 panel-unanimous solves)")

    print("\nwhen do runtime tokens start to matter? attempt cost is $0.00155 gated:")
    for per, lab in ((1, "one LLM call per ATTEMPT"), (8, "one per TARGET (~8 attempts)"),
                     (60, "one per BENCHMARK PASS")):
        add = usd(tokens_in=2000, tokens_out=500) / per
        print(f"  {lab:<32} +${add:.5f}/attempt = {100*add/0.00155:5.1f}% on top")
