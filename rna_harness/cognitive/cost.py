"""Unified economic cost accounting -- DOMAIN-AGNOSTIC.

Every solve attempt carries a cost in USD so tools of different KINDS (classical CPU solvers vs LLM
agents) sit on ONE axis the router can minimise. Constants are rough/public and CONFIGURABLE via env;
only the RELATIVE scale matters for routing. This is the single currency for the whole cognitive core.
"""
from __future__ import annotations

import os

# A commodity vCPU-hour (order-of-magnitude; override with COGCORE_CPU_USD_HR). Only relative scale
# matters for routing decisions, but keeping it near a real price makes the $ numbers interpretable.
CPU_USD_PER_HR = float(os.environ.get("COGCORE_CPU_USD_HR", "0.05"))
CPU_USD_PER_SEC = CPU_USD_PER_HR / 3600.0

# LLM token prices ($ per 1M tokens), rough/public; override per model as needed.
_TOKEN_USD_PER_M = {
    "default": {"in": 0.50, "out": 1.50},
    "z-ai/glm-5.2": {"in": 0.40, "out": 1.60},
}


def solver_cost_usd(wall_s: float, cores: int = 1) -> float:
    """CPU-time cost of a classical solver run = cores x wall_seconds x per-second rate."""
    return max(0.0, cores) * max(0.0, wall_s) * CPU_USD_PER_SEC


def llm_cost_usd(tokens_in: int, tokens_out: int, model: str = "default") -> float:
    """Token cost of an LLM call."""
    p = _TOKEN_USD_PER_M.get(model, _TOKEN_USD_PER_M["default"])
    return (max(0, tokens_in) * p["in"] + max(0, tokens_out) * p["out"]) / 1e6
