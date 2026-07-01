"""LLM (OpenRouter) client + generator.

IMPORTANT framing: the LLMGenerator here is a CONTROLLED BASELINE / TEACHER arm -- a simple
one-shot-or-repair loop over a single model. It is NOT the full autonomous agent. The full
"open-ended RNA problem solving with tools/MCP/web/memory" path is the openclaude agent
driven via the SDK (see experiments/agent_runner). Keeping this thin, model-agnostic loop
lets us (a) get a strong LLM baseline, (b) use Opus-4.8 as a bootstrap teacher, and (c) run
Opus as an in-loop diagnoser with instrumented actuation -- all model-swappable by one env
var (OPENAI/OPENROUTER model id), never with model-specific prompt hacks.
"""
from __future__ import annotations

from .openrouter import OpenRouterClient, LLMResult  # noqa: F401
from .generator import LLMGenerator, parse_sequence  # noqa: F401

__all__ = ["OpenRouterClient", "LLMResult", "LLMGenerator", "parse_sequence"]
