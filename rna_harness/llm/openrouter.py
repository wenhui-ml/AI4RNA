"""Minimal OpenRouter (OpenAI-compatible) chat client with retries + usage tracking.

Model-agnostic by design: pass any OpenRouter model id (anthropic/claude-opus-4.8,
deepseek/deepseek-v4-pro, z-ai/glm-5.2, qwen/...). No provider- or model-specific behavior.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "anthropic/claude-opus-4.8")


def _api_key() -> Optional[str]:
    for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        v = os.environ.get(var)
        if v:
            return v
    return None


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    raw: Dict = field(default_factory=dict)
    error: Optional[str] = None
    reasoning: str = ""          # separate reasoning-trace field (reasoning models); empty otherwise
    reasoning_details: Optional[object] = None  # pass back UNMODIFIED in the next assistant turn so a
                                                # reasoning model continues its chain across turns


class OpenRouterClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: Optional[str] = None,
        max_retries: int = 4,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or _api_key()
        self.max_retries = max_retries
        self.timeout = timeout
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.n_calls = 0

    def available(self) -> bool:
        return bool(self.api_key)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = 1024,
        system: Optional[str] = None,
        reasoning: Optional[Dict] = None,
    ) -> LLMResult:
        """`reasoning` passes OpenRouter's reasoning control, e.g. {"enabled": True} to let a reasoning model
        think. IMPORTANT: `max_tokens` is the TOTAL output cap (reasoning + content); a reasoning model's
        chain can exceed a small cap and leave NO room for content -> empty reply. Pass max_tokens=None to
        omit the cap (use the provider default) when reasoning is enabled. Ignored by non-reasoning models."""
        if not self.api_key:
            return LLMResult(text="", error="no API key (set OPENROUTER_API_KEY or OPENAI_API_KEY)")
        msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        payload = {"model": self.model, "messages": msgs, "temperature": temperature}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if reasoning is not None:
            payload["reasoning"] = reasoning
        body = json.dumps(payload).encode()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/InceptLabs/openclaude_rna",
            "X-Title": "openclaude_rna",
        }
        last_err = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(self.base_url + "/chat/completions", data=body, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    d = json.loads(resp.read().decode())
                msg = d["choices"][0]["message"]
                text = msg.get("content") or ""
                rtext = msg.get("reasoning") or msg.get("reasoning_content") or ""
                usage = d.get("usage", {}) or {}
                pt, ct = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
                self.total_prompt_tokens += pt
                self.total_completion_tokens += ct
                self.n_calls += 1
                return LLMResult(text=text, prompt_tokens=pt, completion_tokens=ct,
                                 model=d.get("model", self.model), raw=d, reasoning=rtext,
                                 reasoning_details=msg.get("reasoning_details"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode()[:400]
                last_err = f"HTTP {e.code}: {detail}"
                if e.code in (429, 500, 502, 503, 529):
                    time.sleep(min(2 ** attempt, 20))
                    continue
                break
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
                time.sleep(min(2 ** attempt, 20))
        return LLMResult(text="", error=last_err or "unknown error")

    def usage(self) -> Dict[str, int]:
        return {
            "n_calls": self.n_calls,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
        }
