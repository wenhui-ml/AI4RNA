"""RNA strategy memory in openclaude's native memdir format.

A MemStore owns a directory that IS an openclaude auto-memory dir (MEMORY.md index + typed
`.md` entries with frontmatter). We write strategies here and point the live agent at it via
CLAUDE_COWORK_MEMORY_PATH_OVERRIDE, so openclaude recalls them through its own
loadMemoryPrompt / findRelevantMemories path -- we extend memory, we don't replace it.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Strategy:
    name: str                    # kebab-case slug (also the .md filename)
    kind: str                    # 'hairpin_motif' | 'gc_closing'
    trigger: Dict[str, object]   # when it applies, e.g. {'loop_len': 4}
    rule: Dict[str, object]      # what to do, e.g. {'motif': 'GAAA'} / {'min_gc_closing': 0.8}
    description: str
    support: int = 1             # number of solved cases it was distilled from

    def key(self) -> tuple:
        return (self.kind, json.dumps(self.trigger, sort_keys=True))

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "kind": self.kind, "trigger": self.trigger,
                "rule": self.rule, "description": self.description, "support": self.support}

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "Strategy":
        return cls(name=d["name"], kind=d["kind"], trigger=d.get("trigger", {}),
                   rule=d.get("rule", {}), description=d.get("description", ""),
                   support=int(d.get("support", 1)))


_FRONT_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
_META_RE = re.compile(r"rna_strategy:\s*(\{.*\})")  # JSON ends at last brace; trailing --> ignored


class MemStore:
    def __init__(self, mem_dir: str):
        self.dir = mem_dir
        os.makedirs(self.dir, exist_ok=True)

    # -- read ------------------------------------------------------------
    def strategies(self) -> List[Strategy]:
        out: List[Strategy] = []
        for fn in sorted(os.listdir(self.dir)):
            if not fn.endswith(".md") or fn == "MEMORY.md":
                continue
            try:
                with open(os.path.join(self.dir, fn)) as f:
                    txt = f.read()
                m = _META_RE.search(txt)
                if m:
                    out.append(Strategy.from_dict(json.loads(m.group(1))))
            except Exception:
                continue
        return out

    # -- write -----------------------------------------------------------
    def _entry_md(self, s: Strategy) -> str:
        return (
            f"---\n"
            f"name: {s.name}\n"
            f"description: {s.description}\n"
            f"metadata:\n  type: reference\n"
            f"---\n\n"
            f"{s.description}\n\n"
            f"Applies when: {json.dumps(s.trigger)}. Design rule: {json.dumps(s.rule)}. "
            f"Learned from {s.support} solved case(s).\n\n"
            f"<!-- rna_strategy: {json.dumps(s.to_dict())} -->\n"
        )

    def write(self, strategies: List[Strategy]) -> None:
        # clear old strategy files (keep it a clean reflection of current knowledge)
        for fn in os.listdir(self.dir):
            if fn.endswith(".md"):
                os.remove(os.path.join(self.dir, fn))
        index = ["# RNA design playbook (learned strategies)\n"]
        for s in strategies:
            with open(os.path.join(self.dir, f"{s.name}.md"), "w") as f:
                f.write(self._entry_md(s))
            index.append(f"- [{s.name}]({s.name}.md) — {s.description} (support {s.support})")
        with open(os.path.join(self.dir, "MEMORY.md"), "w") as f:
            f.write("\n".join(index) + "\n")

    def clear(self) -> None:
        for fn in os.listdir(self.dir):
            if fn.endswith(".md"):
                os.remove(os.path.join(self.dir, fn))

    def render_playbook(self) -> str:
        ss = self.strategies()
        if not ss:
            return "No learned RNA design strategies yet."
        lines = ["# Learned RNA design strategies (playbook):"]
        for s in ss:
            lines.append(f"- {s.description} (applies when {json.dumps(s.trigger)}; support {s.support})")
        return "\n".join(lines)
