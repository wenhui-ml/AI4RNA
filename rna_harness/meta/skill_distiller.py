"""A_theta procedural memory: distill a *transferable method* from a winning solve trace into
an openclaude Skill (`.claude/skills/rna-<class>/SKILL.md`).

This is the corrected memory vehicle. The K_theta path (meta/strategies.py) sank a *fragment of the
answer sequence* (`seq[i+1:j]` hairpin motifs) + declarative bullets into memdir/MEMORY.md -- the
vehicle openclaude's own design explicitly forbids for procedures (src/memdir/memoryTypes.ts
WHAT_NOT_TO_SAVE_SECTION). That content-memory gets over-applied -> uniformity collapse.

Here memory is a PROCEDURE: trigger by target *features*, steps referencing the MCP solver tools,
decision rules. openclaude's native skill-discovery surfaces it to later episodes, which invoke it
by name. It compounds capability, not words, and cannot collapse (it holds no answer content).

RED LINE, structurally enforced (not prompt-hoped):
  1. the trace digest fed to the distiller is PRE-MASKED -- every A/C/G/U run and dot-bracket run is
     blanked to <SEQ>/<STRUCT>, so the model has no answer sequence available to copy;
  2. the emitted SKILL.md is GUARDED -- any residual sequence/structure run rejects/scrubs it.
So a skill is method, never answers.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional, Tuple

import requests

from ..structure_utils import pair_table

# --------------------------------------------------------------------------- target features/class
def _children(pt, i: int, j: int) -> int:
    """Number of child helices directly enclosed by pair (i,j) (>=2 => multiloop)."""
    c, k = 0, i + 1
    while k < j:
        if pt[k] > k:
            c += 1
            k = pt[k] + 1
        else:
            k += 1
    return c


def _has_multiloop(pt, n: int) -> bool:
    for i in range(n):
        j = pt[i]
        if j > i and _children(pt, i, j) >= 2:
            return True
    return False


def _has_tandem(s: str, L: int = 12) -> bool:
    """A dot-bracket substring of length >=L recurring non-overlapping => tandem/fractal target."""
    seen = {}
    for i in range(len(s) - L + 1):
        sub = s[i:i + L]
        if sub in seen and i - seen[sub] >= L:
            return True
        seen.setdefault(sub, i)
    return False


def target_features(target: str) -> Tuple[str, str, str]:
    """(class_key, human features string, length band). Deterministic; drives skill filing/merging."""
    pt = pair_table(target)
    n = len(target)
    pairs = [(i, pt[i]) for i in range(n) if pt[i] > i]
    max_span = max((j - i for i, j in pairs), default=0)
    has_ml = _has_multiloop(pt, n)
    tr = _has_tandem(target)
    long_range = n >= 40 and max_span >= 0.55 * n
    band = "S(<=60)" if n <= 60 else "M(61-120)" if n <= 120 else "L(121-250)" if n <= 250 else "XL(>250)"
    if tr:
        cls = "tandem-repeat"
    elif has_ml:
        cls = "multiloop"
    elif long_range:
        cls = "long-range-pair"
    else:
        cls = "direct"
    feats = (f"length={n}nt (band {band}); n_pairs={len(pairs)}; max_pair_span={max_span}; "
             f"multiloop={has_ml}; tandem_repeat={tr}; dominant_long_range={long_range}")
    return cls, feats, band


# --------------------------------------------------------------------------- red-line masking/guard
_SEQ = re.compile(r"[ACGUacgu]{8,}")
_STRUCT = re.compile(r"[.()\[\]{}]{8,}")
_SEQ_HARD = re.compile(r"[ACGUacgu]{12,}")
_STRUCT_HARD = re.compile(r"[.()\[\]{}]{12,}")


def mask_answers(text: str) -> str:
    """Blank any nucleotide run / dot-bracket run so the answer literally can't be copied."""
    text = _SEQ.sub(lambda m: f"<SEQ:{len(m.group(0))}>", text)
    text = _STRUCT.sub(lambda m: f"<STRUCT:{len(m.group(0))}>", text)
    return text


def guard_skill(md: str) -> Tuple[bool, str]:
    if _SEQ_HARD.search(md):
        return False, "contains a nucleotide sequence"
    if _STRUCT_HARD.search(md):
        return False, "contains a dot-bracket structure"
    if "---" not in md or "name:" not in md:
        return False, "missing frontmatter"
    return True, "ok"


# --------------------------------------------------------------------------- trace digest
def trace_digest(trace_path: str, max_chars: int = 5500) -> str:
    """Compact, MASKED account of how the episode solved it: tool sequence + agent reasoning."""
    tools, texts = [], []
    try:
        with open(trace_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                msg = r.get("message", r)
                content = msg.get("content") if isinstance(msg, dict) else None
                if isinstance(content, str):
                    texts.append(content)
                elif isinstance(content, list):
                    for b in content:
                        if not isinstance(b, dict):
                            continue
                        if b.get("type") == "text":
                            texts.append(b.get("text", ""))
                        elif b.get("type") == "tool_use":
                            inp = b.get("input", {}) or {}
                            knobs = {k: v for k, v in inp.items()
                                     if k in ("n_restarts", "tries", "iters", "n_units") and not isinstance(v, str)}
                            arg = ",".join(f"{k}={v}" for k, v in knobs.items())
                            tools.append(f"{b.get('name')}({arg})")
    except Exception:
        return ""
    reasoning = mask_answers("\n".join(t for t in texts if t.strip()))
    if len(reasoning) > max_chars:  # keep the plan (head) and the endgame fix (tail)
        reasoning = reasoning[: max_chars // 2] + "\n...[snip]...\n" + reasoning[-max_chars // 2:]
    seq = " -> ".join(tools) if tools else "(no tool calls recorded)"
    return f"TOOL SEQUENCE that solved it:\n{seq}\n\nAGENT REASONING (answer content masked):\n{reasoning}"


# --------------------------------------------------------------------------- LLM
_SYSTEM = (
    "You distill a REUSABLE RNA inverse-design METHOD from one winning solve trace into a Claude "
    "Code SKILL.md (a transferable procedure that a fresh agent can invoke on a DIFFERENT target of "
    "the same structural class).\n"
    "ABSOLUTE RULES (a skill is method, NEVER answers):\n"
    "  1. NEVER write any nucleotide sequence (A/C/G/U runs) or dot-bracket structure string. The "
    "trace has these masked as <SEQ>/<STRUCT>; keep everything abstract. No target names.\n"
    "  2. Trigger by target FEATURES only (length band, #multiloops, dominant long-range pairs, "
    "tandem repeats) -- so it transfers.\n"
    "  3. Steps must reference the actual solver TOOLS: mcp__rna_motifs__analyze_target (decomposition "
    "plan), mcp__rna_motifs__inverse_fold_search (the strong search; submit its exact-length output "
    "verbatim), mcp__rna_motifs__finish (diagnosis-guided hill-climb on a near-miss), RnaVerify.\n"
    "  4. Capture the DECISION RULES that made it work (what to do on near-miss / search plateau / "
    "over-pairing / degeneracy), not just a linear recipe.\n"
    "Output ONLY the SKILL.md: '---' YAML frontmatter (name, description, when_to_use, allowed-tools) "
    "then a concise markdown procedure (<= ~40 lines). No preamble, no code fences."
)


def _llm(user: str, model: str, max_tokens: int = 4000) -> Optional[str]:
    # max_tokens raised 1400->4000: at 1400 a merged skill got truncated mid-sentence (shipping half
    # a skill as the whole treatment). A skill is ~40-60 lines; 4000 tokens leaves headroom. We also
    # REJECT a length-truncated completion below so a cut-off skill is never written.
    key = os.environ.get("OPENAI_API_KEY")
    base = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    if not key:
        return None
    resp = requests.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "temperature": 0.2, "max_tokens": max_tokens,
              "messages": [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]},
        timeout=150,
    )
    resp.raise_for_status()
    ch = resp.json()["choices"][0]
    if ch.get("finish_reason") == "length":   # truncated -> refuse; better no skill than half a skill
        return None
    return ch["message"]["content"]


def _ensure_name(md: str, name: str) -> str:
    """Force the frontmatter `name:` to match the skill dir (stable merging + valid invocation)."""
    if re.search(r"(?m)^name:\s*.+$", md):
        return re.sub(r"(?m)^name:\s*.+$", f"name: {name}", md, count=1)
    # inject after the opening '---'
    return re.sub(r"^---\s*\n", f"---\nname: {name}\n", md, count=1)


# --------------------------------------------------------------------------- public API
def distill_skill(out_dir: str, target: str, trace_path: str, model: str) -> Optional[str]:
    """Turn a winning trace into a guarded SKILL.md under <out_dir>/.claude/skills/rna-<class>/.
    Merges into an existing skill for the same class. Returns the skill name, or None if nothing
    usable/clean could be distilled."""
    cls, feats, _ = target_features(target)
    name = f"rna-{cls}"
    skdir = os.path.join(out_dir, ".claude", "skills", name)
    skfile = os.path.join(skdir, "SKILL.md")
    existing = ""
    if os.path.exists(skfile):
        try:
            existing = open(skfile).read()
        except Exception:
            existing = ""
    digest = trace_digest(trace_path)
    if not digest.strip() or "(no tool calls recorded)" in digest and not existing:
        return None
    user = (f"TARGET CLASS: {cls}\nTARGET FEATURES: {feats}\n\n{digest}\n")
    if existing:
        user += ("\nAN EXISTING SKILL for this class already exists. MERGE the new method into it "
                 "(refine, dedup, stay concise). Current content:\n" + existing + "\n")
    user += "\nEmit the complete SKILL.md now. METHOD ONLY -- never a sequence or structure string."
    try:
        md = _llm(user, model)
    except Exception:
        return None
    if not md:
        return None
    md = md.strip()
    md = re.sub(r"^```[a-zA-Z]*\n", "", md)
    md = re.sub(r"\n```$", "", md).strip()
    ok, _reason = guard_skill(md)
    if not ok:  # one scrub pass: drop any offending line, then re-check
        md = "\n".join(l for l in md.splitlines()
                       if not _SEQ_HARD.search(l) and not _STRUCT_HARD.search(l))
        ok, _reason = guard_skill(md)
        if not ok:
            return None
    md = _ensure_name(md, name)
    os.makedirs(skdir, exist_ok=True)
    with open(skfile, "w") as f:
        f.write(md.rstrip() + "\n")
    return name


if __name__ == "__main__":  # offline self-test (no network): features + mask + guard
    t = "(((((....)))))" + "." * 5 + "(((((....)))))"
    print("features:", target_features(t))
    dirty = "The winner was GGGCAAAAGCCC folding to (((((....))))) here."
    print("masked:", mask_answers(dirty))
    print("guard clean:", guard_skill("---\nname: rna-x\n---\nCall inverse_fold_search then finish."))
    print("guard dirty:", guard_skill("---\nname: rna-x\n---\nUse GGGGCCCCAAAA as the stem."))
