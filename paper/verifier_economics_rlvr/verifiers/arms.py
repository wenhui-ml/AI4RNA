"""Cost-repair RLVR: three deterministic verifier arms forming a SOUND ladder.

Design (paper: cheapest-first cascade with epsilon=0 soundness):
  cheap : any number token in the completion is ~equal (tol) to the gold value. Over-accepts (gameable:
          mention the right number anywhere, box something else). ~free.
  strong: the \\boxed{} answer is value/symbolically equal to gold. Objective-aware. Costly.
  cascade: run cheap first; escalate to strong ONLY on cheap-accepts; reward = strong verdict there.
Key invariant: cheap ACCEPT-SET  \\supseteq  strong ACCEPT-SET (a strong-correct boxed value is a number
that appears, so cheap accepts it too) -> cascade == strong on accepts (sound, epsilon=0), paying strong
only on cheap survivors. This is the RLVR analogue of the paper's routing cascade."""
from __future__ import annotations
import hashlib
import re
import time
from fractions import Fraction

_NUM = re.compile(r'-?\d+(?:\.\d+)?(?:/\d+)?|-?\.\d+')


def _to_val(s):
    s = s.strip().replace(',', '').replace('$', '').replace('\\!', '')
    try:
        if '/' in s and s.count('/') == 1:
            return float(Fraction(s))
        return float(s)
    except Exception:
        try:
            import sympy as sp
            return float(sp.N(sp.sympify(s.replace('^', '**'))))
        except Exception:
            return None


def _gold_val(gold):
    m = _NUM.findall(str(gold))
    return _to_val(m[-1]) if m else _to_val(str(gold))


def extract_boxed(text):
    i = text.rfind('\\boxed')
    if i < 0:
        m = re.search(r'(?i)(?:final answer|answer)\s*[:=]?\s*\$?(-?\d[\d,./]*)', text)
        return m.group(1) if m else None
    j = text.find('{', i)
    if j < 0:
        return None
    d, k = 1, j + 1
    while k < len(text) and d:
        d += (text[k] == '{') - (text[k] == '}')
        k += 1
    return text[j + 1:k - 1]


def cheap_verify(completion, gold, tol=1e-6):
    """Loose: any number token ~= gold value. Over-accepts. Returns (accept, cost_s)."""
    t0 = time.perf_counter()
    gv = _gold_val(gold)
    ok = False
    if gv is not None:
        for tok in _NUM.findall(completion):
            v = _to_val(tok)
            if v is not None and abs(v - gv) <= tol + 1e-9 * max(1.0, abs(gv)):
                ok = True
                break
    return ok, time.perf_counter() - t0


def strong_verify(completion, gold):
    """Strict: the boxed answer is value/symbolically equal to gold. Returns (accept, cost_s)."""
    t0 = time.perf_counter()
    b = extract_boxed(completion)
    if b is None:
        return False, time.perf_counter() - t0
    gv, pv = _gold_val(gold), _to_val(b)
    ok = False
    if gv is not None and pv is not None:
        ok = abs(pv - gv) <= 1e-6 + 1e-9 * max(1.0, abs(gv))
    if not ok:
        try:
            import sympy as sp
            ge = sp.sympify(str(gold).replace('^', '**'))
            pe = sp.sympify(str(b).replace('^', '**'))
            ok = bool(sp.simplify(ge - pe) == 0)
        except Exception:
            ok = (str(b).strip() == str(gold).strip())
    return ok, time.perf_counter() - t0


def cascade_verify(completion, gold):
    """Cheapest-first: cheap gate, escalate to strong on cheap-accept. Sound wrt strong on accepts."""
    ca, cc = cheap_verify(completion, gold)
    if not ca:
        return False, cc, {"escalated": False}
    sa, sc = strong_verify(completion, gold)
    return sa, cc + sc, {"escalated": True}


def _stable_coin(completion, gold, salt="dial"):
    """Deterministic uniform in [0,1) from content — reproducible and identical across dist workers
    (unlike Python's per-process-salted hash()). Same completion -> same escalation decision."""
    h = hashlib.sha256(("%s\x00%s\x00%s" % (salt, gold, completion)).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2.0 ** 64


def reward_and_metrics(arm, completion, gold, dial_p=None):
    """Unified reward + per-sample instrumentation for J (TPR/FPR vs strong-gold), gaming, cost.

    Arms: cheap / strong / cascade, plus 'dial' — a risk-calibrated knob that escalates a cheap-accept
    to strong with probability dial_p (a deterministic content coin). dial_p=0 == cheap, dial_p=1 ==
    cascade; intermediate p traces the continuous cost x robustness frontier. Escalation only ever
    happens on cheap-accepts (cheap-rejects are certain strong-rejects, so never worth a strong call)."""
    ca, cc = cheap_verify(completion, gold)
    sa, sc = strong_verify(completion, gold)   # ground-truth strong verdict for J/gaming logging
    escalated = False
    if arm == "cheap":
        acc, cost = ca, cc
    elif arm == "strong":
        acc, cost = sa, sc
    elif arm == "cascade":
        acc, cost, escalated = (sa if ca else False), (cc + (sc if ca else 0.0)), ca
    elif arm == "dial":
        if dial_p is None or not (0.0 <= dial_p <= 1.0):
            raise ValueError("dial arm requires dial_p in [0,1], got %r" % (dial_p,))
        if not ca:
            acc, cost = False, cc                       # cheap reject -> reject, no strong call
        elif _stable_coin(completion, gold) < dial_p:
            acc, cost, escalated = sa, cc + sc, True    # escalate: pay strong, take strong verdict
        else:
            acc, cost = True, cc                        # trust cheap-accept (may reward a gamed sample)
    else:
        raise ValueError(arm)
    return {"reward": 1.0 if acc else 0.0, "accept": acc,
            "cheap_accept": ca, "strong_accept": sa,
            "gamed": bool(ca and not sa),
            "escalated": bool(escalated),
            "reward_cost_s": cost,
            "audit_cost_s": cc + sc}
