"""AI4Catalysis-style rich, server-rendered pages for the openclaude_rna dashboard.

Self-contained (NO CDN / external JS): saturation curves and RNA secondary-structure diagrams
are rendered as inline SVG so reports work offline. Wired into app.py as extra routes:

    /ui                 runs list (verdict badge, episodes/solved, actions)
    /report/{run}       per-run report: verdict banner + overview + saturation SVG
                        + per-episode ranking + RNA structure (target vs achieved MFE) arcs
    /trace/{run}        per-episode agent tool-call trace (from traces/*.jsonl)
    /memory/{run}       injected memory snapshots + learned strategy playbook

Reads campaign_state.json (partial/killed/in-progress runs show their episodes too), plus
campaign_summary.json / summary.json / scorecard.json when present.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

# ------------------------------------------------------------------ styling (self-contained)
CSS = """
:root{--bg:#fafafa;--card:#fff;--border:#e6e8ec;--text:#1a1a2e;--muted:#6b7280;--accent:#1976d2;
 --ok:#2e7d32;--okl:#4caf50;--warn:#b07d00;--warnl:#f9a825;--fail:#c62828;--faill:#e53935;}
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
 margin:0;background:var(--bg);color:var(--text);font-size:14px;line-height:1.5}
a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}
.topbar{display:flex;align-items:center;justify-content:space-between;padding:14px 28px;
 background:var(--card);border-bottom:1px solid var(--border);position:sticky;top:0;z-index:10}
.topbar h1{font-size:1.15rem;font-weight:700;margin:0} .topbar h1 a{color:var(--muted);margin-right:8px}
.topbar .lnk{font-size:.9rem}
header.verdict{padding:22px 32px;color:#fff}
header.verdict .big{font-size:1.9rem;font-weight:700}
header.verdict small{display:block;font-size:.9rem;font-weight:400;margin-top:8px;opacity:.95;max-width:1100px}
header.verdict.success{background:linear-gradient(90deg,#2e7d32,#4caf50)}
header.verdict.partial{background:linear-gradient(90deg,#b07d00,#f9a825)}
header.verdict.failure{background:linear-gradient(90deg,#b71c1c,#e53935)}
header.verdict.neutral{background:linear-gradient(90deg,#37474f,#607d8b)}
main{max-width:1280px;margin:0 auto;padding:24px 32px 80px}
section{margin:30px 0}
section h2{font-size:1.25rem;border-left:4px solid var(--accent);padding-left:10px;margin:0 0 12px}
.card{background:var(--card);padding:14px 16px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.06);border:1px solid var(--border)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px}
table{width:100%;border-collapse:collapse;background:var(--card);border-radius:8px;overflow:hidden;
 box-shadow:0 1px 3px rgba(0,0,0,.06)}
th,td{padding:8px 12px;text-align:left;font-size:.9rem;border-bottom:1px solid var(--border);vertical-align:top}
th{background:#f0f2f5;font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.03em;color:var(--muted)}
tr.solved td{background:#f1f8e9} tr.unsolved td{background:#fff5f5}
.badge{display:inline-block;padding:2px 9px;border-radius:11px;font-size:.76rem;font-weight:600;color:#fff}
.badge.success{background:var(--ok)} .badge.partial{background:var(--warnl);color:#222}
.badge.failure{background:var(--fail)} .badge.neutral{background:var(--muted)}
.mono,code{font-family:ui-monospace,"JetBrains Mono","SF Mono",Menlo,monospace;font-size:.85rem}
.seq{font-family:ui-monospace,monospace;font-size:.8rem;word-break:break-all}
.muted{color:var(--muted)} .kv{display:inline-block;margin:2px 18px 2px 0}
details{background:var(--card);padding:8px 14px;margin:6px 0;border-radius:6px;border:1px solid var(--border)}
details>summary{cursor:pointer;font-weight:600}
pre{background:#f6f8fa;padding:10px 12px;border-radius:6px;overflow:auto;font-size:.8rem;max-height:460px}
.scroll{overflow-x:auto} .chip{display:inline-block;padding:1px 7px;border-radius:8px;font-size:.72rem;margin:1px}
.chip.g{background:#c8e6c9;color:#1b5e20}.chip.r{background:#ffcdd2;color:#b71c1c}.chip.y{background:#fff3c4;color:#7a5c00}
td.win{background:#e8f5e9;font-weight:700} td.lose{color:#9e9e9e} .col-b{border-left:2px solid var(--border)}
.legend span{margin-right:14px;font-size:.82rem} .sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:4px;vertical-align:middle}
"""

_ARM_COLOR = {"with_memory": "#2e7d32", "no_memory": "#78909c", "permuted_memory": "#c62828"}
_TOOL_RE = re.compile(r'"name"\s*:\s*"([A-Za-z0-9_]+)"')


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def page(title: str, body: str) -> str:
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{esc(title)}</title>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<style>{CSS}</style></head><body>"
            f"<div class='topbar'><h1><a href='/ui'>&#8592;</a> {esc(title)}</h1>"
            f"<span class='lnk'><a href='/ui'>Runs</a> &middot; <a href='/'>Live tabs</a></span></div>"
            f"{body}</body></html>")


# ------------------------------------------------------------------ data loading
def _read_json(p: str) -> Optional[dict]:
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _state_records(run_dir: str) -> Tuple[List[Dict], int]:
    """Per-episode records from campaign_state.json (works for partial/killed runs)."""
    st = _read_json(os.path.join(run_dir, "campaign_state.json"))
    if not st:
        return [], 0
    recs = list(st.get("done", {}).values())
    recs.sort(key=lambda r: (r.get("arm", ""), r.get("epoch", 0), r.get("n", 0)))
    return recs, st.get("n_episodes", len(recs))


def _agent_records(run_dir: str):
    """Normalize an agent_runner summary.json (single-pass agent eval) into the campaign
    per-episode record shape, so the same report machinery renders it."""
    s = _read_json(os.path.join(run_dir, "summary.json"))
    if not s or not s.get("model") or not isinstance(s.get("records"), list):
        return None
    recs = []
    for r in s["records"]:
        a = r.get("actuation") or {}
        recs.append({
            "arm": "agent", "epoch": 0,
            "task_id": r.get("task_id", ""), "family": r.get("family", ""), "n": r.get("n", 0),
            "solved": r.get("solved", False), "reward": r.get("reward", 0.0),
            "oracle_calls": r.get("n_rna_tool_calls"), "n_mcp_calls": r.get("n_mcp_calls"),
            "cross_oracle": r.get("cross_oracle"), "A": a.get("A"), "A_memory": None,
            "final_sequence": r.get("final_sequence") or "", "elapsed_s": r.get("elapsed_s"),
            "trace_path": r.get("trace_path"), "model": r.get("model"),
        })
    return recs, s


def _load_records(run_dir: str) -> Tuple[List[Dict], int, Optional[str]]:
    """Return (records, n_episodes, kind) where kind in {'campaign','agent','agent_live',None}.

    'agent_live' = an agent run still IN PROGRESS (traces present, no summary.json yet): its
    per-task records are scored on the fly from traces/*.jsonl so the report is a live scoreboard.
    """
    recs, n = _state_records(run_dir)
    if recs:
        return recs, n, "campaign"
    ar = _agent_records(run_dir)
    if ar:
        return ar[0], len(ar[0]), "agent"
    live = _live_agent_records(run_dir)
    if live:
        return live[0], len(live[0]), "agent_live"
    return [], 0, None


# ------------------------------------------------------------------ live scoring (in-progress runs)
# Cache keyed by trace path -> {mtime,size,rec}. A finished trace never changes, so it is folded
# exactly once; only new/growing traces are (re)scored on each page load. Read-only.
_LIVE_CACHE: Dict[str, Dict] = {}
_E100_BY_SAFE: Optional[Dict[str, object]] = None


def _e100_map():
    global _E100_BY_SAFE
    if _E100_BY_SAFE is None:
        from ..benchmarks import eterna100_tasks
        tasks = eterna100_tasks(ids=None, max_len=400)
        _E100_BY_SAFE = {re.sub(r"[^A-Za-z0-9_.-]", "_", t.task_id): t for t in tasks}
    return _E100_BY_SAFE


def _score_one_trace(path: str, fn: str):
    """Score a single finished trace with the SAME extractor + folding oracle the runner uses
    (verifier.verify), so live numbers match the eventual summary.json exactly. Returns a record
    in the normalized per-episode shape, or None if the trace can't be matched/scored."""
    from ..experiments.agent_runner import _extract_final_sequence, _count_tool_calls
    from ..verifier import verify
    key = fn[:-6] if fn.endswith(".jsonl") else fn
    m = _e100_map()
    task = m.get(key)
    if task is None:
        cand = [t for k, t in m.items() if key.startswith(k) or k.startswith(key)]
        task = cand[0] if cand else None
    if task is None:
        return None
    try:
        with open(path) as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
    except OSError:
        return None
    seq = _extract_final_sequence(lines, task.n)
    v = verify(task, seq, compute_cross_oracle=False) if seq else None
    calls = _count_tool_calls(lines)
    return {
        "arm": "agent", "epoch": 0, "task_id": task.task_id,
        "family": getattr(task, "family", ""), "n": task.n,
        "solved": bool(v.solved) if v else False, "reward": (v.reward if v else 0.0),
        "oracle_calls": sum(c for k, c in calls.items() if k.startswith("Rna")),
        "n_mcp_calls": sum(c for k, c in calls.items() if k.startswith("mcp__")),
        "cross_oracle": None, "A": None, "A_memory": None,
        "final_sequence": seq or "", "elapsed_s": None, "trace_path": path, "model": "",
    }


def _live_agent_records(run_dir: str):
    """For an in-progress agent run (no summary.json, traces present): fold-score every finished
    trace on the fly. Returns (records, live_summary) or None. Cached per-trace by (mtime,size)."""
    if os.path.exists(os.path.join(run_dir, "summary.json")):
        return None  # finished run -> use the authoritative summary path instead
    tdir = os.path.join(run_dir, "traces")
    if not os.path.isdir(tdir):
        return None
    tracefiles = sorted(f for f in os.listdir(tdir) if f.endswith(".jsonl"))
    if not tracefiles:
        return None
    recs: List[Dict] = []
    for fn in tracefiles:
        path = os.path.join(tdir, fn)
        try:
            stt = os.stat(path)
        except OSError:
            continue
        c = _LIVE_CACHE.get(path)
        if c and c["mtime"] == stt.st_mtime and c["size"] == stt.st_size:
            rec = c["rec"]
        else:
            try:
                rec = _score_one_trace(path, fn)
            except Exception:
                rec = None
            _LIVE_CACHE[path] = {"mtime": stt.st_mtime, "size": stt.st_size, "rec": rec}
        if rec is not None:
            recs.append(rec)
    if not recs:
        return None
    solved = sum(1 for r in recs if r.get("solved"))
    n = len(recs)
    model = _run_model(run_dir) or ""
    summary = {
        "model": model, "n_tasks": n, "solved": solved,
        "solve_rate": round(solved / n, 4),
        "mean_reward": round(sum(r.get("reward", 0.0) for r in recs) / n, 4),
        "mean_rna_calls": round(sum((r.get("oracle_calls") or 0) for r in recs) / n, 2),
        "_live": True,
    }
    recs.sort(key=lambda r: r.get("n", 0))
    return recs, summary


def _length_bands(recs: List[Dict]) -> str:
    """Compact solved/total-by-length HTML table (the capability-boundary view)."""
    bands = [("≤45", 0, 45), ("46–70", 46, 70), ("71–100", 71, 100),
             ("101–150", 101, 150), ("151–250", 151, 250), (">250", 251, 10 ** 9)]
    cells = []
    for lbl, lo, hi in bands:
        b = [r for r in recs if lo <= r.get("n", 0) <= hi]
        if not b:
            continue
        sv = sum(1 for r in b if r.get("solved"))
        rate = sv / len(b)
        col = "var(--ok)" if rate >= 0.6 else ("var(--warnl)" if rate >= 0.3 else "var(--fail)")
        cells.append(f"<span class='kv'>{lbl} nt: <b style='color:{col}'>{sv}/{len(b)}</b></span>")
    return "".join(cells)


def _run_summary(run_dir: str) -> Dict:
    recs, n_ep, kind = _load_records(run_dir)
    solved = sum(1 for r in recs if r.get("solved"))
    arms: Dict[str, Dict] = {}
    for r in recs:
        a = arms.setdefault(r.get("arm", "?"), {"n": 0, "solved": 0})
        a["n"] += 1
        a["solved"] += 1 if r.get("solved") else 0
    return {"n_episodes": n_ep, "n_records": len(recs), "solved": solved, "kind": kind,
            "solve_rate": (solved / len(recs)) if recs else None, "arms": arms,
            "has_campaign_summary": os.path.exists(os.path.join(run_dir, "campaign_summary.json")),
            "has_scorecard": os.path.exists(os.path.join(run_dir, "scorecard.json")),
            "has_summary": os.path.exists(os.path.join(run_dir, "summary.json"))}


def _curves(run_dir: str) -> Dict[str, List[Dict]]:
    cs = _read_json(os.path.join(run_dir, "campaign_summary.json"))
    if cs and cs.get("curves"):
        return cs["curves"]
    ss = _read_json(os.path.join(run_dir, "saturation_summary.json"))
    if ss and ss.get("curves"):
        return ss["curves"]
    # derive from state (partial runs)
    recs, _ = _state_records(run_dir)
    curves: Dict[str, Dict[int, List[Dict]]] = {}
    for r in recs:
        curves.setdefault(r.get("arm", "?"), {}).setdefault(r.get("epoch", 0), []).append(r)
    out: Dict[str, List[Dict]] = {}
    for arm, epochs in curves.items():
        pts = []
        for ep in sorted(epochs):
            rows = epochs[ep]
            sv = [x for x in rows if x.get("solved")]
            oc = [x["oracle_calls"] for x in sv if x.get("oracle_calls") is not None]
            am = [x["A_memory"] for x in rows if x.get("A_memory") is not None]
            pts.append({"epoch": ep, "n": len(rows),
                        "solve_rate": round(len(sv) / len(rows), 4) if rows else 0,
                        "mean_oracle_calls_solved": round(sum(oc) / len(oc), 2) if oc else None,
                        "A_memory": round(sum(am) / len(am), 4) if am else None})
        out[arm] = pts
    return out


# ------------------------------------------------------------------ RNA structure (inline SVG)
def _pairs(struct: str):
    from ..structure_utils import base_pairs
    try:
        return set(base_pairs(struct))
    except Exception:
        return set()


_FOLD_CACHE: Dict[str, str] = {}


def _fold_mfe(seq: str) -> str:
    if not seq:
        return ""
    key = seq.upper().replace("T", "U")
    if key in _FOLD_CACHE:
        return _FOLD_CACHE[key]
    try:
        import RNA
        md = RNA.md(); md.temperature = 37.0; md.dangles = 2
        st, _ = RNA.fold_compound(key, md).mfe()
    except Exception:
        st = ""
    _FOLD_CACHE[key] = st
    return st


def svg_arc(target: str, mfe: str, seq: str = "", solved=None, reason: str = "") -> str:
    """Diagnostic RNA arc diagram (inline SVG, no JS/CDN). THREE modes:
      * solved      -> ONE green arc set (target == achieved MFE), 'bp-distance 0'.
      * diagnostic  -> target arcs ABOVE (green=correct / amber-dashed=missing), achieved MFE
                       arcs BELOW (green=correct / red=spurious). The point of the view.
      * target-only -> target arcs in GREY when the agent submitted NO sequence (shows the
                       structure the agent failed to design).
    Arcs are ELLIPTICAL, height scaled to fit, so deeply-nested large targets never clip."""
    n = max(len(target), len(mfe))
    if n == 0:
        return "<p class='muted'>no structure</p>"
    tp, mp = _pairs(target), _pairs(mfe)
    correct, missing, spurious = tp & mp, tp - mp, mp - tp
    has_fold = bool(seq) and bool(mfe)
    is_solved = solved if solved is not None else (has_fold and not missing and not spurious)
    mode = "target_only" if not has_fold else ("solved" if is_solved else "diagnostic")
    draw_below = mode == "diagnostic"

    step = max(6, min(16, int(1100 / n)))
    W = n * step + 40
    x = lambda i: 20 + i * step + step / 2
    used = (tp | mp) if draw_below else tp
    maxrx = max(((x(j) - x(i)) / 2) for i, j in used) if used else 20
    MAXH = 160.0
    scale = min(1.0, MAXH / maxrx) if maxrx else 1.0  # flatten tall arcs to fit; keeps nesting
    ah = maxrx * scale + 14
    ax = ah + 20
    H = (ah * 2 + 46) if draw_below else (ax + 24)

    def arc(i, j, up, color, dashed=False):
        rx = (x(j) - x(i)) / 2
        ry = rx * scale
        sweep = 1 if up else 0
        dash = " stroke-dasharray='4 3'" if dashed else ""
        return (f"<path d='M {x(i):.1f} {ax:.1f} A {rx:.1f} {ry:.1f} 0 0 {sweep} {x(j):.1f} {ax:.1f}' "
                f"fill='none' stroke='{color}' stroke-width='1.6'{dash}/>")

    parts = [f"<svg viewBox='0 0 {W} {H:.0f}' width='{min(W,1180)}' style='max-width:100%;background:#fff;border:1px solid #e6e8ec;border-radius:6px'>"]
    parts.append(f"<line x1='20' y1='{ax:.1f}' x2='{W-20}' y2='{ax:.1f}' stroke='#cfd8dc'/>")
    if mode == "target_only":
        for i, j in sorted(tp):
            parts.append(arc(i, j, True, "#90a4ae"))
    elif mode == "solved":
        for i, j in sorted(tp):
            parts.append(arc(i, j, True, "#2e7d32"))
    else:  # diagnostic
        for i, j in sorted(tp):  # target above
            parts.append(arc(i, j, True, "#2e7d32" if (i, j) in correct else "#f9a825",
                             dashed=(i, j) in missing))
        for i, j in sorted(mp):  # achieved below
            parts.append(arc(i, j, False, "#2e7d32" if (i, j) in correct else "#e53935"))
    if seq and len(seq) == n and step >= 9:  # sequence letters on the axis
        for i, ch in enumerate(seq.upper()):
            parts.append(f"<text x='{x(i):.1f}' y='{ax+13:.1f}' font-size='{min(step,11)}' "
                         f"text-anchor='middle' font-family='monospace' fill='#455a64'>{ch}</text>")
    parts.append("</svg>")

    if mode == "target_only":
        legend = ("<div class='legend'><span><i class='sw' style='background:#90a4ae'></i>target structure</span>"
                  f"<span class='muted'>&#9888; agent produced no sequence within budget &middot; "
                  f"{len(tp)} base pairs, {n} nt</span></div>")
    elif mode == "solved":
        legend = ("<div class='legend'><span><i class='sw' style='background:#2e7d32'></i>all pairs correct</span>"
                  "<span class='muted'>solved &middot; bp-distance 0 (target = achieved MFE)</span></div>")
    else:
        bpd = len(missing) + len(spurious)
        note = (f"<span class='muted'>bp-distance {bpd} &middot; target &#9650; / achieved &#9660;</span>"
                if not reason else
                f"<span style='color:#c62828'>&#9888; {esc(reason)}</span>"
                f"<span class='muted'> &middot; bp-distance {bpd}</span>")
        legend = ("<div class='legend'>"
                  "<span><i class='sw' style='background:#2e7d32'></i>correct</span>"
                  "<span><i class='sw' style='background:#f9a825'></i>missing (target only, dashed)</span>"
                  "<span><i class='sw' style='background:#e53935'></i>spurious (fold only)</span>"
                  + note + "</div>")
    return "".join(parts) + legend


# ------------------------------------------------------------------ saturation (inline SVG)
def _svg_panel(series: Dict[str, List[Tuple[float, float]]], title: str, ymin=None, ymax=None) -> str:
    pts_all = [y for s in series.values() for _, y in s if y is not None]
    if not pts_all:
        return f"<div class='muted'>{esc(title)}: no data</div>"
    xs_all = [x for s in series.values() for x, y in s if y is not None]
    x0, x1 = min(xs_all), max(xs_all) if xs_all else (0, 1)
    y0 = ymin if ymin is not None else min(pts_all)
    y1 = ymax if ymax is not None else max(pts_all)
    if y1 == y0:
        y1 = y0 + 1
    if x1 == x0:
        x1 = x0 + 1
    W, H, ml, mt = 420, 240, 46, 24
    pw, ph = W - ml - 14, H - mt - 30
    sx = lambda x: ml + (x - x0) / (x1 - x0) * pw
    sy = lambda y: mt + (1 - (y - y0) / (y1 - y0)) * ph
    p = [f"<svg viewBox='0 0 {W} {H}' width='{W}' style='max-width:100%'>"]
    p.append(f"<text x='{ml}' y='16' font-size='12' font-weight='700' fill='#1a1a2e'>{esc(title)}</text>")
    # axes + gridlines
    for k in range(5):
        gy = mt + ph * k / 4
        val = y1 - (y1 - y0) * k / 4
        p.append(f"<line x1='{ml}' y1='{gy:.1f}' x2='{W-14}' y2='{gy:.1f}' stroke='#eceff1'/>")
        p.append(f"<text x='{ml-6}' y='{gy+3:.1f}' font-size='9' text-anchor='end' fill='#90a4ae'>{val:.2f}</text>")
    p.append(f"<line x1='{ml}' y1='{mt}' x2='{ml}' y2='{mt+ph}' stroke='#b0bec5'/>")
    p.append(f"<line x1='{ml}' y1='{mt+ph}' x2='{W-14}' y2='{mt+ph}' stroke='#b0bec5'/>")
    for x, _ in (series[list(series)[0]] if series else []):
        p.append(f"<text x='{sx(x):.1f}' y='{mt+ph+14}' font-size='9' text-anchor='middle' fill='#90a4ae'>e{int(x)}</text>")
    for arm, s in series.items():
        col = _ARM_COLOR.get(arm, "#1976d2")
        pl = [(sx(x), sy(y)) for x, y in s if y is not None]
        if len(pl) >= 2:
            p.append(f"<polyline points='{' '.join(f'{a:.1f},{b:.1f}' for a,b in pl)}' fill='none' stroke='{col}' stroke-width='2'/>")
        for a, b in pl:
            p.append(f"<circle cx='{a:.1f}' cy='{b:.1f}' r='3' fill='{col}'/>")
    p.append("</svg>")
    return "".join(p)


def svg_saturation(curves: Dict[str, List[Dict]]) -> str:
    solve = {arm: [(pt["epoch"], pt.get("solve_rate")) for pt in pts] for arm, pts in curves.items()}
    oc = {arm: [(pt["epoch"], pt.get("mean_oracle_calls_solved")) for pt in pts] for arm, pts in curves.items()}
    legend = "<div class='legend' style='margin:8px 0'>" + "".join(
        f"<span><i class='sw' style='background:{_ARM_COLOR.get(a,'#1976d2')}'></i>{esc(a)}</span>" for a in curves) + "</div>"
    return (legend + "<div style='display:flex;gap:20px;flex-wrap:wrap'>"
            + _svg_panel(solve, "solve rate vs epoch", 0, 1)
            + _svg_panel(oc, "oracle-calls to solve vs epoch") + "</div>")


# ------------------------------------------------------------------ trace parsing
def _parse_trace(path: str) -> Dict:
    out: Dict = {"tool_calls": [], "target": "", "final_sequence": "", "memory": None, "result": None}
    try:
        lines = open(path).read().splitlines()
    except Exception:
        return out
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            ev = json.loads(ln)
        except Exception:
            continue
        t = ev.get("type")
        if t == "rna_memory_snapshot":
            out["memory"] = ev
            continue
        if t == "result":
            out["result"] = {k: ev.get(k) for k in ("num_turns", "total_cost_usd", "duration_ms", "result")}
        msg = ev.get("message") if isinstance(ev.get("message"), dict) else ev
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_use":
                    inp = b.get("input") or {}
                    if isinstance(inp, dict) and inp.get("target"):
                        out["target"] = inp["target"]
                    arg = ""
                    if isinstance(inp, dict):
                        arg = inp.get("sequence") or inp.get("target") or json.dumps(inp)[:60]
                    out["tool_calls"].append({"name": b.get("name", "?"), "arg": str(arg)[:80]})
                elif b.get("type") == "text":
                    m = re.search(r"FINAL_SEQUENCE:\s*([ACGUacgu]+)", b.get("text", ""))
                    if m:
                        out["final_sequence"] = m.group(1).upper().replace("T", "U")
    return out


def _trace_for(run_dir: str, task_id: str) -> Optional[str]:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", task_id)
    p = os.path.join(run_dir, "traces", f"{safe}.jsonl")
    return p if os.path.exists(p) else None


# ------------------------------------------------------------------ verdict
def _verdict(run_dir: str) -> Tuple[str, str, str]:
    """Return (css_class, headline, explanation)."""
    s = _run_summary(run_dir)
    curves = _curves(run_dir)
    if "with_memory" in curves and "no_memory" in curves and curves["with_memory"] and curves["no_memory"]:
        wm, nm = curves["with_memory"][-1], curves["no_memory"][-1]
        woc, noc = wm.get("mean_oracle_calls_solved"), nm.get("mean_oracle_calls_solved")
        if woc is not None and noc is not None:
            better = woc < noc
            css = "success" if better else "partial"
            expl = (f"Memory effect at final epoch: with_memory reaches solutions in "
                    f"{woc} oracle-calls vs no_memory {noc} "
                    f"({'memory HELPS (fewer calls)' if better else 'no clear gain yet'}); "
                    f"A_memory={wm.get('A_memory')}. Solve {s['solved']}/{s['n_records']}.")
            return css, f"Memory {'HELPS' if better else 'inconclusive'}", expl
    if s["solve_rate"] is not None:
        r = s["solve_rate"]
        css = "success" if r >= 0.6 else ("partial" if r >= 0.3 else "failure")
        if s.get("kind") == "agent_live":  # run still in progress -> live-scored, partial
            css = "neutral"
            return css, f"🔴 运行中 · 已评 {s['solved']}/{s['n_records']} ({r:.0%})", \
                (f"实时统计:已完成 {s['n_records']} 题,当前解出 {s['solved']} 题(solve-rate {r:.1%})。"
                 "数字随运行滚动更新,整轮跑完后写入最终 summary。")
        return css, f"Solved {s['solved']}/{s['n_records']} ({r:.0%})", \
            f"{s['n_episodes']} episodes recorded; solve-rate {r:.1%}."
    if s["has_scorecard"]:
        return "neutral", "Scorecard", "3-panel scorecard available below."
    return "neutral", "Run", "No episode records; showing available artifacts."


# ------------------------------------------------------------------ pages
def _kind(run_dir: str, files: List[str]) -> str:
    if "campaign_state.json" in files or "campaign_summary.json" in files:
        return "campaign"
    if "scorecard.json" in files:
        return "scorecard"
    if "saturation_summary.json" in files:
        return "saturation"
    if "summary.json" in files:
        return "agent/classical"
    return "run"


# plain-language run categories (ordered): key -> (icon+title, one-line question/explanation)
_CATS = [
    ("eval", "🧪 AI 模型能力评测", "问题:某个 AI 模型「独立设计 RNA」的能力有多强?(在 Eterna100 谜题上一次性尝试,不给记忆)"),
    ("meta", "📈 元学习实验:记忆是否让智能体越做越好", "问题:让智能体把解题经验存进「记忆」反复练,它是否用更少的尝试次数就能解题?"),
    ("score", "🏆 综合成绩单", "把所有能力汇总成一张三面板成绩单:Eterna100 基准排行 · 混合约束设计 · 智能体能力与记忆。"),
    ("found", "🔬 地基与确定性验证", "验证这个领域是否「确定性可复现」,以及「看诊断再动手 vs 乱试」是否真有用(对照实验)。"),
    ("debug", "🧫 冒烟 / 调试运行(可忽略)", "开发过程中的小规模测试,没有正式结论,列在这里仅为完整。"),
    ("other", "其他运行", ""),
]


def _run_model(run_dir: str) -> str:
    s = _read_json(os.path.join(run_dir, "summary.json")) or {}
    return s.get("model") or ""


def _categorize(name: str, files: List[str], kind: Optional[str], s: Dict) -> str:
    low = name.lower()
    if "smoke" in low:
        return "debug"
    if "scorecard.json" in files:
        return "score"
    if name == "zero_credit_main" or "meta_summary.json" in files:
        return "found"
    if kind in ("agent", "agent_live"):
        return "eval"
    if kind == "campaign":
        return "meta"
    if s["n_records"] == 0 and s.get("kind") is None:
        return "debug"  # incomplete / traces-only helper run
    return "other"


def _plain_desc(name: str, cat: str, s: Dict, headline: str, model: str, ntr: int) -> str:
    n, solved = s["n_records"], s["solved"]
    rate = f"{s['solve_rate']:.0%}" if s.get("solve_rate") is not None else "—"
    m = esc(model.split("/")[-1]) if model else "某模型"
    if cat == "eval":
        return (f"AI 模型 <b>{m}</b> 独立设计 RNA:在 <b>{n}</b> 道 Eterna100 谜题上解对 "
                f"<b>{solved}</b> 道(<b>{rate}</b>)。点 Report 看逐题结果 + 每道题的「目标形状 vs 实际折叠」对比图。")
    if cat == "meta":
        if "HELP" in headline.upper() or "MEMORY" in headline.upper():
            return ("记忆实验:让智能体把经验存进记忆反复练 —— 结果显示 <b>记忆有帮助</b>"
                    "(解题所需尝试次数随练习下降)。点 Report 看自我改进曲线。")
        return f"元学习 campaign:共 <b>{n}</b> 次尝试,解对 <b>{solved}</b> 次。点 Report 看各「记忆臂」对比曲线。"
    if cat == "score":
        return "综合成绩单:三面板 —— Eterna100 基准排行(基线算法解出率)· 混合约束设计 · 智能体能力与记忆增益。"
    if cat == "found":
        if name == "zero_credit_main":
            return ("地基验证:数据/奖励/轨迹三种「确定性」均 ≈ 1;并做「看诊断再动手 vs 随机乱试 vs 丢弃诊断」"
                    "的对照,证明<b>让诊断驱动下一步</b>确实提升解题率。")
        return "经典(非 AI)记忆实验对照:用本地搜索验证「记忆能降低搜索成本」这一机制本身。"
    if cat == "debug":
        return "小规模冒烟 / 调试测试(开发用,无正式结论)。"
    return f"运行 {esc(name)}(共 {n} 次尝试,解对 {solved})。"


def runs_index(runs_root: str) -> str:
    names = []
    buckets: Dict[str, List[str]] = {k: [] for k, _, _ in _CATS}
    for name in sorted(os.listdir(runs_root)):
        d = os.path.join(runs_root, name)
        if not os.path.isdir(d):
            continue
        names.append(name)
        files = os.listdir(d)
        s = _run_summary(d)
        css, headline, _ = _verdict(d)
        ntr = len(os.listdir(os.path.join(d, "traces"))) if os.path.isdir(os.path.join(d, "traces")) else 0
        cat = _categorize(name, files, s.get("kind"), s)
        model = _run_model(d)
        desc = _plain_desc(name, cat, s, headline, model, ntr)
        scale = []
        if s["n_records"]:
            scale.append(f"{s['n_records']} 次尝试")
        if ntr:
            scale.append(f"{ntr} 段录像")
        scale_txt = " · ".join(scale) or "—"
        actions = f"<a href='/report/{esc(name)}'>Report</a>"
        if ntr:
            actions += f" · <a href='/trace/{esc(name)}'>Trace</a>"
        if os.path.exists(os.path.join(d, "campaign_state.json")) or any(x.startswith("memdir") for x in files):
            actions += f" · <a href='/memory/{esc(name)}'>Memory</a>"
        row = (f"<tr><td><a href='/report/{esc(name)}'><b>{esc(name)}</b></a><br>"
               f"<span class='badge {css}'>{esc(headline)}</span></td>"
               f"<td>{desc}<div class='muted' style='margin-top:3px'>规模:{scale_txt}</div></td>"
               f"<td>{actions}</td></tr>")
        buckets[cat].append(row)

    # hero + glossary
    hero = ("<section><h2>这是什么?</h2><div class='card'>"
            "<p><b>openclaude_rna</b> 是一个让 <b>AI 智能体自动设计 RNA 序列</b>的实验平台。"
            "给它一个目标「形状」(RNA 二级结构,如发夹/茎环),它要设计出一条能折叠成该形状的碱基序列(A/C/G/U);"
            "由 <b>ViennaRNA 折叠算法</b>当裁判验证对不对。下面每一行是一次「实验运行」,按用途分成几类。</p>"
            "<p class='muted'><b>红线</b>:所有「成功/解出」都仅指「在折叠算法裁判下设计正确」,不代表任何体内或生物功能。</p>"
            "</div></section>")
    glossary = ("<details class='card'><summary>看不懂表格?点这里看术语说明</summary>"
                "<ul style='margin:8px 0 0;padding-left:20px;line-height:1.9'>"
                "<li><b>解出 / Solved 8/10</b>:设计的序列其「最低能量折叠」= 目标形状;10 道里解对 8 道。</li>"
                "<li><b>Memory HELPS</b>:记忆实验的结论 —— 智能体累积经验后更省力(尝试次数下降)。</li>"
                "<li><b>尝试(episode)</b>:对一道题的一次完整设计尝试。</li>"
                "<li><b>录像(trace)</b>:智能体每一步操作的完整记录,可在 Trace 页回放。</li>"
                "<li><b>Report / Trace / Memory</b>:详情报告 / 操作录像 / 它学到的记忆。</li>"
                "<li><b>Run / Scorecard</b>(灰标):辅助运行 / 综合成绩单。</li>"
                "</ul></details>")

    sections = [hero, glossary]
    for key, title, expl in _CATS:
        rows = buckets.get(key) or []
        if not rows:
            continue
        table = ("<div class='scroll'><table><thead><tr><th style='width:22%'>运行</th>"
                 "<th>说明与结果</th><th style='width:16%'>操作</th></tr></thead><tbody>"
                 + "".join(rows) + "</tbody></table></div>")
        block = f"<h2>{title}</h2>" + (f"<p class='muted'>{expl}</p>" if expl else "") + table
        if key == "debug":  # collapse the noise
            sections.append(f"<section><details><summary style='font-size:1.1rem;font-weight:700'>"
                            f"{title} <span class='muted' style='font-weight:400'>(点开)</span></summary>"
                            f"{('<p class=muted>'+expl+'</p>') if expl else ''}{table}</details></section>")
        else:
            sections.append(f"<section>{block}</section>")

    opts = "".join(f"<option>{esc(nm)}</option>" for nm in names)
    compare_form = (
        "<section><h2>&#9878; 对比两次运行</h2>"
        "<p class='muted'>选两个「AI 模型能力评测」运行(同一批题),看谁解得多、谁更省、逐题胜负。</p>"
        "<form action='/compare' method='get' class='card'>"
        f"A <select name='a'>{opts}</select> &nbsp; vs &nbsp; B <select name='b'>{opts}</select> &nbsp; "
        "<button type='submit'>对比 &rarr;</button></form></section>")

    body = "<main>" + "".join(sections) + compare_form + "</main>"
    return page("openclaude_rna 实验总览", body)


def _episode_rows(recs: List[Dict]) -> str:
    rows = []
    for r in sorted(recs, key=lambda x: (not x.get("solved"), x.get("n", 0))):
        cls = "solved" if r.get("solved") else "unsolved"
        badge = "success" if r.get("solved") else "failure"
        seq = r.get("final_sequence") or ""
        rows.append(
            f"<tr class='{cls}'><td>{esc(r.get('arm',''))}</td><td>{r.get('epoch','')}</td>"
            f"<td>{esc(str(r.get('task_id','')).replace('eterna100:',''))}</td><td>{r.get('n','')}</td>"
            f"<td><span class='badge {badge}'>{'solved' if r.get('solved') else 'unsolved'}</span></td>"
            f"<td>{_fmt(r.get('reward'))}</td><td>{_fmt(r.get('oracle_calls'))}</td>"
            f"<td>{_fmt(r.get('A_memory'))}</td>"
            f"<td class='seq'>{esc(seq[:48])}{'&hellip;' if len(seq)>48 else ''}</td></tr>")
    return ("<div class='scroll'><table><thead><tr><th>arm</th><th>epoch</th><th>task</th><th>n</th>"
            "<th>result</th><th>reward</th><th>oracle&nbsp;calls</th><th>A_memory</th>"
            "<th>final sequence</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def _fmt(x) -> str:
    if x is None:
        return "&mdash;"
    if isinstance(x, float):
        return f"{x:.3f}" if abs(x) < 100 else f"{x:.1f}"
    return esc(x)


def _agent_episode_rows(recs: List[Dict]) -> str:
    """Richer per-task table for agent runs (RNA calls / MCP calls / cross-oracle / A / elapsed)."""
    rows = []
    for r in sorted(recs, key=lambda x: (not x.get("solved"), x.get("n", 0))):
        cls = "solved" if r.get("solved") else "unsolved"
        badge = "success" if r.get("solved") else "failure"
        seq = r.get("final_sequence") or ""
        el = r.get("elapsed_s")
        rows.append(
            f"<tr class='{cls}'><td>{esc(str(r.get('task_id','')).replace('eterna100:',''))}</td>"
            f"<td>{r.get('n','')}</td>"
            f"<td><span class='badge {badge}'>{'solved' if r.get('solved') else 'unsolved'}</span></td>"
            f"<td>{_fmt(r.get('reward'))}</td><td>{_fmt(r.get('oracle_calls'))}</td>"
            f"<td>{_fmt(r.get('n_mcp_calls'))}</td><td>{_fmt(r.get('cross_oracle'))}</td>"
            f"<td>{_fmt(r.get('A'))}</td><td>{(str(int(el))+'s') if isinstance(el,(int,float)) else '&mdash;'}</td>"
            f"<td class='seq'>{esc(seq[:44])}{'&hellip;' if len(seq)>44 else ''}</td></tr>")
    return ("<div class='scroll'><table><thead><tr><th>task</th><th>n</th><th>result</th><th>reward</th>"
            "<th>RNA&nbsp;calls</th><th>MCP&nbsp;calls</th><th>cross&#8209;oracle</th><th>A</th>"
            "<th>elapsed</th><th>final sequence</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def _structure_gallery(run_dir: str, run: str, recs: List[Dict], num: int) -> str:
    solved_ep = [r for r in recs if r.get("solved") and r.get("final_sequence")]
    unsolved_seq = [r for r in recs if not r.get("solved") and r.get("final_sequence")]
    unsolved_noseq = [r for r in recs if not r.get("solved") and not r.get("final_sequence")]
    if not (solved_ep or unsolved_seq or unsolved_noseq):
        return ""
    try:
        emap = _e100_map()
    except Exception:
        emap = {}

    def card(r) -> str:
        seq = r.get("final_sequence") or ""
        tid = r.get("task_id", "")
        task = emap.get(re.sub(r"[^A-Za-z0-9_.-]", "_", tid))
        # authoritative target from the dataset (never fold-vs-self); trace-parse only if unknown.
        tp = _trace_for(run_dir, tid)
        target = (getattr(task, "target", "") if task else "") or (_parse_trace(tp)["target"] if tp else "") or ""
        tlen = getattr(task, "n", 0) or r.get("n") or len(target)
        mfe = _fold_mfe(seq) if seq else ""
        if not target and mfe:
            target = mfe  # last resort only when no known target exists at all
        if not target and not mfe:
            return ""  # nothing renderable (no target and no sequence)
        # honest reason WHY an unsolved case may still look clean (bp-distance 0)
        reason = ""
        if not r.get("solved") and seq:
            if len(seq) != tlen:
                reason = f"序列长度 {len(seq)} ≠ 目标 {tlen} nt(长度不符 → 判负)"
            elif mfe and not (_pairs(target) ^ _pairs(mfe)):
                reason = "配对拓扑全对,但未通过唯一最低能(uMFE)判定:目标不是唯一 MFE(简并基态)"
        tag = "success" if r.get("solved") else "failure"
        seq_html = (f"<div class='seq muted' style='margin:4px 0'>{esc(seq)}</div>" if seq
                    else "<div class='seq muted' style='margin:4px 0'>(未提交序列)</div>")
        return (f"<div class='card'><b>{esc(str(tid).replace('eterna100:',''))}</b> "
                f"({r.get('n')} nt) <span class='badge {tag}'>{'solved' if r.get('solved') else 'unsolved'}</span>"
                + seq_html + svg_arc(target, mfe, seq, solved=r.get("solved"), reason=reason) + "</div>")

    # first-principles: draw EVERY case, grouped by category (nothing hidden / no sampling)
    out = [f"<section><h2>{num} &middot; RNA 二级结构 &mdash; 目标 vs 实际折叠(<b>每题一张</b>)</h2>"]
    out.append(_arc_reading_guide())
    groups = [
        ("&#10060; 失败(有提交序列)&mdash; 最有诊断价值", unsolved_seq, True),
        ("&#9888;&#65039; 失败(未提交序列)&mdash; 只画目标结构", unsolved_noseq, True),
        ("&#9989; 解出(正对照)", solved_ep, False),
    ]
    for title, grp, open_default in groups:
        if not grp:
            continue
        cards = "".join(c for c in (card(r) for r in grp) if c)
        if not cards:
            continue
        op = " open" if open_default else ""
        out.append(f"<details{op}><summary style='font-size:1.02rem'>{title} &mdash; "
                   f"<b>{len(grp)}</b> 题(点击展开/收起)</summary><div style='margin-top:10px'>"
                   + cards + "</div></details>")
    out.append("</section>")
    return "".join(out)


def _arc_reading_guide() -> str:
    """Plain-language on-page explainer so a newcomer can read the arc diagrams in 30s."""
    return (
        "<div class='card' style='background:#f0f7ff;border-color:#bcd8f5'>"
        "<b>&#128161; 怎么看这些弧线图(30 秒上手)</b>"
        "<ul style='margin:8px 0 8px 18px;padding:0'>"
        "<li>横线是 RNA 序列(左&rarr;右 = 5'&rarr;3'),每个字母一个碱基。<b>一条弧 = 两个碱基配对</b>。</li>"
        "<li><b>一圈套一圈</b>(嵌套弧)= 一段螺旋(茎);<b>最里层弧下方、没有弧盖住的那几个碱基</b> = 环(loop,不配对)"
        " —— 环里常是一串 <b>A</b>,因为 A 最不容易乱配对,能稳稳把环撑开(这是正确且常见的设计手法)。</li>"
        "<li>弧在<b>线上方 &#9650; = 目标</b>结构(想要的);<b>线下方 &#9660; = 实际</b>折叠(提交的序列真正折成的)。"
        "解出时两者一致,只画一组绿弧。</li></ul>"
        "<b>颜色含义:</b> "
        "<span class='chip g'>绿 = 对(目标和实际都有这个配对)</span> "
        "<span class='chip y'>橙虚线 = 漏了(目标有、实际没折出)</span> "
        "<span class='chip r'>红 = 多了(实际有、目标没有的错误配对)</span>"
        "<div style='margin-top:8px'><b>solved(解出)的严格定义:</b> 提交序列折叠后<b>完全等于目标</b> —— "
        "① 长度一致 &nbsp;②&nbsp;最低能结构(MFE)== 目标 &nbsp;③&nbsp;该结构唯一(uMFE)。三者缺一即 <b>unsolved</b>。</div>"
        "<div class='muted' style='margin-top:5px'>&#9888; 注意:<b>bp-distance 0(配对全对)也可能判负</b> —— "
        "如序列长度差 1(off-by-one),或目标不是唯一最低能结构(简并基态)。图注会写明真实原因。</div>"
        "</div>")


def report(runs_root: str, run: str) -> str:
    d = os.path.join(runs_root, run)
    if not os.path.isdir(d):
        return page("not found", "<main><p>run not found</p></main>")
    css, headline, expl = _verdict(d)
    s = _run_summary(d)
    recs, _, kind = _load_records(d)
    mem_link = (" &middot; <a href='/memory/" + esc(run) + "'>Memory</a>") if (
        os.path.exists(os.path.join(d, "campaign_state.json"))
        or any(x.startswith("memdir") for x in os.listdir(d))) else ""
    body = [f"<header class='verdict {css}'><div class='big'>{esc(run)} &mdash; {esc(headline)}</div>"
            f"<small>{esc(expl)}</small></header>",
            f"<div class='topbar' style='top:0'><span></span>"
            f"<span class='lnk'><a href='/ui'>&#8592; runs</a> &middot; "
            f"<a href='/trace/{esc(run)}'>Trace</a>{mem_link}</span></div>",
            "<main>"]

    _intro = {
        "agent": "🧪 <b>AI 模型能力评测</b>:给这个 AI 模型一批 RNA 折叠谜题(每题给一个目标「形状」,"
                 "让它设计出能折叠成该形状的碱基序列 A/C/G/U),看它独立解对几道。绿=解出、红=未解出;"
                 "下方有每道题「目标形状 vs 实际折叠」的弧线对比图。",
        "campaign": "📈 <b>元学习实验</b>:让智能体反复练同一批谜题、把经验存进记忆,"
                    "看它是否「越做越省」(解题所需尝试次数随练习下降)。",
    }.get("agent" if kind == "agent_live" else kind, "")
    if _intro:
        body.append(f"<div class='card' style='margin:16px 0 4px'>{_intro}</div>")

    if kind in ("agent", "agent_live"):
        live = kind == "agent_live"
        if live:
            # live scoreboard: metrics folded on the fly from traces (no summary.json yet)
            lr = _live_agent_records(d)
            sm = lr[1] if lr else {}
            recs = lr[0] if lr else recs
            body.insert(len(body), "<script>setTimeout(function(){location.reload()},20000)</script>")
            body.append("<div class='card' style='margin:8px 0;border-left:4px solid #c62828'>"
                        "🔴 <b>运行中 (LIVE)</b> — 本页指标由后台<b>实时折叠裁判</b>对已完成的每条 trace "
                        "现算得到,<b>每 20 秒自动刷新</b>;整轮跑完后会落盘为最终 <code>summary.json</code>。"
                        "长题排在后面,solve-rate 会随长题进入而变化。</div>")
        else:
            # single-pass AGENT capability report (agent_runner summary.json)
            sm = _read_json(os.path.join(d, "summary.json")) or {}
        title = "1 &middot; Agent capability" + (" (实时)" if live else "")
        body.append(f"<section><h2>{title}</h2><div class='card'>"
                    f"<span class='kv'>model <b>{esc(sm.get('model') or run.split('_')[0])}</b></span>"
                    f"<span class='kv'>solve-rate <b>{(s['solve_rate'] or 0):.0%}</b> "
                    f"({s['solved']}/{s['n_records']}{' 已评' if live else ''})</span>"
                    f"<span class='kv'>mean reward <b>{_fmt(sm.get('mean_reward'))}</b></span>"
                    + ("" if live else
                       f"<span class='kv'>agent_A <b>{_fmt(sm.get('agent_A'))}</b></span>"
                       f"<span class='kv'>A_model <b>{_fmt(sm.get('agent_A_model'))}</b></span>")
                    + f"<span class='kv'>mean RNA calls <b>{_fmt(sm.get('mean_rna_calls'))}</b></span>"
                    + ("" if live else f"<span class='kv'>tasks w/ repair <b>{_fmt(sm.get('n_tasks_with_repair'))}</b></span>")
                    + "<div style='margin-top:8px'><b>按长度分档(能力边界):</b> "
                    + (_length_bands(recs) or "<span class='muted'>—</span>") + "</div>"
                    "</div></section>")
        rt = "2 &middot; Per-task results" + (" — 实时逐题(短→长)" if live else "")
        body.append(f"<section><h2>{rt}</h2>" + _agent_episode_rows(recs) + "</section>")
        body.append(_structure_gallery(d, run, recs, 3))
    else:
        # campaign report (meta-learning arms / epochs)
        body.append("<section><h2>1 &middot; Overview</h2><div class='card'>"
                    f"<span class='kv'>episodes <b>{s['n_episodes']}</b></span>"
                    f"<span class='kv'>records <b>{s['n_records']}</b></span>"
                    f"<span class='kv'>solved <b>{s['solved']}</b></span>")
        if s["solve_rate"] is not None:
            body.append(f"<span class='kv'>solve-rate <b>{s['solve_rate']:.1%}</b></span>")
        for arm, a in s["arms"].items():
            body.append(f"<span class='kv'>{esc(arm)}: {a['solved']}/{a['n']}</span>")
        body.append("</div></section>")
        curves = _curves(d)
        if curves:
            body.append("<section><h2>2 &middot; Self-improvement (saturation) curves</h2>"
                        "<div class='card'>" + svg_saturation(curves) + "</div></section>")
        if recs:
            body.append("<section><h2>3 &middot; Per-episode results</h2>" + _episode_rows(recs) + "</section>")
        body.append(_structure_gallery(d, run, recs, 4))

    # common trailing sections
    sc = _read_json(os.path.join(d, "scorecard.json"))
    if sc:
        body.append("<section><h2>Scorecard</h2><div class='card'><pre>"
                    + esc(json.dumps(sc.get("panels", sc), indent=2)[:6000]) + "</pre></div></section>")
    figdir = os.path.join(d, "figures")
    if os.path.isdir(figdir):
        imgs = "".join(f"<img src='/fig/{esc(run)}/{esc(f)}' style='max-width:460px;margin:6px;border:1px solid #e6e8ec;border-radius:6px'>"
                       for f in sorted(os.listdir(figdir)) if f.endswith(".png"))
        if imgs:
            body.append(f"<section><h2>Figures</h2><div class='card'>{imgs}</div></section>")
    body.append("</main>")
    return page(f"{run} report", "".join(body))


def trace(runs_root: str, run: str, task: Optional[str] = None) -> str:
    d = os.path.join(runs_root, run)
    tdir = os.path.join(d, "traces")
    if not os.path.isdir(tdir):
        return page(f"{run} trace", "<main><p class='muted'>no traces</p></main>")
    tracefiles = sorted(f for f in os.listdir(tdir) if f.endswith(".jsonl"))
    body = [f"<main><section><h2>Traces &mdash; {esc(run)}</h2>"]
    body.append("<div class='card'>" + " &middot; ".join(
        f"<a href='/trace/{esc(run)}?task={esc(f[:-6])}'>{esc(f[:-6])}</a>" for f in tracefiles) + "</div>")
    sel = None
    if task:
        cand = os.path.join(tdir, task + ".jsonl")
        sel = cand if os.path.exists(cand) else None
    if sel is None and tracefiles:
        sel = os.path.join(tdir, tracefiles[0])
    if sel:
        pt = _parse_trace(sel)
        body.append(f"<h3 class='mono'>{esc(os.path.basename(sel))}</h3>")
        if pt["result"]:
            body.append(f"<div class='card'><span class='kv'>turns {pt['result'].get('num_turns')}</span>"
                        f"<span class='kv'>cost ${_fmt(pt['result'].get('total_cost_usd'))}</span>"
                        f"<span class='kv'>final_seq <code>{esc(pt['final_sequence'])}</code></span></div>")
        rows = "".join(
            f"<tr><td>{i+1}</td><td><span class='badge neutral'>{esc(c['name'])}</span></td>"
            f"<td class='mono'>{esc(c['arg'])}</td></tr>" for i, c in enumerate(pt["tool_calls"]))
        body.append("<div class='scroll'><table><thead><tr><th>#</th><th>tool</th><th>arg</th></tr>"
                    f"</thead><tbody>{rows}</tbody></table></div>")
        # tool-call histogram
        from collections import Counter
        hist = Counter(c["name"] for c in pt["tool_calls"])
        body.append("<div class='card' style='margin-top:10px'>tool usage: " + " ".join(
            f"<span class='chip {'g' if k.startswith('Rna') else ('y' if k.startswith('mcp__') else '')}'>{esc(k)}&times;{v}</span>"
            for k, v in hist.most_common()) + "</div>")
    body.append("</section></main>")
    return page(f"{run} trace", "".join(body))


def memory(runs_root: str, run: str) -> str:
    d = os.path.join(runs_root, run)
    body = [f"<main><section><h2>Memory &amp; learned playbook &mdash; {esc(run)}</h2>"]
    # learned playbook (memdir MEMORY.md)
    found = False
    for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if name.startswith("memdir"):
            mp = os.path.join(d, name, "MEMORY.md")
            if os.path.exists(mp):
                found = True
                body.append(f"<div class='card'><b>{esc(name)}/MEMORY.md</b> "
                            f"<span class='muted'>(the playbook openclaude injects)</span>"
                            f"<pre>{esc(open(mp).read()[:6000])}</pre></div>")
    # per-episode injected memory snapshots (from traces)
    tdir = os.path.join(d, "traces")
    snaps = []
    if os.path.isdir(tdir):
        for f in sorted(os.listdir(tdir)):
            if not f.endswith(".jsonl"):
                continue
            pt = _parse_trace(os.path.join(tdir, f))
            m = pt.get("memory")
            if m:
                snaps.append((f[:-6], m))
    if snaps:
        body.append("<h3>Per-episode injected-memory snapshots</h3>"
                    "<p class='muted'>Exactly what the agent saw each episode (auditable from the trace).</p>")
        for tid, m in snaps:
            inj = m.get("injected")
            body.append(f"<details><summary>{esc(tid)} &mdash; injected={inj} "
                        f"chars={m.get('n_chars')} sha={esc(str(m.get('sha256',''))[:10])}</summary>"
                        f"<pre>{esc((m.get('content') or '(empty)')[:4000])}</pre></details>")
    if not found and not snaps:
        body.append("<p class='muted'>No memory directory or snapshots for this run.</p>")
    body.append("</section></main>")
    return page(f"{run} memory", "".join(body))


# ------------------------------------------------------------------ two-run comparison
def _cmp_metrics(run_dir: str) -> Dict:
    recs, _, _ = _load_records(run_dir)
    s = _read_json(os.path.join(run_dir, "summary.json")) or {}
    solved = sum(1 for r in recs if r.get("solved"))
    n = len(recs)
    return {
        "model": s.get("model") or "?",
        "n": n, "solved": solved,
        "solve_rate": s.get("solve_rate", (round(solved / n, 4) if n else None)),
        "mean_reward": s.get("mean_reward"),
        "agent_A": s.get("agent_A"),
        "agent_A_model": s.get("agent_A_model"),
        "mean_rna_calls": s.get("mean_rna_calls"),
        "by_task": {r.get("task_id"): r for r in recs},
    }


def _wl(va, vb, higher_better=True) -> Tuple[str, str]:
    if va is None or vb is None or va == vb:
        return "", ""
    a_wins = (va > vb) if higher_better else (va < vb)
    return ("win", "lose") if a_wins else ("lose", "win")


def _run_cells(r, rescls: str, border=False) -> str:
    bcls = "col-b " if border else ""
    if not r:
        return (f"<td class='{bcls}{rescls}'>&mdash;</td><td>&mdash;</td><td>&mdash;</td>"
                f"<td>&mdash;</td><td>&mdash;</td>")
    bd = "success" if r.get("solved") else "failure"
    lab = "solved" if r.get("solved") else "unsolved"
    return (f"<td class='{bcls}{rescls}'><span class='badge {bd}'>{lab}</span></td>"
            f"<td>{_fmt(r.get('reward'))}</td><td>{_fmt(r.get('oracle_calls'))}</td>"
            f"<td>{_fmt(r.get('n_mcp_calls'))}</td><td>{_fmt(r.get('A'))}</td>")


def compare(runs_root: str, a: str, b: str) -> str:
    da, db = os.path.join(runs_root, a), os.path.join(runs_root, b)
    if not (os.path.isdir(da) and os.path.isdir(db)):
        return page("compare", "<main><p class='muted'>one or both runs not found</p></main>")
    ma, mb = _cmp_metrics(da), _cmp_metrics(db)

    # overall verdict = who wins more of the headline metrics
    win_a = win_b = 0
    for key, hb in [("solve_rate", True), ("mean_reward", True), ("agent_A", True), ("mean_rna_calls", False)]:
        ca, cb = _wl(ma.get(key), mb.get(key), hb)
        win_a += ca == "win"
        win_b += cb == "win"
    if win_a > win_b:
        vcss, headline = "success", f"{esc(ma['model'])} leads ({win_a}–{win_b} headline metrics)"
    elif win_b > win_a:
        vcss, headline = "success", f"{esc(mb['model'])} leads ({win_b}–{win_a} headline metrics)"
    else:
        vcss, headline = "neutral", "even split"

    body = [f"<header class='verdict {vcss}'><div class='big'>Model comparison &mdash; {headline}</div>"
            f"<small>A = {esc(ma['model'])} &nbsp;vs&nbsp; B = {esc(mb['model'])} &middot; "
            f"same task set (n={max(ma['n'], mb['n'])}) &middot; identical config</small></header>",
            f"<div class='topbar' style='top:0'><span></span><span class='lnk'>"
            f"<a href='/ui'>&#8592; runs</a> &middot; <a href='/report/{esc(a)}'>report A</a> &middot; "
            f"<a href='/report/{esc(b)}'>report B</a></span></div>", "<main>"]

    # 1 summary metrics
    body.append("<section><h2>1 &middot; Summary metrics</h2><div class='scroll'><table>"
                f"<thead><tr><th>metric</th><th>A &middot; {esc(ma['model'])}</th>"
                f"<th class='col-b'>B &middot; {esc(mb['model'])}</th><th>winner</th></tr></thead><tbody>")
    for key, hb, dirn in [("solve_rate", True, "higher"), ("solved", True, "higher"),
                          ("mean_reward", True, "higher"), ("agent_A", True, "higher"),
                          ("agent_A_model", True, "higher"), ("mean_rna_calls", False, "lower")]:
        va, vb = ma.get(key), mb.get(key)
        ca, cb = _wl(va, vb, hb)
        winner = ma["model"] if ca == "win" else (mb["model"] if cb == "win" else "—")
        body.append(f"<tr><td>{key} <span class='muted'>({dirn} better)</span></td>"
                    f"<td class='{ca}'>{_fmt(va)}</td><td class='col-b {cb}'>{_fmt(vb)}</td>"
                    f"<td>{esc(winner)}</td></tr>")
    body.append("</tbody></table></div></section>")

    # 2 per-task
    tids = sorted(set(ma["by_task"]) | set(mb["by_task"]),
                  key=lambda t: (ma["by_task"].get(t) or mb["by_task"].get(t) or {}).get("n", 0))
    a_only = b_only = both = neither = 0
    trows = []
    for t in tids:
        ra, rb = ma["by_task"].get(t), mb["by_task"].get(t)
        sa = bool(ra and ra.get("solved"))
        sb = bool(rb and rb.get("solved"))
        wa = wb = ""
        if sa and not sb:
            winner, wa, wb, a_only = esc(ma["model"]), "win", "lose", a_only + 1
        elif sb and not sa:
            winner, wa, wb, b_only = esc(mb["model"]), "lose", "win", b_only + 1
        elif sa and sb:
            both += 1
            ca_, cb_ = (ra or {}).get("oracle_calls"), (rb or {}).get("oracle_calls")
            edge = ""
            if ca_ is not None and cb_ is not None and ca_ != cb_:
                edge = f" (edge {esc(ma['model'] if ca_ < cb_ else mb['model'])}, fewer calls)"
            winner = "both" + edge
        else:
            winner, neither = "both failed", neither + 1
        n = (ra or rb or {}).get("n", "")
        trows.append(f"<tr><td>{esc(str(t).replace('eterna100:',''))}</td><td>{n}</td>"
                     + _run_cells(ra, wa) + _run_cells(rb, wb, border=True)
                     + f"<td>{winner}</td></tr>")
    body.append("<section><h2>2 &middot; Per-task head-to-head</h2>"
                f"<p class='muted'>A solved-only: <b>{a_only}</b> &middot; B solved-only: <b>{b_only}</b> "
                f"&middot; both solved: <b>{both}</b> &middot; neither: <b>{neither}</b></p>"
                "<div class='scroll'><table><thead><tr><th>task</th><th>n</th>"
                "<th>A&middot;result</th><th>A&middot;rew</th><th>A&middot;calls</th><th>A&middot;mcp</th><th>A&middot;A</th>"
                "<th class='col-b'>B&middot;result</th><th>B&middot;rew</th><th>B&middot;calls</th><th>B&middot;mcp</th><th>B&middot;A</th>"
                "<th>winner</th></tr></thead><tbody>" + "".join(trows) + "</tbody></table></div></section>")
    body.append("<p class='muted'>Red line: design success under the ViennaRNA folding oracle only.</p></main>")
    return page(f"compare {a} vs {b}", "".join(body))
