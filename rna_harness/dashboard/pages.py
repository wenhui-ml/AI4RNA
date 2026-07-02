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
    """Return (records, n_episodes, kind) where kind in {'campaign','agent',None}."""
    recs, n = _state_records(run_dir)
    if recs:
        return recs, n, "campaign"
    ar = _agent_records(run_dir)
    if ar:
        return ar[0], len(ar[0]), "agent"
    return [], 0, None


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


def _fold_mfe(seq: str) -> str:
    try:
        import RNA
        md = RNA.md(); md.temperature = 37.0; md.dangles = 2
        st, _ = RNA.fold_compound(seq.upper().replace("T", "U"), md).mfe()
        return st
    except Exception:
        return ""


def svg_arc(target: str, mfe: str, seq: str = "") -> str:
    """Arc diagram: TARGET pairs as arcs ABOVE the axis (grey/green), ACHIEVED MFE pairs BELOW
    (green if correct, red if spurious). Missing target pairs are amber-dashed above. No JS/CDN."""
    n = max(len(target), len(mfe))
    if n == 0:
        return "<p class='muted'>no structure</p>"
    tp, mp = _pairs(target), _pairs(mfe)
    correct, missing, spurious = tp & mp, tp - mp, mp - tp
    step = max(6, min(16, int(1100 / n)))
    W = n * step + 40
    maxr = max((j - i) for i, j in (tp | mp)) * step / 2 if (tp | mp) else 20
    ah = min(180, maxr + 14)
    H = ah * 2 + 46
    ax = ah + 20  # baseline y
    x = lambda i: 20 + i * step + step / 2

    def arc(i, j, up, color, dashed=False):
        r = (x(j) - x(i)) / 2
        sweep = 1 if up else 0
        dash = " stroke-dasharray='4 3'" if dashed else ""
        return (f"<path d='M {x(i):.1f} {ax} A {r:.1f} {r:.1f} 0 0 {sweep} {x(j):.1f} {ax}' "
                f"fill='none' stroke='{color}' stroke-width='1.6'{dash}/>")

    parts = [f"<svg viewBox='0 0 {W} {H}' width='{min(W,1180)}' style='max-width:100%;background:#fff;border:1px solid #e6e8ec;border-radius:6px'>"]
    parts.append(f"<line x1='20' y1='{ax}' x2='{W-20}' y2='{ax}' stroke='#cfd8dc'/>")
    for i, j in sorted(tp):  # target above
        parts.append(arc(i, j, True, "#2e7d32" if (i, j) in correct else "#f9a825",
                         dashed=(i, j) in missing))
    for i, j in sorted(mp):  # achieved below
        parts.append(arc(i, j, False, "#2e7d32" if (i, j) in correct else "#e53935"))
    if seq and len(seq) == n and step >= 9:  # sequence letters on the axis
        for i, ch in enumerate(seq.upper()):
            parts.append(f"<text x='{x(i):.1f}' y='{ax+13}' font-size='{min(step,11)}' "
                         f"text-anchor='middle' font-family='monospace' fill='#455a64'>{ch}</text>")
    parts.append("</svg>")
    legend = ("<div class='legend'>"
              "<span><i class='sw' style='background:#2e7d32'></i>correct</span>"
              "<span><i class='sw' style='background:#f9a825'></i>missing (target only)</span>"
              "<span><i class='sw' style='background:#e53935'></i>spurious (fold only)</span>"
              f"<span class='muted'>bp-distance {len(missing)+len(spurious)} &middot; target &#9650; / achieved &#9660;</span></div>")
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


def runs_index(runs_root: str) -> str:
    rows = []
    for name in sorted(os.listdir(runs_root)):
        d = os.path.join(runs_root, name)
        if not os.path.isdir(d):
            continue
        files = os.listdir(d)
        s = _run_summary(d)
        css, headline, _ = _verdict(d)
        ntr = len(os.listdir(os.path.join(d, "traces"))) if os.path.isdir(os.path.join(d, "traces")) else 0
        rows.append(
            f"<tr><td><a href='/report/{esc(name)}'><b>{esc(name)}</b></a></td>"
            f"<td>{esc(_kind(d, files))}</td>"
            f"<td><span class='badge {css}'>{esc(headline)}</span></td>"
            f"<td>{s['n_records'] or '&mdash;'}{(' / '+str(s['solved'])+' solved') if s['n_records'] else ''}</td>"
            f"<td>{ntr or '&mdash;'}</td>"
            f"<td><a href='/report/{esc(name)}'>Report</a>"
            f"{' &middot; <a href=/trace/'+esc(name)+'>Trace</a>' if ntr else ''}"
            f"{' &middot; <a href=/memory/'+esc(name)+'>Memory</a>' if os.path.exists(os.path.join(d,'campaign_state.json')) or any(x.startswith('memdir') for x in files) else ''}"
            f"</td></tr>")
    body = (f"<main><section><h2>Runs</h2><div class='scroll'><table>"
            f"<thead><tr><th>Run</th><th>Type</th><th>Verdict</th><th>Episodes</th>"
            f"<th>Traces</th><th>Actions</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
            f"<p class='muted' style='margin-top:10px'>Red line: design success under the ViennaRNA "
            f"folding oracle only &mdash; never in-vivo function.</p></section></main>")
    return page("openclaude_rna runs", body)


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
    unsolved_ep = [r for r in recs if not r.get("solved") and r.get("final_sequence")]
    gallery = solved_ep[:5] + unsolved_ep[:3]
    if not gallery:
        return ""
    out = [f"<section><h2>{num} &middot; RNA secondary structure &mdash; target vs achieved MFE</h2>"]
    for r in gallery:
        seq = r["final_sequence"]
        tp = _trace_for(run_dir, r.get("task_id", ""))
        target = (_parse_trace(tp)["target"] if tp else "") or ""
        mfe = _fold_mfe(seq)
        if not target and mfe:
            target = mfe
        tag = "success" if r.get("solved") else "failure"
        out.append(f"<div class='card'><b>{esc(str(r.get('task_id','')).replace('eterna100:',''))}</b> "
                   f"({r.get('n')} nt) <span class='badge {tag}'>{'solved' if r.get('solved') else 'unsolved'}</span>"
                   f"<div class='seq muted' style='margin:4px 0'>{esc(seq)}</div>"
                   + svg_arc(target, mfe, seq) + "</div>")
    out.append("</section>")
    return "".join(out)


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

    if kind == "agent":
        # single-pass AGENT capability report (agent_runner summary.json)
        sm = _read_json(os.path.join(d, "summary.json")) or {}
        body.append("<section><h2>1 &middot; Agent capability</h2><div class='card'>"
                    f"<span class='kv'>model <b>{esc(sm.get('model','?'))}</b></span>"
                    f"<span class='kv'>solve-rate <b>{(s['solve_rate'] or 0):.0%}</b> "
                    f"({s['solved']}/{s['n_records']})</span>"
                    f"<span class='kv'>mean reward <b>{_fmt(sm.get('mean_reward'))}</b></span>"
                    f"<span class='kv'>agent_A <b>{_fmt(sm.get('agent_A'))}</b></span>"
                    f"<span class='kv'>A_model <b>{_fmt(sm.get('agent_A_model'))}</b></span>"
                    f"<span class='kv'>mean RNA calls <b>{_fmt(sm.get('mean_rna_calls'))}</b></span>"
                    f"<span class='kv'>tasks w/ repair <b>{_fmt(sm.get('n_tasks_with_repair'))}</b></span>"
                    "</div></section>")
        body.append("<section><h2>2 &middot; Per-task results</h2>" + _agent_episode_rows(recs) + "</section>")
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
