"""FastAPI dashboard for the openclaude_rna meta-learning harness.

Serves run summaries, figures, the four-coordinate / actuation / meta results, agent traces,
and a live log tail. Read-only. Launch:

    conda run -n rna python -m rna_harness.dashboard.app --host 0.0.0.0 --port 8811

Then open http://<server-ip>:8811 . NOTE: binding to 0.0.0.0 with no auth exposes it to
anyone who can reach the port (explicitly requested).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("RNA_HARNESS_DIR", os.path.abspath(os.path.join(_HERE, "..", "..")))
RUNS = os.path.join(REPO, "rna_harness", "runs")
RESULTS = os.path.join(REPO, "rna_harness", "results")
# extra dirs whose *.log files may be tailed (colon-separated)
LOG_DIRS = [d for d in os.environ.get("RNA_DASHBOARD_LOG_DIRS", "").split(":") if d] + [RUNS, RESULTS]

app = FastAPI(title="openclaude_rna dashboard")


def _safe_under(path: str, roots: List[str]) -> bool:
    rp = os.path.realpath(path)
    return any(rp == os.path.realpath(r) or rp.startswith(os.path.realpath(r) + os.sep) for r in roots)


def _valid_run(run: str) -> bool:
    """A run is valid iff it is a direct entry of RUNS (prevents path traversal while still
    allowing symlinked runs, which realpath-based _safe_under would wrongly reject)."""
    if not run or "/" in run or "\\" in run or ".." in run:
        return False
    try:
        return run in set(os.listdir(RUNS)) and os.path.isdir(os.path.join(RUNS, run))
    except OSError:
        return False


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# Off-mission runs hidden from the web display (archived per the Paper C mainline: the brute-force
# decompose / number-chasing probes are 本末倒置, orthogonal to the de-scaffolding + mode-efficacy
# thesis). Files stay on disk and remain reachable directly via /report/<run>; only the listing hides.
# Goal (2026-07-13): the web shows ONLY accurate, key experiments. Hidden (files kept on disk,
# still reachable at /report/<run>; only the listing/figures hide):
#   *decomp*   brute-force decompose number-chasing probes (glm52_decomp_hardcore31[_refix])
#   *_smoke    smoke tests (glm52_decomp_smoke shows a misleading 1.000@3-task)
#   *tail35*   P1 spectrum-law hard-35 subset replicates (0.09-0.23) -- real science but small-n /
#              low-absolute, easily misread out of context; raw data stays on disk for the P1 work
#   self_revise*  first capability-positive result but no summary.json -> renders as a bare card
#   p1_logs / funsearch_rna  raw log dir / abandoned empty FunSearch probe (no result)
#   glm52_eterna_memON  single memory-ON run; 0.65 next to the memory-null mainline reads as "+7 memory"
# (User-confirmed 2026-07-13: archive the borderline tail35 / self_revise / memON groups; keep the
# clean model baselines + meta-ladder memory-null + p1_floor_full100 + the orchestration-81 card.)
_ARCHIVED_SUBSTR = ("decomp", "_smoke", "tail35", "self_revise")
_ARCHIVED_EXACT = {"p1_logs", "funsearch_rna", "glm52_eterna_memON"}


def _archived_from_display(name: str) -> bool:
    return name in _ARCHIVED_EXACT or any(s in name for s in _ARCHIVED_SUBSTR)


def _list_runs() -> List[Dict]:
    out = []
    if not os.path.isdir(RUNS):
        return out
    for name in sorted(os.listdir(RUNS)):
        if _archived_from_display(name):
            continue
        d = os.path.join(RUNS, name)
        if not os.path.isdir(d):
            continue
        entry: Dict = {"name": name, "files": sorted(os.listdir(d))}
        entry["summary"] = _read_json(os.path.join(d, "summary.json"))
        entry["meta_summary"] = _read_json(os.path.join(d, "meta_summary.json"))
        figdir = os.path.join(d, "figures")
        entry["figures"] = sorted(os.listdir(figdir)) if os.path.isdir(figdir) else []
        tdir = os.path.join(d, "traces")
        entry["n_traces"] = len(os.listdir(tdir)) if os.path.isdir(tdir) else 0
        # live flag: a progress file touched in the last 90s => the run is actively producing output
        import time as _t
        _mt = 0.0
        for _f in ("epoch_progress.json", "scorecard.json", "summary.json", "rows_partial.json",
                   "campaign_state.json", "solved.json"):
            _p = os.path.join(d, _f)
            if os.path.exists(_p):
                _mt = max(_mt, os.path.getmtime(_p))
        # live if a progress file was touched recently. Threshold > episode TIMEOUT (1500s): with a
        # few workers all mid-long-episode, progress writes can be ~1 episode apart, so 90s falsely
        # read genuinely-running arms as dead. 1900s covers the worst-case gap.
        entry["live"] = bool(_mt and (_t.time() - _mt) < 1900)
        # count RLVR rows if present
        for f in ("rft.jsonl", "grpo.jsonl", "trajectories.jsonl", "agentic_rlvr.jsonl"):
            p = os.path.join(d, f)
            if os.path.exists(p):
                entry.setdefault("datasets", {})[f] = sum(1 for _ in open(p))
        out.append(entry)
    return out


@app.get("/health")
def health():
    return {"ok": True, "repo": REPO, "runs_dir": RUNS, "n_runs": len(_list_runs())}


@app.get("/api/runs")
def api_runs():
    return {"runs": _list_runs()}


@app.get("/api/scorecard")
def api_scorecard():
    cards = []
    if os.path.isdir(RUNS):
        for name in sorted(os.listdir(RUNS)):
            p = os.path.join(RUNS, name, "scorecard.json")
            d = _read_json(p) if os.path.exists(p) else None
            if d:
                cards.append({"name": name, "scorecard": d})
    return {"scorecards": cards}


@app.get("/api/campaigns")
def api_campaigns():
    cards = []
    if os.path.isdir(RUNS):
        for name in sorted(os.listdir(RUNS)):
            d = _read_json(os.path.join(RUNS, name, "campaign_summary.json"))
            if d:
                cards.append({"name": name, "summary": d})
    return {"campaigns": cards}


@app.get("/api/orchestration")
def api_orchestration():
    """Plan-A orchestration result: base model + oracle-verified calls to the external SOTA tool
    (DesiRNA). Reported SEPARATELY from single-model solve rates -- 81 is base 58 + tool 23, it
    measures the tool/pipeline, NOT model learning. Returns None if the sweep hasn't run."""
    d = _read_json(os.path.join(RUNS, "desirna_residual", "desirna_residual.json"))
    if not d:
        return {"orchestration": None}
    rows = d.get("rows", [])
    unsolved = [r for r in rows if not r.get("solved")]
    A = [r for r in unsolved if (r.get("len") or 0) <= 214 and (r.get("bp") or 99) <= 2]
    D = [r for r in unsolved if r.get("pid") == 100]
    B = [r for r in unsolved if r not in A and r not in D]
    def pids(rs): return sorted(x["pid"] for x in rs)
    return {"orchestration": {
        "model_solved": d.get("model_solved"),
        "tool_solved_of_unsolved": d.get("n_desirna_solved_of_unsolved"),
        "n_unsolved_before_tool": len(rows),
        "orchestrated_total": d.get("orchestrated_total"),
        "config": d.get("config"),
        "residual": {"n": len(unsolved),
                     "A_small_near_miss": pids(A), "B_large_compute": pids(B),
                     "D_undesignable": pids(D)}}}


@app.get("/fig/{run}/{fname}")
def figure(run: str, fname: str):
    if os.path.basename(fname) != fname or ".." in fname:
        raise HTTPException(403, "bad filename")
    bases = ([os.path.join(RUNS, run, "figures")] if _valid_run(run) else []) + [RESULTS]
    for base in bases:
        p = os.path.join(base, fname)
        if os.path.exists(p):  # run is a validated RUNS entry (symlinks allowed) or RESULTS
            return FileResponse(p)
    raise HTTPException(404, "figure not found")


@app.get("/api/loglist")
def loglist():
    files = []
    for root in LOG_DIRS:
        if not os.path.isdir(root):
            continue
        for dirpath, _, fnames in os.walk(root):
            for fn in fnames:
                if fn.endswith((".log", ".jsonl")) or fn == "summary.json":
                    p = os.path.join(dirpath, fn)
                    try:
                        files.append({"path": p, "size": os.path.getsize(p), "mtime": os.path.getmtime(p)})
                    except OSError:
                        pass
    files.sort(key=lambda x: -x["mtime"])
    return {"files": files[:200]}


@app.get("/api/logtail")
def logtail(path: str = Query(...), n: int = Query(120, le=2000)):
    if not _safe_under(path, LOG_DIRS):
        raise HTTPException(403, "path not allowed")
    if not os.path.exists(path):
        raise HTTPException(404, "not found")
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = min(size, n * 400)
            f.seek(size - block)
            data = f.read().decode("utf-8", "replace")
        lines = data.splitlines()[-n:]
        return {"path": path, "size": size, "lines": lines}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e))


@app.get("/ui", response_class=HTMLResponse)
def ui_runs():
    from . import pages
    return pages.runs_index(RUNS)


@app.get("/report/{run}", response_class=HTMLResponse)
def ui_report(run: str):
    from . import pages
    if not _valid_run(run):
        raise HTTPException(404, "run not found")
    return pages.report(RUNS, run)


@app.get("/trace/{run}", response_class=HTMLResponse)
def ui_trace(run: str, task: Optional[str] = Query(None)):
    from . import pages
    if not _valid_run(run):
        raise HTTPException(404, "run not found")
    return pages.trace(RUNS, run, task)


@app.get("/memory/{run}", response_class=HTMLResponse)
def ui_memory(run: str):
    from . import pages
    if not _valid_run(run):
        raise HTTPException(404, "run not found")
    return pages.memory(RUNS, run)


@app.get("/compare", response_class=HTMLResponse)
def ui_compare(a: str = Query(...), b: str = Query(...)):
    from . import pages
    if not (_valid_run(a) and _valid_run(b)):
        raise HTTPException(404, "run not found")
    return pages.compare(RUNS, a, b)


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


HTML_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>openclaude_rna 实验总览</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 :root{--bg:#fafafa;--card:#fff;--border:#e6e8ec;--text:#1a1a2e;--muted:#6b7280;--accent:#1976d2;--g:#2e7d32;--r:#c62828;--y:#b07d00}
 body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;margin:0;background:var(--bg);color:var(--text)}
 header{background:var(--card);padding:14px 22px;border-bottom:1px solid var(--border);position:sticky;top:0;z-index:5;display:flex;align-items:center;justify-content:space-between}
 h1{font-size:18px;margin:0} .sub{color:var(--muted);font-size:12px;margin-left:8px}
 .lnk{background:var(--accent);color:#fff!important;padding:7px 14px;border-radius:6px;font-weight:700;font-size:13px}
 .intro{background:var(--card);border-bottom:1px solid var(--border);padding:12px 22px;font-size:13.5px;line-height:1.7}
 .tabs{margin:12px 22px 0} .tab{display:inline-block;padding:7px 15px;cursor:pointer;border:1px solid var(--border);border-bottom:none;border-radius:8px 8px 0 0;background:#eef1f5;color:var(--muted);margin-right:3px;font-size:14px}
 .tab.active{background:var(--card);color:var(--accent);font-weight:700}
 .panel{margin:0 22px 30px;border:1px solid var(--border);border-radius:0 10px 10px 10px;padding:18px;background:var(--card)}
 .card{border:1px solid var(--border);border-radius:8px;padding:14px 16px;margin:12px 0;background:var(--card);box-shadow:0 1px 3px rgba(0,0,0,.05)}
 .name{font-weight:700;color:var(--accent);font-size:15px} h3{margin:4px 0} h4{margin:16px 0 4px;font-size:1rem}
 .note{color:#374151;font-size:12.5px;line-height:1.75;margin:6px 0 8px;background:#eef4fb;border-left:3px solid var(--accent);padding:9px 12px;border-radius:0 6px 6px 0}
 table{border-collapse:collapse;margin:6px 0;font-size:13px;background:var(--card)}
 td,th{border:1px solid var(--border);padding:6px 12px;text-align:left} th{background:#f0f2f5;font-weight:600;color:#374151}
 .g{color:var(--g);font-weight:700}.r{color:var(--r);font-weight:700}.y{color:var(--y);font-weight:600}
 img{max-width:520px;border:1px solid var(--border);border-radius:6px;margin:8px 8px 0 0;background:#fff}
 pre{background:#f6f8fa;border:1px solid var(--border);border-radius:6px;padding:10px;overflow:auto;max-height:60vh;font-size:12px;line-height:1.4}
 select,button{background:#fff;color:var(--text);border:1px solid var(--border);border-radius:6px;padding:6px 10px} button{cursor:pointer}
 a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}
 .kv{display:inline-block;margin-right:16px} details{margin:6px 0} summary{cursor:pointer;font-weight:600;color:var(--accent)}
 ul.gloss{margin:8px 0 0;padding-left:20px;line-height:1.9;font-size:13px}
</style></head><body>
<header><div><h1>🧬 openclaude_rna</h1><span class="sub">让 AI 智能体自动设计 RNA 序列 · 实时看板</span></div>
 <a href="/ui" class="lnk">📋 运行总览 / 详细报告 →</a></header>
<div class="intro">
 <b>这是什么:</b>给 AI 一个目标「形状」(RNA 二级结构),它要设计出能折叠成该形状的碱基序列(A/C/G/U);由 <b>ViennaRNA 折叠算法</b>当裁判验证。下面几个标签是不同角度的汇总结果。
 <b>红线:</b>所有「成功/解出」仅指「在折叠算法裁判下设计正确」,不代表任何体内或生物功能。
 <details><summary>看不懂表里的词?点这里(术语表)</summary>
 <ul class="gloss">
  <li><b>solved / 解出</b>:设计的序列,其「最低能量折叠」= 目标形状。</li>
  <li><b>uMFE</b>:更严 —— 目标是「唯一」最低能量解(无并列),更稳健。</li>
  <li><b>NED(系综缺陷)</b>:折叠「不干脆」的程度(0~1,<b>越低越好</b>)。</li>
  <li><b>A / A_model / A_harness(驱动保真度)</b>:「正确诊断有没有真的驱动下一步编辑」(0~1)。A_model=模型照做的程度;A_harness=框架有没有如实传达诊断(丢掉=0)。</li>
  <li><b>A_memory</b>:解答有没有采纳「记忆」里的策略(0~1)。</li>
  <li><b>mean evals→solve / oracle-calls→solve</b>:平均要试几次(调用几次裁判)才解出,<b>越低越省</b>。</li>
  <li><b>epoch</b>:把整批题练一遍;<b>episode/尝试</b>:对一道题的一次设计尝试。</li>
  <li>颜色:<span class="g">绿=好</span> / <span class="y">黄=中</span> / <span class="r">红=差</span>。</li>
 </ul></details>
</div>
<div class="tabs">
 <span class="tab active" data-t="scorecard" onclick="tab('scorecard')">🏆 成绩单</span>
 <span class="tab" data-t="campaigns" onclick="tab('campaigns')">📈 记忆实验</span>
 <span class="tab" data-t="overview" onclick="tab('overview')">🧭 概览</span>
 <span class="tab" data-t="runs" onclick="tab('runs')">📂 运行</span>
 <span class="tab" data-t="figures" onclick="tab('figures')">📊 图表</span>
 <span class="tab" data-t="logs" onclick="tab('logs')">📜 日志(实时)</span>
</div>
<div id="scorecard" class="panel"></div>
<div id="campaigns" class="panel" style="display:none"></div>
<div id="overview" class="panel" style="display:none"></div>
<div id="runs" class="panel" style="display:none"></div>
<div id="figures" class="panel" style="display:none"></div>
<div id="logs" class="panel" style="display:none">
 <p class="note">智能体运行时的实时日志尾部。选一个文件查看;可勾选自动刷新。</p>
 <select id="logsel"></select> <button onclick="loadlog()">刷新</button>
 <label class="sub"><input type="checkbox" id="auto" checked> 自动(3秒)</label>
 <pre id="logbox">选择一个日志…</pre>
</div>
<script>
let RUNS=[]; let SCORECARD=[]; let ORCH=null;
const SOTA_V2=[{m:'DesiRNA 24h/题(经典极限)',v:0.97,c:'#2e7d32'},{m:'DesiRNA <1min(同预算·最可比)',v:0.85,c:'#66bb6a'},{m:'NEMO 原生',v:0.94,c:'#78909c'},{m:'LLM方法SOTA(Gautam)',v:0.80,c:'#d97706'}];
function tab(t){document.querySelectorAll('.tab').forEach(e=>e.classList.toggle('active',e.dataset.t==t));
 ['scorecard','campaigns','overview','runs','figures','logs'].forEach(p=>document.getElementById(p).style.display=p==t?'block':'none');}
function fmt(x){return x==null?'—':(typeof x=='number'?(Math.round(x*1000)/1000):x)}
function cls(v){return v==null?'':(v>=0.6?'g':(v>=0.3?'y':'r'))}
function note(t){return '<p class="note">'+t+'</p>'}
async function load(){
 const h=await (await fetch('/health')).json();
 document.getElementById('health')&&(document.getElementById('health').textContent='');
 RUNS=(await (await fetch('/api/runs')).json()).runs;
 renderOverview(); renderRuns();
 try{const sc=(await (await fetch('/api/scorecard')).json()).scorecards; SCORECARD=sc; renderScorecard(sc);}catch(e){}
 try{const cp=(await (await fetch('/api/campaigns')).json()).campaigns; renderCampaigns(cp);}catch(e){}
 try{ORCH=(await (await fetch('/api/orchestration')).json()).orchestration;}catch(e){}
 renderFigures();
}
function svgBars(title,items,opt){
 opt=opt||{}; items=items.filter(i=>i&&i.value!=null);
 if(!items.length) return '';
 const mx=Math.max.apply(null,items.map(i=>i.value));
 const ymax=opt.ymax!=null?opt.ymax:(mx<=1?Math.min(1,mx*1.35+0.05):mx*1.15);
 const W=520,H=300,ml=52,mr=16,mt=42,mb=64,pw=W-ml-mr,ph=H-mt-mb;
 const gap=pw/items.length, bw=Math.min(96,gap*0.6);
 const sy=v=>mt+ph*(1-(v/ymax));
 let s=`<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:540px;height:auto;background:#fff;border:1px solid #e6e8ec;border-radius:8px">`;
 s+=`<text x="${W/2}" y="24" text-anchor="middle" font-size="14" font-weight="700" fill="#1a1a2e">${title}</text>`;
 for(let k=0;k<=4;k++){const gy=mt+ph*k/4,val=ymax*(1-k/4);
  s+=`<line x1="${ml}" y1="${gy}" x2="${W-mr}" y2="${gy}" stroke="#eceff1"/>`;
  s+=`<text x="${ml-6}" y="${gy+4}" text-anchor="end" font-size="10" fill="#90a4ae">${val.toFixed(ymax<=1?2:0)}</text>`;}
 s+=`<line x1="${ml}" y1="${mt+ph}" x2="${W-mr}" y2="${mt+ph}" stroke="#b0bec5"/>`;
 items.forEach((it,i)=>{const cx=ml+gap*(i+0.5),x=cx-bw/2,v=it.value,y=sy(v),h=Math.max(0,mt+ph-y);
  const col=it.color||(opt.colorByValue?(v>=0.6?'#2e7d32':(v>=0.3?'#f9a825':'#c62828')):'#4c78a8');
  s+=`<rect x="${x}" y="${y}" width="${bw}" height="${h}" fill="${col}" rx="2"/>`;
  s+=`<text x="${cx}" y="${y-5}" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1a2e">${v.toFixed(opt.dec==null?2:opt.dec)}</text>`;
  s+=`<text x="${cx}" y="${mt+ph+16}" text-anchor="middle" font-size="11" fill="#374151">${it.label}</text>`;});
 // reference lines: draw at true y; LEFT-anchor labels (away from the right-side bars) and
 // de-collide vertically so close refs (e.g. 0.97/0.94, 0.85/0.80) never overlap each other.
 const refs=(opt.refs||[]).filter(rf=>rf.v<=ymax).slice().sort((a,b)=>b.v-a.v);
 let lastLy=-1e9;
 refs.forEach(rf=>{const y=sy(rf.v);const col=rf.c||'#c62828';
  s+=`<line x1="${ml}" y1="${y}" x2="${W-mr}" y2="${y}" stroke="${col}" stroke-width="1.4" stroke-dasharray="5 4"/>`;
  let ly=y-3; if(ly<lastLy+11.5) ly=lastLy+11.5; lastLy=ly;
  s+=`<text x="${ml+5}" y="${ly}" text-anchor="start" font-size="10" font-weight="700" fill="${col}">${rf.m} ${rf.v.toFixed(2)}</text>`;});
 s+='</svg>';
 return `<div class="card" style="flex:1 1 360px;min-width:320px">${s}${opt.caption?('<p class="note">'+opt.caption+'</p>'):''}</div>`;
}
function renderCampaigns(cps){
 const el=document.getElementById('campaigns'); if(!el)return;
 let intro='<h3>📈 记忆实验:记忆是否让智能体越做越好?</h3>'+note('让智能体反复练同一批题、把经验存进「记忆」。每行是一个 <b>epoch</b>(练一遍)。看 <b>oracle-calls→solve</b>(解题所需尝试次数)是否随 epoch <b>下降</b> —— 下降=记忆在起作用。臂:<b>with_memory</b>=用记忆 · <b>no_memory</b>=不用(对照)· <b>permuted_memory</b>=故意用错的记忆(理应帮不上)。');
 if(!cps||!cps.length){el.innerHTML=intro+'<div class="card">还没有记忆实验数据。</div>';return;}
 let h=intro;
 for(const c of cps){h+=`<div class="card"><div class="name">${c.name} — 共 ${c.summary.n_episodes} 次尝试</div>`;
  for(const [arm,curve] of Object.entries(c.summary.curves||{})){
   h+=`<div style="margin:6px 0"><b>${arm}</b><table><tr><th>epoch(第几遍)</th><th>解出率</th><th>解题所需尝试次数 ↓越低越好</th><th>A_memory</th><th>已学策略数</th></tr>`+
   curve.map(e=>`<tr><td>${e.epoch}</td><td class="${cls(e.solve_rate)}">${fmt(e.solve_rate)}</td><td class="${arm=='with_memory'?'g':''}">${fmt(e.mean_oracle_calls_solved)}</td><td>${fmt(e.A_memory)}</td><td>${fmt(e.n_strategies)}</td></tr>`).join('')+`</table></div>`;}
  h+=`</div>`;}
 el.innerHTML=h;
}
// 🔧 编排求解卡片(基座模型 + 调用外部 SOTA 工具 DesiRNA)。单列呈现,绝不与"模型自身能力"混为一柱。
// 成绩单页(首屏)与图表页共用,保证一打开就能看到 81 这个 headline。
function orchHTML(){
 if(!(ORCH && ORCH.orchestrated_total!=null)) return '';
 const o=ORCH;const R=o.residual||{};
 const A=(R.A_small_near_miss||[]).length,B=(R.B_large_compute||[]).length,D=(R.D_undesignable||[]).length;
 // 图:两根柱 + 仅保留两条最可比的 SOTA 参照(同预算 DesiRNA 0.85、LLM 方法 0.80);
 // 24h 极限 0.97 / NEMO 0.94 挪到右侧说明,避免虚线标签在图上挤成一团。
 const bars=svgBars('基座模型独立  →  + 编排调用 DesiRNA',[
   {label:'GLM-5.2 独立',value:o.model_solved/100,color:'#4c78a8'},
   {label:'+编排 DesiRNA',value:o.orchestrated_total/100,color:'#2e7d32'}],
  {ymax:1,dec:2,refs:[{m:'DesiRNA <1min·同预算',v:0.85,c:'#2e7d32'},{m:'LLM 方法 SOTA',v:0.80,c:'#d97706'}],
   caption:'柱=解出率;两条虚线=同分钟级预算下最可比的公开 SOTA。'});
 // 右侧信息块:大字 headline + 拆解 + 残差三桶 + 一行红色告警。分区块,不再堆成一段。
 const info=`<div class="card" style="flex:1 1 300px;min-width:290px">
   <div style="font-size:30px;font-weight:800;color:#2e7d32;line-height:1">${o.orchestrated_total}<span style="font-size:15px;color:#6b7280;font-weight:600">/100</span></div>
   <div class="sub" style="margin:3px 0 10px"><b>${o.model_solved}</b> 基座模型独立 &nbsp;+&nbsp; <b>${o.tool_solved_of_unsolved}</b> 智能体调用 DesiRNA 回收(${o.n_unsolved_before_tool} 未解中)</div>
   <table style="width:100%"><tr><th>编排后残差 ${R.n} 题</th><th>数</th><th>说明</th></tr>
   <tr><td>A 小题近失 bp≤2</td><td class="y">${A}</td><td class="sub">最后一毫米,可更长 -t/收官算子再推</td></tr>
   <tr><td>B 大题算力受限</td><td class="y">${B}</td><td class="sub">需更多算力预算</td></tr>
   <tr><td>D 不可设计</td><td>${D}</td><td class="sub">样本永不折叠到目标,非搜索失败</td></tr>
   </table>
   <div style="margin-top:10px;background:#fdecea;border-left:3px solid #c62828;padding:8px 11px;border-radius:0 6px 6px 0;font-size:12.5px;line-height:1.6"><b class="r">⚠ 这测的是"调用工具"这条工程链路,不是模型学到的能力(全程无 self-improvement)</b> —— "直接调 DesiRNA" = 绿区 null 的部署侧陈述。</div>
  </div>`;
 const foot=note('口径:每条候选序列都经我们<b>自己的 oracle</b> 复验,绝不信工具自报;DesiRNA 能量模型已对齐 (-p 2004)。其余 SOTA 参照:24h 经典极限 <b>0.97</b>、NEMO 原生 <b>0.94</b>(均非同预算,故不画在图上)。');
 return `<h4>🔧 编排求解(基座模型 + 调用 SOTA 工具 DesiRNA)· 当前 headline</h4>
  <div style="display:flex;flex-wrap:wrap;gap:14px;align-items:flex-start">${bars}${info}</div>${foot}`;
}
function renderScorecard(cards){
 let intro='<h3>🏆 综合成绩单</h3>'+note('把所有能力汇总成三块:① 传统算法在 100 道公认 RNA 谜题上的成绩(定「地板线」)· ② 带额外约束时的设计能力 · ③ AI 智能体的「照诊断行动」与「记忆」能力。');
 const orch=orchHTML();   // headline: 编排求解 81(基座 + 调工具),放最上方,一打开就看到
 if(!cards||!cards.length){document.getElementById('scorecard').innerHTML=intro+orch+'<div class="card">还没有成绩单数据。</div>';return;}
 let h=intro+orch;
 // === 元学习三臂对照 overlay(frozen / K_θ / A_θ)—— the money comparison ===
 const armMeta={'meta_ladder_frozen':{label:'frozen 对照(种子不变)',color:'#868e96'},
   'meta_ladder_accumulate':{label:'K_θ 内容记忆(memdir bullets)',color:'#e8590c'},
   'meta_ladder_askill':{label:'A_θ 过程记忆(Skills)',color:'#2f9e44'}};
 const arms=cards.filter(c=>armMeta[c.name]&&c.scorecard.panels&&c.scorecard.panels.capability_ladder&&(c.scorecard.panels.capability_ladder.epochs||[]).length);
 if(arms.length>=2){const tgt=80,base=65;let maxEp=1;
  arms.forEach(c=>c.scorecard.panels.capability_ladder.epochs.forEach(e=>{if(e.epoch>maxEp)maxEp=e.epoch;}));
  const W=640,H=270,pL=44,pB=40,pT=24,pR=230;
  const xs=e=>pL+(W-pL-pR)*(maxEp<2?0.5:(e-1)/(maxEp-1));
  const ys=v=>pT+(H-pT-pB)*(1-v/100);
  const yt=ys(tgt).toFixed(1),yb=ys(base).toFixed(1);let lines='',legend='';
  arms.forEach((c,ai)=>{const m=armMeta[c.name];const byEp={};
   c.scorecard.panels.capability_ladder.epochs.forEach(e=>{byEp[e.epoch]=Math.max(byEp[e.epoch]||0,e.solved);});
   const pts=Object.keys(byEp).map(Number).sort((a,b)=>a-b);
   const poly=pts.map(ep=>`${xs(ep).toFixed(1)},${ys(byEp[ep]).toFixed(1)}`).join(' ');
   lines+=`<polyline points="${poly}" fill="none" stroke="${m.color}" stroke-width="2.5"/>`+
    pts.map(ep=>`<circle cx="${xs(ep).toFixed(1)}" cy="${ys(byEp[ep]).toFixed(1)}" r="3.5" fill="${m.color}"/>`).join('');
   const cur=pts.length?byEp[pts[pts.length-1]]:'?';
   legend+=`<text x="${W-pR+10}" y="${pT+18+ai*20}" font-size="11" fill="${m.color}">■ ${m.label} · 当前 ${cur}</text>`;});
  const svg=`<svg viewBox="0 0 ${W} ${H}" style="max-width:100%;height:auto;display:block">
    <line x1="${pL}" y1="${ys(0)}" x2="${W-pR}" y2="${ys(0)}" stroke="#ccc"/><line x1="${pL}" y1="${pT}" x2="${pL}" y2="${ys(0)}" stroke="#ccc"/>
    <text x="${pL-6}" y="${(ys(0)+3).toFixed(1)}" font-size="10" text-anchor="end" fill="#888">0</text>
    <text x="${pL-6}" y="${(ys(100)+3).toFixed(1)}" font-size="10" text-anchor="end" fill="#888">100</text>
    <line x1="${pL}" y1="${yt}" x2="${W-pR}" y2="${yt}" stroke="#e03131" stroke-width="1.3" stroke-dasharray="6 4"/>
    <text x="${pL+6}" y="${(+yt-5)}" font-size="10" fill="#e03131">▲ LLM法 SOTA ~80 (Gautam 2026)</text>
    <line x1="${pL}" y1="${yb}" x2="${W-pR}" y2="${yb}" stroke="#adb5bd" stroke-width="1" stroke-dasharray="3 3"/>
    <text x="${pL+6}" y="${(+yb-5)}" font-size="10" fill="#868e96">基线 65</text>
    ${lines}${legend}
    ${Array.from({length:maxEp},(_,i)=>i+1).map(ep=>`<text x="${xs(ep).toFixed(1)}" y="${(ys(0)+16).toFixed(1)}" font-size="10" text-anchor="middle" fill="#888">e${ep}</text>`).join('')}
  </svg>`;
  h+=`<div class="card"><div class="name">🧬 元学习三臂对照:哪种「记忆表征」在复利?</div>`+
   note('同一 agent、同一 memdir 种子;<b>唯一差别 = epoch 间积累的记忆表征</b>。<b>A_θ(Skills,绿)</b>=可迁移的推理/搜索/规划<b>过程</b> · <b>K_θ(memdir,橙)</b>=内容/事实 bullet(易被过度套用→坍缩)· <b>frozen(灰)</b>=不积累(对照)。epoch1 三臂应重合;之后 <b>A_θ 与 frozen 的差 = 过程记忆带来的元学习增益</b>。')+svg+`</div>`;}
 for(const c of cards){const p=c.scorecard.panels||{};
  const _am=armMeta[c.name];
  const _role=_am?` <span style="color:${_am.color};font-weight:700">— ${_am.label} 臂</span>`:'';
  h+=`<div class="card"><div class="name">🏆 ${c.name}${_role}</div><div class="sub">红线:仅在折叠算法裁判下判成功,不涉体内功能。${_am?' · 本卡是三臂对照之一(见顶部叠加图),不是旧实验。':''}</div>`;
  const cl=p.capability_ladder;
  if(cl&&cl.epochs&&cl.epochs.length){const eps=cl.epochs,tgt=80;
   const _last=eps[eps.length-1];
   const live=(_last&&_last.provisional)?`<div style="margin:6px 0 2px;color:#e8590c;font-weight:600">🔴 正在攻:${_last.current_task||'?'} (${_last.current_len||'?'}nt) · 全集第 <b>${_last.current_idx_100||'?'}/${_last.n}</b> 道 · 未解集第 <b>${_last.attempted||0}/${_last.to_attempt||'?'}</b> 道</div>`:'';
   const W=560,H=230,pL=42,pB=34,pT=22,pR=140;
   const xs=i=>pL+(W-pL-pR)*(eps.length<2?0.5:i/(eps.length-1));
   const ys=v=>pT+(H-pT-pB)*(1-v/100);
   const poly=eps.map((e,i)=>`${xs(i).toFixed(1)},${ys(e.solved).toFixed(1)}`).join(' ');
   const dots=eps.map((e,i)=>`<circle cx="${xs(i).toFixed(1)}" cy="${ys(e.solved).toFixed(1)}" r="4" fill="#2f9e44"/><text x="${xs(i).toFixed(1)}" y="${(ys(e.solved)-9).toFixed(1)}" font-size="12" font-weight="700" text-anchor="middle" fill="#2f9e44">${e.solved}</text>`).join('');
   const yt=ys(tgt).toFixed(1);
   const svg=`<svg viewBox="0 0 ${W} ${H}" style="max-width:100%;height:auto;display:block">
     <line x1="${pL}" y1="${ys(0)}" x2="${W-pR}" y2="${ys(0)}" stroke="#ccc"/>
     <line x1="${pL}" y1="${pT}" x2="${pL}" y2="${ys(0)}" stroke="#ccc"/>
     <text x="${pL-6}" y="${(ys(0)+3).toFixed(1)}" font-size="10" text-anchor="end" fill="#888">0</text>
     <text x="${pL-6}" y="${(ys(100)+3).toFixed(1)}" font-size="10" text-anchor="end" fill="#888">100</text>
     <line x1="${pL}" y1="${yt}" x2="${W-pR}" y2="${yt}" stroke="#e03131" stroke-width="1.5" stroke-dasharray="6 4"/>
     <text x="${W-pR+6}" y="${yt}" font-size="11" fill="#e03131">▲ LLM法 SOTA ~${tgt}</text>
     <text x="${W-pR+6}" y="${(+yt+14)}" font-size="10" fill="#e03131">经典 DesiRNA 85@1min·97@24h</text>
     <polyline points="${poly}" fill="none" stroke="#2f9e44" stroke-width="2.5"/>${dots}
     ${eps.map((e,i)=>`<text x="${xs(i).toFixed(1)}" y="${(ys(0)+16).toFixed(1)}" font-size="10" text-anchor="middle" fill="#888">e${e.epoch}</text>`).join('')}
   </svg>`;
   h+=`<h4>🚀 解题能力阶梯 — 逐 epoch 提升(${cl.benchmark||'Eterna100-V2'})</h4>`+
    note((cl.benchmark||'').includes('autonomous')
      ? '目标:量化拿掉硬编码策略后的<b>自主解题地板</b>。本梯 = <b>自主 AGENT</b>(glm-5.2,<b>RNA_AUTONOMOUS=1:中性工具(仅 fold/verify/diagnose/通用搜索)+ 空记忆 + 无策略</b>)—— 工具层不再喂「分解→拼接→修复」套路,agent 必须<b>自己识别问题、自组装搜索</b>。绿点=解对题数;红虚线=<b>LLM法 SOTA ~80</b>(Gautam 2026;经典 DesiRNA <b>同预算(&lt;1min)~85</b> / 1h~95 / 24h 97、V1 已饱和 100)。这是<b>诚实基线</b>,预期<b>低于</b>带拐杖的 ~76;二者之差 = 之前被硬编码策略替 agent 干掉的部分。从基线 65 起、只做错的 35 题;token 为估算。'
      : '目标:通过<b>不断改进→测试</b>让解题率逐 epoch 上升。每个 epoch 叠加一个新技术、在<b>固定预算</b>下重测整批题(固定预算=提升来自策略而非堆算力)。绿线=解对题数;红虚线=<b>LLM法 SOTA ~80</b>(Gautam 2026;经典 DesiRNA <b>同预算(&lt;1min)~85</b> / 1h~95 / 24h 97、V1 已饱和 100)。solved 只增不减。本梯 = <b>meta-learning AGENT</b>(glm-5.2,全工具+策略记忆);从 with-memory 基线 <b>65</b> 起、只迭代做错的 35 题;token 为<b>估算</b>(glm/OpenRouter 报 usage=0,按 trace 字符/4)。')+
    live+svg+
    `<table><tr><th>epoch</th><th>新增技术</th><th>解对/100</th><th>解出率</th><th>本轮新解出</th><th>耗时</th></tr>`+
    eps.map(e=>`<tr><td>${e.epoch}</td><td>${e.note||('L'+e.level)}</td><td><b class="${e.solved>=tgt?'g':''}">${e.solved}</b></td><td>${fmt(e.solve_rate)}</td><td class="sub">${(e.new_this_epoch||[]).slice(0,10).join(', ')}${(e.new_this_epoch||[]).length>10?'…':''}</td><td>${fmt(e.epoch_secs)}s</td></tr>`).join('')+
    `</table>`;}
  const s=p.standard_leaderboard;
  if(s){h+=`<h4>① Eterna100 基准:传统算法的「地板线」</h4>`+
   note('100 道公认 RNA 谜题上,两个<b>传统基线算法</b>各解对多少道 —— 这是 AI 要超越的地板(本 harness 预算下的测量,非文献上限)。<b>solved/100</b>=解对几道;<b>uMFE</b>=其中「唯一最优解」几道(更稳);<b>NED</b>=折叠不干脆程度(越低越好);<b>mean s</b>=每题耗时秒。<br>📌 <b>公开 SOTA 标尺</b>:V2 经典榜首 <b>DesiRNA 97/100</b>(NAR 2025;<b>同预算&lt;1min 仅 ~85</b>,24h 才 97)· V2 LLM 方法 SOTA <b>~80</b>(Gautam 2026)· V1 已饱和 100/100。要超越需口径对齐(见「图表」页)。')+
   `<table><tr><th>方法</th><th>解对/100</th><th>解出率</th><th>uMFE</th><th>NED(全部)</th><th>NED(解出)</th><th>每题秒</th></tr>`+
   Object.entries(s.by_method).map(([m,v])=>`<tr><td>${m}</td><td><b>${v.solved}</b></td><td>${fmt(v.solved_rate)}</td><td>${v.umfe}</td><td>${fmt(v.mean_NED_all)}</td><td>${fmt(v.mean_NED_solved)}</td><td>${fmt(v.mean_secs)}</td></tr>`).join('')+`</table>`;}
  const hy=p.hybrid_seam;
  if(hy){h+=`<h4>② 混合约束设计(${hy.n_tasks} 题 / ${hy.n_base} 基础题)</h4>`+
   note('在「结构目标」之外再加约束(GC 含量 / 固定片段 / 禁用序列 / 编码蛋白)时的成绩。<b>solved</b>=结构对且约束全满足;<b>struct match</b>=只看结构;<b>constraints ok</b>=只看约束。⚠ <b>coding=0</b> 说明「编码蛋白」约束目前没搞定(已知短板)。')+
   `<table><tr><th>约束类型</th><th>解出(结构+约束)</th><th>结构对</th><th>约束满足</th></tr>`+
   Object.entries(hy.by_profile).map(([pr,v])=>`<tr><td>${pr}</td><td class="${cls(v.solved_rate)}">${fmt(v.solved_rate)}</td><td>${fmt(v.struct_match_rate)}</td><td>${fmt(v.constraints_ok_rate)}</td></tr>`).join('')+`</table>`;}
  const ag=p.agent_capability;
  if(ag){const a=ag.actuation_ablation||{};
   h+=`<h4>③ 智能体能力:核心对照实验(${ag.n_ablation_tasks} 题)</h4>`+
   note('把「正确诊断」喂给智能体让它<b>照着改</b>(diagnosis)vs <b>随机乱改</b>(random)vs 框架<b>偷偷丢掉诊断</b>(silent_fallback)。<b>A</b>=诊断真的驱动了下一步编辑的程度。<b>看点:diagnosis 的 A 最高、也解得最多 → 证明「照诊断行动」确实有用</b>(这就是项目的核心论点)。')+
   `<table><tr><th>模式</th><th>解出率</th><th>A(驱动保真度)</th><th>A_model</th><th>A_harness</th></tr>`+
   Object.entries(a).map(([m,v])=>{const lbl={diagnosis:'照诊断改',random:'随机乱改',silent_fallback:'丢掉诊断'}[m]||m;return `<tr><td>${lbl} <span class="sub">${m}</span></td><td>${fmt(v.solve_rate)}</td><td class="${cls(v.A)}">${fmt(v.A)}</td><td>${fmt(v.A_model)}</td><td>${fmt(v.A_harness)}</td></tr>`}).join('')+`</table>`;
   if(ag.meta_arms){h+=`<h4>④ 记忆是否提升效率</h4>`+
    note('<b>with_memory</b>=用记忆 · <b>no_memory</b>=不用(对照)· <b>permuted</b>=故意用错的记忆。<b>平均尝试次数→解出</b>越低越省。看点:with_memory 远低于 no_memory → 记忆把试错砍到约 1/3;错记忆(permuted)帮不上 → 增益来自「正确知识」。')+
    `<table><tr><th>记忆臂</th><th>解出率</th><th>平均尝试次数→解出 ↓</th><th>A_memory</th><th>已学策略</th></tr>`+
    Object.entries(ag.meta_arms).map(([m,v])=>{const lbl={with_memory:'用记忆',no_memory:'不用记忆(对照)',permuted_memory:'错记忆(对照)'}[m]||m;return `<tr><td>${lbl}</td><td>${fmt(v.solve_rate)}</td><td class="${m=='with_memory'?'g':''}">${fmt(v.mean_evals_solved)}</td><td>${fmt(v.A_memory)}</td><td>${fmt(v.n_strategies_learned)}</td></tr>`}).join('')+`</table>`;}}
  h+=`</div>`;}
 document.getElementById('scorecard').innerHTML=h;
}
function renderOverview(){
 let h='<h3>🧭 概览:各运行的头条结果</h3>'+note('每张卡片是一次实验运行的一句话结论。想看某次运行的完整报告(逐题表 + RNA 结构图),点右上角「运行总览 / 详细报告」。');
 for(const r of RUNS){
  const s=r.summary||{}, m=r.meta_summary;
  if(s.ablation){h+=`<div class="card"><div class="name">${r.name} — 驱动保真度对照</div>`+
   note('「照诊断改 vs 随机 vs 丢弃诊断」的对照;并附「数据/奖励/轨迹」三种确定性(均应 ≈ 1)。')+
   `<table><tr><th>模式</th><th>解出率</th><th>A</th><th>A_model</th><th>A_harness</th></tr>`+
   Object.entries(s.ablation).map(([k,v])=>`<tr><td>${k}</td><td class="${cls(v.solve_rate)}">${fmt(v.solve_rate)}</td><td class="${cls(v.A)}">${fmt(v.A)}</td><td>${fmt(v.A_model)}</td><td>${fmt(v.A_harness)}</td></tr>`).join('')+`</table>`;
   if(s.coords)h+=`<div class="kv">数据确定性 <b class="g">${fmt(s.coords.D_sym)}</b></div><div class="kv">奖励确定性 <b class="g">${fmt(s.coords.R_det)}</b></div><div class="kv">轨迹确定性 <b class="g">${fmt(s.coords.T_trace)}</b></div>`;
   if(s.baselines)h+='<div class="kv">基线: '+Object.entries(s.baselines).map(([k,v])=>`${k} ${fmt(v.solve_rate_best_of_seeds)}`).join(', ')+'</div>';
   h+=`</div>`;}
  if(m&&m.arms){h+=`<div class="card"><div class="name">${r.name} — 记忆臂对比</div>
   <table><tr><th>记忆臂</th><th>解出率</th><th>平均尝试→解出 ↓</th><th>A_memory</th><th>策略数</th></tr>`+
   Object.entries(m.arms).map(([k,v])=>`<tr><td>${k}</td><td>${fmt(v.solve_rate)}</td><td class="${v.arm=='with_memory'?'g':''}">${fmt(v.mean_evals_solved)}</td><td>${fmt(v.A_memory)}</td><td>${fmt(v.n_strategies_learned)}</td></tr>`).join('')+`</table></div>`;}
  if(s.model){h+=`<div class="card"><div class="name">${r.live?'🔴 <span style="color:#e8590c">运行中</span> · ':''}${r.name} — AI 模型能力(${s.model})</div>
   <div class="kv">解出率 <b>${fmt(s.solve_rate)}</b></div><div class="kv">平均奖励 <b>${fmt(s.mean_reward)}</b></div>
   <div class="kv">agent_A(驱动保真度) <b class="${cls(s.agent_A)}">${fmt(s.agent_A)}</b></div><div class="kv">A_model <b>${fmt(s.agent_A_model)}</b></div>
   <div style="margin-top:6px"><a href="/report/${encodeURIComponent(r.name)}">看完整报告 →</a></div></div>`;}
 }
 document.getElementById('overview').innerHTML=h;
}
function renderRuns(){
 let h='<h3>📂 所有运行(文件视图)</h3>'+note('原始文件视图。想要按用途分类 + 白话说明的版本,请用右上角「运行总览」。');
 const _runs=[...RUNS].sort((a,b)=>(b.live?1:0)-(a.live?1:0));
 for(const r of _runs){h+=`<div class="card"><div class="name">${r.live?'🔴 <span style="color:#e8590c">运行中</span> · ':''}<a href="/report/${encodeURIComponent(r.name)}">${r.name}</a></div>
  <div class="sub">文件: ${(r.files||[]).join(', ')}</div>`+
  (r.datasets?('<div class="kv">数据集: '+Object.entries(r.datasets).map(([k,v])=>`${k}=${v} 行`).join(', ')+'</div>'):'')+
  (r.n_traces?`<div class="kv">操作录像: ${r.n_traces} 段</div>`:'')+`</div>`;}
 document.getElementById('runs').innerHTML=h;
}
function renderFigures(){
 let h='<h3>📊 图表(动态渲染)</h3>'+note('由实验数据现场渲染(内联 SVG)。<b>重要:只有跑在「同一批题」上的运行才放进同一张对比图</b> —— 不同题集(全100题 / 随机10题 / 小demo)解出率不可直接比,分开呈现以免误导。');
 // 🎯 公开 SOTA 标尺(要超越的数字)
 h+=`<div class="card">🎯 <b>Eterna100 公开 SOTA(要超越的标尺 · 带引用)</b>
  <div class="scroll"><table><tr><th>关键数字</th><th>值</th><th>出处</th></tr>
  <tr><td>V1 榜首(已饱和)</td><td><b>100/100</b></td><td>Montparnasse (Cazenave 2026, arXiv:2606.07562, 嵌套MC/GNRPA) · DesiRNA (Wirecki, NAR 2025, REMC)</td></tr>
  <tr><td><b>V2 榜首(经典 SOTA)</b></td><td><b class="g">97/100</b><span class="sub"> @24h</span></td><td>DesiRNA · NAR 2025 gkae1306(Turner2004, 开源 github.com/fryzjergda/DesiRNA;<b>分层预算 85@&lt;1min · 95@&lt;1h · 97@24h</b> —— 同分钟级预算下 ~85)</td></tr>
  <tr><td>V2 · NEMO 原生</td><td>~94/100</td><td>NEMO · Portela 2018 / Koodli 2021(经典 MC)</td></tr>
  <tr><td><b>V2 · LLM 方法 SOTA</b>(对我们最可比)</td><td><b class="y">~79–81/100</b></td><td>Qwen2.5-0.5B(SL+RL)· Gautam, arXiv:2602.12470 (2026)</td></tr>
  </table></div>`+note('<b>纠正两处常见误传</b>:① 传说的「<b>99/100</b>」是 <b>Montparnasse 在 V1</b> 的时限快照(10,240s/题内 99;100/100 需给最难的 Gladius #90 单独跑 24,108s)——<b>是 V1、不是 V2</b>。② 「NEMO 严格定预算 <b>90</b>」其实是 Gautam 2026 那篇 LLM 论文里把 NEMO 当<b>固定采样预算基线</b>跑出来的数,<b>非 NEMO 真实水平(~94)、更非 solver 榜首</b>;NEMO 是 Portela 2018,不是 Gautam。<br>结论:<b>V1 已饱和(100)</b>;<b>V2 经典天花板 = DesiRNA 97,但要 24h/题</b>;<b>LLM 路线直接对标 = ~80(Gautam)</b>。口径对齐:V2 题集 + ViennaRNA/Turner2004 + 精确 MFE + 报告预算。<br>⏱️ <b>预算对齐是关键</b>:DesiRNA 的 97 是 <b>24h/题</b>数,分层为 <b>85@&lt;1min · 95@&lt;1h · 97@24h</b>;故与本 harness(分钟级)最公平的经典对比是 <b>~85</b>,而非 97 —— 21 分那个差里,只有 ~9 是同预算差,其余 ~12 是额外 24h 才买到的最后 2 道。');
 h+=orchHTML();   // 🔧 编排求解卡片(定义见 orchHTML;图表页与成绩单页共用)
 const sc=(SCORECARD&&SCORECARD[0])?SCORECARD[0].scorecard:null;
 const lbl={diagnosis:'照诊断改',random:'随机乱改',silent_fallback:'丢掉诊断'};
 // 确定性三坐标:一行陈述(三根 1.00 的柱图无信息量)
 const cr=RUNS.find(r=>r.summary&&r.summary.coords);
 if(cr){const c=cr.summary.coords;
  h+=`<div class="card"><b>✓ 确定性三坐标</b>:数据 <span class="g">${fmt(c.D_sym)}</span> · 奖励 <span class="g">${fmt(c.R_det)}</span> · 轨迹 <span class="g">${fmt(c.T_trace)}</span> —— 均 = 1.00。RNA 是「确定性正对照」,把「驱动保真度」隔离成唯一变量。</div>`;}
 let charts='';
 // 驱动保真度 gap(核心)
 if(sc&&sc.panels&&sc.panels.agent_capability&&sc.panels.agent_capability.actuation_ablation){const ab=sc.panels.agent_capability.actuation_ablation;
  charts+=svgBars('① 解出率:照诊断改 vs 随机乱改 vs 丢掉诊断', Object.entries(ab).map(([m,v])=>({label:lbl[m]||m,value:v.solve_rate})),
   {caption:'同一批30题、只改「反馈方式」。照诊断行动解得最多 → 「让诊断驱动下一步」确实有用(项目核心论点)。'});
  charts+=svgBars('② 驱动保真度 A(诊断是否真的控制了编辑)', Object.entries(ab).map(([m,v])=>({label:lbl[m]||m,value:v.A})),
   {ymax:1,colorByValue:true,caption:'A 越高=越照诊断行动;框架丢弃诊断 → A=0,解出率塌回随机水平。'});}
 // 评测运行:按「题目签名」分组,只在同题集内对比
 const evals=RUNS.filter(r=>r.summary&&r.summary.model&&r.summary.solve_rate!=null&&Array.isArray(r.summary.records));
 const groups={};
 for(const r of evals){const sig=r.summary.records.map(x=>x.task_id).sort().join('|');(groups[sig]=groups[sig]||[]).push(r);}
 // 全 Eterna100(100题):经典基线 = 地板;AI 全100运行跑完自动并入
 const full=evals.filter(r=>r.summary.n_tasks>=100);
 if(sc&&sc.panels&&sc.panels.standard_leaderboard){const bm=sc.panels.standard_leaderboard.by_method||{};
  let items=Object.entries(bm).map(([m,v])=>({label:m,value:v.solved_rate,color:'#90a4ae'}));
  full.forEach(r=>items.push({label:r.summary.model.split('/').pop(),value:r.summary.solve_rate,color:'#4c78a8'}));
  charts+=svgBars('全 Eterna100-V2(100 题):AI vs 经典基线 vs 公开 SOTA', items,
   {ymax:1,refs:SOTA_V2,caption:'柱 = 解出率:<b>灰</b>=两个传统(非 AI)算法地板线(rnainverse=ViennaRNA 经典逆折叠、local_search=本地搜索;本 harness 预算下的测量,非文献上限)、<b>蓝</b>=AI 模型。<b>虚线 = 公开 SOTA 标尺</b>:深绿=DesiRNA 0.97(24h/题·经典极限)、浅绿=DesiRNA ~0.85(&lt;1min·<b>同预算最可比</b>)、灰=NEMO 原生 0.94、橙=LLM 方法 0.80。同分钟级预算下经典对标是 <b>~85</b>,不是 97。'+(full.length?'':' AI 全 100 题跑完后其柱自动加入,即可直接对着虚线看差距。')});}
 // 可比子集:同一题集且 ≥2 个运行(如 sonnet-5 vs deepseek 同 10 题)
 for(const sig in groups){const g=groups[sig];
  if(g.length>=2 && g[0].summary.n_tasks<100){
   charts+=svgBars(`同批 ${g[0].summary.n_tasks} 道题:AI 模型对比(严格可比)`,
    g.map(r=>({label:r.summary.model.split('/').pop(),value:r.summary.solve_rate})),
    {ymax:1,colorByValue:true,caption:'同一批题、同配置 —— 唯一严格可比的模型对照。逐题胜负见顶部「对比两次运行」。'});}}
 h+=`<div style="display:flex;flex-wrap:wrap;gap:14px">${charts||'<div class="card">暂无可渲染数据</div>'}</div>`;
 // 小规模 demo 运行:题集小且各不相同,不参与对比,仅提示
 const small=evals.filter(r=>r.summary.n_tasks<10);
 if(small.length){h+=note('另有小规模 demo 运行:'+small.map(r=>`${r.name}(${r.summary.n_tasks}题)`).join('、')+' —— 题目少且各不相同,解出率不可与上面对比,故不入图(可在「概览」单独查看)。');}
 document.getElementById('figures').innerHTML=h;
}
async function loadLogList(){
 const j=await (await fetch('/api/loglist')).json();
 const sel=document.getElementById('logsel'); const cur=sel.value;
 sel.innerHTML=j.files.map(f=>`<option value="${f.path}">${f.path.split('/').slice(-2).join('/')} (${(f.size/1024|0)}k)</option>`).join('');
 if(cur)sel.value=cur;
}
async function loadlog(){
 const p=document.getElementById('logsel').value; if(!p)return;
 const j=await (await fetch('/api/logtail?n=200&path='+encodeURIComponent(p))).json();
 const b=document.getElementById('logbox'); b.textContent=(j.lines||[]).join('\\n'); b.scrollTop=b.scrollHeight;
}
setInterval(()=>{if(document.getElementById('auto').checked && document.getElementById('logs').style.display!='none')loadlog();},3000);
setInterval(load,8000); setInterval(loadLogList,8000);
load(); loadLogList();
</script></body></html>"""


def main(argv=None) -> int:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8811)
    args = ap.parse_args(argv)
    print(f"[dashboard] serving {REPO} on http://{args.host}:{args.port}  (runs={RUNS})", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
