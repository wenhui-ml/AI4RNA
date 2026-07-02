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


def _list_runs() -> List[Dict]:
    out = []
    if not os.path.isdir(RUNS):
        return out
    for name in sorted(os.listdir(RUNS)):
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
let RUNS=[]; let SCORECARD=[];
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
function renderScorecard(cards){
 let intro='<h3>🏆 综合成绩单</h3>'+note('把所有能力汇总成三块:① 传统算法在 100 道公认 RNA 谜题上的成绩(定「地板线」)· ② 带额外约束时的设计能力 · ③ AI 智能体的「照诊断行动」与「记忆」能力。');
 if(!cards||!cards.length){document.getElementById('scorecard').innerHTML=intro+'<div class="card">还没有成绩单数据。</div>';return;}
 let h=intro;
 for(const c of cards){const p=c.scorecard.panels||{};
  h+=`<div class="card"><div class="name">🏆 ${c.name}</div><div class="sub">红线:仅在折叠算法裁判下判成功,不涉体内功能。</div>`;
  const s=p.standard_leaderboard;
  if(s){h+=`<h4>① Eterna100 基准:传统算法的「地板线」</h4>`+
   note('100 道公认 RNA 谜题上,两个<b>传统基线算法</b>各解对多少道 —— 这是 AI 要超越的地板。<b>solved/100</b>=解对几道;<b>uMFE</b>=其中「唯一最优解」几道(更稳);<b>NED</b>=折叠不干脆程度(越低越好);<b>mean s</b>=每题耗时秒。')+
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
  if(s.model){h+=`<div class="card"><div class="name">${r.name} — AI 模型能力(${s.model})</div>
   <div class="kv">解出率 <b>${fmt(s.solve_rate)}</b></div><div class="kv">平均奖励 <b>${fmt(s.mean_reward)}</b></div>
   <div class="kv">agent_A(驱动保真度) <b class="${cls(s.agent_A)}">${fmt(s.agent_A)}</b></div><div class="kv">A_model <b>${fmt(s.agent_A_model)}</b></div>
   <div style="margin-top:6px"><a href="/report/${encodeURIComponent(r.name)}">看完整报告 →</a></div></div>`;}
 }
 document.getElementById('overview').innerHTML=h;
}
function renderRuns(){
 let h='<h3>📂 所有运行(文件视图)</h3>'+note('原始文件视图。想要按用途分类 + 白话说明的版本,请用右上角「运行总览」。');
 for(const r of RUNS){h+=`<div class="card"><div class="name"><a href="/report/${encodeURIComponent(r.name)}">${r.name}</a></div>
  <div class="sub">文件: ${(r.files||[]).join(', ')}</div>`+
  (r.datasets?('<div class="kv">数据集: '+Object.entries(r.datasets).map(([k,v])=>`${k}=${v} 行`).join(', ')+'</div>'):'')+
  (r.n_traces?`<div class="kv">操作录像: ${r.n_traces} 段</div>`:'')+`</div>`;}
 document.getElementById('runs').innerHTML=h;
}
function renderFigures(){
 let h='<h3>📊 图表(动态渲染)</h3>'+note('由实验数据现场渲染(内联 SVG)。<b>重要:只有跑在「同一批题」上的运行才放进同一张对比图</b> —— 不同题集(全100题 / 随机10题 / 小demo)解出率不可直接比,分开呈现以免误导。');
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
  charts+=svgBars('全 Eterna100(100 题):AI 模型 vs 经典基线', items,
   {ymax:1,caption:'同为「全100题」(含大量 >100nt 难题)。<b>灰 = 两个传统(非 AI)算法的「地板线」</b>:rnainverse(ViennaRNA 自带的经典逆折叠)与 local_search(本地搜索);数值为<b>本 harness 预算下的测量</b>,非文献上限(加大重启数会更高)。蓝 = AI 模型。'+(full.length?'':' 任一 AI 模型跑完全 100 题后,其柱会自动并入此图与地板线对比。')});}
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
