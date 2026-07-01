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
    for base in (os.path.join(RUNS, run, "figures"), RESULTS):
        p = os.path.join(base, fname)
        if os.path.exists(p) and _safe_under(p, [RUNS, RESULTS]):
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


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


HTML_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>openclaude_rna dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:#0f1419;color:#e6edf3}
 header{background:#161b22;padding:14px 20px;border-bottom:1px solid #30363d;position:sticky;top:0;z-index:5}
 h1{font-size:18px;margin:0;display:inline-block} .sub{color:#8b949e;font-size:12px;margin-left:10px}
 .tabs{margin:10px 20px 0} .tab{display:inline-block;padding:6px 14px;cursor:pointer;border:1px solid #30363d;border-bottom:none;border-radius:6px 6px 0 0;background:#161b22;color:#8b949e}
 .tab.active{background:#0d1117;color:#e6edf3;font-weight:600}
 .panel{margin:0 20px 30px;border:1px solid #30363d;border-radius:0 8px 8px 8px;padding:16px;background:#0d1117}
 .card{border:1px solid #30363d;border-radius:8px;padding:12px 14px;margin:10px 0;background:#161b22}
 .name{font-weight:700;color:#58a6ff} table{border-collapse:collapse;margin:6px 0;font-size:13px}
 td,th{border:1px solid #30363d;padding:4px 10px;text-align:left} th{background:#161b22}
 .g{color:#3fb950}.r{color:#f85149}.y{color:#d29922} img{max-width:520px;border:1px solid #30363d;border-radius:6px;margin:8px 8px 0 0;background:#fff}
 pre{background:#010409;border:1px solid #30363d;border-radius:6px;padding:10px;overflow:auto;max-height:60vh;font-size:12px;line-height:1.4}
 select,button{background:#161b22;color:#e6edf3;border:1px solid #30363d;border-radius:6px;padding:5px 8px}
 .kv{display:inline-block;margin-right:16px}
</style></head><body>
<header><h1>🧬 openclaude_rna</h1><span class="sub">RNA inverse-design meta-learning harness — live dashboard</span>
 <span id="health" class="sub"></span></header>
<div class="tabs">
 <span class="tab active" data-t="scorecard" onclick="tab('scorecard')">🏆 Scorecard</span>
 <span class="tab" data-t="campaigns" onclick="tab('campaigns')">📈 Campaign</span>
 <span class="tab" data-t="overview" onclick="tab('overview')">Overview</span>
 <span class="tab" data-t="runs" onclick="tab('runs')">Runs</span>
 <span class="tab" data-t="figures" onclick="tab('figures')">Figures</span>
 <span class="tab" data-t="logs" onclick="tab('logs')">Logs (live)</span>
</div>
<div id="scorecard" class="panel"></div>
<div id="campaigns" class="panel" style="display:none"></div>
<div id="overview" class="panel" style="display:none"></div>
<div id="runs" class="panel" style="display:none"></div>
<div id="figures" class="panel" style="display:none"></div>
<div id="logs" class="panel" style="display:none">
 <select id="logsel"></select> <button onclick="loadlog()">refresh</button>
 <label class="sub"><input type="checkbox" id="auto" checked> auto (3s)</label>
 <pre id="logbox">select a log…</pre>
</div>
<script>
let RUNS=[];
function tab(t){document.querySelectorAll('.tab').forEach(e=>e.classList.toggle('active',e.dataset.t==t));
 ['scorecard','campaigns','overview','runs','figures','logs'].forEach(p=>document.getElementById(p).style.display=p==t?'block':'none');}
function fmt(x){return x==null?'—':(typeof x=='number'?(Math.round(x*1000)/1000):x)}
function cls(v){return v==null?'':(v>=0.6?'g':(v>=0.3?'y':'r'))}
async function load(){
 const h=await (await fetch('/health')).json();
 document.getElementById('health').textContent=' • '+h.n_runs+' runs • '+h.repo;
 RUNS=(await (await fetch('/api/runs')).json()).runs;
 renderOverview(); renderRuns(); renderFigures();
 try{const sc=(await (await fetch('/api/scorecard')).json()).scorecards; renderScorecard(sc);}catch(e){}
 try{const cp=(await (await fetch('/api/campaigns')).json()).campaigns; renderCampaigns(cp);}catch(e){}
}
function renderCampaigns(cps){
 const el=document.getElementById('campaigns'); if(!el)return;
 if(!cps||!cps.length){el.innerHTML='<div class="card">No meta-learning campaign yet. Run <code>python -m rna_harness.experiments.campaign</code>.</div>';return;}
 let h='<h3>Meta-learning saturation (phase-2 self-improvement)</h3>';
 for(const c of cps){h+=`<div class="card"><div class="name">${c.name} — ${c.summary.n_episodes} episodes</div>`;
  for(const [arm,curve] of Object.entries(c.summary.curves||{})){
   h+=`<div style="margin:6px 0"><b>${arm}</b><table><tr><th>epoch</th><th>solve</th><th>oracle-calls→solve</th><th>A_memory</th><th>#strategies</th></tr>`+
   curve.map(e=>`<tr><td>${e.epoch}</td><td class="${cls(e.solve_rate)}">${fmt(e.solve_rate)}</td><td class="${arm=='with_memory'?'g':''}">${fmt(e.mean_oracle_calls_solved)}</td><td>${fmt(e.A_memory)}</td><td>${fmt(e.n_strategies)}</td></tr>`).join('')+`</table></div>`;}
  h+=`</div>`;}
 el.innerHTML=h;
}
function renderScorecard(cards){
 if(!cards||!cards.length){document.getElementById('scorecard').innerHTML='<div class="card">No scorecard yet. Run <code>python -m rna_harness.experiments.leaderboard</code>.</div>';return;}
 let h='';
 for(const c of cards){const p=c.scorecard.panels||{};
  h+=`<div class="card"><div class="name">🏆 ${c.name}</div><div class="sub">red line: ${c.scorecard.red_line||''}</div>`;
  const s=p.standard_leaderboard;
  if(s){h+=`<h4>Panel 1 — ${s.benchmark} (${s.n_puzzles} puzzles, budget: ${s.budget})</h4>
   <table><tr><th>method</th><th>solved/100</th><th>solved rate</th><th>uMFE</th><th>mean NED (all)</th><th>mean NED (solved)</th><th>mean s</th></tr>`+
   Object.entries(s.by_method).map(([m,v])=>`<tr><td>${m}</td><td><b>${v.solved}</b></td><td>${fmt(v.solved_rate)}</td><td>${v.umfe}</td><td>${fmt(v.mean_NED_all)}</td><td>${fmt(v.mean_NED_solved)}</td><td>${fmt(v.mean_secs)}</td></tr>`).join('')+`</table>`;}
  const hy=p.hybrid_seam;
  if(hy){h+=`<h4>Panel 2 — hybrid-constraint seam (${hy.n_tasks} tasks / ${hy.n_base} base)</h4>
   <table><tr><th>profile</th><th>solved</th><th>struct match</th><th>constraints ok</th></tr>`+
   Object.entries(hy.by_profile).map(([pr,v])=>`<tr><td>${pr}</td><td class="${cls(v.solved_rate)}">${fmt(v.solved_rate)}</td><td>${fmt(v.struct_match_rate)}</td><td>${fmt(v.constraints_ok_rate)}</td></tr>`).join('')+`</table>`;}
  const ag=p.agent_capability;
  if(ag){const a=ag.actuation_ablation||{};
   h+=`<h4>Panel 3 — agent capability (actuation ablation, ${ag.n_ablation_tasks} tasks)</h4>
   <table><tr><th>mode</th><th>solve</th><th>A</th><th>A_model</th><th>A_harness</th></tr>`+
   Object.entries(a).map(([m,v])=>`<tr><td>${m}</td><td>${fmt(v.solve_rate)}</td><td class="${cls(v.A)}">${fmt(v.A)}</td><td>${fmt(v.A_model)}</td><td>${fmt(v.A_harness)}</td></tr>`).join('')+`</table>`;
   if(ag.meta_arms){h+=`<h4>meta-learning arms (oracle-calls-to-solve, A_memory)</h4>
    <table><tr><th>arm</th><th>solve</th><th>mean evals→solve</th><th>A_memory</th><th>strategies</th></tr>`+
    Object.entries(ag.meta_arms).map(([m,v])=>`<tr><td>${m}</td><td>${fmt(v.solve_rate)}</td><td class="${m=='with_memory'?'g':''}">${fmt(v.mean_evals_solved)}</td><td>${fmt(v.A_memory)}</td><td>${fmt(v.n_strategies_learned)}</td></tr>`).join('')+`</table>`;}}
  h+=`</div>`;}
 document.getElementById('scorecard').innerHTML=h;
}
function renderOverview(){
 let h='<h3>Headline results</h3>';
 for(const r of RUNS){
  const s=r.summary||{}, m=r.meta_summary;
  if(s.ablation){h+=`<div class="card"><div class="name">${r.name} — actuation ablation</div>
   <table><tr><th>mode</th><th>solve</th><th>A</th><th>A_model</th><th>A_harness</th></tr>`+
   Object.entries(s.ablation).map(([k,v])=>`<tr><td>${k}</td><td class="${cls(v.solve_rate)}">${fmt(v.solve_rate)}</td><td class="${cls(v.A)}">${fmt(v.A)}</td><td>${fmt(v.A_model)}</td><td>${fmt(v.A_harness)}</td></tr>`).join('')+`</table>`;
   if(s.coords)h+=`<div class="kv">D_sym <b class="g">${fmt(s.coords.D_sym)}</b></div><div class="kv">R_det <b class="g">${fmt(s.coords.R_det)}</b></div><div class="kv">T_trace <b class="g">${fmt(s.coords.T_trace)}</b></div>`;
   if(s.baselines)h+='<div class="kv">baselines: '+Object.entries(s.baselines).map(([k,v])=>`${k} ${fmt(v.solve_rate_best_of_seeds)}`).join(', ')+'</div>';
   h+=`</div>`;}
  if(m&&m.arms){h+=`<div class="card"><div class="name">${r.name} — meta-learning arms</div>
   <table><tr><th>arm</th><th>solve</th><th>mean evals→solve</th><th>A_memory</th><th>strategies</th></tr>`+
   Object.entries(m.arms).map(([k,v])=>`<tr><td>${k}</td><td>${fmt(v.solve_rate)}</td><td class="${v.arm=='with_memory'?'g':''}">${fmt(v.mean_evals_solved)}</td><td>${fmt(v.A_memory)}</td><td>${fmt(v.n_strategies_learned)}</td></tr>`).join('')+`</table></div>`;}
  if(s.model){h+=`<div class="card"><div class="name">${r.name} — agent (${s.model})</div>
   <div class="kv">solve_rate <b>${fmt(s.solve_rate)}</b></div><div class="kv">mean_reward <b>${fmt(s.mean_reward)}</b></div>
   <div class="kv">agent_A <b class="${cls(s.agent_A)}">${fmt(s.agent_A)}</b></div><div class="kv">A_model <b>${fmt(s.agent_A_model)}</b></div></div>`;}
 }
 document.getElementById('overview').innerHTML=h||'no runs yet';
}
function renderRuns(){
 let h='';
 for(const r of RUNS){h+=`<div class="card"><div class="name">${r.name}</div>
  <div class="sub">files: ${(r.files||[]).join(', ')}</div>`+
  (r.datasets?('<div class="kv">datasets: '+Object.entries(r.datasets).map(([k,v])=>`${k}=${v} rows`).join(', ')+'</div>'):'')+
  (r.n_traces?`<div class="kv">agent traces: ${r.n_traces}</div>`:'')+`</div>`;}
 document.getElementById('runs').innerHTML=h||'no runs';
}
function renderFigures(){
 let h='';
 for(const r of RUNS){for(const f of (r.figures||[])){h+=`<img src="/fig/${encodeURIComponent(r.name)}/${encodeURIComponent(f)}" title="${r.name}/${f}">`;}}
 document.getElementById('figures').innerHTML=h||'no figures yet';
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
