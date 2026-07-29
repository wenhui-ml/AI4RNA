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
from pydantic import BaseModel

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


@app.get("/api/surface")
def api_surface():
    """The capability-cost surface: both benchmarks on ONE measured object.

    Capability and efficiency used to be reported as two incomparable multiples ("1.69x capability" vs
    "2.23x more expensive"). They are two readings of one curve,

        S(B) = max over {b_i} of sum_i [1 - (1-p_i)^{b_i}]   s.t.  sum_i b_i c_i <= B

    with p_i the per-trajectory success rate under INDEPENDENT adjudication (three-oracle panel on the
    crossing set, unique-MFE on Eterna) and c_i the cost of one trajectory. Budget is USD at cost.py's
    rate. S_inf = #{p_i > 0} is capability; the budget scale is efficiency; the gap to the flat policy is
    scheduler slack. Curves are precomputed by runs/ceiling/{metric,eterna_analyze}.py; this only reshapes
    them for drawing. Returns None if that branch's runs are absent.
    """
    C = os.path.join(RUNS, "ceiling")
    m = _read_json(os.path.join(C, "metric.json"))
    e = _read_json(os.path.join(C, "eterna_sb.json"))
    if not m:
        return {"surface": None}
    R = m.get("cpu_usd_per_sec", 0.05 / 3600.0)

    def thin(grid, S, n=20):
        k = max(1, len(grid) // n)
        return [[round(float(g), 7), round(float(s), 3)] for g, s in list(zip(grid, S))[::k]]

    cross = {}
    for a, lab in (("on_400", "闸门开 (ShapeKnots 入环)"), ("off_150", "闸门关 · 150 次预算"),
                   ("off_400", "闸门关 · 400 次预算 (证伪对照)")):
        d = m.get(a)
        if not d:
            continue
        cross[a] = {"label": lab, "s_inf": d["s_inf"], "n": 60,
                    "pass_usd": round(d.get("pass_usd", 0.0), 4),
                    "pts": thin([g * R for g in d["grid"]], d["S"])}
    eterna = {}
    if e:
        for k, lab in (("remc_gate|umfe", "REMC + 唯一-MFE 闸门"), ("remc_plain|umfe", "REMC 原样")):
            d = e.get(k)
            if not d:
                continue
            eterna[k.split("|")[0]] = {"label": lab, "s_inf": d["s_inf"], "n": 100,
                                       "pass_usd": round(float(sum(d["c"])) * R, 4),
                                       "pts": thin([g * R for g in d["grid"]], d["S"])}
    return {"surface": {
        "cpu_usd_per_sec": R, "cross": cross, "eterna": eterna,
        # DesiRNA under the SAME panel, from cc_ascent/desirna_seqs/panel_always_sota.json:
        # 43/60 under pKiss alone, 0/60 unanimous, at 180 s x 10 replicas x 60 targets.
        "sota": {"label": "DesiRNA (SOTA)", "s_inf": 0, "n": 60, "pass_usd": 1.50,
                 "note": "pKiss 下 43/60,三 oracle 一致 0/60,预算是我们的 1.6 倍"},
        "fixed": [["3 / 60", 0.0021, 0.0022, "0.96×"], ["6 / 60", 0.0079, 0.0069, "1.15×"],
                  ["8 / 60", 0.0168, 0.0115, "1.47×"], ["10 / 60", 0.0355, 0.0176, "2.02×"],
                  ["12 / 60", 0.9705, 0.0266, "36.5×"], ["15 / 60", None, 0.0520, "∞"],
                  ["20 / 60", None, 0.1760, "∞"]],
        "slack": [["闸门关", 3.25, 10.39, "3.20×"], ["闸门开", 5.50, 17.42, "3.17×"]],
        "unit": [["一次求解尝试 (加闸门)", 0.00155, "≈ 一次小的 LLM 调用"],
                 ["一次 LLM 调用 2k/0.5k", 0.00175, "≈ 一次求解尝试"],
                 ["一个三 oracle 一致解", 0.01696, "≈ 一次 20k token 调用"],
                 ["一次 LLM 调用 20k/4k", 0.01600, "≈ 一个完整的解"],
                 ["跑完 60 靶交叉基准一遍", 0.09325, "≈ 一个 agent episode"]],
        "caveats": [
            "Eterna 闸门臂被混淆:StrictGate 在任何内部失败时都重试,不只在 umfe 拒绝时。平均 2.27 次尝试但只有 0.08 次拒绝,800 次里仅 45 次真的发生拒绝 —— 增益大部分来自多重启,不是闸门。交叉基准那个没有这个问题(同预算证伪臂零差异,p=1)。",
            "S_inf 由 8 个 seed 估出,是下界:真实 p=0.10 的靶在 8 个 seed 上全输的概率是 0.43。两个坐标采样需求相反(S_inf 要少靶多 seed,成本尺度要多靶少 seed),一次扫描估不好两个。",
            "两个基准、一个算子族、短靶。本分支约 31 次假设检验中三个过 Bonferroni:闸门 p=7.6e-06、轨迹剂量反应 p=2e-04 / 1e-03、Eterna 集合目标 p=6.5e-12。轨迹臂的解数比值(p=0.0225)过不了。",
        ]}}


@app.get("/api/rssd")
def api_rssd():
    """RSSD experiment board, organised by the three mainlines (categories) with key-focus headlines.
    Reads the custom result JSONs each rssd_* experiment writes; missing files -> status 'pending'.
    Everything is computed server-side into render-ready strings so the UI just draws cards."""
    def rj(run, fname):
        return _read_json(os.path.join(RUNS, run, fname))

    def x_cheaper(sota, c):
        return f"{sota / c:.0f}×" if c and c > 0 else "—"

    # ---- category C: agent self-authoring (north-star) ----
    agent_rows = []
    for label, run in (("人工基线 v1", None), ("gpt-4o", "rssd_agent_openai_gpt-4o"),
                       ("GLM-5.2 (关推理)", "rssd_agent_z-ai_glm-5.2"),
                       ("GLM-5.2 (开推理)", "rssd_agent_glm_reason_on"),
                       ("GLM-5.2 (Eterna/路线A)", "rssd_agent_nested")):
        if run is None:
            agent_rows.append({"model": label, "skill": "—", "held": "96/125", "x": "13×",
                               "discovered": "是(基准)", "note": "手写技能"})
            continue
        d = rj(run, "agent_result.json")
        if not d:
            agent_rows.append({"model": label, "status": "pending"}); continue
        h, s = d.get("held", {}), d.get("held_always_sota", {})
        sota_c = s.get("cost") or 0
        code = ""
        try:
            code = open(os.path.join(RUNS, run, "authored_skill.py")).read()
        except OSError:
            pass
        disc = "✅ 发现交叉互补" if "fix_crossing" in code or "s_list[i] = 'G'" in code else \
               ("✅ 发现方法" if (h.get("skill_solved", 0) or 0) > 0.3 * (h.get("n") or 1) else "❌ 未发现")
        agent_rows.append({
            "model": label,
            "skill": f"{h.get('skill_solved','?')}/{h.get('n','?')}",
            "held": f"{h.get('solved','?')}/{h.get('n','?')}",
            "x": x_cheaper(sota_c, h.get("cost")),
            "discovered": disc,
            "note": f"弃权丢 {h.get('solves_lost','?')} 解" if run != "rssd_agent_nested" else "路线A:嵌套域"})

    # ---- category B: efficiency (real threshold) ----
    B = []
    comp = rj("rssd_compounding", "compounding.json")
    if comp:
        h = (comp.get("arms") or {}).get("held", {})
        sota = (h.get("always_sota") or {}).get("cost") or 0
        ab = h.get("abstain") or {}
        best = min((v for v in ab.values() if v.get("solves_lost") == 0),
                   key=lambda v: v.get("cost", 9e9), default=None)
        orc = h.get("oracle_abstain") or {}
        B.append({"name": "B1/G1 留出集 + 弃权消融", "purpose": "学到的路由在同质量下降本?来自哪个部件?",
                  "result": (f"弃权 {x_cheaper(sota, best['cost'])} @ 0损失 (便宜级联仅 "
                             f"{x_cheaper(sota, (h.get('cold') or {}).get('cost'))});oracle 上界 "
                             f"{x_cheaper(sota, orc.get('cost'))}") if best else "—",
                  "status": "done"})
    st = rj("rssd_stream", "stream.json")
    if st:
        B.append({"name": "F4 在线流式复利", "purpose": "冷启动边学边降本(时间维度)?",
                  "result": f"cost/solve 跌 {st.get('drop_pct',0):.0f}%,累计 "
                            f"{x_cheaper(st.get('cum_sota'), st.get('cum_stream'))}(丢 {st.get('lost_per_stream',0):.1f}/流)",
                  "status": "done"})
    ec = rj("rssd_eterna_cost", "eterna_cost.json")
    if ec:
        h = ec.get("held_result", {})
        sota = (h.get("always_sota") or {}).get("cost") or 0
        casc = (h.get("cascade") or {}).get("cost")
        B.append({"name": "F2 Eterna 跨域成本", "purpose": "成本论题跨数据集成立?",
                  "result": f"留出集 {x_cheaper(sota, casc)} 便宜、0 丢失(n={ec.get('n_puzzles','?')})",
                  "status": "done"})
    vc = rj("rssd_verifier_cost", "verifier_cost.json")
    if vc and vc.get("results"):
        res, ids = vc["results"], vc.get("ids", [])
        scr = vc.get("config", {}).get("screens", [])
        def agg(scv):
            sv = sum(1 for i in ids if res.get(f"{i}:{scv}", {}).get("solved"))
            ow = sum(res.get(f"{i}:{scv}", {}).get("oracle_s", 0) for i in ids)
            return sv, ow
        base = agg(-1) if -1 in scr else (None, None)
        parts = []
        for scv in scr:
            sv, ow = agg(scv)
            parts.append(f"{'无筛' if scv < 0 else '≤%d' % scv}:{sv}解/{ow:.0f}s")
        B.append({"name": "F5 验证器成本路由", "purpose": "便宜 oracle 预筛砍贵 oracle 成本?",
                  "result": " | ".join(parts) + "(弱杠杆:砍成本但丢几解)", "status": "done"})

    # ---- category A: capability ceiling ----
    A = []
    des = rj("rssd_designability", "designability.json")
    if des:
        rec, stuck = des.get("recovered", []), des.get("stuck", [])
        A.append({"name": "E1 可设计性证书", "purpose": "71 残差是搜索限制还是真墙?",
                  "result": f"{len(rec)}/{len(rec)+len(stuck)} 被 5× 重搜索救回 → "
                            f"{'全是可设计性墙,天花板 180/251(认证)' if not rec else '部分可救'}",
                  "status": "done"})
    con = rj("rssd_constrained", "constrained.json")
    if con:
        cracked, broke = con.get("residual_cracked", []), con.get("solved_broke", [])
        A.append({"name": "E2 约束设计", "purpose": "丢了 IUPAC 约束是否低估了 180?",
                  "result": f"约束救回残差 {len(cracked)}/71,搞坏已解 {len(broke)}/20 → "
                            f"{'180 不是低估' if not cracked else '约束能救 %d' % len(cracked)}",
                  "status": "done"})
    bud = rj("rssd_eterna_budget", "budget_curve.json")
    if bud:
        sa = bud.get("solved_at", {})
        hard = bud.get("hard", [])
        seq = " → ".join(f"{len(sa.get(str(b), sa.get(b, [])))}" for b in (bud.get("config", {}).get("budgets") or [60, 600, 1800]))
        A.append({"name": "E5 Eterna 86→97 预算曲线", "purpose": "到 97 是不是纯预算?",
                  "result": f"硬集 {len(hard)} 题:逐级加预算解出 {seq};剩 {len(bud.get('still_unsolved',[]))} 需数小时 → 差距主要是预算",
                  "status": "done"})

    return {"rssd": {"categories": [
        {"id": "C", "title": "🌟 北极星 · agent 自写方法",
         "headline": "把'写方法'交给 LLM agent:GLM-5.2 自发现假结命门(交叉互补),留出集 91/125、62× 便宜;gpt-4o 不行 → 方法发现是模型能力问题。",
         "table": agent_rows},
        {"id": "B", "title": "⚙️ 效率 · 真门槛(同质量降本)",
         "headline": "学到的编排/弃权在同质量下省 8–62× 算力、随时间复利(31.6×)、跨域成立(2.7×)→ 认知核不是壳,真门槛已过。",
         "experiments": B},
        {"id": "A", "title": "🎯 能力 · 天花板认证",
         "headline": "假结 71 残差是可设计性墙(重搜索+约束都救不回);Eterna 86→97 差距主要是预算。",
         "experiments": A},
    ]}}


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


class ChatIn(BaseModel):
    message: str
    budget: Optional[str] = None


@app.post("/api/chat")
def api_chat(inp: ChatIn):
    """The product entry point: a request in words (or dot-bracket, or a sequence to check) -> a CERTIFICATE.

    Routing is deliberate and cheap to explain:
      * a dot-bracket string            -> design for it
      * a sequence AND a dot-bracket    -> check that sequence (free, no search)
      * anything else                   -> the intent layer translates it, and code assembles the structure
    Default budget is 'cheap' (one 5 s tier): the measured curve answers about 72% of targets there in ~9 s
    for about $0.002, which is the right default for an interactive box. A caller can ask for more.
    """
    from ..cognitive.certify import (BUDGETS, certify_design, certify_sequence, classify, render,
                                     to_display)
    msg = (inp.message or "").strip()
    if not msg:
        return {"kind": "error", "text": "empty message"}
    # answer in the language the question was asked in; a Chinese question answered in English undoes the
    # accessibility this layer exists for
    lang = "zh" if any("\u4e00" <= ch <= "\u9fff" for ch in msg) else "en"
    budget = inp.budget if inp.budget in BUDGETS else "cheap"

    toks = [t for t in msg.replace(",", " ").split() if t]
    db = [t for t in toks if set(t) <= set("().[]{}<>") and any(c in t for c in "([{<")]
    sq = [t for t in toks if len(t) > 3 and set(t.upper()) <= set("ACGUT")]

    if db and sq:
        # A sequence AND a structure means "check this", always. Falling through to design when the lengths
        # disagree would silently answer a DIFFERENT question -- the exact failure this product exists to
        # prevent -- so a mismatch is reported as a mismatch.
        c = certify_sequence(sq[0].upper().replace("T", "U"), db[0])
        return {"kind": "check", "lang": lang, "text": render(c),
                "display": to_display(c, lang), "certificate": c.to_dict()}
    if db and not sq:
        kind, err = classify(db[0])
        if err:
            return {"kind": "error", "text": f"that dot-bracket is not valid: {err}"}
        c = certify_design(db[0], budget=budget)
        return {"kind": "design", "lang": lang, "text": render(c),
                "display": to_display(c, lang), "certificate": c.to_dict()}

    from ..cognitive.intent import design_from_request, render as irender
    spec, target, c = design_from_request(msg, budget=budget)
    if spec.error:
        hint = ""
        if "translation unavailable" in (spec.error or ""):
            hint = ("\n\nThe natural-language layer needs a model API key (OPENROUTER_API_KEY). Without one "
                    "you can still use this box by pasting a dot-bracket structure, e.g. ((((....)))) , or a "
                    "sequence and its target together to check it.")
        return {"kind": "error", "text": f"{spec.error}{hint}", "spec": spec.to_dict()}
    return {"kind": "design", "lang": lang, "text": irender(spec, target, c),
            "spec": spec.to_dict(), "target": target,
            "display": (to_display(c, lang) if c else None),
            "understood": spec.interpretation,
            "certificate": (c.to_dict() if c else None)}


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
 body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;margin:0;background:var(--bg);color:var(--text);
      display:flex;flex-direction:column;min-height:100vh}
 header{background:var(--card);padding:14px 22px;border-bottom:1px solid var(--border);position:sticky;top:0;z-index:5;display:flex;align-items:center;justify-content:space-between}
 h1{font-size:18px;margin:0} .sub{color:var(--muted);font-size:12px;margin-left:8px}
 .lnk{background:var(--accent);color:#fff!important;padding:7px 14px;border-radius:6px;font-weight:700;font-size:13px}
 .intro{background:var(--card);border-bottom:1px solid var(--border);padding:12px 22px;font-size:13.5px;line-height:1.7}
 .tabs{margin:12px 22px 0} .tab{display:inline-block;padding:7px 15px;cursor:pointer;border:1px solid var(--border);border-bottom:none;border-radius:8px 8px 0 0;background:#eef1f5;color:var(--muted);margin-right:3px;font-size:14px}
 .tab.active{background:var(--card);color:var(--accent);font-weight:700}
 .panel{margin:0 22px 30px;border:1px solid var(--border);border-radius:0 10px 10px 10px;padding:18px;background:var(--card);
        flex:1 1 auto;min-height:0}
 /* the chat is an app pane, not a document: the log takes whatever height is left and the composer stays put */
 #chat{display:flex;flex-direction:column}
 #chatlog{flex:1 1 auto;min-height:220px;overflow:auto}
 #chatrow{flex:0 0 auto}
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
 <span class="tab active" data-t="chat" onclick="tab('chat')">💬 对话</span>
 <span class="tab" data-t="scorecard" onclick="tab('scorecard')">🏆 成绩单</span>
 <span class="tab" data-t="surface" onclick="tab('surface')">🎯 能力-成本平面</span>
 <span class="tab" data-t="rssd" onclick="tab('rssd')">🧬 认知核/RSSD</span>
 <span class="tab" data-t="campaigns" onclick="tab('campaigns')">📈 记忆实验</span>
 <span class="tab" data-t="overview" onclick="tab('overview')">🧭 概览</span>
 <span class="tab" data-t="runs" onclick="tab('runs')">📂 运行</span>
 <span class="tab" data-t="figures" onclick="tab('figures')">📊 图表</span>
 <span class="tab" data-t="logs" onclick="tab('logs')">📜 日志(实时)</span>
</div>
<div id="chat" class="panel">
 <p class="note">用一句话描述你要的 RNA 结构,或直接粘贴 dot-bracket;也可以同时给"序列 + 结构"来检查一条已有序列。
 返回的是<b>证书</b>而不是序列:谁判的、过了哪一级、花了多少、以及它<b>不</b>主张什么。</p>
 <div id="chatlog" style="border:1px solid var(--border);border-radius:10px;padding:14px;background:#fff"></div>
 <div id="chatrow" style="display:flex;gap:8px;margin-top:10px">
  <input id="chatin" placeholder="例:一个 6 对茎、4 nt 环的发夹    或    ((((....))))" 
         style="flex:1;padding:10px 12px;border:1px solid var(--border);border-radius:8px;font-size:14px"
         onkeydown="if(event.key==='Enter')send()">
  <select id="chatbudget" style="padding:10px;border:1px solid var(--border);border-radius:8px">
   <option value="cheap">便宜(5s 档)</option><option value="standard">标准(5/15/60/180s)</option>
   <option value="best_effort">尽力(加 600/1800s)</option>
  </select>
  <button onclick="send()" style="padding:10px 18px;border:0;border-radius:8px;background:#2563eb;color:#fff;cursor:pointer">发送</button>
 </div>
</div>
<div id="scorecard" class="panel" style="display:none"></div>
<div id="surface" class="panel" style="display:none"></div>
<div id="rssd" class="panel" style="display:none"></div>
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
 ['chat','scorecard','surface','rssd','campaigns','overview','runs','figures','logs'].forEach(p=>document.getElementById(p).style.display=p==t?'block':'none');}
function esc(t){const d=document.createElement('div'); d.textContent=t==null?'':t; return d.innerHTML;}
function bubble(html,accent){const d=document.createElement('div'); d.style.cssText='margin:12px 0;display:flex';
 const b=document.createElement('div');
 b.style.cssText='max-width:88%;padding:13px 16px;border-radius:12px;background:#fff;border:1px solid var(--border);'
  +'font-size:14px;line-height:1.7'+(accent?';border-left:5px solid '+accent:'');
 b.innerHTML=html; d.appendChild(b);
 const L=document.getElementById('chatlog'); L.appendChild(d); L.scrollTop=L.scrollHeight; return b;}
function mine(txt){const d=document.createElement('div'); d.style.cssText='margin:12px 0;display:flex;justify-content:flex-end';
 const b=document.createElement('div');
 b.style.cssText='max-width:80%;padding:10px 14px;border-radius:12px;background:#2563eb;color:#fff;font-size:14px';
 b.textContent=txt; d.appendChild(b);
 const L=document.getElementById('chatlog'); L.appendChild(d); L.scrollTop=L.scrollHeight;}
function copySeq(el){const s=el.dataset.seq; navigator.clipboard.writeText(s).then(()=>{
 const o=el.textContent; el.textContent='已复制'; setTimeout(()=>el.textContent=o,1200);});}
const VCOL={certified:'#16a34a',knot_confirmed:'#65a30d',cannot_certify:'#d97706',
            not_found:'#dc2626',invalid_input:'#dc2626'};
function card(j){
 const d=j.display; const zh=(j.lang||'zh')=='zh';
 if(!d) return {html:'<b>'+esc(zh?'没能完成':'could not complete')+'</b><br>'+esc(j.text||''),accent:'#dc2626'};
 const col=VCOL[d.verdict]||'#94a3b8'; let h='';
 h+='<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">'
   +'<span style="background:'+col+';color:#fff;padding:3px 11px;border-radius:20px;font-weight:600;font-size:13px">'
   +esc(d.headline)+'</span>';
 if(d.plain) h+='<span style="color:var(--muted);font-size:13px">'+esc(d.plain)+'</span>';
 h+='</div>';
 if(j.understood) h+='<div style="color:var(--muted);font-size:13px;margin-bottom:8px">'
   +esc(zh?'理解为:':'understood as: ')+esc(j.understood)+'</div>';
 if(d.sketch) h+='<pre style="margin:10px 0;padding:12px 14px;background:#0f172a;color:#e2e8f0;border-radius:8px;'
   +'font-size:13px;line-height:1.35;overflow-x:auto">'+esc(d.sketch)+'</pre>';
 if(d.sequence){h+='<div style="margin:10px 0;padding:10px 12px;background:#f8fafc;border:1px solid var(--border);border-radius:8px">'
   +'<div style="font-size:12px;color:var(--muted);margin-bottom:4px">'+esc(zh?'设计出的序列':'sequence')+'</div>'
   +'<code style="font-size:14px;letter-spacing:.5px;word-break:break-all">'+esc(d.sequence)+'</code>'
   +' <span onclick="copySeq(this)" data-seq="'+esc(d.sequence)+'" style="cursor:pointer;color:#2563eb;font-size:12px;margin-left:8px">'
   +esc(zh?'复制':'copy')+'</span></div>';}
 if(d.target) h+='<div style="font-size:12px;color:var(--muted)">'+esc(zh?'目标结构式：':'target: ')
   +'<code>'+esc(d.target)+'</code></div>';
 if(d.criterion){h+='<div style="margin-top:10px"><b>'+esc(zh?'通过的判据':'criterion')+'</b>：'
   +'<code>'+esc(d.criterion.code)+'</code>'
   +(d.criterion.is_strongest?' <span style="color:#16a34a;font-size:12px">'+esc(zh?'（三级中最强）':'(strongest rung)')+'</span>':'')
   +'<div style="color:var(--muted);font-size:13px;margin-top:3px">'+esc(d.criterion.means)+'</div>';
  if(d.criterion.also&&d.criterion.also.length)
   h+='<div style="color:var(--muted);font-size:12.5px">'+esc(zh?'同时也满足：':'also passes: ')
     +d.criterion.also.map(esc).join('；')+'</div>';
  h+='</div>';}
 if(d.operating_point) h+='<div style="margin-top:8px;font-size:12.5px;color:var(--muted)">'
   +esc(d.operating_point.text)+(d.operating_point.scope?'<br>'+esc(d.operating_point.scope):'')+'</div>';
 if(d.why) h+='<div style="margin-top:10px;font-size:13px"><b>'+esc(zh?'为什么':'why')+'</b>：'+esc(d.why)+'</div>';
 h+='<div style="margin-top:10px;font-size:12.5px;color:var(--muted)">'
   +esc(zh?'用了 ':'cost ')+'$'+(d.cost_usd||0).toFixed(5)+esc(zh?'，等待 ':', waited ')
   +(d.wait_s||0).toFixed(1)+'s'+(d.checked_by?esc(zh?'，判定者:':', checked by ')+esc(d.checked_by):'')+'</div>';
 if(d.audit) h+='<details style="margin-top:10px"><summary style="font-size:12.5px;color:var(--muted);cursor:pointer">'
   +esc(zh?'可复现性 / 审查信息':'reproducibility')+'</summary>'
   +'<div style="font-size:12px;color:var(--muted);line-height:1.7;margin-top:6px">'
   +esc(zh?'生成时间：':'generated ')+esc(d.audit.generated_utc||'')+'　'+esc(zh?'随机种子：':'seed ')
   +esc(String(d.audit.seed))+'　'+esc(d.audit.schema||'')+'<br>'
   +Object.entries(d.audit.versions||{}).filter(([k])=>!['schema'].includes(k))
     .map(([k,v])=>esc(k)+' '+esc(v)).join('　')
   +(d.audit.verify_cmd?'<br>'+esc(zh?'复核这条结论（确定性，任何人任何时候结果相同）：':'verify (deterministic): ')
     +'<br><code style="font-size:11.5px;word-break:break-all">'+esc(d.audit.verify_cmd)+'</code>'
     +' <span onclick="copySeq(this)" data-seq="'+esc(d.audit.verify_cmd)+'" style="cursor:pointer;color:#2563eb">'
     +esc(zh?'复制':'copy')+'</span>':'')
   +'<br>'+esc(zh?'重跑搜索（随机，可能给出另一条序列）：':'re-run search (stochastic): ')
   +'<br><code style="font-size:11.5px;word-break:break-all">'+esc(d.audit.rerun_search||'')+'</code>'
   +'<div style="margin-top:5px;color:#7c2d12">'+esc(d.audit.reproducibility||'')+'</div>'
   +'</div></details>';
 if(d.note) h+='<div style="margin-top:10px;padding:8px 11px;background:#fff7ed;border-left:3px solid #f59e0b;'
   +'border-radius:6px;font-size:12.5px;color:#7c2d12">'+esc(d.note)+'</div>';
 return {html:h,accent:col};}
async function send(){const i=document.getElementById('chatin'); const m=i.value.trim(); if(!m)return;
 i.value=''; mine(m);
 const w=bubble('<span style="color:var(--muted)">正在设计并验证…（便宜档约 9 秒）</span>','#94a3b8');
 try{
  const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({message:m,budget:document.getElementById('chatbudget').value})});
  const j=await r.json(); const c=card(j);
  w.innerHTML=c.html; w.parentElement.firstChild.style.borderLeft='5px solid '+c.accent;
 }catch(e){w.innerHTML='<span style="color:#dc2626">请求失败: '+esc(String(e))+'</span>';}}
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
 try{const rs=(await (await fetch('/api/rssd')).json()).rssd; renderRSSD(rs);}catch(e){document.getElementById('rssd').innerHTML=note('RSSD 数据加载失败');}
 try{const sf=(await (await fetch('/api/surface')).json()).surface; renderSurface(sf);}catch(e){document.getElementById('surface').innerHTML=note('平面数据加载失败');}
 renderFigures();
}
// ---- capability-cost surface: one curve, two orthogonal coordinates -------------------------
const SC={s1:'#2a78d6',s2:'#eb6834',s3:'#1baf7a',neg:'#c62828',gl:'#e7e5df',tx:'#52514e'};
function usd(v){return v>=1?'$'+v.toFixed(2):v>=0.01?'$'+v.toFixed(3):'$'+v.toFixed(5)}
function svgLine(series,ymax,ylab,W,H){
 const M={t:14,r:230,b:44,l:54},pw=W-M.l-M.r,ph=H-M.t-M.b;
 const all=series.flatMap(s=>s.pts), x0=Math.log10(Math.min(...all.map(p=>p[0]))),
       x1=Math.log10(Math.max(...all.map(p=>p[0])));
 const X=v=>M.l+(Math.log10(v)-x0)/(x1-x0)*pw, Y=v=>M.t+ph-v/ymax*ph;
 let g='';
 for(let e=Math.ceil(x0);e<=x1;e++){const x=X(Math.pow(10,e));
  g+=`<line x1="${x}" x2="${x}" y1="${M.t}" y2="${M.t+ph}" stroke="${SC.gl}"/>`+
     `<text x="${x}" y="${H-24}" text-anchor="middle" font-size="11" fill="${SC.tx}">${usd(Math.pow(10,e))}</text>`;}
 for(let k=0;k<=4;k++){const v=ymax*k/4,y=Y(v);
  g+=`<line x1="${M.l}" x2="${M.l+pw}" y1="${y}" y2="${y}" stroke="${SC.gl}"/>`+
     `<text x="${M.l-9}" y="${y+4}" text-anchor="end" font-size="11" fill="${SC.tx}">${Math.round(v)}</text>`;}
 g+=`<text x="${M.l+pw/2}" y="${H-6}" text-anchor="middle" font-size="11.5" fill="${SC.tx}">整个基准的预算 (美元, 对数轴)</text>`+
    `<text x="13" y="${M.t+ph/2}" text-anchor="middle" font-size="11.5" fill="${SC.tx}" transform="rotate(-90 13 ${M.t+ph/2})">${ylab}</text>`;
 series.forEach(s=>{const d=s.pts.map((p,i)=>(i?'L':'M')+X(p[0])+' '+Y(p[1])).join(' ');
  const L=s.pts[s.pts.length-1];
  g+=`<path d="${d}" fill="none" stroke="${s.c}" stroke-width="2" stroke-linejoin="round"/>`+
     `<circle cx="${X(L[0])}" cy="${Y(L[1])}" r="4" fill="${s.c}" stroke="#fff" stroke-width="2"/>`+
     `<text x="${X(L[0])+10}" y="${Y(L[1])+4}" font-size="12" font-weight="600" fill="#1a1a2e">${s.label} · ${L[1].toFixed(0)}</text>`;});
 return `<svg width="${W}" height="${H}" role="img" aria-label="${ylab}">${g}</svg>`;
}
function renderSurface(sf){
 const el=document.getElementById('surface');
 if(!sf){el.innerHTML=note('本分支尚无 ceiling 运行数据(需要 runs/ceiling/metric.json)');return;}
 const cr=sf.cross,et=sf.eterna;
 let h=`<div class="card"><h2>🎯 能力-成本平面</h2>`+
  note('把"效果"和"效率"放在<b>同一个被测对象</b>上,不再是两个不可比的倍数。曲线 <code>S(B)=max Σ[1-(1-p_i)^b_i]</code>,预算约束 <code>Σ b_i c_i ≤ B</code>。'+
       '<b>S<sub>∞</sub>=#{p_i>0}</b> 是能力,预算尺度是效率,离"每靶一条轨迹"的差距是调度余量。所有解都由<b>独立裁决</b>(交叉基准用三 oracle panel,Eterna 用唯一-MFE),没有任何一臂能给自己打分。')+`</div>`;

 // ---- the plane
 const P=[];
 if(cr.off_150) P.push({x:cr.off_150.pass_usd,y:cr.off_150.s_inf/60*100,c:SC.s2,lab:'交叉 · 闸门关',n:`${cr.off_150.s_inf}/60`});
 if(cr.off_400) P.push({x:cr.off_400.pass_usd,y:cr.off_400.s_inf/60*100,c:SC.s3,lab:'证伪对照 · 2.5倍预算',n:`${cr.off_400.s_inf}/60`});
 if(cr.on_400)  P.push({x:cr.on_400.pass_usd, y:cr.on_400.s_inf/60*100, c:SC.s1,lab:'交叉 · ShapeKnots 闸门',n:`${cr.on_400.s_inf}/60`});
 if(et.remc_plain) P.push({x:et.remc_plain.pass_usd,y:et.remc_plain.s_inf,c:SC.s2,lab:'Eterna · REMC 原样',n:`${et.remc_plain.s_inf}/100`});
 if(et.remc_gate)  P.push({x:et.remc_gate.pass_usd, y:et.remc_gate.s_inf, c:SC.s1,lab:'Eterna · 加闸门',n:`${et.remc_gate.s_inf}/100`});
 P.push({x:sf.sota.pass_usd,y:0,c:SC.neg,lab:'DesiRNA (SOTA) · 交叉',n:'0/60'});
 const W=980,H=420,M={t:20,r:230,b:48,l:58},pw=W-M.l-M.r,ph=H-M.t-M.b;
 const lx=v=>M.l+(Math.log10(v)-Math.log10(0.02))/(Math.log10(3)-Math.log10(0.02))*pw, ly=v=>M.t+ph-v/70*ph;
 let g='';
 [0.02,0.05,0.1,0.3,1,3].forEach(e=>{const x=lx(e);
  g+=`<line x1="${x}" x2="${x}" y1="${M.t}" y2="${M.t+ph}" stroke="${SC.gl}"/>`+
     `<text x="${x}" y="${H-26}" text-anchor="middle" font-size="11" fill="${SC.tx}">$${e}</text>`;});
 for(let v=0;v<=70;v+=10){const y=ly(v);
  g+=`<line x1="${M.l}" x2="${M.l+pw}" y1="${y}" y2="${y}" stroke="${SC.gl}"/>`+
     `<text x="${M.l-9}" y="${y+4}" text-anchor="end" font-size="11" fill="${SC.tx}">${v}%</text>`;}
 g+=`<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#9aa0a6"/></marker></defs>`;
 const arw=(a,b,dash)=>`<line x1="${lx(a.x)}" y1="${ly(a.y)}" x2="${lx(b.x)}" y2="${ly(b.y)}" stroke="#9aa0a6" stroke-width="2" marker-end="url(#ah)"${dash?' stroke-dasharray="5 4"':''}/>`;
 if(P.length>=3){g+=arw(P[0],P[1],true)+arw(P[0],P[2],false);}
 if(P.length>=5){g+=arw(P[3],P[4],false);}
 P.forEach(p=>{g+=`<circle cx="${lx(p.x)}" cy="${ly(p.y)}" r="7" fill="${p.c}" stroke="#fff" stroke-width="2"/>`+
   `<text x="${lx(p.x)+12}" y="${ly(p.y)+4}" font-size="12" font-weight="600" fill="#1a1a2e">${p.lab} · ${p.n}</text>`;});
 g+=`<text x="${M.l+pw/2}" y="${H-6}" text-anchor="middle" font-size="11.5" fill="${SC.tx}">跑完一遍基准的成本 (美元, 对数轴)</text>`+
    `<text x="14" y="${M.t+ph/2}" text-anchor="middle" font-size="11.5" fill="${SC.tx}" transform="rotate(-90 14 ${M.t+ph/2})">天花板:基准中可达的比例</text>`;
 h+=`<div class="card"><h3>1 · 每个改动把我们推向哪里</h3>`+
   note('纵轴是<b>能力天花板</b>,横轴是跑一遍基准的<b>成本</b>。一支箭头是一个改动:<b>竖直向上</b>买到能力,<b>水平向右</b>只是花钱。虚线那支是<b>证伪对照</b>—— 同一个算子给 2.5 倍 oracle 预算,只往右不往上,这就是"多给算力买不到东西"的直接证据。')+
   `<div style="overflow-x:auto"><svg width="${W}" height="${H}">${g}</svg></div></div>`;

 // ---- curves
 const cs=[];
 if(cr.on_400) cs.push({pts:cr.on_400.pts,c:SC.s1,label:'闸门开'});
 if(cr.off_150)cs.push({pts:cr.off_150.pts,c:SC.s2,label:'闸门关·150'});
 if(cr.off_400)cs.push({pts:cr.off_400.pts,c:SC.s3,label:'闸门关·400'});
 h+=`<div class="card"><h3>2 · 交叉基准:两条曲线会交叉</h3>`+
   note('低于约 $0.0013 便宜的系统更好,高于它加闸门的系统更好,而且差距<b>无上界</b>。'+
        '这就是为什么"1.69倍能力"和"贵 2.23 倍"看起来像互相矛盾的两件事 —— 它们是<b>同一条曲线读在不同位置</b>。')+
   `<div style="overflow-x:auto">${svgLine(cs,24,'三 oracle 一致解 (共 60)',980,360)}</div></div>`;
 const es=[];
 if(et.remc_gate) es.push({pts:et.remc_gate.pts,c:SC.s1,label:'加闸门'});
 if(et.remc_plain)es.push({pts:et.remc_plain.pts,c:SC.s2,label:'原样'});
 if(es.length) h+=`<div class="card"><h3>3 · Eterna100:天花板输在长度上</h3>`+
   note('和交叉基准不同,Eterna <b>几乎一发就饱和</b>:8 个 seed 只比一发多 1.1–1.2 倍(交叉基准是 3.7 倍)。'+
        '天花板丢在长靶上 —— 250nt 以上只有 9/25 可达,250nt 以下是 53/75。')+
   `<div style="overflow-x:auto">${svgLine(es,70,'唯一-MFE 解 (共 100)',980,330)}</div></div>`;

 // ---- tables
 const fx=sf.fixed.map(r=>`<tr><td>${r[0]}</td><td>${r[1]==null?'达不到':usd(r[1])}</td><td>${usd(r[2])}</td><td class="g">${r[3]}</td></tr>`).join('');
 h+=`<div class="card"><h3>4 · 达到固定质量目标的成本</h3>`+
  note('同一条曲线的<b>效率读法</b>。优势随质量目标增长,在弱系统的天花板处发散。')+
  `<table><tr><th>解出的靶</th><th>闸门关</th><th>闸门开</th><th>倍数</th></tr>${fx}</table></div>`;
 const sl=sf.slack.map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td class="y">${r[3]}</td></tr>`).join('');
 h+=`<div class="card"><h3>5 · 第三个坐标:调度余量</h3>`+
  note('"每靶一条轨迹"离<b>同预算下的最优分配</b>有多远。两个系统数值几乎相同,所以调度和能力是<b>经验上独立</b>的。轨迹实验已实现其中 1.6–2.6 倍,<b>剩下的还没拿</b>。')+
  `<table><tr><th>系统</th><th>平摊策略</th><th>同预算最优</th><th>余量</th></tr>${sl}</table></div>`;
 const un=sf.unit.map(r=>`<tr><td>${r[0]}</td><td>${usd(r[1])}</td><td>${r[2]}</td></tr>`).join('');
 h+=`<div class="card"><h3>6 · 一美元买到什么</h3>`+
  note('<b>规模比倍数更重要。</b>在基准规模上 CPU 几乎免费,主导成本是 R&amp;D 循环的 token;在生产规模上反过来。'+
       '本分支所有效率倍数都测在<b>第二种</b>规模,而项目实际处在<b>第一种</b>。每靶调一次 LLM 加 14%(可行),每次尝试调一次加 113%(不可行)。')+
  `<table><tr><th>单位</th><th>成本</th><th>等价于</th></tr>${un}</table></div>`;
 h+=`<div class="card"><h3>7 · 不声称什么</h3>`+
  sf.caveats.map(c=>`<p class="note" style="border-left-color:#c62828;background:#fdf0f0">${c}</p>`).join('')+`</div>`;
 el.innerHTML=h;
}
function renderRSSD(rs){
 const el=document.getElementById('rssd'); if(!rs){el.innerHTML=note('暂无 RSSD 数据');return;}
 let h=`<div class="card"><h2>🧬 认知核 / RSSD 实验板</h2>`+
   note('把 RNA 求解器当工具,验证"经济地编排 + agent 自发现方法"。三条主线:🌟 北极星(agent 自写方法)、⚙️ 效率(同质量降本)、🎯 能力(天花板)。全部防自欺:训练/评估不相交,pKiss/MFE 亲自复验。')+`</div>`;
 (rs.categories||[]).forEach(c=>{
   h+=`<div class="card"><h3>${c.title}</h3><p class="note" style="font-size:13px;color:#333"><b>重点:</b>${c.headline}</p>`;
   if(c.table){ // 北极星对比表
     h+=`<table style="width:100%"><tr><th>方法作者</th><th>技能单独解出</th><th>留出集总解出</th><th>vs always-SOTA</th><th>是否自发现方法</th><th>备注</th></tr>`;
     c.table.forEach(r=>{ if(r.status=='pending'){h+=`<tr><td>${r.model}</td><td colspan="5" class="y">运行中/待出</td></tr>`;return;}
       const dc=(r.discovered||'').indexOf('✅')>=0?'g':((r.discovered||'').indexOf('❌')>=0?'r':'');
       h+=`<tr><td><b>${r.model}</b></td><td>${r.skill}</td><td>${r.held}</td><td class="g">${r.x}</td><td class="${dc}">${r.discovered}</td><td class="sub">${r.note||''}</td></tr>`;});
     h+=`</table>`;
   }
   (c.experiments||[]).forEach(e=>{
     const st=e.status=='done'?'<span class="g">✓ 完成</span>':'<span class="y">运行中</span>';
     h+=`<div style="margin:8px 0;padding:8px 10px;border-left:3px solid var(--accent);background:#f8fafc">
       <b>${e.name}</b> ${st}<br><span class="sub">目的:${e.purpose}</span><br><span style="font-size:13px">结果:${e.result}</span></div>`;
   });
   h+=`</div>`;
 });
 el.innerHTML=h;
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
