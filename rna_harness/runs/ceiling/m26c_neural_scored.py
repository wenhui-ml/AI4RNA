#!/usr/bin/env python3
"""M26c: score a model that contains no thermodynamics on the same 4317 interventions.

Declaration m26c_neural_scored_declaration.json, frozen and pushed at c22b1ddd BEFORE this file
existed. Offline, $0 -- local weights, local GPU.

M26/M26b covered four parameter sets but one implementation: one ensemble definition, one
pseudoknot treatment, one nearest-neighbour frame. RibonanzaNet-SS shares none of that. It is a
transformer trained on roughly two million chemical-mapping profiles, and its ct_predictor head
emits a per-pair score directly from pairwise features. If the residual survives it too, the
reading moves from "not this folding implementation" to "not either of the two model classes we
evaluated" -- and no further, which the declaration writes down in advance.

The leakage question comes first. RibonanzaNet was trained on Eterna OpenKnot; this corpus IS
OpenKnot 7a. The weights predate the OK7a release by eight months, so the prior is zero, but a
prior is not evidence: L4 (the mutants are new no matter what) runs here, and L1/L2/L3 run in
m26c_leakage.py once the training sequences are on disk.
"""
from __future__ import annotations

import argparse, collections, json, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m26_model_scored as M26

RNET = "/data/wenhui/RibonanzaNet"
WEIGHTS = "/data/wenhui/rnet_weights/RibonanzaNet-SS.pt"
NBINS, NPERM, NPLACEBO = 5, 2000, 12
DELTAS = (0.00, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60)
SELF_MAX, P_GATE = 0.15, 0.005
CACHE = os.path.join(HERE, "m26c_pneural_cache.npz")


# ------------------------------------------------------------------ the neural pair scorer
def load_model():
    """RibonanzaNet backbone + the (1,64) ct_predictor head, as a base-pair scorer."""
    sys.path.insert(0, RNET)
    import torch, torch.nn as nn, yaml
    import Network as NW

    # the repo's Functions.load_config_from_yaml drags in polars; the config is a flat dict.
    class Cfg:
        def __init__(self, d):
            self.__dict__.update(d)

    with open(os.path.join(RNET, "configs", "pairwise.yaml")) as f:
        cfg = Cfg(yaml.safe_load(f))
    cfg.dropout = 0.0

    class SS(NW.RibonanzaNet):
        def __init__(self, config):
            super().__init__(config)
            self.ct_predictor = nn.Linear(config.pairwise_dimension, 1)

        def pair_logits(self, src):
            """mirrors RibonanzaNet.forward but returns the symmetrised pairwise head."""
            B, L = src.shape
            mask = torch.ones(B, L, device=src.device, dtype=src.dtype)
            x = self.encoder(src).reshape(B, L, -1)
            pf = self.outer_product_mean(x)
            pf = pf + self.pos_encoder(x)
            for layer in self.transformer_encoder:
                x, pf = layer(x, pf, mask)
            pf = pf + pf.permute(0, 2, 1, 3)          # symmetrise before the head
            return self.ct_predictor(pf).squeeze(-1)  # B,L,L

    m = SS(cfg)
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    missing, unexpected = m.load_state_dict(sd, strict=False)
    m.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = m.to(dev)
    return m, dev, sorted(missing), sorted(unexpected)


TOK = {"A": 0, "C": 1, "G": 2, "U": 3}


def score_sequences(seqs, log):
    """P_neural[(seq, i, j)] for every sequence, cached on disk so the GPU pass runs once."""
    import torch
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        cache = {k: v for k, v in zip(z["keys"], z["mats"])}
        if all(s in cache for s in seqs):
            log(f"  P_neural 取自缓存 {CACHE}({len(cache)} 条序列)")
            return cache, None, None
    m, dev, missing, unexpected = load_model()
    log(f"  设备 {dev};权重缺失键 {len(missing)},多余键 {len(unexpected)}")
    if missing:
        log(f"  **缺失键样例** {missing[:5]}")
    nonstd = collections.Counter()
    cache, t0 = {}, time.time()
    with torch.no_grad():
        for n, s in enumerate(seqs):
            ids = []
            for ch in s:
                if ch not in TOK:
                    nonstd[ch] += 1
                ids.append(TOK.get(ch, 4))
            src = torch.tensor([ids], dtype=torch.long, device=dev)
            lg = m.pair_logits(src)
            p = torch.sigmoid(lg)[0].float().cpu().numpy().astype(np.float32)
            cache[s] = p
            if (n + 1) % 500 == 0:
                log(f"    {n+1}/{len(seqs)}  {time.time()-t0:.0f}s")
    np.savez_compressed(CACHE, keys=np.array(list(cache.keys()), dtype=object),
                        mats=np.array(list(cache.values()), dtype=object))
    log(f"  非标准字符 {dict(nonstd) if nonstd else '无'}")
    return cache, missing, dict(nonstd)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import random as _r
    LOG = []

    def log(s=""):
        print(s, flush=True); LOG.append(s)

    log("=== M26c:给一个不含热力学的模型记分 ===")
    log("  声明 c22b1ddd。离线 $0(本地权重 + 本地 GPU)。\n")

    # ---------------- P1: byte-identical admission set to M26
    recs, dropped = M26.build()
    prev = json.load(open(os.path.join(HERE, "m26_model_scored.json")))
    OUT = {"n": len(recs), "n_dropped": len(dropped), "m26_n": prev["n"]}
    log("=== P1 纳入集合必须与 M26 逐字节相同 ===")
    log(f"  M26c 纳入 {len(recs)}  M26 纳入 {prev['n']}  排除 {len(dropped)}")
    if len(recs) != prev["n"]:
        OUT["verdict"] = "void_admission_set_differs"
        json.dump(OUT, open(os.path.join(HERE, "m26c_neural_scored.json"), "w"),
                  ensure_ascii=False, indent=1)
        log("  **P1 不过 ⇒ 作废,不许与 M26 的数并列比较。**"); return 1
    log("  **P1 通过**\n")

    # ---------------- neural scoring
    log("=== 神经模型打分 ===")
    seqs = sorted({r["mut_seq"] for r in recs} | {r["wt_seq"] for r in recs})
    log(f"  唯一序列 {len(seqs)},长度 {min(map(len,seqs))}~{max(map(len,seqs))}")
    cache, missing, nonstd = score_sequences(seqs, log)
    OUT["n_unique_seq"] = len(seqs)
    OUT["weights_missing_keys"] = len(missing) if missing is not None else "cached"
    OUT["nonstandard_chars"] = nonstd
    for r in recs:
        i, j = min(r["a"], r["b"]), max(r["a"], r["b"])
        r["nmut"] = float(cache[r["mut_seq"]][i, j])
        r["nwt"] = float(cache[r["wt_seq"]][i, j])
        r["ndp"] = r["nmut"] - r["nwt"]

    # ---------------- P6: symmetry, range, index convention
    log("\n=== P6 对称性 / 取值域 / 索引约定 ===")
    probe = recs[:10]
    asym = max(abs(cache[r["mut_seq"]][r["a"], r["b"]] - cache[r["mut_seq"]][r["b"], r["a"]])
               for r in probe)
    vals = np.array([r["nmut"] for r in recs])
    rng0 = np.random.default_rng(4242)
    off = []
    for r in recs[:400]:
        M = cache[r["mut_seq"]]; L = M.shape[0]
        x, y = rng0.integers(0, L), rng0.integers(0, L)
        if abs(int(x) - int(y)) > 3:
            off.append(float(M[min(x, y), max(x, y)]))
    on_mean, off_mean = float(vals.mean()), float(np.mean(off))
    log(f"  最大不对称 {asym:.2e}(须 <1e-5)")
    log(f"  取值域 [{vals.min():.4f}, {vals.max():.4f}]  SD {vals.std():.4f}")
    log(f"  目标配对均分 {on_mean:.4f}  vs  随机位置对 {off_mean:.4f}  比 {on_mean/max(off_mean,1e-9):.1f}×")
    OUT["P6"] = {"max_asym": float(asym), "range": [float(vals.min()), float(vals.max())],
                 "sd": float(vals.std()), "on_target_mean": round(on_mean, 4),
                 "off_target_mean": round(off_mean, 4)}
    if asym > 1e-5 or vals.max() > 1.0 or vals.min() < 0.0 or vals.std() < 1e-4:
        OUT["verdict"] = "K6_output_is_not_a_probability"
        json.dump(OUT, open(os.path.join(HERE, "m26c_neural_scored.json"), "w"),
                  ensure_ascii=False, indent=1)
        log("  **K6 触发:先查头的用法与对称化,修正前不许报任何留存数。**"); return 1
    if on_mean <= off_mean:
        log("  **shape_2 预警:目标配对分数不高于随机位置 —— 索引约定可疑,N1 必须连同这条一起读。**")
    log("  **P6 通过**\n")

    sb = lambda r: (r["design"], min(r["depth"], 3), r["crossed"])

    # ---------------- N5 first: are the two models even different?
    log("=== N5 两个模型是不是同一类 ===")
    pv = np.array([r["pmut"] for r in recs]); nv = np.array([r["nmut"] for r in recs])
    pear = float(np.corrcoef(pv, nv)[0, 1])
    rp, rn = pv.argsort().argsort(), nv.argsort().argsort()
    spear = float(np.corrcoef(rp, rn)[0, 1])
    OUT["N5"] = {"pearson": round(pear, 4), "spearman": round(spear, 4)}
    log(f"  P_vienna vs P_neural   Pearson {pear:+.4f}  Spearman {spear:+.4f}")
    same_class = max(abs(pear), abs(spear)) > 0.95
    OUT["N5_same_class"] = bool(same_class)
    log(f"  **{'shape_3:相关>0.95 ⇒ 不构成独立的第二类,N2 降级' if same_class else '两模型不同类 ⇒ N2 可读作跨模型类'}**\n")

    # ---------------- N3 calibration
    log("=== N3 神经模型标定(描述性)===")
    edges = [float(np.quantile(nv, q)) for q in np.linspace(0, 1, NBINS + 1)[1:-1]]
    for r in recs:
        r["nbin"] = int(np.searchsorted(edges, r["nmut"], side="right"))
    by = collections.defaultdict(list)
    for r in recs:
        by[sb(r)].append(r)
    cen = {}
    for _s, rs in by.items():
        mu = float(np.mean([x["rf"] for x in rs]))
        for x in rs:
            cen[id(x)] = x["rf"] - mu
    cal = []
    for k in range(NBINS):
        s = [r for r in recs if r["nbin"] == k]
        cal.append({"bin": k, "n": len(s),
                    "p_mean": round(float(np.mean([r["nmut"] for r in s])), 4),
                    "rf_mean": round(float(np.mean([r["rf"] for r in s])), 4),
                    "rf_centred": round(float(np.mean([cen[id(r)] for r in s])), 4)})
        log(f"  箱{k}: n={len(s):4d}  P_neural {cal[-1]['p_mean']:.4f}  "
            f"rescue {cal[-1]['rf_mean']:+.4f}  层内去均值 {cal[-1]['rf_centred']:+.4f}")
    mono = all(cal[i]["rf_centred"] <= cal[i + 1]["rf_centred"] for i in range(NBINS - 1))
    OUT["N3"] = {"bins": cal, "monotone": bool(mono), "edges": [round(e, 4) for e in edges]}
    log(f"  **单调 {'是' if mono else '否'}**\n")

    # ---------------- N1 skill, and L4 leakage backstop in the same table
    log("=== N1 神经模型有没有技能(并列 L4:WT 与突变体分开看)===")
    n1 = {}
    for xk, nm in (("nmut", "P_neural(突变体)"), ("nwt", "P_neural(WT)"),
                   ("ndp", "ΔP_neural = 突变体−WT")):
        s, n = M26.slope_in(recs, sb, xk)
        p = M26.slope_perm_p(recs, sb, xk, s, 700 + abs(hash(xk)) % 997, nperm=500)
        n1[xk] = {"slope": round(s, 5), "n": n, "p_perm": round(p, 5)}
        log(f"  rescue 对 {nm:22s} 层内斜率 {s:+.5f}  置换 p {p:.5f}  n={n}")
    OUT["N1"] = n1
    vslope = prev["D1"]["pmut"]["slope"]
    log(f"  ViennaRNA 同一臂的斜率 {vslope:+.5f}  ⇒ 神经模型 "
        f"{'更高' if n1['nmut']['slope'] > vslope else '更低'}(押注 (3) "
        f"{'对' if n1['nmut']['slope'] > vslope else '错'})")
    log("  **L4:WT 与突变体上的技能相当 ⇒ 泄漏不是解释;差距悬殊 ⇒ 必须查泄漏。**")
    OUT["L4"] = {"skill_mut": n1["nmut"]["slope"], "skill_wt": n1["nwt"]["slope"],
                 "vienna_skill": vslope}
    skill = n1["nmut"]["p_perm"] <= P_GATE and n1["nmut"]["slope"] > 0
    OUT["K3_no_skill"] = bool(not skill)
    log(f"  **{'神经模型有技能 ⇒ N2 可解读' if skill else 'K3:无技能 ⇒ N2 不许读成【超出模型】'}**\n")

    # ---------------- N2 primary
    log("=== N2 主臂:固定神经模型的预测之后,身份还剩多少 ===")
    sd = lambda r: (r["design"], min(r["depth"], 3), r["crossed"], r["nbin"])
    d0, w0, _ = M26.matched(recs, "CG", "AU", sb)
    d1v, w1, _ = M26.matched(recs, "CG", "AU", sd)
    if d1v is None or w1 < 100:
        OUT["verdict"] = "void_no_overlap"
        json.dump(OUT, open(os.path.join(HERE, "m26c_neural_scored.json"), "w"),
                  ensure_ascii=False, indent=1)
        log("  K4 触发,重叠不足,不放宽 caliper。"); return 0
    keep = M26.retained_recs(recs, "CG", "AU", sd)
    dsame, wsame, _ = M26.matched(keep, "CG", "AU", sb)
    null = np.sort(np.array([M26.perm(recs, "CG", "AU", sd, _r.Random(a.seed + i)) or 0.0
                             for i in range(NPERM)]))
    pv1 = (float((null >= d1v).sum()) + 1) / (NPERM + 1)
    ci = M26.cluster_ci(recs, "CG", "AU", sd, a.seed)
    ret = d1v / dsame
    log(f"  全体 · 无模型维      {d0:+.4f}  权重 {w0:.0f}")
    log(f"  留存子集 · 无模型维  {dsame:+.4f}  权重 {wsame:.0f}   <== 同群体分母(R2)")
    log(f"  **CDE(固定 P_neural) {d1v:+.4f}**  权重 {w1:.0f}  置换 p {pv1:.5f}  零SD {null.std():.4f}")
    log(f"  **同群体留存 {100*ret:.1f}%**   经神经模型中介 = {100*(1-ret):+.1f}%")
    log(f"  cluster bootstrap {ci}")

    # ---------------- P2 placebo
    byp = collections.defaultdict(list)
    for r in recs:
        byp[r["puzzle"]].append(r)
    pl = []
    for k in range(NPLACEBO):
        rng = np.random.default_rng(9100 + k)
        sh = []
        for _q, rs in byp.items():
            vals2 = [r["nmut"] for r in rs]
            ix = rng.permutation(len(vals2))
            for r, j in zip(rs, ix):
                sh.append({**r, "nbin": int(np.searchsorted(edges, vals2[j], side="right"))})
        kp = M26.retained_recs(sh, "CG", "AU", sd)
        v, _w, _s = M26.matched(sh, "CG", "AU", sd)
        b, _w2, _s2 = M26.matched(kp, "CG", "AU", sb)
        if v is not None and b:
            pl.append(v / b)
    pl = np.array(pl); pm, ps = float(pl.mean()), float(pl.std(ddof=1))
    z = (ret - pm) / (ps or 1.0)
    med = 1 - ret; mci = [med - 1.96 * ps, med + 1.96 * ps]
    log(f"  P2 伪中介留存 {100*pm:.1f}% ± {100*ps:.1f}%   真值 z = {z:+.2f}")
    log(f"  ⇒ 经神经模型中介的比例 = {100*med:+.1f}% [{100*mci[0]:+.1f}%, {100*mci[1]:+.1f}%]")
    OUT["N2"] = {"cde": round(d1v, 4), "same_pop_baseline": round(dsame, 4),
                 "all_no_model_dim": round(d0, 4), "retention": round(ret, 4),
                 "p_perm": round(pv1, 5), "cluster_ci": ci, "weight": w1,
                 "placebo_mean": round(pm, 4), "placebo_sd": round(ps, 4), "z": round(z, 2),
                 "mediated": round(med, 4), "mediated_ci": [round(x, 4) for x in mci]}

    # ---------------- P3 ladder on the null
    log("\n=== P3 梯级(建在零上)===")
    sub = [r for r in recs if r["pt"] in ("CG", "AU")]
    bysd = collections.defaultdict(list)
    for r in sub:
        bysd[sd(r)].append(r)
    thr = float(np.quantile(null, .995))
    lad = {}
    for dlt in DELTAS:
        hits = 0
        for t in range(20):
            rng = np.random.default_rng(5000 + t)
            inj = []
            for _s, rs in bysd.items():
                y = np.array([r["rf"] for r in rs], float)
                y = y[rng.permutation(len(y))]
                for r, v in zip(rs, y):
                    inj.append({**r, "rf": float(v) + (dlt if r["pt"] == "CG" else 0.0)})
            v2, _w, _s2 = M26.matched(inj, "CG", "AU", sd)
            hits += (v2 is not None and v2 >= thr)
        lad[dlt] = hits / 20
    OUT["P3_ladder"] = {str(k): v for k, v in lad.items()}
    log("  " + "  ".join(f"{k:.3f}:{v:.2f}" for k, v in lad.items()))
    detect = min((k for k, v in lad.items() if v >= 0.8), default=None)
    log(f"  δ=0 自检 {lad[0.0]:.2f}(须<={SELF_MAX})  检出阈值 {detect}")
    if lad[0.0] > SELF_MAX:
        OUT["verdict"] = "K5_ladder_self_check_failed"
        json.dump(OUT, open(os.path.join(HERE, "m26c_neural_scored.json"), "w"),
                  ensure_ascii=False, indent=1)
        log("  **K5 触发,作废。**"); return 1

    # ---------------- N4 joint conditioning on BOTH models
    log("\n=== N4 同时固定两类模型的预测 ===")
    pe3 = [float(np.quantile(pv, q)) for q in (1 / 3, 2 / 3)]
    ne3 = [float(np.quantile(nv, q)) for q in (1 / 3, 2 / 3)]
    for r in recs:
        r["vb3"] = int(np.searchsorted(pe3, r["pmut"], side="right"))
        r["nb3"] = int(np.searchsorted(ne3, r["nmut"], side="right"))
    sj = lambda r: (r["design"], min(r["depth"], 3), r["crossed"], r["vb3"], r["nb3"])
    dj, wj, _ = M26.matched(recs, "CG", "AU", sj)
    if dj is None or wj < 100:
        log(f"  K4:联合分层后权重和 {wj if wj else 0:.0f} < 100 ⇒ 该臂作废,不放宽。")
        OUT["N4"] = {"verdict": "void_no_overlap", "weight": (wj or 0)}
    else:
        kj = M26.retained_recs(recs, "CG", "AU", sj)
        bj, wbj, _ = M26.matched(kj, "CG", "AU", sb)
        nullj = np.sort(np.array([M26.perm(recs, "CG", "AU", sj, _r.Random(a.seed + 3000 + i)) or 0.0
                                  for i in range(NPERM)]))
        pj = (float((nullj >= dj).sum()) + 1) / (NPERM + 1)
        rj = dj / bj
        log(f"  留存子集 · 无模型维  {bj:+.4f}  权重 {wbj:.0f}")
        log(f"  **CDE(同时固定 P_vienna 与 P_neural) {dj:+.4f}**  权重 {wj:.0f}  置换 p {pj:.5f}")
        log(f"  **同群体留存 {100*rj:.1f}%**   两类模型合起来中介 = {100*(1-rj):+.1f}%")
        OUT["N4"] = {"cde": round(dj, 4), "same_pop_baseline": round(bj, 4),
                     "retention": round(rj, 4), "p_perm": round(pj, 5), "weight": wj}

    # ---------------- verdict
    if OUT.get("K3_no_skill"):
        verdict = "neural_model_has_no_skill_here"
    elif ret >= 0.5 and pv1 <= P_GATE and abs(z) <= 1.96:
        verdict = "identity_beyond_two_model_classes"
    elif ret <= 0.2:
        verdict = "neural_model_absorbs_identity"
    else:
        verdict = "intermediate_report_as_is"
    OUT["verdict"] = verdict
    stakes = {
        "1_L1_exact_zero": "deferred_to_m26c_leakage",
        "2_N1_positive_skill": bool(skill),
        "3_neural_slope_beats_vienna": bool(n1["nmut"]["slope"] > vslope),
        "4_N3_monotone": bool(mono),
        "5_N2_retention_ge_50": bool(ret >= 0.5),
        "6_mediated_ci_contains_zero": bool(mci[0] <= 0 <= mci[1]),
        "7_N5_corr_below_095": bool(not same_class),
        "8_N4_joint_retention_ge_40": (bool(OUT["N4"].get("retention", 0) >= 0.4)
                                       if "retention" in OUT["N4"] else "void"),
    }
    OUT["stakes"] = stakes
    log(f"\n=== 裁定: {verdict} ===")
    log("押注逐条: " + json.dumps(stakes, ensure_ascii=False))
    json.dump(OUT, open(os.path.join(HERE, "m26c_neural_scored.json"), "w"),
              ensure_ascii=False, indent=1)
    open(os.path.join(HERE, "m26c_neural_scored.log"), "w").write("\n".join(LOG) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
