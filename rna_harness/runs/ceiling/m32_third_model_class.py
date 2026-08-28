#!/usr/bin/env python3
"""M32: score a THIRD model class on the same 4317 interventions.

Declaration m32_third_model_class_declaration.json, frozen and pushed at 614b8d15 BEFORE this
file existed. Offline, $0 -- a locally built EternaFold binary, CPU only.

A networked review of the manuscript found its heaviest defect: it scores two predictors and
concludes about two model CLASSES. EternaFold splits the confound those two leave whole. Its
feature family is CONTRAfold's, which is the nearest-neighbour frame in learned form; its
parameters are fit to Eterna chemical-mapping data rather than to optical melting. So:

  if the identity separation belongs to being NEURAL      -> EternaFold behaves like ViennaRNA
  if it belongs to being FIT TO PROBING DATA              -> EternaFold behaves like RibonanzaNet

Staked S2 says the first. K2 says that if the second happens, "neural versus thermodynamic" is
refused everywhere including the abstract. The expensive outcome is written down first.

P1 is the probe that matters most: this script recomputes ViennaRNA's identity contrast through
its own code path and requires it to equal m26v's to 4 decimals. Without that, a new number from
a new script is not comparable to the old ones and none of this may be reported.
"""
from __future__ import annotations

import collections, json, os, subprocess, sys, tempfile, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, "/data/wenhui/openclaude_rna/rna_harness/tools")
import m26_model_scored as M26
import m31_which_flips_rescue as M31

EF = "/tmp/claude-1000/-data-wenhui-openclaude-rna/6e34da6a-b0bb-4401-a731-1ad021750f1c/scratchpad/EternaFold"
BIN = os.path.join(EF, "src", "contrafold")
PARAMS = os.path.join(EF, "parameters", "EternaFoldParams.v1")
CACHE = os.path.join(HERE, "m32_eterna_cache.npz")     # 42 MB, regenerable in ~6 min, gitignored
# Provenance of the third predictor, so a reader can rebuild the exact binary:
#   git clone https://github.com/eternagame/EternaFold.git   (commit 87b9aac5, 2026-06-16)
#   cd EternaFold/src && make            -> ./contrafold  (md5 5e1306e54be09835a383c1d3c41573dc)
#   parameters/EternaFoldParams.v1                          md5 febc7352ae048d9edd1ba541a3d0b9cc
EF_PROVENANCE = {
    "repo": "https://github.com/eternagame/EternaFold.git",
    "commit": "87b9aac55cee14fd562049d08f7b92d3131f10ce",
    "commit_date": "2026-06-16",
    "params_md5": "febc7352ae048d9edd1ba541a3d0b9cc",
    "binary_md5": "5e1306e54be09835a383c1d3c41573dc",
    "build": "cd src && make",
}
CUTOFF = "0.0001"          # posteriors below this are not emitted; censoring is counted, not hidden
CHUNK = 250
LOG = []


def log(s=""):
    print(s, flush=True); LOG.append(s)


# ---------------------------------------------------------------- EternaFold posteriors
def parse_post(path):
    """the posterior file: one line per position, `i base j:p j:p ...`, 1-based. -> dict[(i,j)]=p"""
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            i = int(parts[0])
            for tok in parts[2:]:
                j, p = tok.split(":")
                j = int(j)
                if i < j:
                    out[(i, j)] = float(p)
    return out


def score_sequences(seqs):
    """P(i,j) for every unique sequence. Cached, because the corpus does not change."""
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        # np.load returns an object array whose elements are already dicts on some numpy
        # builds and 0-d arrays on others. M32's own run wrote the cache and never read it,
        # so this branch was untested until M32r exercised it.
        cached = {str(k): (v.item() if hasattr(v, "item") else v)
                  for k, v in zip(z["keys"], z["maps"])}
        if all(s in cached for s in seqs):
            log(f"  缓存命中 {len(seqs)} 条,跳过重算")
            return cached
    log(f"  EternaFold 打分 {len(seqs)} 条唯一序列(cutoff {CUTOFF})")
    res, t0 = {}, time.time()
    for c0 in range(0, len(seqs), CHUNK):
        chunk = seqs[c0:c0 + CHUNK]
        din, dout = tempfile.mkdtemp(), tempfile.mkdtemp()
        names = []
        for i, s in enumerate(chunk):
            n = f"q{i}.fa"; names.append(n)
            open(os.path.join(din, n), "w").write(f">q{i}\n{s}\n")
        r = subprocess.run([BIN, "predict", "--params", PARAMS, "--posteriors", CUTOFF, dout]
                           + [os.path.join(din, n) for n in names],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"contrafold failed: {r.stderr[-400:]}")
        for s, n in zip(chunk, names):
            fp = os.path.join(dout, n)
            res[s] = parse_post(fp) if os.path.exists(fp) else {}
        if c0 % (CHUNK * 4) == 0:
            log(f"    {c0 + len(chunk)}/{len(seqs)}  {time.time() - t0:.0f}s")
    np.savez_compressed(CACHE, keys=np.array(list(res), dtype=object),
                        maps=np.array([res[k] for k in res], dtype=object))
    log(f"  完成,用时 {time.time() - t0:.0f}s,写入 {os.path.basename(CACHE)}")
    return res


def boot_ci(vals, lo=0.025, hi=0.975):
    return M31.boot_ci(vals, lo, hi)


# ---------------------------------------------------------------- identity contrast, M26's estimator
def contrast_arm(recs, key, seed, nperm=2000, nboot=2000):
    sb = lambda r: (r["design"], min(r["depth"], 3), r["crossed"])
    obs, w, used = M26.matched(recs, "CG", "AU", sb, key=key)
    pz = sorted({r["puzzle"] for r in recs})
    byp = collections.defaultdict(list)
    for r in recs:
        byp[r["puzzle"]].append(r)
    rng = np.random.default_rng(seed); bs = []
    for _ in range(nboot):
        ix = rng.integers(0, len(pz), len(pz))
        s = [r for q in ix for r in byp[pz[q]]]
        v, den, _ = M26.matched(s, "CG", "AU", sb, key=key)
        if v is not None and den:
            bs.append(v)
    ci = boot_ci(bs)
    rp = np.random.default_rng(seed + 1); ge = 0
    for _ in range(nperm):
        v = M26.perm(recs, "CG", "AU", sb, rp, key=key)
        if v is not None and abs(v) >= abs(obs):
            ge += 1
    return {"effect": round(obs, 5), "ci": ci, "p_perm": (ge + 1) / (nperm + 1),
            "weight": float(w), "n_strata": used}


def main():
    log("=== M32:第三个模型类 —— EternaFold(学出来的最近邻框架)===\n")
    OUT = {"binary": BIN, "params": os.path.basename(PARAMS), "cutoff": float(CUTOFF),
           "eternafold_provenance": EF_PROVENANCE}

    # P0 -- the binary must fold a known hairpin before it touches the corpus
    log("--- P0 二进制健全性 ---")
    # with ONE input file contrafold writes the posteriors to that PATH as a file; with several it
    # treats the path as a directory. Give it two so the directory branch is the one under test.
    d, o = tempfile.mkdtemp(), tempfile.mkdtemp()
    open(os.path.join(d, "h.fa"), "w").write(">h\nGGGGAAAACCCC\n")
    open(os.path.join(d, "h2.fa"), "w").write(">h2\nGGGGAAAACCCC\n")
    subprocess.run([BIN, "predict", "--params", PARAMS, "--posteriors", "0.001", o,
                    os.path.join(d, "h.fa"), os.path.join(d, "h2.fa")],
                   capture_output=True, text=True)
    hp = parse_post(os.path.join(o, "h.fa"))
    p_close = hp.get((1, 12), 0.0)
    log(f"  GGGGAAAACCCC 的 1-12 配对后验 {p_close:.4f}")
    OUT["P0_hairpin_close"] = round(p_close, 4)
    if p_close < 0.5:
        log("  !! P0 未通过:二进制折不出一个已知发夹,后面全部作废")
        OUT["VOID"] = "P0_failed"
        json.dump(OUT, open(os.path.join(HERE, "m32_third_model_class.json"), "w"),
                  indent=2, ensure_ascii=False)
        return 1

    recs, _ = M26.build()
    log(f"\n纳入 {len(recs)} 次干预")
    seqs = sorted({r["wt_seq"] for r in recs} | {r["mut_seq"] for r in recs})
    post = score_sequences(seqs)

    # attach. EternaFold emits nothing below CUTOFF; that is a floor, and it is counted.
    cens_m = cens_w = 0
    for r in recs:
        i, j = min(r["a"], r["b"]) + 1, max(r["a"], r["b"]) + 1
        em = post[r["mut_seq"]].get((i, j))
        ew = post[r["wt_seq"]].get((i, j))
        cens_m += em is None; cens_w += ew is None
        r["emut"] = 0.0 if em is None else em
        r["ewt"] = 0.0 if ew is None else ew
        r["edp"] = r["emut"] - r["ewt"]
    cov = 1.0 - cens_m / len(recs)
    log(f"\n--- S1 覆盖率 ---")
    log(f"  突变体在 cutoff 之上的比例 {cov:.4f}(被截断 {cens_m}),野生型被截断 {cens_w}")
    OUT["S1_coverage"] = round(cov, 4)
    OUT["censored_mut"] = cens_m; OUT["censored_wt"] = cens_w
    if cov < 0.95:
        log("  !! K1 触发:覆盖率 < 95%,本次运行作废,任何一臂都不许解读")
        OUT["VOID"] = "K1_coverage"
        json.dump(OUT, open(os.path.join(HERE, "m32_third_model_class.json"), "w"),
                  indent=2, ensure_ascii=False)
        return 1

    # ---- P1: the same estimator must reproduce ViennaRNA's published contrast
    log("\n--- P1 估计量等价(本脚本重算 ViennaRNA,必须对上 m26v 的四位小数)---")
    ref = json.load(open(os.path.join(HERE, "m26v_alpha_verify.json")))
    here_v = contrast_arm(recs, "pmut", seed=3201, nperm=200, nboot=300)
    ref_v = ref["pmut"]["m26"]
    dv = abs(here_v["effect"] - ref_v)
    log(f"  m26v {ref_v:+.5f}   本脚本 {here_v['effect']:+.5f}   差 {dv:.2e}")
    OUT["P1_equivalence"] = {"m26v": ref_v, "here": here_v["effect"], "abs_diff": dv,
                             "passed": bool(dv < 1e-4)}
    if dv >= 1e-4:
        log("  !! K4 触发:新代码路径不是旧估计量,本次运行不许报告任何数")
        OUT["VOID"] = "K4_estimator_drift"
        json.dump(OUT, open(os.path.join(HERE, "m32_third_model_class.json"), "w"),
                  indent=2, ensure_ascii=False)
        return 1
    log("  P1 通过")

    # ---- A: the identity contrast, EternaFold
    log("\n=== A 身份对照(配平后 C-G 减 A-U),三个模型并列 ===")
    A = {}
    for nm, key in (("measured_rescue", "rf"), ("P_vienna", "pmut"), ("P_eterna", "emut")):
        A[nm] = contrast_arm(recs, key, seed=3210)
        e = A[nm]
        log(f"  {nm:16s} {e['effect']:+.5f}  CI {e['ci']}  置换 p {e['p_perm']:.5f}  "
            f"权重 {e['weight']:.0f}  层 {e['n_strata']}")
    A["P_neural_from_m26v"] = {"effect": ref["nmut"]["m26"], "ci": ref["nmut"]["ci"]}
    log(f"  {'P_neural(取自 m26v)':16s} {ref['nmut']['m26']:+.5f}  CI {ref['nmut']['ci']}")
    OUT["A_identity_contrast"] = A

    # S2 / K2: which of the two does EternaFold look like?
    e_eff, v_eff, n_eff = A["P_eterna"]["effect"], A["P_vienna"]["effect"], ref["nmut"]["m26"]
    e_ci = A["P_eterna"]["ci"]
    same_sign_as_vienna = (e_eff < 0) == (v_eff < 0)
    separates = e_ci[0] is not None and e_ci[0] > 0        # measured direction is POSITIVE
    log(f"\n--- S2 / K2 ---")
    log(f"  ViennaRNA {v_eff:+.5f} | EternaFold {e_eff:+.5f} | RibonanzaNet {n_eff:+.5f}")
    log(f"  与 ViennaRNA 同号 {same_sign_as_vienna};区间整体为正(按实测方向分离){separates}")
    OUT["S2"] = {"same_sign_as_vienna": bool(same_sign_as_vienna),
                 "separates_in_measured_direction": bool(separates),
                 "held": bool(same_sign_as_vienna and not separates)}
    OUT["K2_fires"] = bool(separates)
    if separates:
        log("  !! K2 触发:EternaFold 按实测方向分离。『神经 vs 热力学』的说法必须全文改写"
            "为『学自探针数据 vs 拟合自光学熔解』,摘要在内。")

    # ---- B: the within-design ranking arm, via M31's own estimator
    log("\n=== B 设计格内排名(用 M31 的估计量,加进 EternaFold 两列)===")
    M31.PREDS = (("P_vienna", "pmut"), ("P_neural", "nmut"), ("depth_baseline", "depth"),
                 ("P_eterna", "emut"), ("dP_eterna", "edp"))
    z = np.load(os.path.join(HERE, "m26c_pneural_cache.npz"), allow_pickle=True)
    pn = {k: v for k, v in zip(z["keys"], z["mats"])}
    for r in recs:
        i, j = min(r["a"], r["b"]), max(r["a"], r["b"])
        r["nmut"] = float(pn[r["mut_seq"]][i, j])
    cells = M31.make_cells(recs)
    log(f"  {len(cells)} 个设计格,{sum(len(v) for _k, v in cells)} 次干预")
    B = M31.arm_A(cells, "格内", seed=3220)
    OUT["B_within_design"] = B

    # ---- C: EternaFold minus the free baseline, with an interval ON THE DIFFERENCE (rule 68)
    log("\n=== C 减去免费基线,区间打在差上(第 68 条)===")
    ry = M31.ry_map(cells)
    pz = sorted({rs[0]["puzzle"] for _k, rs in cells})
    bypz = collections.defaultdict(list)
    for k, rs in cells:
        bypz[rs[0]["puzzle"]].append((k, rs))
    rng = np.random.default_rng(3230); dif = []
    for _ in range(2000):
        ix = rng.integers(0, len(pz), len(pz))
        sub = [c for q in ix for c in bypz[pz[q]]]
        a, _ = M31.weighted_rho(sub, "emut", ry)
        b, _ = M31.weighted_rho(sub, "depth", ry)
        if a is not None and b is not None:
            dif.append(a - b)
    ci_d = boot_ci(dif)
    pt = B["P_eterna"]["rho"] - B["depth_baseline"]["rho"]
    contains0 = ci_d[0] is not None and ci_d[0] <= 0 <= ci_d[1]
    log(f"  EternaFold {B['P_eterna']['rho']:+.5f} - 深度 {B['depth_baseline']['rho']:+.5f} "
        f"= {pt:+.5f}   CI {ci_d}   含 0 {contains0}")
    OUT["C_vs_depth"] = {"diff": round(pt, 5), "ci_diff": ci_d, "contains_zero": bool(contains0)}
    OUT["S4"] = {"held": bool(contains0)}
    OUT["K3_fires"] = bool(not contains0 and ci_d[0] is not None and ci_d[0] > 0)

    # ---- D: how much of EternaFold's ordering IS depth (the m31s quantity)
    log("\n=== D 它排出来的序里有多少就是深度 ===")
    rdep = {k: M31.cz(M31.rankdata([r["depth"] for r in rs])) for k, rs in cells}
    gap, _ = M31.weighted_rho(cells, "emut", rdep)
    log(f"  与深度的格内秩一致 {gap:+.5f};与结果的 {B['P_eterna']['rho']:+.5f};"
        f"差 {abs(gap - B['P_eterna']['rho']):.5f}")
    OUT["D_depth_share"] = {"rho_with_depth": round(gap, 5),
                            "rho_with_outcome": B["P_eterna"]["rho"],
                            "gap": round(abs(gap - B["P_eterna"]["rho"]), 5)}
    OUT["S5"] = {"held": bool(abs(gap - B["P_eterna"]["rho"]) < 0.06)}

    # ---- S3
    lo, hi = sorted((B["P_vienna"]["rho"], B["P_neural"]["rho"]))
    OUT["S3"] = {"held": bool(lo <= B["P_eterna"]["rho"] <= hi),
                 "vienna": B["P_vienna"]["rho"], "eterna": B["P_eterna"]["rho"],
                 "neural": B["P_neural"]["rho"]}

    # ---- P2: EternaFold is not a clean control and the artifact says so
    OUT["P2_leakage_note"] = (
        "EternaFold's parameters were fit to Eterna cloud-lab chemical mapping. This corpus is an "
        "Eterna OpenKnot round. The published EternaFold parameters predate this round, so these "
        "molecules were not in its training data, but the DESIGN POPULATION is the same one and "
        "EternaFold must not be described as innocent of it. It is a third model class, not a "
        "clean-room control.")

    log("\n=== 押注结算 ===")
    for k in ("S1", "S2", "S3", "S4", "S5"):
        v = OUT.get(k)
        if k == "S1":
            ok = cov >= 0.95
        else:
            ok = bool(v and v.get("held"))
        log(f"  {k}  {'成立' if ok else '**未成立**'}")
    OUT["verdict"] = ("third_class_behaves_like_the_nearest_neighbour_frame"
                      if OUT["S2"]["held"] else "REFRAME_REQUIRED_learned_vs_fitted")
    log(f"\n裁定:{OUT['verdict']}")

    json.dump(OUT, open(os.path.join(HERE, "m32_third_model_class.json"), "w"),
              indent=2, ensure_ascii=False)
    open(os.path.join(HERE, "m32_run.out"), "w").write("\n".join(LOG) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
