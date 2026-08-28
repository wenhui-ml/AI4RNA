#!/usr/bin/env python3
"""M38: does a riboSNitch-style mutation-effect score beat the free depth integer?

Declaration m38_ribosnitch_baseline_declaration.json, frozen and pushed BEFORE this file
existed. Offline, $0 -- ViennaRNA 2.7.2 locally on the interventions M26 already builds.
No API, no GPU, no network, no new data.

WHY THIS RUN EXISTS. A literature pass found that scoring a folding algorithm by its predicted
response to a sequence change is an established task in this field (riboSNitch prediction;
Corley et al., NAR 2015). The manuscript's sharpest sentence is that no predictor beats a free
integer -- helix depth -- at ranking which compensatory flips rescue, and the obvious reply is
that none of the three scored predictors was BUILT to answer a question about a mutation. A
riboSNitch-style score was. So it is scored here, on the same interventions, in the same cells,
by the same statistic, against the same free baseline.

WHAT IS AND IS NOT BEING TESTED (declaration K4). The scores below are the standard riboSNitch
quantities computed from the same folding model the paper already scores. They are NOT remuRNA,
RNAsnp or SNPfold, and no sentence may say those tools were tested.

DECLARED DIRECTION, frozen before any number existed: a compensatory flip that RESCUES preserves
the global structure, so a structural-change score should be SMALL where rescue is LARGE. The
sign is NEGATIVE. Every score is negated before ranking, so a positive rho below means the
declaration was right.

The statistics are not reimplemented. M31's cells, weighting, clustered bootstrap and rule-80
within-cell permutation are imported and reused, so this arm cannot diverge from the arm it is
compared against.
"""
from __future__ import annotations

import collections, json, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np
import RNA

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m26_model_scored as M26
import m31_which_flips_rescue as M31

DECL = os.path.join(HERE, "m38_ribosnitch_baseline_declaration.json")
OUTJ = os.path.join(HERE, "m38_ribosnitch_baseline.json")
D = json.load(open(DECL))
NMIN = D["constants"]["NMIN"]
WINDOW = D["constants"]["WINDOW"]
SEED = D["constants"]["SEED"]

# name in the declaration -> the key holding the NEGATED score, so higher = more rescue predicted
SCORES = (("d_bppm", "rs_bppm"), ("d_mfe", "rs_mfe"), ("d_ens", "rs_ens"),
          ("d_bppm_local", "rs_local"), ("depth_baseline", "depth"))
LOG = []


def log(s=""):
    print(s, flush=True); LOG.append(s)


# ------------------------------------------------------------------ folding quantities
def fold_all(seq):
    """dense upper-triangle bpp matrix, MFE pair set, ensemble free energy. One pf per sequence."""
    fc = RNA.fold_compound(seq)
    mfe_ss, _mfe_e = fc.mfe()
    fc.exp_params_rescale(_mfe_e)
    _pf_ss, g_ens = fc.pf()
    bpp = fc.bpp()                                  # 1-based, upper triangle, list of lists
    n = len(seq)
    P = np.zeros((n, n), float)
    for i in range(1, n + 1):
        row = bpp[i]
        for j in range(i + 1, n + 1):
            p = row[j]
            if p:
                P[i - 1, j - 1] = p
    return P, pairset(mfe_ss), float(g_ens)


def pairset(ss):
    st, out = [], set()
    for k, c in enumerate(ss):
        if c == "(":
            st.append(k)
        elif c == ")":
            if st:
                out.add((st.pop(), k))
    return out


def d_bppm(Pw, Pm):
    return float(np.sqrt(((Pw - Pm) ** 2).sum()))


def d_bppm_dict(Pw, Pm):
    """检查2 对拍: the SAME quantity accumulated a different way -- pair by pair through a dict,
    never touching the array difference. Must agree with d_bppm to 1e-10."""
    acc = {}
    n = Pw.shape[0]
    for A, s in ((Pw, 1.0), (Pm, -1.0)):
        nz = np.nonzero(A)
        for i, j in zip(nz[0].tolist(), nz[1].tolist()):
            acc[(i, j)] = acc.get((i, j), 0.0) + s * float(A[i, j])
    return float(np.sqrt(sum(v * v for v in acc.values())))


def d_bppm_local(Pw, Pm, a, b, w=WINDOW):
    n = Pw.shape[0]
    idx = sorted(set(list(range(max(0, a - w), min(n, a + w + 1)))
                     + list(range(max(0, b - w), min(n, b + w + 1)))))
    ix = np.ix_(idx, idx)
    return float(np.sqrt(((Pw[ix] - Pm[ix]) ** 2).sum()))


# ------------------------------------------------------------------ streaming driver
# Memory note: a first version cached one dense n-by-n matrix per unique sequence and reached
# 1.5 GB in forty seconds on its way to roughly 11 GB. Nothing is cached now. One wild type is
# folded, its mutants are folded one at a time, each distance is taken immediately and both
# matrices are dropped. Two matrices are live at any moment.
def score_all(bywt):
    """fold, measure, discard. Returns the three arm-D check maxima alongside."""
    chk = {"identity_max": 0.0, "obweight_max": 0.0, "symmetry_max": 0.0,
           "n_identity": 0, "n_obweight": 0, "n_symmetry": 0}
    t0, done, total = time.time(), 0, sum(len(v) for v in bywt.values())
    for wi, wt in enumerate(sorted(bywt)):
        Pw, Mw, Gw = fold_all(wt)
        # 检查1 零点恒等式, on every wild type: each score must be exactly 0 against itself.
        chk["identity_max"] = max(chk["identity_max"], abs(d_bppm(Pw, Pw)),
                                  abs(d_bppm_local(Pw, Pw, 0, 1)), float(len(Mw ^ Mw)))
        chk["n_identity"] += 1
        for k, r in enumerate(bywt[wt]):
            Pm, Mm, Gm = fold_all(r["mut_seq"])
            a, b = min(r["a"], r["b"]), max(r["a"], r["b"])
            dg = d_bppm(Pw, Pm)
            r["rs_bppm"] = -dg
            r["rs_mfe"] = -float(len(Mw ^ Mm))
            r["rs_ens"] = -abs(Gm - Gw)
            r["rs_local"] = -d_bppm_local(Pw, Pm, a, b)
            if wi < 12 and k < 2:
                # 检查2 对拍: the same number accumulated pair by pair through a dict, never
                # touching the array difference. 检查3 对称性 on the same pairs.
                chk["obweight_max"] = max(chk["obweight_max"], abs(dg - d_bppm_dict(Pw, Pm)))
                chk["symmetry_max"] = max(chk["symmetry_max"], abs(dg - d_bppm(Pm, Pw)))
                chk["n_obweight"] += 1; chk["n_symmetry"] += 1
            del Pm, Mm
            done += 1
            if done % 500 == 0:
                log(f"    ...{done}/{total}  ({time.time()-t0:.0f}s)")
        del Pw, Mw
    return chk


def grade_verification(chk):
    """K1: any failure voids the run and no number from it may be quoted."""
    log("=== D 验证(在读任何数之前判)===")
    a = chk["identity_max"] == 0.0
    b = chk["obweight_max"] < 1e-10
    c = chk["symmetry_max"] < 1e-12
    log(f"  检查1 零点恒等式 d(wt,wt)==0,{chk['n_identity']} 个野生型:最大 "
        f"{chk['identity_max']:.3e}  {'通过' if a else '**失败**'}")
    log(f"  检查2 对拍(稠密数组 vs 逐对字典),{chk['n_obweight']} 对:最大偏差 "
        f"{chk['obweight_max']:.3e}  {'通过' if b else '**失败**'}")
    log(f"  检查3 对称性 d(a,b)==d(b,a),{chk['n_symmetry']} 对:最大偏差 "
        f"{chk['symmetry_max']:.3e}  {'通过' if c else '**失败**'}")
    ok = bool(a and b and c)
    log(f"  → {'验证通过,可以读数' if ok else '**K1 触发:本次运行作废**'}\n")
    return ok, {**chk, "pass": ok}


def save(OUT):
    json.dump(OUT, open(OUTJ, "w"), indent=1, ensure_ascii=False)
    open(OUTJ.replace(".json", ".log"), "w").write("\n".join(LOG) + "\n")


def main():
    t0 = time.time()
    log("=== M38:文献里早就有的对照 —— riboSNitch 式突变效应分数 vs 免费的深度整数 ===")
    log("  声明已冻结并推送于本文件存在之前。离线 $0,不用新数据,不联网。")
    log("  声明的方向是**负**:补偿翻转若 rescue 则保住结构 ⇒ 结构变化分数应当小。")
    log("  所有分数取负后再排名,所以下面的 rho 为正 = 声明的方向对了。\n")
    OUT = {"declaration": os.path.basename(DECL), "constants": D["constants"]}

    recs, _dr = M26.build()
    log(f"  M26 纳入 {len(recs)} 次干预,{len({r['design'] for r in recs})} 个设计格")

    bywt = collections.defaultdict(list)
    for r in recs:
        bywt[r["wt_seq"]].append(r)
    L = [len(s) for s in bywt]
    log(f"  野生型 {len(bywt)} 条,长度 {min(L)}-{max(L)} nt;要折叠 {len(bywt)+len(recs)} 次")

    chk = score_all(bywt)
    log(f"  折叠与打分完成 {time.time()-t0:.0f}s\n")
    ok, ver = grade_verification(chk)
    OUT["D_verification"] = ver
    if not ok:
        OUT["verdict"] = "void_K1_verification_failed"; save(OUT); return 1

    cells = M31.make_cells(recs, nmin=NMIN)
    ncr = sum(len(v) for _k, v in cells)
    log(f"  至少 {NMIN} 个翻转的格 {len(cells)},覆盖 {ncr} 次干预 ({100*ncr/len(recs):.1f}%)")
    OUT["n_records"] = len(recs); OUT["n_cells"] = len(cells); OUT["n_records_in_cells"] = ncr
    if len(cells) < 8:
        log("  **K0:存活格 < 8 ⇒ 本臂作废**")
        OUT["verdict"] = "void_K0_too_few_cells"; save(OUT); return 1

    # ---------------- A: within-design ranking, M31's statistic verbatim
    log("\n=== A 格内排名(与 M31 同一个统计量、同一批格、同一个自助与置换)===")
    M31.PREDS = SCORES                      # the only thing that changes is WHICH columns
    A = M31.arm_A(cells, "格内", seed=SEED % 100000)
    OUT["A_within_design"] = A

    # ---------------- B: head to head against the free integer
    log("\n=== B 与免费深度基线按格配对相减(claim 依赖的就是这一臂)===")
    ry = M31.ry_map(cells)
    pz = sorted({rs[0]["puzzle"] for _k, rs in cells})
    bypz = collections.defaultdict(list)
    for k, rs in cells:
        bypz[rs[0]["puzzle"]].append((k, rs))
    B = {}
    for nm, key in SCORES:
        if key == "depth" or A.get(nm) is None:
            continue
        rng = np.random.default_rng(SEED + 7); dif = []
        for _ in range(M31.NBOOT):
            ix = rng.integers(0, len(pz), len(pz))
            sub = [c for q in ix for c in bypz[pz[q]]]
            a_, _ = M31.weighted_rho(sub, key, ry)
            b_, _ = M31.weighted_rho(sub, "depth", ry)
            if a_ is not None and b_ is not None:
                dif.append(a_ - b_)
        ci = M31.boot_ci(dif)
        pt = A[nm]["rho"] - A["depth_baseline"]["rho"]
        beats = bool(ci[0] is not None and ci[0] > 0)
        B[nm] = {"diff": round(pt, 5), "ci": ci, "beats_depth": beats}
        log(f"  {nm:16s} - 深度 = {pt:+.5f}   CI {ci}   "
            f"{'**打赢了深度(K3)**' if beats else '没有打赢'}")
    OUT["B_vs_free_depth"] = B

    # ---------------- the staked predictions, graded in the declaration's own words
    log("\n=== 声明里下的注,按声明自己的措辞判 ===")
    st = {}
    glob = [nm for nm, _k in SCORES if nm not in ("depth_baseline", "d_bppm_local")]
    resolved = sorted(nm for nm in glob if A.get(nm) and A[nm]["ci"][0] is not None
                      and A[nm]["ci"][0] > 0)
    wrongside = sorted(nm for nm in glob if A.get(nm) and A[nm]["ci"][1] is not None
                       and A[nm]["ci"][1] < 0)
    st["1"] = {"text": D["staked"]["1"], "pass": bool(resolved),
               "detail": f"在声明方向上离零的:{resolved or '无'}"}
    st["2"] = {"text": D["staked"]["2"], "pass": not any(v["beats_depth"] for v in B.values()),
               "detail": f"打赢深度的:{[k for k, v in B.items() if v['beats_depth']] or '无'}"}
    st["3"] = {"text": D["staked"]["3"],
               "pass": bool(A["d_bppm_local"]["rho"] >= A["d_bppm"]["rho"]),
               "detail": f"局部 {A['d_bppm_local']['rho']:+.5f} vs 全局 {A['d_bppm']['rho']:+.5f}"}
    st["4"] = {"text": D["staked"]["4"],
               "pass": bool((A["d_bppm"]["rho"] >= 0) == (A["d_mfe"]["rho"] >= 0)),
               "detail": f"d_bppm {A['d_bppm']['rho']:+.5f}, d_mfe {A['d_mfe']['rho']:+.5f}"}
    for k in sorted(st):
        log(f"  下注{k} {'成立' if st[k]['pass'] else '**未成立**'} —— {st[k]['detail']}")
        log(f"        『{st[k]['text']}』")
    OUT["staked"] = st

    if wrongside:
        log(f"\n  **K2 触发:{wrongside} 的区间整个落在声明方向的反面。**")
        log("  按声明,这些以『方向失败』的措辞报告,故事不跟着符号走。")
    OUT["K2_wrong_side"] = wrongside

    if any(v["beats_depth"] for v in B.values()):
        v = "K3_a_purpose_built_score_beats_the_free_integer"
        log("\n  **K3 触发:有分数打赢了免费整数。这是正向结果,论文必须按正向结果写:**")
        log("  深度那句话要限定到『不是为这个任务造的预测器』,并把这个分数写成第一个打赢的。")
    elif resolved:
        v = "purpose_built_score_reads_the_intervention_but_does_not_beat_the_free_integer"
    else:
        v = "purpose_built_score_does_not_resolve_from_zero_in_the_declared_direction"
    OUT["verdict"] = v
    log(f"\n判决:{v}")
    log(f"用时 {time.time()-t0:.0f}s")
    save(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
