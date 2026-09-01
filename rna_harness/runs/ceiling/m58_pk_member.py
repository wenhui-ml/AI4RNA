#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M58: put a pseudoknot-capable nearest-neighbour implementation in the panel.

Declaration m58_pk_member_declaration.json was frozen and pushed before this file existed.

The corpus is an OpenKnot round.  Every implementation scored in Section 4.2 is pseudoknot-blind
in its released configuration, so every panel-level sentence in the manuscript is, as it stands, a
statement about a pseudoknot-blind panel applied to a pseudoknot corpus.  ShapeKnots is the same
energy model the panel's RNAstructure member already uses, with the pseudoknot search switched on
-- which is not an independent fifth opinion and the declaration says so.  That is the point: it
holds thermodynamics fixed and moves only the one capability in question.

NO probing data is passed.  M54 restrained ShapeKnots with the construct's own 2A3 profile because
it was building a reference; a scored panel member must see sequence only, or the outcome's own
experiment sits on both sides of the comparison.
"""
import collections, csv, json, math, os, subprocess, sys, tempfile, time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DECL = os.path.join(HERE, "m58_pk_member_declaration.json")
OUTJ = os.path.join(HERE, "m58_pk_member.json")
FCACHE = os.path.join(HERE, "m58_shapeknots_seqonly_cache.json")
RCACHE = os.path.join(HERE, "m55_recs_cache.json")
RS_BIN = "/home/ubuntu/miniconda3/envs/rnastr/bin/ShapeKnots"
RS_DATA = "/home/ubuntu/miniconda3/envs/rnastr/share/rnastructure/data_tables"
NWORK = 100                     # ~92 of this box's 256 cores were held by other tenants when
                                # this ran, so the binding constraint was leaving the box idle
                                # headroom, not the standing 192-thread self-cap
LOG = []


def log(s=""):
    print(s, flush=True); LOG.append(s)


def _plain(o):
    import numpy as np
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def save(OUT):
    OUT["log"] = LOG
    json.dump(OUT, open(OUTJ, "w"), indent=1, ensure_ascii=False, default=_plain)
    log(f"\n写出 {OUTJ}")


# --------------------------------------------------------------------- the folder
def _all_ct(path):
    """every structure block in the CT file, lowest-free-energy first.

    ShapeKnots writes a SET.  The agreement arm takes only block 0; the per-pair score is the
    fraction of blocks containing the pair, and V6 checks that this reduces to the indicator
    wherever exactly one block comes back."""
    out = []
    with open(path) as f:
        while True:
            head = f.readline()
            if not head.strip():
                break
            n = int(head.split()[0])
            s = set()
            for _ in range(n):
                p = f.readline().split()
                if len(p) < 5:
                    return out
                i, j = int(p[0]), int(p[4])
                if j and i < j:
                    s.add((i - 1, j - 1))
            out.append(s)
    return out


def _fold_one(seq):
    """sequence -> (sequence, [pair set per structure], ok).  Sequence only: no -sh, ever."""
    env = dict(os.environ, DATAPATH=RS_DATA, OMP_NUM_THREADS="1")
    with tempfile.TemporaryDirectory() as d:
        sf = os.path.join(d, "s.seq"); cf = os.path.join(d, "s.ct")
        open(sf, "w").write(";\ns\n" + seq + "1\n")
        r = subprocess.run([RS_BIN, sf, cf], capture_output=True, text=True, env=env)
        if r.returncode != 0 or not os.path.exists(cf):
            return seq, [], False
        return seq, _all_ct(cf), True


def build_cache(seqs):
    got, fail, t0 = {}, [], time.time()
    with ProcessPoolExecutor(max_workers=NWORK) as ex:
        for k, (s, sts, ok) in enumerate(ex.map(_fold_one, seqs, chunksize=1), 1):
            if ok and sts:
                got[s] = [sorted(map(list, x)) for x in sts]
            else:
                fail.append(s)
            if k % 500 == 0:
                log(f"    ShapeKnots {k}/{len(seqs)}  {time.time()-t0:.0f}s")
    log(f"    ShapeKnots 完成 {len(got)}/{len(seqs)},失败 {len(fail)},用时 {time.time()-t0:.0f}s")
    return got, fail


def main_cache():
    recs = json.load(open(RCACHE))["recs"]
    seqs = sorted({r["mut_seq"] for r in recs} | {r["wt_seq"] for r in recs})
    log(f"=== M58 折叠缓存:{len(seqs)} 条唯一序列,纯序列输入(不传探测数据)===")
    got, fail = build_cache(seqs)
    json.dump({"structs": got, "failed": fail}, open(FCACHE, "w"))
    log(f"写出 {FCACHE}")
    return 0




# ===================================================================== the analysis
def _load_cache():
    c = json.load(open(FCACHE))
    return ({s: [{tuple(p) for p in st} for st in v] for s, v in c["structs"].items()},
            c["failed"])


def freq(sts, a, b):
    """the declared per-pair score: fraction of ShapeKnots' own returned structures holding (a,b)"""
    return sum(1 for st in sts if (a, b) in st) / len(sts)


def main_analyse():
    import numpy as np
    import m53_probing_reference as M53
    from m53_probing_reference import (pairs_of, f1, wrho, pct, matched, within_sd, wild_types,
                                       NMIN, SEED, NBOOT, PUB_RANK, PUB_DEPTH, SF)
    t0 = time.time()
    log("=== M58:面板里放进一个能算假结的最近邻实现 ===")
    log("  声明已冻结并推送于本文件存在之前。全离线,$0,不传探测数据。")
    OUT = {"declaration": os.path.basename(DECL), "V": {},
           "cost": "offline, $0 -- local binary, no API, no probing data passed",
           "what_shapeknots_is_here": "the panel's own RNAstructure energy model with the "
                                      "pseudoknot search on; declared as dependent, not independent"}

    S, failed = _load_cache()
    recs = [{**r, "design": tuple(r["design"])} for r in json.load(open(RCACHE))["recs"]]
    log(f"  折叠缓存 {len(S)} 条序列,失败 {len(failed)};干预 {len(recs)}")
    if len(failed) > 0.01 * (len(S) + len(failed)):
        OUT["VOID"] = f"K1: {len(failed)} 条折叠失败,超过 1%"; save(OUT); return 1

    wt = wild_types()
    PUB = {"rf": 0.05390, "nmut": 0.06050, "pmut": -0.03040, "emut": -0.00569}

    # ---------------------------------------------------------------- V, before any arm
    log("\n=== V 验证(先于任何 arm)===")
    _s0 = sorted(S)[0]
    v1 = (f1(S[_s0][0], S[_s0][0]) == 1.0 and f1(S[_s0][0], set()) == 0.0)
    log(f"  V1 自身 F1 {f1(S[_s0][0], S[_s0][0]):.1f}  对空 {f1(S[_s0][0], set()):.1f}"
        f"  -> {'通过' if v1 else '失败'}")

    # V2: the same binary M54 called, so its no-data control must come back bit for bit
    c54 = json.load(open(os.path.join(HERE, "m54_shapeknots_cache.json")))
    nod = {tuple(json.loads(k)): {tuple(p) for p in v} for k, v in c54["no_data"].items()}
    chk = [(k, wt[k]["sequence"]) for k in nod if k in wt and wt[k]["sequence"] in S]
    ndiff = sum(1 for k, s in chk if S[s][0] != nod[k])
    log(f"  V2 与 M54 的无数据对照逐条比较 {len(chk)} 条,不同 {ndiff} 条"
        f"  -> {'通过' if ndiff == 0 else '失败'}")
    if ndiff:
        OUT["V"]["V2_reproduces_m54"] = False
        OUT["VOID"] = "K2: 与 M54 的无数据结构不一致,两个结果不能放进同一个面板"
        save(OUT); return 1

    # V4 well-formedness, and the non-canonical fraction REPORTED rather than filtered
    CAN = {"AU", "UA", "GC", "CG", "GU", "UG"}
    bad_len = bad_deg = 0; nc = tot = 0
    seq_of = {}
    for r in recs:
        seq_of[r["mut_seq"]] = r["mut_seq"]; seq_of[r["wt_seq"]] = r["wt_seq"]
    for s, sts in S.items():
        st = sts[0]; seen = collections.Counter()
        for i, j in st:
            if not (0 <= i < j < len(s)):
                bad_len += 1
            seen[i] += 1; seen[j] += 1
            tot += 1
            if s[i] + s[j] not in CAN:
                nc += 1
        if seen and max(seen.values()) > 1:
            bad_deg += 1
    v4 = (bad_len == 0 and bad_deg == 0)
    log(f"  V4 越界配对 {bad_len}  一个碱基配两次的分子 {bad_deg}"
        f"  非典型配对占比 {nc/max(1,tot):.5f}(报告而非过滤)  -> {'通过' if v4 else '失败'}")

    # V5 determinism
    rngd = np.random.default_rng(SEED)
    samp = [sorted(S)[i] for i in rngd.choice(len(S), 20, replace=False)]
    redo = dict(_fold_one(s)[:2] for s in samp)
    v5 = all({tuple(p) for p in redo[s][0]} == S[s][0] for s in samp)
    log(f"  V5 重跑 20 条,最大差异 {0 if v5 else 1}  -> {'通过' if v5 else '失败'}")

    # V6 the per-pair score's zero point
    one = [s for s in S if len(S[s]) == 1]
    v6dev = 0.0
    for r in recs:
        if r["mut_seq"] in S and len(S[r["mut_seq"]]) == 1:
            f_ = freq(S[r["mut_seq"]], r["a"], r["b"])
            ind = 1.0 if (r["a"], r["b"]) in S[r["mut_seq"]][0] else 0.0
            v6dev = max(v6dev, abs(f_ - ind))
    log(f"  V6 只返回一个结构的序列 {len(one)} 条;频率与指示的最大偏差 {v6dev:.1e}"
        f"  -> {'通过' if v6dev == 0.0 else '失败'}")
    v6 = (v6dev == 0.0)

    # V3 reverse control -- the published numbers, from this file's own code
    for r in recs:
        r["_s"] = SF(r)
        r["kmut"] = freq(S[r["mut_seq"]], r["a"], r["b"]) if r["mut_seq"] in S else None
    miss = sum(1 for r in recs if r["kmut"] is None)
    log(f"  未能打分的干预 {miss}/{len(recs)}")
    d3 = {}
    for k in ("rf", "nmut", "pmut", "emut"):
        c_, _w, _u = matched(recs, k, SF)
        d3[k] = abs(c_ - PUB[k])
        log(f"  V3 {k:6s} 我 {c_:+.5f}  论文 {PUB[k]:+.5f}  |Δ| {d3[k]:.2e}")
    bycell = collections.defaultdict(list)
    for r in recs:
        bycell[r["design"]].append(r)
    cells = [(k, v) for k, v in bycell.items() if len(v) >= NMIN]
    rk = {n: wrho(cells, kx) for n, kx in
          (("RibonanzaNet", "nmut"), ("ViennaRNA", "pmut"), ("EternaFold", "emut"))}
    dep = wrho(cells, "depth")
    for n in PUB_RANK:
        log(f"  V3 rank {n:14s} 我 {rk[n]:+.5f}  论文 {PUB_RANK[n]:+.4f}")
    log(f"  V3 rank depth          我 {dep:+.5f}  论文 {PUB_DEPTH:+.4f}")

    # V3 also has to reproduce M54's four built-reference agreements, on M26's 158 cells
    ref = {tuple(json.loads(k)): {tuple(p) for p in v} for k, v in c54["with_data"].items()}
    rnet, rnet_f1 = M53.rnet_structures(wt)
    vi, rs, ef = M53.model_structures(wt)

    def win(k, prs):
        d = wt[k]; ss, se = int(float(d["sub_start"])), int(float(d["sub_end"]))
        return {(i, j) for i, j in prs if ss - 1 <= i < se and ss - 1 <= j < se}

    def struct_for(nm, k):
        s = wt[k]["sequence"]
        return {"ViennaRNA": vi.get(s), "RNAstructure": rs.get(s), "EternaFold": ef.get(s),
                "RibonanzaNet": rnet.get(k),
                "ShapeKnots": ("".join([]) if False else None)}[nm]

    acells = [k for k in sorted({r["design"] for r in recs}) if k in wt]
    log(f"  一致性用 M26 纳入的设计格 {len(acells)}")
    PUB_AG = json.load(open(os.path.join(HERE, "m54_probing_reference.json")))["A1_ranking"]

    def agree_of(pairs_for, ref_of):
        v = [f1(pairs_for(k), ref_of(k)) for k in acells if pairs_for(k) is not None]
        return float(np.mean(v)), len(v)

    ag_built, ag_decl = {}, {}
    for nm in ("ViennaRNA", "RNAstructure", "EternaFold", "RibonanzaNet"):
        ag_built[nm] = agree_of(lambda k, n=nm: win(k, pairs_of(struct_for(n, k))),
                                lambda k: win(k, ref[k]))[0]
        ag_decl[nm] = agree_of(lambda k, n=nm: win(k, pairs_of(struct_for(n, k))),
                               lambda k: win(k, pairs_of(wt[k]["target_structure"])))[0]
    dag = max(abs(ag_built[n] - PUB_AG["agreement_built"][n]) for n in ag_built)
    log(f"  V3 built 参考下四个一致性 最大偏差 {dag:.2e}")
    v3 = (max(d3.values()) < 1e-4
          and all(abs(rk[n] - PUB_RANK[n]) < 5e-4 for n in PUB_RANK)
          and abs(dep - PUB_DEPTH) < 5e-4 and dag < 5e-5)
    OUT["V"] = {"V1_zero_point": bool(v1), "V2_reproduces_m54": True, "V4_well_formed": bool(v4),
                "V4_noncanonical_fraction": round(nc / max(1, tot), 5),
                "V5_deterministic": bool(v5), "V6_frequency_zero_point": bool(v6),
                "V3_reverse_control": bool(v3), "n_unscored": miss,
                "n_single_structure_sequences": len(one)}
    if not (v1 and v4 and v5 and v6 and v3):
        OUT["VOID"] = "K2: V 未全部通过"; save(OUT); return 1
    log("  V 全部通过 —— 现在才读 ShapeKnots 的列")

    # ---------------------------------------------------------------- A1 agreement
    log("\n=== A1 一致性:把 ShapeKnots 放进面板,对两个参考各算一次 ===")
    sk_b = agree_of(lambda k: win(k, S[wt[k]["sequence"]][0]), lambda k: win(k, ref[k]))
    sk_d = agree_of(lambda k: win(k, S[wt[k]["sequence"]][0]),
                    lambda k: win(k, pairs_of(wt[k]["target_structure"])))
    ag_built["ShapeKnots"], ag_decl["ShapeKnots"] = sk_b[0], sk_d[0]
    for nm in sorted(ag_built, key=lambda n: -ag_built[n]):
        log(f"  {nm:14s} built 参考 {ag_built[nm]:.5f}   声明靶标 {ag_decl[nm]:.5f}")
    lead_b = max(ag_built, key=lambda n: ag_built[n])
    lead_d = max(ag_decl, key=lambda n: ag_decl[n])
    log(f"  built 参考下的第一名 {lead_b};声明靶标下的第一名 {lead_d}")
    OUT["A1_agreement"] = {"built": {k: round(v, 5) for k, v in ag_built.items()},
                           "declared": {k: round(v, 5) for k, v in ag_decl.items()},
                           "leader_built": lead_b, "leader_declared": lead_d,
                           "n_cells": sk_b[1],
                           "shapeknots_exceeds_ribonanzanet_built":
                               bool(ag_built["ShapeKnots"] > ag_built["RibonanzaNet"])}

    # ---------------------------------------------------------------- A2 level arm
    log("\n=== A2 level 臂:ShapeKnots 的集合频率 vs 免费的整数 ===")
    kk = wrho(cells, "kmut")
    bpz = collections.defaultdict(list)
    for k, v in cells:
        bpz[v[0]["puzzle"]].append((k, v))
    pzz = sorted(bpz); rng2 = np.random.default_rng(SEED + 3)
    dr = []
    for _ in range(NBOOT):
        ix = rng2.integers(0, len(pzz), len(pzz))
        cs = [x for q in ix for x in bpz[pzz[q]]]
        a_, b_ = wrho(cs, "depth"), wrho(cs, "kmut")
        if a_ is not None and b_ is not None:
            dr.append(a_ - b_)
    ci2 = pct(dr)
    log(f"  ShapeKnots level {kk:+.5f};depth {dep:+.5f};三个已发表 "
        f"{', '.join(f'{n} {rk[n]:+.4f}' for n in PUB_RANK)}")
    log(f"  depth − ShapeKnots  {dep-kk:+.5f}  {ci2}"
        f"  {'含 0' if ci2[0] <= 0 <= ci2[1] else '不含 0'}")
    OUT["A2_level"] = {"shapeknots": round(kk, 5), "depth": round(dep, 5),
                       "depth_minus": round(dep - kk, 5), "ci95": ci2,
                       "ci_contains_zero": bool(ci2[0] <= 0 <= ci2[1]),
                       "depth_wins": bool(dep > kk)}

    # ---------------------------------------------------------------- A3 identity contrast
    log("\n=== A3 身份对比:同一批干预,同一套分层 ===")
    ck, wk, sk = matched(recs, "kmut", SF)
    sdk = within_sd(recs, "kmut", SF)
    rng3 = np.random.default_rng(SEED + 5)
    bypz = collections.defaultdict(list)
    for r in recs:
        bypz[r["puzzle"]].append(r)
    pz3 = sorted(bypz); dk = []
    for _ in range(NBOOT):
        ix = rng3.integers(0, len(pz3), len(pz3))
        rr = [x for q in ix for x in bypz[pz3[q]]]
        c_, _w, _u = matched(rr, "kmut", SF)
        if c_ is not None:
            dk.append(c_)
    ci3 = pct(dk)
    log(f"  对比 {ck:+.5f}  权重 {wk:.0f}  层 {sk}  层内 SD {sdk:.4f}  标准化 {ck/sdk:+.5f}")
    log(f"  A3 ShapeKnots 对比 {ck:+.5f}  95% {ci3}"
        f"  {'含 0' if ci3[0] <= 0 <= ci3[1] else '不含 0'}")
    OUT["A3_identity"] = {"contrast": round(ck, 5), "within_sd": round(sdk, 4),
                          "contrast_in_sd": round(ck / sdk, 5), "ci95": ci3,
                          "ci_contains_zero": bool(ci3[0] <= 0 <= ci3[1]),
                          "weight": round(wk, 1), "strata": sk}

    # ---------------------------------------------------------------- A4 budget, via M44v's code
    log("\n=== A4 预算臂:用 M44v 自己的 recover() ===")
    import m44v_verify as M44V
    ext = os.path.join(HERE, "m58_scores_with_shapeknots.csv")
    rows = list(csv.DictReader(open(os.path.join(HERE, "m43_predictor_scores.csv"))))
    assert len(rows) == len(recs), "打分表与记录缓存行数不一致"
    for x, r in zip(rows, recs):
        assert (x["puzzle"], x["method"]) == r["design"] and int(x["a"]) == r["a"] \
               and int(x["b"]) == r["b"], "打分表与记录缓存未按行对齐"
        x["kmut"] = f"{r['kmut']:.6f}"
    with open(ext, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    M44V.IMPL = ("ViennaRNA", "EternaFold", "RNAstructure", "RibonanzaNet", "ShapeKnots")
    M44V.COL = {"ViennaRNA": "pmut", "EternaFold": "emut", "RNAstructure": "smut",
                "RibonanzaNet": "nmut", "ShapeKnots": "kmut"}
    M44V.SCORES = ext
    bcells = M44V.cells_from_csv()
    sgn = M44V.depth_sign(bcells)
    log(f"  M44v 的格 {len(bcells)};depth 方向 {sgn:+.0f}")
    A4 = {"budgets": {}}
    for frac in M44V.BUDGETS:
        row = {}
        for rule in ("depth", "ShapeKnots", "RibonanzaNet", "ViennaRNA", "EternaFold",
                     "RNAstructure"):
            row[rule] = round(M44V.recover(bcells, rule, sgn, frac), 6)
        row["shapeknots_minus_depth"] = round(row["ShapeKnots"] - row["depth"], 6)
        row["shapeknots_beats_depth"] = bool(row["ShapeKnots"] > row["depth"])
        A4["budgets"][str(frac)] = row
        log(f"  预算 {frac}: depth {row['depth']:.4f}  ShapeKnots {row['ShapeKnots']:.4f}"
            f"  差 {row['shapeknots_minus_depth']:+.4f}"
            f"  {'ShapeKnots 更好' if row['shapeknots_beats_depth'] else 'depth 不输'}")
    tight = str(min(M44V.BUDGETS))
    A4["tightest"] = tight
    OUT["A4_budget"] = A4

    # ---------------------------------------------------------------- stakes
    log("\n=== 预注册记分 ===")
    St = {"S1": bool(ag_built["ShapeKnots"] <= ag_built["RibonanzaNet"]),
          "S2": bool(dep > kk and not (ci2[0] <= 0 <= ci2[1])),
          "S3": bool(ci3[0] <= 0 <= ci3[1]),
          "S4": bool(not A4["budgets"][tight]["shapeknots_beats_depth"])}
    log(f"  S1  {'成立' if St['S1'] else '**失败**'}   ShapeKnots {ag_built['ShapeKnots']:.5f}"
        f" vs RibonanzaNet {ag_built['RibonanzaNet']:.5f}(built 参考)")
    log(f"  S2  {'成立' if St['S2'] else '**失败**'}   depth − ShapeKnots {dep-kk:+.5f} {ci2}")
    log(f"  S3  {'成立' if St['S3'] else '**失败**'}   身份对比 {ck:+.5f} {ci3}")
    log(f"  S4  {'成立' if St['S4'] else '**失败**'}   最紧预算 {tight}")
    met = sum(St.values())
    verdict = ("panel_conclusions_survive_a_pseudoknot_capable_member" if met == 4 else
               "a_pseudoknot_capable_member_moves_the_panel")
    OUT["stakes"] = St; OUT["n_met"] = met; OUT["verdict"] = verdict
    log(f"\n  裁定:{verdict}   {met}/4 成立   用时 {time.time()-t0:.0f}s")
    save(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main_cache() if "--build-cache" in sys.argv else main_analyse())
