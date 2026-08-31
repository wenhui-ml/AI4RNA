#!/usr/bin/env python3
"""M43: does the intervention readout reorder the predictors that structure agreement ranks?

Declaration `m43_reorder_declaration.json`, frozen and pushed at 425f3271 BEFORE this file
existed. Offline, $0 -- RNAstructure 6.6 and ViennaRNA local, EternaFold and RibonanzaNet from
their on-disk caches, no API and no new GPU time.

DECLARATION ERRATUM (recorded, not edited away). The declaration's `definitions` block asserts
that "EternaFold enters this paper as ViennaRNA's langdon2018 parameter set rather than as its own
binary" and that it is therefore a parameter variant. THAT IS WRONG. M32 builds and runs the real
EternaFold binary -- github.com/eternagame/EternaFold at 87b9aac5, binary md5
5e1306e54be09835a383c1d3c41573dc, verified on disk before this file was written. The error came
from reading `rna_harness/oracles/registry.py`, which does carry a langdon2018 ViennaRNA oracle,
but that panel drives design search and is not what the manuscript scores. The RULE in the
declaration stands unchanged; only its application to EternaFold was wrong, and it moves the count
the right way: this paper already has THREE independent implementations and RNAstructure is a
fourth. Staked predictions are judged against the rule, not against the erratum.

A second thing this run fixes, because it is a reproducibility hole rather than a result: the
EternaFold and RibonanzaNet pair probabilities live in gitignored caches (42 MB and 606 MB) and a
binary in a session scratchpad, so no number in the manuscript that depends on them can be
reproduced from the branch alone. This run writes the per-intervention scores for every predictor
as a small committed table, which is what actually has to be archived.
"""
from __future__ import annotations

import collections, csv, json, math, os, subprocess, sys, tempfile, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DECL = json.load(open(os.path.join(HERE, "m43_reorder_declaration.json")))
C = DECL["constants"]
NBOOT, SEED, MIN_CELL, CEILING = C["NBOOT"], C["SEED"], C["MIN_CELL"], C["CEILING_FROM_M41"]
OUTJ = os.path.join(HERE, "m43_reorder.json")
SCORES = os.path.join(HERE, "m43_predictor_scores.csv")

RS_BIN = "/home/ubuntu/miniconda3/envs/rnastr/bin"
RS_DATA = "/home/ubuntu/miniconda3/envs/rnastr/share/rnastructure/data_tables"
BENCHFILE = "/data/wenhui/openknot_data/Data/OpenKnotBench_data.v4.5.1.csv"
MAXPROC = 32                       # well inside the 160-core cap; RNAstructure is single-threaded
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.append(s)


# ---------------------------------------------------------------- RNAstructure, the 4th engine
def _rs_one(seq):
    """(pair-probability dict, MFE dot-bracket) for one sequence. Runs in a worker process."""
    env = dict(os.environ, DATAPATH=RS_DATA)
    with tempfile.TemporaryDirectory() as d:
        fa = os.path.join(d, "s.fa")
        open(fa, "w").write(">s\n" + seq.replace("T", "U") + "\n")
        pfs, txt, ct, dot = (os.path.join(d, x) for x in ("s.pfs", "s.txt", "s.ct", "s.dot"))
        subprocess.run([f"{RS_BIN}/partition", fa, pfs], env=env, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([f"{RS_BIN}/ProbabilityPlot", pfs, txt, "--text"], env=env, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        bpp = {}
        with open(txt) as f:
            f.readline(); f.readline()                      # length line, then the header
            for line in f:
                p = line.split()
                if len(p) == 3:
                    bpp[(int(p[0]), int(p[1]))] = 10.0 ** (-float(p[2]))
        subprocess.run([f"{RS_BIN}/Fold", fa, ct, "-mfe"], env=env, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([f"{RS_BIN}/ct2dot", ct, "1", dot], env=env, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        db = open(dot).read().splitlines()[2].strip()
    return seq, bpp, db


def rnastructure_all(seqs):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    bpp, mfe, t0 = {}, {}, time.time()
    with ProcessPoolExecutor(max_workers=MAXPROC) as ex:
        for k, (s, b, d) in enumerate(ex.map(_rs_one, seqs, chunksize=8)):
            bpp[s], mfe[s] = b, d
            if (k + 1) % 500 == 0:
                log(f"    RNAstructure {k+1}/{len(seqs)}  {time.time()-t0:.0f}s")
    log(f"    RNAstructure 完成 {len(seqs)} 条,用时 {time.time()-t0:.0f}s")
    return bpp, mfe


# ---------------------------------------------------------------- structures and F1
CLOSER = {")": "(", "]": "[", "}": "{", ">": "<"}


def parse_db(db):
    """dot-bracket -> set of pairs, with ONE STACK PER BRACKET TYPE.

    This corpus is a pseudoknot benchmark: every declared target here uses `[`, and the released
    structures also use `{}` and `<>`. A single shared stack pairs a `)` with the most recent `[`
    and silently returns a structure that is not the one on the page. It corrupts the predicted
    structures too, and worst for the model that actually predicts pseudoknots -- which is exactly
    the shape of a reorder result. Found by the V4 cross-check against the corpus's own RNet_F1
    column, which disagreed by up to 1.0000.
    """
    st = {k: [] for k in "([{<"}
    out = set()
    for k, ch in enumerate(db):
        if ch in st:
            st[ch].append(k)
        elif ch in CLOSER:
            s = st[CLOSER[ch]]
            if s:
                out.add((s.pop(), k))
    return out


def f1(pred, true):
    if not pred and not true:
        return 1.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 2 * p * r / (p + r)


def eternafold_structures(seqs):
    """EternaFold's own output, from the binary M32 recorded. Its MEA structure is native."""
    import m32_third_model_class as M32
    out = {}
    with tempfile.TemporaryDirectory() as d:
        for k, s in enumerate(seqs):
            fa = os.path.join(d, "s.fa")
            open(fa, "w").write(">s\n" + s + "\n")
            r = subprocess.run([M32.BIN, "predict", "--params", M32.PARAMS, fa],
                               capture_output=True, text=True)
            lines = [l.strip() for l in r.stdout.splitlines() if l.strip()]
            db = next((l for l in reversed(lines) if set(l) <= set(".()[]{}<>")), None)
            if db and len(db) == len(s):
                out[s] = db
            if (k + 1) % 50 == 0:
                log(f"    EternaFold 结构 {k+1}/{len(seqs)}")
    return out


# ---------------------------------------------------------------- rankings
def kendall_tau(a, b):
    """tau-b between two orderings given as score dicts over the same names."""
    ks = sorted(a)
    n, con, dis = len(ks), 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[ks[i]] - a[ks[j]]) * (b[ks[i]] - b[ks[j]])
            con += s > 0; dis += s < 0
    tot = con + dis
    return None if tot == 0 else round((con - dis) / tot, 4)


def stage1():
    import m26_model_scored as M26
    import m31_which_flips_rescue as M31
    t0 = time.time()
    out = {"declaration": "m43_reorder_declaration.json", "cost": "offline, $0",
           "declaration_erratum": {
               "field": "definitions.why_this_matters_here",
               "asserted": "EternaFold enters this paper as ViennaRNA's langdon2018 parameter "
                           "set and is therefore a parameter variant, not an implementation",
               "actual": "M32 builds and runs the real EternaFold binary (CONTRAfold codebase), "
                         "github.com/eternagame/EternaFold @ 87b9aac5, binary md5 "
                         "5e1306e54be09835a383c1d3c41573dc, verified on disk",
               "cause": "read from rna_harness/oracles/registry.py, which carries a langdon2018 "
                        "ViennaRNA oracle for design search and is not what the manuscript scores",
               "effect": "the rule stands; its application was wrong. The paper already has three "
                         "independent implementations and RNAstructure is a fourth, not a third.",
               "declaration_not_edited": True}}

    # ---- the interventions, and ViennaRNA's probabilities with them
    log("=== 建立干预集合(M26.build,turner2004)")
    recs, _dr = M26.build()
    log(f"  干预 {len(recs)}  设计格 {len(set(r['design'] for r in recs))}")

    # ---- RibonanzaNet and EternaFold from their caches
    z = np.load(os.path.join(HERE, "m26c_pneural_cache.npz"), allow_pickle=True)
    pn = {str(k): v for k, v in zip(z["keys"], z["mats"])}
    ze = np.load(os.path.join(HERE, "m32_eterna_cache.npz"), allow_pickle=True)
    pe = {str(k): (v.item() if hasattr(v, "item") else v) for k, v in zip(ze["keys"], ze["maps"])}
    log(f"  缓存:RibonanzaNet {len(pn)} 条,EternaFold {len(pe)} 条")

    # ---- RNAstructure, the fourth implementation
    uniq = sorted({r["mut_seq"] for r in recs} | {r["wt_seq"] for r in recs})
    log(f"=== RNAstructure 6.6:唯一序列 {len(uniq)}")
    rs_bpp, rs_mfe = rnastructure_all(uniq)

    miss = collections.Counter()
    for r in recs:
        a, b = min(r["a"], r["b"]), max(r["a"], r["b"])
        m = pn.get(r["mut_seq"])
        r["nmut"] = float(m[a, b]) if m is not None else float("nan")
        e = pe.get(r["mut_seq"])
        r["emut"] = float(e.get((a + 1, b + 1), 0.0)) if e is not None else float("nan")
        s = rs_bpp.get(r["mut_seq"])
        r["smut"] = float(s.get((a + 1, b + 1), 0.0)) if s is not None else float("nan")
        for k in ("pmut", "nmut", "emut", "smut"):
            if r[k] != r[k]:
                miss[k] += 1
    out["coverage"] = {k: {"missing": miss[k], "frac": round(miss[k] / len(recs), 5)}
                       for k in ("pmut", "nmut", "emut", "smut")}
    log(f"  覆盖缺失 {dict(miss)}")
    dropped = [k for k in ("pmut", "nmut", "emut", "smut") if miss[k] / len(recs) > 0.05]
    out["dropped_for_coverage"] = dropped
    if dropped:
        log(f"  ** 覆盖率不足被剔除:{dropped}(按 declaration 的 kill,不做插补)**")

    IMPL = [("ViennaRNA", "pmut"), ("EternaFold", "emut"),
            ("RNAstructure", "smut"), ("RibonanzaNet", "nmut")]
    IMPL = [(n, k) for n, k in IMPL if k not in dropped]

    # ---- R_resp: this paper's readout, pooled exactly as M31 pools it
    log("\n=== R_resp 干预响应排名(格内加权 Spearman,按靶标自助)")
    cells = M31.make_cells(recs, nmin=MIN_CELL)
    ry = M31.ry_map(cells)
    pz = sorted({rs[0]["puzzle"] for _k, rs in cells})
    bypz = collections.defaultdict(list)
    for k, rs in cells:
        bypz[rs[0]["puzzle"]].append((k, rs))
    R_resp, resp_boot = {}, {}
    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, len(pz), len(pz)) for _ in range(NBOOT)]
    for nm, key in IMPL:
        obs, _per = M31.weighted_rho(cells, key, ry)
        bs = []
        for ix in idx:
            sub = [c for q in ix for c in bypz[pz[q]]]
            v, _ = M31.weighted_rho(sub, key, ry)
            if v is not None:
                bs.append(v)
        R_resp[nm] = round(obs, 5)
        resp_boot[nm] = np.array(bs)
        log(f"  {nm:14s} rho = {obs:+.5f}   CI {M31.boot_ci(bs)}   "
            f"去衰减 {obs/CEILING:+.5f}")
    out["R_resp"] = {n: {"rho": R_resp[n], "ci": M31.boot_ci(resp_boot[n].tolist()),
                         "rho_disattenuated": round(R_resp[n] / CEILING, 5)} for n, _ in IMPL}
    out["n_cells"] = len(cells)
    out["n_targets"] = len(pz)
    return out, recs, cells, IMPL, rs_mfe, bypz, pz, ry, idx, t0


def stage_agree(recs, IMPL, rs_mfe):
    """R_agree: the INCUMBENT ranking. Each implementation's own native structure for the design
    cell's wild type, against the declared target, scored by F1 over pairs INSIDE the design
    window -- outside it the constructs carry adapters and barcodes nobody designed."""
    import RNA
    import m16_rescue as M16
    import m32_third_model_class as M32
    log("\n=== R_agree 常规结构一致性(每个实现自己的原生结构,同一批野生型)")
    csv.field_size_limit(10 ** 9)
    cellinfo = {}
    for d in csv.DictReader(open(M26_SRC)):
        if d["mutA"] in ("", "NA", "nan") and d["mutB"] in ("", "NA", "nan"):
            cellinfo[(d["puzzle"], d["method"])] = d
    used = sorted({r["design"] for r in recs})
    cells = [(k, cellinfo[k]) for k in used if k in cellinfo]
    log(f"  设计格 {len(cells)}(有野生型且被干预集合用到)")

    # RibonanzaNet's native structure ships with the benchmark file, keyed by the full sequence.
    # The two files released together use DIFFERENT coordinates: OK7a's target_structure spans the
    # whole 177 nt construct, the benchmark file's RNet_structure and target_structure span the
    # design region alone. The structure is therefore shifted into full coordinates by sub_start-1
    # before anything is compared, and the shift is checked against the file's own RNet_F1 column.
    # The two files released together share NO full sequence -- 0 of 160 -- because they carry
    # different adapters around the same designs. They share the DESIGN REGION exactly, 160 of
    # 160, so that is the key. RNet_structure spans the design region alone and is shifted into
    # OK7a's coordinates with OK7a's own sub_start.
    rnet_by_design, rnet_f1_by_design = {}, {}
    if os.path.exists(BENCHFILE):
        for d in csv.DictReader(open(BENCHFILE)):
            ds, st = d.get("design_sequence"), d.get("RNet_structure")
            if not ds or not st or st in ("", "NA", "nan") or ds in rnet_by_design:
                continue
            if len(st) != len(ds):
                continue
            rnet_by_design[ds] = st
            if d.get("RNet_F1") not in (None, "", "NA", "nan"):
                rnet_f1_by_design[ds] = float(d["RNet_F1"])
    rnet, rnet_f1 = {}, {}
    for _k, d in cells:
        ss_, se_ = int(float(d["sub_start"])), int(float(d["sub_end"]))
        ds = d["sequence"][ss_ - 1:se_]
        st = rnet_by_design.get(ds)
        if st is None:
            continue
        full = ["."] * len(d["sequence"])
        for k_, ch in enumerate(st):                  # copy verbatim, brackets and all
            full[ss_ - 1 + k_] = ch
        rnet[d["sequence"]] = "".join(full)
        if ds in rnet_f1_by_design:
            rnet_f1[d["sequence"]] = rnet_f1_by_design[ds]
    log(f"  RNet_structure 覆盖 {sum(1 for _k, d in cells if d['sequence'] in rnet)}/{len(cells)}"
        f"  (按设计区序列连接;两个文件的全长序列重叠为 0)")

    wts = [d["sequence"] for _k, d in cells]
    ef = eternafold_structures(wts)
    scores = collections.defaultdict(list)
    per = []
    for k, d in cells:
        ss, se = int(float(d["sub_start"])), int(float(d["sub_end"]))
        w = d["sequence"]
        tgt = {(i, j) for i, j in M16.parse_pairs(d["target_structure"]).items()
               if i < j and ss - 1 <= i < se and ss - 1 <= j < se}
        if not tgt:
            continue
        got = {}
        for nm, _key in IMPL:
            db = None
            if nm == "ViennaRNA":
                db = RNA.fold(w)[0]
            elif nm == "RNAstructure":
                db = rs_mfe.get(w)
            elif nm == "EternaFold":
                db = ef.get(w)
            elif nm == "RibonanzaNet":
                db = rnet.get(w)
            if db is None or len(db) != len(w):
                continue
            pr = {(i, j) for i, j in parse_db(db) if ss - 1 <= i < se and ss - 1 <= j < se}
            v = f1(pr, tgt)
            scores[nm].append(v); got[nm] = round(v, 5)
        if len(got) == len(IMPL):
            per.append({"design": list(k), "puzzle": d["puzzle"], "f1": got})
    chk = [(v["f1"]["RibonanzaNet"], rnet_f1[d["sequence"]])
           for (k, d), v in zip(cells, per) if "RibonanzaNet" in v["f1"] and d["sequence"] in rnet_f1] \
        if len(per) == len(cells) else []
    if chk:
        dev = float(np.max(np.abs([a - b for a, b in chk])))
        log(f"  V4 我们算的 RibonanzaNet F1 与文件自带的 RNet_F1 列:最大偏差 {dev:.4f}(n={len(chk)})")
    same = [f1(parse_db(RNA.fold(w)[0]), parse_db(rs_mfe[w])) for w in wts if w in rs_mfe]
    log(f"  V2 ViennaRNA 与 RNAstructure 的 MFE 结构 F1 中位 "
        f"{np.median(same):.4f}(同一个近邻模型的两个独立实现,低了说明接线错了)")
    R_agree = {nm: round(float(np.mean(v)), 5) for nm, v in scores.items() if v}
    v2 = round(float(np.median(same)), 4)
    for nm, _k in IMPL:
        log(f"  {nm:14s} mean F1 = {R_agree.get(nm)}   (n={len(scores[nm])})")
    return R_agree, per, v2


def stage_reorder(R_agree, R_resp, per_agree, cells, IMPL, bypz, pz, ry, idx):
    """E2. Rankings side by side, tau between them, and for every pair that changes order a
    bootstrap interval on the difference of differences, resampled by target."""
    log("\n=== E2 重排检验")
    names = [n for n, _ in IMPL if n in R_agree and n in R_resp]
    rk = lambda d: {n: i + 1 for i, n in enumerate(sorted(names, key=lambda x: -d[x]))}
    ra, rr = rk(R_agree), rk(R_resp)
    log("  " + " ".join(f"{n}(agree {ra[n]} / resp {rr[n]})" for n in names))
    tau = kendall_tau({n: R_agree[n] for n in names}, {n: R_resp[n] for n in names})
    log(f"  Kendall tau(R_agree, R_resp) = {tau}")

    agree_by_pz = collections.defaultdict(list)
    for row in per_agree:
        agree_by_pz[row["puzzle"]].append(row)
    swaps = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A, B = names[i], names[j]
            da, dr = R_agree[A] - R_agree[B], R_resp[A] - R_resp[B]
            if (da > 0) == (dr > 0):
                continue                                    # same order on both readouts
            dd = []
            for ix in idx:
                sub = [c for q in ix for c in bypz[pz[q]]]
                va, _ = _wr(sub, A, B, ry)
                rows = [x for q in ix for x in agree_by_pz[pz[q]]]
                if not rows or va is None:
                    continue
                ga = float(np.mean([x["f1"][A] - x["f1"][B] for x in rows]))
                dd.append(va - ga)
            ci = _ci(dd)
            resolved = bool(dd) and (ci[0] > 0 or ci[1] < 0)
            swaps.append({"pair": [A, B], "agree_diff": round(da, 5), "resp_diff": round(dr, 5),
                          "dd_ci": ci, "resolved": resolved})
            log(f"  {A} vs {B}: agree {da:+.5f}  resp {dr:+.5f}  "
                f"差之差 CI {ci}  -> {'重排(区间不含 0)' if resolved else 'unresolved(区间含 0)'}")
    if not swaps:
        log("  两个读数给出同一顺序,没有可检验的换序对")
    return {"ranking_agree": ra, "ranking_resp": rr, "kendall_tau": tau, "swaps": swaps,
            "n_resolved_swaps": sum(1 for s in swaps if s["resolved"])}


M26_SRC = "/data/wenhui/openknot_data/Data/OK7a_M2R_data.v4.5.1.csv"
KEYOF = {}


def _wr(cells, A, B, ry):
    import m31_which_flips_rescue as M31
    a, _ = M31.weighted_rho(cells, KEYOF[A], ry)
    b, _ = M31.weighted_rho(cells, KEYOF[B], ry)
    return (None if (a is None or b is None) else a - b), None


def _ci(v):
    import m31_which_flips_rescue as M31
    return M31.boot_ci(list(v)) if v else [None, None]


def arm_V(pn):
    """printed BEFORE any ranking."""
    log("=== V 臂 验证(先于任何排名)")
    V = {}
    s, b, db = _rs_one("GGGAAACCCAAAGGGAAACCC")
    V["V1_rnastructure_smoke"] = {"n_pairs": len(b), "mfe": db,
                                  "max_p": round(max(b.values()), 4) if b else None}
    log(f"  V1 RNAstructure 冒烟:{db}  最大配对概率 {V['V1_rnastructure_smoke']['max_p']}")
    V["V2_note"] = ("ViennaRNA-vs-RNAstructure MFE agreement is measured in stage_agree, on the "
                    "same wild types both readouts use, and reported as V2_mfe_agreement")
    # V3: the cached neural probabilities reproduce M26c's published per-record values
    m26c = json.load(open(os.path.join(HERE, "m26c_neural_scored.json")))
    V["V3_neural_cache_present"] = len(pn)
    log(f"  V3 RibonanzaNet 缓存 {len(pn)} 条序列(M26c 写入,本次只读)")
    return V


def _variant_job(name):
    """one parameter set, in a FRESH process.

    `RNA.params_load_RNA_<set>()` has no effect in a process that has already built a fold
    compound: called a second time it silently leaves the model unchanged, and every parameter set
    then returns the previous one's numbers. M26b is correct only because it dispatches each set
    to its own ProcessPoolExecutor worker. Doing it in-process here produced Turner1999 == Turner2004
    to five decimals, which is what a no-op looks like when it looks like a result.
    """
    import RNA
    import m26_model_scored as M26
    getattr(RNA, f"params_load_RNA_{name}")()
    rv, _ = M26.build()
    return name, [((r["design"], r["a"], r["b"]), r["pmut"]) for r in rv]


def arm_E3(recs, cells, ry, bypz, pz, idx):
    """the parameter variants, in their own table. They are NOT predictors."""
    import m31_which_flips_rescue as M31
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    log("\n=== E3 参数变体(同一个 ViennaRNA 引擎换参数文件,不计入实现数)")
    out = {}
    base = {(r["design"], r["a"], r["b"]): r for r in recs}
    names = ("Turner1999", "Andronescu2007")
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as ex:
        results = dict(ex.map(_variant_job, names))
    for name in names:
        rv = [{"design": k[0], "a": k[1], "b": k[2], "pmut": v} for k, v in results[name]]
        got = 0
        for r in rv:
            t = base.get((r["design"], r["a"], r["b"]))
            if t is not None:
                t[f"v_{name}"] = r["pmut"]; got += 1
        KEYOF[name] = f"v_{name}"
        obs, _ = M31.weighted_rho(cells, f"v_{name}", ry)
        bs = []
        for ix in idx[:500]:
            sub = [c for q in ix for c in bypz[pz[q]]]
            v, _ = M31.weighted_rho(sub, f"v_{name}", ry)
            if v is not None:
                bs.append(v)
        out[name] = {"rho": round(obs, 5), "ci": M31.boot_ci(bs), "matched_records": got}
        log(f"  {name:16s} rho = {obs:+.5f}  CI {out[name]['ci']}")
    return out


def run():
    out, recs, cells, IMPL, rs_mfe, bypz, pz, ry, idx, t0 = stage1()
    KEYOF.update(dict(IMPL))
    z = np.load(os.path.join(HERE, 'm26c_pneural_cache.npz'), allow_pickle=True)
    out['V'] = arm_V({str(k) for k in z['keys']})
    R_agree, per_agree, v2 = stage_agree(recs, IMPL, rs_mfe)
    out["R_agree"] = R_agree
    out["V"]["V2_mfe_agreement_vienna_vs_rnastructure"] = v2
    out["E2"] = stage_reorder(R_agree, {n: out["R_resp"][n]["rho"] for n, _ in IMPL},
                              per_agree, cells, IMPL, bypz, pz, ry, idx)
    out["E3_parameter_variants"] = arm_E3(recs, cells, ry, bypz, pz, idx)
    impl_span = max(out["R_resp"][n]["rho"] for n, _ in IMPL) - \
        min(out["R_resp"][n]["rho"] for n, _ in IMPL)
    vr = [v["rho"] for v in out["E3_parameter_variants"].values()] + [out["R_resp"]["ViennaRNA"]["rho"]]
    out["span"] = {"implementations": round(impl_span, 5),
                   "parameter_variants": round(max(vr) - min(vr), 5) if len(vr) > 1 else None}
    log(f"\n  实现之间的跨度 {out['span']['implementations']:+.5f}   "
        f"参数变体之间 {out['span']['parameter_variants']}")

    # the reproducibility artifact: per-intervention scores, small enough to commit
    with open(SCORES, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["puzzle", "method", "a", "b", "pair_type", "depth", "crossed", "rescue"]
                   + [k for _n, k in IMPL])
        for r in recs:
            w.writerow([r["puzzle"], r["design"][1], r["a"], r["b"], r["pt"], r["depth"],
                        int(r["crossed"]), f"{r['rf']:.6f}"]
                       + [f"{r[k]:.6f}" for _n, k in IMPL])
    log(f"  写出逐干预分数表 {os.path.basename(SCORES)}({len(recs)} 行)")
    out["elapsed_s"] = round(time.time() - t0, 1)
    out["log"] = LOG
    json.dump(out, open(OUTJ, "w"), ensure_ascii=False, indent=1)
    log(f"写出 {OUTJ}  用时 {out['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(run())
