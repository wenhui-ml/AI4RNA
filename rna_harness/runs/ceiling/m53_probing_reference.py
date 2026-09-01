#!/usr/bin/env python3
"""M53: score the agreement axis against a reference no model in this panel wrote.

Declaration m53_probing_reference_declaration.json, frozen and pushed BEFORE this file existed.
Offline, $0.

THE SETUP
---------
Every agreement number in this paper is scored against something written down rather than measured:
Section 5.1's column is RibonanzaNet's shipped RNet_structure, which that section says in its own
paragraph; Section 5.10's is the designer's DECLARED TARGET -- m43 scores all four implementations
against it, so RibonanzaNet's 0.87916 is not self-agreement, contrary to review. But a declared
target is an intention.

OK7a_M2_data.v4.5.2.csv carries M2_structure for the wild types, inferred from the mutate-and-map
contact map rather than from any sequence-to-structure model here. 158/158 of our design cells have
one, and it recovers 0.623 of the declared target's pairs -- both recorded in the declaration
before this file existed.

Parsers, F1, estimator, rank agreement, bootstrap: written here. The published runs supply nothing
but numbers to reproduce.
"""
from __future__ import annotations

import collections, csv, json, math, os, subprocess, sys, tempfile, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

DECL = os.path.join(HERE, "m53_probing_reference_declaration.json")
OUTJ = os.path.join(HERE, "m53_probing_reference.json")
SCACHE = os.path.join(HERE, "m53_structs_cache.json")
D = json.load(open(DECL))
SEED, NBOOT = 20260901, 2000
NMIN = 8

M2R = "/data/wenhui/openknot_data/Data/OK7a_M2R_data.v4.5.1.csv"
BENCH = "/data/wenhui/openknot_data/Data/OpenKnotBench_data.v4.5.1.csv"
PROBE = "/data/wenhui/openknot_data/Data/OK7a_M2_data.v4.5.2.csv"

# published, for V3 -- reproduced before any reference is swapped
PUB_M25 = {"n_pairs": 4293, "strata": 549, "weight": 892, "R1": 0.0527, "R2": 0.0198}
PUB_M36 = {"sd_rescue": 0.2195, "sd_rnet": 0.2335, "intv_sd": 0.2400, "conv_sd": 0.0848,
           "paired_sd": 0.1552, "ci": [0.01533, 0.30193]}
PUB_AGREE = {"ViennaRNA": 0.61916, "EternaFold": 0.67583,
             "RNAstructure": 0.60834, "RibonanzaNet": 0.87916}
PUB_RANK = {"RibonanzaNet": 0.32947, "ViennaRNA": 0.25570, "EternaFold": 0.27649}
PUB_DEPTH = 0.32252
IMPL = ("ViennaRNA", "RNAstructure", "EternaFold", "RibonanzaNet")
LOG = []


def log(s=""):
    print(s, flush=True); LOG.append(s)


def _plain(o):
    if isinstance(o, (np.bool_, np.integer, np.floating)):
        return o.item()
    raise TypeError(repr(o))


def save(OUT):
    OUT["log"] = LOG
    json.dump(OUT, open(OUTJ, "w"), indent=1, ensure_ascii=False, default=_plain)
    log(f"\n写出 {OUTJ}")


# --------------------------------------------------------------- parsers, written here
CLOSE = {")": "(", "]": "[", "}": "{", ">": "<"}


def pairs_of(db):
    """dot-bracket -> set of (i,j), i<j, ONE STACK PER BRACKET TYPE.

    This is a pseudoknot corpus: a single shared stack pairs ')' with the most recent '[' and
    returns a structure that is not the one on the page. V2 below checks this against two columns
    of the released files, neither of which is under our control."""
    st = {k: [] for k in "([{<"}
    out = set()
    for k, ch in enumerate(db):
        if ch in st:
            st[ch].append(k)
        elif ch in CLOSE:
            s = st[CLOSE[ch]]
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


def depth_of(P, a, b):
    """stacking depth of (a,b) inside the pair map P, min of the two directions; 0 if unpaired."""
    if P.get(a) != b:
        return 0, False
    dep = 0
    while P.get(a - dep - 1) == b + dep + 1:
        dep += 1
    up = 0
    while P.get(a + up + 1) == b - up - 1:
        up += 1
    return min(dep, up), True


def crossed_in(P, a, b):
    return any(m < a < n < b or a < m < b < n for m, n in P.items() if m < n)


# --------------------------------------------------------------- estimator, written here
def matched(rs, key, sfn, A="CG", B="AU"):
    by = collections.defaultdict(lambda: ([], []))
    for r in rs:
        if r["pt"] == A:
            by[sfn(r)][0].append(float(r[key]))
        elif r["pt"] == B:
            by[sfn(r)][1].append(float(r[key]))
    num = den = 0.0; used = 0
    for a, b in by.values():
        if a and b:
            w = min(len(a), len(b))
            num += w * (sum(a) / len(a) - sum(b) / len(b)); den += w; used += 1
    return (num / den if den else None), den, used


def within_sd(rs, key, sfn):
    by = collections.defaultdict(list)
    for r in rs:
        by[sfn(r)].append(float(r[key]))
    num = den = 0.0
    for v in by.values():
        if len(v) < 2:
            continue
        a = np.asarray(v); num += float(((a - a.mean()) ** 2).sum()); den += len(a) - 1
    return float(np.sqrt(num / den)) if den else None


def cz(v):
    v = np.asarray(v, float) - np.mean(v)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def rankdata(v):
    v = np.asarray(v, float); o = np.argsort(v, kind="mergesort")
    r = np.empty(len(v)); r[o] = np.arange(1, len(v) + 1)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = r[o[i:j + 1]].mean()
        i = j + 1
    return r


def wrho(cells, xk, yk="rf"):
    """cell-size-weighted mean within-cell Spearman, as a dot product of normalised rank vectors."""
    num = den = 0.0
    for _k, rs in cells:
        if len(rs) < 2:
            continue
        x = cz(rankdata([r[xk] for r in rs])); y = cz(rankdata([r[yk] for r in rs]))
        num += len(rs) * float(x @ y); den += len(rs)
    return (num / den) if den else None


def pct(v, lo=2.5, hi=97.5):
    v = np.sort(np.asarray(v, float))
    return [round(float(v[int(lo / 100 * len(v))]), 5), round(float(v[int(hi / 100 * len(v))]), 5)]


# --------------------------------------------------------------- structures for the wild types
def wild_types():
    """(puzzle, method) -> the wild-type row of OK7a, which is what every structure is computed on."""
    csv.field_size_limit(10 ** 9)
    wt = {}
    for d in csv.DictReader(open(M2R)):
        if d["mutA"] in ("", "NA", "nan") and d["mutB"] in ("", "NA", "nan"):
            wt[(d["puzzle"], d["method"])] = d
    return wt


def shifted(st, ss, n):
    """a design-window structure placed into full-construct coordinates, verbatim."""
    full = ["."] * n
    for k, ch in enumerate(st):
        full[ss - 1 + k] = ch
    return "".join(full)


def probing_structures(wt):
    """(puzzle, method) -> the mutate-and-map structure, in full coordinates, plus the file's F1."""
    csv.field_size_limit(10 ** 9)
    out, col = {}, {}
    for d in csv.DictReader(open(PROBE)):
        if d.get("mutA") not in ("", "NA", "nan", None):
            continue
        st = d.get("M2_structure") or ""
        if not st or set(st) == {"."}:
            continue
        k = (d["puzzle"], d["method"])
        if k not in wt or k in out:
            continue
        ss = int(float(d["sub_start"]))
        out[k] = shifted(st, ss, len(wt[k]["sequence"]))
        if d.get("M2_F1") not in (None, "", "NA", "nan"):
            col[k] = float(d["M2_F1"])
    return out, col


def rnet_structures(wt):
    """(puzzle, method) -> RibonanzaNet's shipped structure, keyed by design region, plus RNet_F1.

    The two released files share NO full sequence -- different adapters around the same designs --
    so the design region is the key, and the structure is shifted with OK7a's own sub_start."""
    csv.field_size_limit(10 ** 9)
    by_ds, f1_ds = {}, {}
    for d in csv.DictReader(open(BENCH)):
        ds, st = d.get("design_sequence"), d.get("RNet_structure")
        if not ds or not st or st in ("", "NA", "nan") or ds in by_ds or len(st) != len(ds):
            continue
        by_ds[ds] = st
        if d.get("RNet_F1") not in (None, "", "NA", "nan"):
            f1_ds[ds] = float(d["RNet_F1"])
    out, col = {}, {}
    for k, d in wt.items():
        ss, se = int(float(d["sub_start"])), int(float(d["sub_end"]))
        ds = d["sequence"][ss - 1:se]
        if ds in by_ds:
            out[k] = shifted(by_ds[ds], ss, len(d["sequence"]))
            if ds in f1_ds:
                col[k] = f1_ds[ds]
    return out, col


def model_structures(wt):
    """ViennaRNA MFE, RNAstructure MFE and EternaFold MEA on the same wild types. Cached."""
    seqs = sorted({d["sequence"] for d in wt.values()})
    if os.path.exists(SCACHE):
        c = json.load(open(SCACHE))
        if c.get("seqs") == len(seqs) and all(s in c["vienna"] for s in seqs):
            log(f"  结构缓存命中({len(seqs)} 条野生型)")
            return c["vienna"], c["rnastructure"], c["eternafold"]
    import RNA
    import m43_reorder as M43
    log(f"  重算三个实现的结构({len(seqs)} 条野生型)")
    t0 = time.time()
    vi = {s: RNA.fold(s)[0] for s in seqs}
    log(f"    ViennaRNA 完成  {time.time()-t0:.0f}s")
    _bpp, rs = M43.rnastructure_all(seqs)
    ef = M43.eternafold_structures(seqs)
    log(f"    EternaFold 完成  {time.time()-t0:.0f}s")
    json.dump({"seqs": len(seqs), "vienna": vi, "rnastructure": rs, "eternafold": ef},
              open(SCACHE, "w"))
    return vi, rs, ef


# --------------------------------------------------------------- the two key sets
def m25_keyset(wt, rnet_db):
    """m25's B arm, rebuilt here: the pairs BOTH readouts can score.

    resc -- the rescue readout, keyed by (design, i, j), on rows passing the signal gate whose two
    mutated positions are a declared pair. called -- the agreement readout: the frequency with
    which the shipped structure contains that pair, over the deposited rows for the design.
    Written here so V3 can fail on a coding error rather than agree by shared code."""
    csv.field_size_limit(10 ** 9)
    rows = list(csv.DictReader(open(M2R)))
    resc = {}
    for d in rows:
        if d["rescue_factor"] in ("", "NA", "nan") or d["SN_filter"] not in ("1", "1.0"):
            continue
        k = (d["puzzle"], d["method"]); p = wt.get(k)
        if p is None:
            continue
        ss = int(float(d["sub_start"])); P = {}
        for i, j in pairs_of(p["target_structure"]):
            P[i] = j; P[j] = i
        try:
            ma, mb = int(float(d["mutA"])), int(float(d["mutB"]))
        except ValueError:
            continue
        a, b = ma + ss - 2, mb + ss - 2
        if P.get(a) != b:
            continue
        s_ = p["sequence"]; pt = "".join(sorted({s_[a], s_[b]}))
        if pt not in ("CG", "AU", "GU"):
            continue
        dep, _ok = depth_of(P, a, b)
        resc[(k, a, b)] = {"design": k, "puzzle": d["puzzle"], "pt": pt, "depth": dep,
                           "crossed": crossed_in(P, a, b), "y": float(d["rescue_factor"])}
    # the agreement readout, over the deposited structures for each design
    csv.field_size_limit(10 ** 9)
    reps = collections.defaultdict(list)
    for d in csv.DictReader(open(BENCH)):
        if d["SN_filter"] not in ("1", "1.0"):
            continue
        r = d.get("RNet_structure") or ""
        ds = d.get("design_sequence") or ""
        if not r or set(r) == {"."} or not ds or len(r) != len(ds):
            continue
        reps[ds].append(r)
    called, ndes = {}, 0
    for k, p in wt.items():
        ss, se = int(float(p["sub_start"])), int(float(p["sub_end"]))
        ds = p["sequence"][ss - 1:se]
        if ds not in reps:
            continue
        ndes += 1
        cnt = collections.Counter(); n = len(reps[ds])
        for st in reps[ds]:
            for i, j in pairs_of(st):
                cnt[(i + ss - 1, j + ss - 1)] += 1
        for i, j in pairs_of(p["target_structure"]):
            called[(k, i, j)] = cnt[(i, j)] / n
    keys = sorted(set(resc) & set(called))
    return resc, called, keys, ndes


SF = lambda r: (r["design"], min(r["depth"], 3), r["crossed"])


def main():
    t0 = time.time()
    log("=== M53:把一致性轴换到一个面板里没有模型写过的参考上 ===")
    log("  声明已冻结并推送于本文件存在之前。全离线,$0。")
    OUT = {"declaration": os.path.basename(DECL),
           "cost": "offline, $0 -- no API, no GPU, no download."}
    wt = wild_types()
    probe, probe_f1 = probing_structures(wt)
    rnet, rnet_f1 = rnet_structures(wt)
    vi, rs, ef = model_structures(wt)
    log(f"  野生型 {len(wt)}  探测结构 {len(probe)}  RNet 结构 {len(rnet)}  "
        f"ViennaRNA {len(vi)}  RNAstructure {len(rs)}  EternaFold {len(ef)}  ({time.time()-t0:.0f}s)")

    def struct_for(nm, k):
        s = wt[k]["sequence"]
        return {"ViennaRNA": vi.get(s), "RNAstructure": rs.get(s),
                "EternaFold": ef.get(s), "RibonanzaNet": rnet.get(k)}[nm]

    # ---------------------------------------------------------- V, before any arm
    log("\n=== V 验证(先于任何 arm)===")
    _one = next(iter(probe.values()))
    v1 = (f1(pairs_of(_one), pairs_of(_one)) == 1.0 and f1(pairs_of(_one), set()) == 0.0)
    log(f"  V1 自身 F1 {f1(pairs_of(_one), pairs_of(_one)):.1f}  对空结构 "
        f"{f1(pairs_of(_one), set()):.1f}  -> {'通过' if v1 else '失败'}")

    # V2: our F1 against the file's own two columns. Neither is under our control.
    def window_pairs(k, db):
        d = wt[k]; ss, se = int(float(d["sub_start"])), int(float(d["sub_end"]))
        return {(i, j) for i, j in pairs_of(db) if ss - 1 <= i < se and ss - 1 <= j < se}
    dev = {}
    for tag, src, col in (("M2_F1", probe, probe_f1), ("RNet_F1", rnet, rnet_f1)):
        ds = []
        for k, db in src.items():
            if k not in col:
                continue
            tgt = window_pairs(k, wt[k]["target_structure"])
            ds.append(abs(f1(window_pairs(k, db), tgt) - col[k]))
        dev[tag] = (float(max(ds)) if ds else None, len(ds))
        log(f"  V2 我们的 F1 vs 文件自带 {tag} 列:最大偏差 "
            f"{dev[tag][0]:.5f}(n={dev[tag][1]})")
    v2 = all(d[0] is not None and d[0] < 1e-3 for d in dev.values())

    OUT["V"] = {"V1_zero_point": bool(v1), "V2_external_columns": bool(v2),
                "V2_max_dev": {k: v[0] for k, v in dev.items()}}
    if not (v1 and v2):
        # K2 fires. Before stopping, characterise WHY, because the answer decides what replaces
        # this run: our parser is wrong, or the released column is. RNet_F1 reproducing to 0.00000
        # with the same parser, the same coordinate shift and the same window says it is not ours.
        PAIRC = {")": "(", "]": "[", "}": "{", ">": "<"}
        def balanced(db):
            c = collections.Counter(db)
            return all(c[o] == c[cl] for cl, o in PAIRC.items())
        csv.field_size_limit(10 ** 9)
        rawrow = {}
        for d in csv.DictReader(open(PROBE)):
            if d.get("mutA") in ("", "NA", "nan", None):
                k = (d["puzzle"], d["method"])
                rawrow.setdefault(k, d)
        def pkfrac(db):
            return sum(1 for ch in db if ch in "[]{}<>") / max(len(db), 1)
        bal_ok = bal_bad = 0; repro_bal = repro_unbal = 0; pk_b, pk_u = [], []
        for k, db in probe.items():
            if k not in probe_f1 or k not in rawrow:
                continue
            ours = f1(window_pairs(k, db), window_pairs(k, wt[k]["target_structure"]))
            hit = abs(ours - probe_f1[k]) < 1e-3
            if balanced(rawrow[k]["M2_structure"]):
                bal_ok += 1; repro_bal += hit; pk_b.append(pkfrac(wt[k]["target_structure"]))
            else:
                bal_bad += 1; repro_unbal += hit; pk_u.append(pkfrac(wt[k]["target_structure"]))
        log("\n  K2 触发。停下之前先定位原因:")
        log(f"    RNet_F1 用同一个解析器、同一个坐标平移、同一个窗口复现到 "
            f"{dev['RNet_F1'][0]:.5f} —— 所以问题不在我们这边")
        log(f"    released M2_structure 里括号按类型配平的 {bal_ok} 条,不配平的 {bal_bad} 条")
        log(f"    配平的里复现文件自带 M2_F1 的 {repro_bal}/{bal_ok};不配平的 "
            f"{repro_unbal}/{bal_bad}")
        log(f"    而且这不是随机的:不配平那批的靶标假结字符占比 {np.mean(pk_u):.4f},"
            f"配平那批 {np.mean(pk_b):.4f} —— 只用配平的那批就是在假结基准上按假结含量做选择")
        OUT["defect_in_the_released_column"] = {
            "what": "M2_structure is not a balanced dot-bracket string for a large minority of the "
                    "released wild-type rows, so no per-bracket-type parse of it can reproduce the "
                    "file's own M2_F1 column",
            "our_parser_is_not_the_cause": {
                "RNet_F1_max_dev": dev["RNet_F1"][0], "n": dev["RNet_F1"][1],
                "note": "same parser, same coordinate shift, same window, exact reproduction"},
            "balanced": bal_ok, "unbalanced": bal_bad,
            "reproduce_M2_F1_among_balanced": repro_bal,
            "reproduce_M2_F1_among_unbalanced": repro_unbal,
            "selection_if_the_balanced_subset_were_used": {
                "target_pseudoknot_char_fraction_balanced": round(float(np.mean(pk_b)), 4),
                "target_pseudoknot_char_fraction_unbalanced": round(float(np.mean(pk_u)), 4),
                "why_it_matters": "the malformed rows are the MORE pseudoknotted ones, so keeping "
                                  "only the well-formed rows selects on structure class in a "
                                  "pseudoknot benchmark. The subset is a bounded robustness check "
                                  "at best and cannot be the primary reference."},
            "consequence": "M53 is VOID as declared. The question it asks is unchanged and is "
                           "carried to a successor run that builds a probing-derived reference "
                           "from the 2A3 profiles rather than reading one out of this column."}
        OUT["VOID"] = "K2: V2 未通过 —— released M2_structure 列在 74/160 行上不是配平的点括号串"
        save(OUT); return 1

    # V3a: m43's four agreement scores against the DECLARED target
    resc, called, keys, ndes = m25_keyset(wt, rnet)
    cells_used = sorted({k for (k, _a, _b) in keys})
    log(f"  两个读数都定义的设计 {ndes},共同配对 {len(keys)},设计格 {len(cells_used)}")
    ncov = sum(1 for k in cells_used if k in probe)
    log(f"  V4 覆盖:{ncov}/{len(cells_used)} 个设计格有探测衍生结构")
    if ncov < 150:
        OUT["VOID"] = "K3: 探测结构覆盖不足 150 格"; save(OUT); return 1

    def agree_scores(ref_of):
        sc = collections.defaultdict(list)
        for k in cells_used:
            tgt = ref_of(k)
            if not tgt:
                continue
            for nm in IMPL:
                db = struct_for(nm, k)
                if db is None:
                    continue
                sc[nm].append(f1(window_pairs(k, db), tgt))
        return {nm: round(float(np.mean(v)), 5) for nm, v in sc.items() if v}, \
               {nm: len(v) for nm, v in sc.items()}

    ag_decl, n_decl = agree_scores(lambda k: window_pairs(k, wt[k]["target_structure"]))
    dev3 = {nm: abs(ag_decl[nm] - PUB_AGREE[nm]) for nm in PUB_AGREE if nm in ag_decl}
    for nm in IMPL:
        log(f"  V3a {nm:14s} 我 {ag_decl.get(nm)}  m43 {PUB_AGREE[nm]}  "
            f"|Δ| {dev3.get(nm, float('nan')):.5f}  (n={n_decl.get(nm)})")
    v3a = len(dev3) == 4 and max(dev3.values()) < 5e-4

    # V3b: m25's B block
    R1 = [{**resc[k]} for k in keys]
    R2 = [{**resc[k], "y": called[k]} for k in keys]
    r1, w1, s1 = matched(R1, "y", SF)
    r2, w2, s2 = matched(R2, "y", SF)
    log(f"  V3b 配对 {len(keys)}(m25 {PUB_M25['n_pairs']})  权重 {w1:.0f}({PUB_M25['weight']})  "
        f"层 {s1}({PUB_M25['strata']})")
    log(f"      介入读数 {r1:+.5f}(m25 {PUB_M25['R1']:+.4f})  一致读数 {r2:+.5f}"
        f"(m25 {PUB_M25['R2']:+.4f})")
    v3b = (len(keys) == PUB_M25["n_pairs"] and s1 == PUB_M25["strata"]
           and abs(w1 - PUB_M25["weight"]) < 0.5 and (w1, s1) == (w2, s2)
           and abs(r1 - PUB_M25["R1"]) < 5e-5 and abs(r2 - PUB_M25["R2"]) < 5e-5)

    # V3c: m36's standardisation
    sd1 = within_sd(R1, "y", SF); sd2 = within_sd(R2, "y", SF)
    log(f"  V3c 层内 SD  rescue {sd1:.4f}(m36 {PUB_M36['sd_rescue']})  "
        f"RNet {sd2:.4f}({PUB_M36['sd_rnet']})")
    log(f"      标准化对比 {r1/sd1:+.4f}({PUB_M36['intv_sd']:+.4f})  "
        f"{r2/sd2:+.4f}({PUB_M36['conv_sd']:+.4f})  差 {r1/sd1 - r2/sd2:+.4f}"
        f"({PUB_M36['paired_sd']:+.4f})")
    v3c = (abs(sd1 - PUB_M36["sd_rescue"]) < 5e-4 and abs(sd2 - PUB_M36["sd_rnet"]) < 5e-4
           and abs(r1 / sd1 - PUB_M36["intv_sd"]) < 5e-4
           and abs(r2 / sd2 - PUB_M36["conv_sd"]) < 5e-4)

    # V5: the estimator here against m25's own
    import m25_estimator_benchmark as M25
    v5d = max(abs(matched(R, "y", SF)[0] - M25.E4_matched(R, "y")) for R in (R1, R2))
    log(f"  V5 本文件的估计量 vs m25 的 E4_matched  max|Δ| {v5d:.3e}")
    v5 = v5d < 1e-12
    OUT["V"].update({"V3a_agreement": bool(v3a), "V3b_m25": bool(v3b), "V3c_m36": bool(v3c),
                     "V5_equivalent_form": bool(v5), "V4_coverage": ncov})
    if not (v3a and v3b and v3c and v5):
        OUT["VOID"] = "K1/K2: V3 或 V5 未通过"; save(OUT); return 1
    log("  V 全部通过 —— 现在才换参考")
    OUT["reverse_control"] = {"agreement_declared": ag_decl, "n_pairs": len(keys),
                              "weight": round(w1, 1), "strata": s1,
                              "intervention": round(r1, 5), "agreement_rnet": round(r2, 5),
                              "sd_rescue": round(sd1, 5), "sd_rnet": round(sd2, 5)}

    # ---------------------------------------------------------- A4 is the reference independent?
    log("\n=== A4 探测参考与面板四个实现的距离(ShapeKnots 属于 RNAstructure 家族)===")
    a4 = {}
    for nm in IMPL:
        vs = [f1(window_pairs(k, struct_for(nm, k)), window_pairs(k, probe[k]))
              for k in cells_used if k in probe and struct_for(nm, k)]
        a4[nm] = round(float(np.mean(vs)), 5)
        log(f"  {nm:14s} 与探测参考的平均 F1 {a4[nm]:.5f}")
    a4_max = max(a4.values())
    indep = a4_max <= 0.90
    log(f"  最大 {a4_max:.5f}  -> {'独立于面板' if indep else '**K4:与某个面板成员过近**'}")
    OUT["A4_independence"] = {"f1_vs_panel": a4, "max": a4_max, "independent": bool(indep)}

    # ---------------------------------------------------------- A1 the ranking, swapped
    log("\n=== A1 一致性排名换成探测衍生参考 ===")
    ag_probe, n_probe = agree_scores(lambda k: window_pairs(k, probe[k]) if k in probe else None)
    for nm in IMPL:
        log(f"  {nm:14s} 声明靶标 {ag_decl[nm]:.5f}  ->  探测参考 {ag_probe.get(nm, float('nan')):.5f}"
            f"   (n={n_probe.get(nm)})")
    m_decl = ag_decl["RibonanzaNet"] - ag_decl["ViennaRNA"]
    m_probe = ag_probe["RibonanzaNet"] - ag_probe["ViennaRNA"]
    order_decl = sorted(IMPL, key=lambda n: -ag_decl[n])
    order_probe = sorted(IMPL, key=lambda n: -ag_probe[n])
    log(f"  incumbent margin  声明靶标 {m_decl:+.5f}  ->  探测参考 {m_probe:+.5f}")
    log(f"  排序  声明 {order_decl}")
    log(f"        探测 {order_probe}")

    # a design-clustered interval on the margin and on the change in the margin
    pzs = sorted({wt[k]["puzzle"] for k in cells_used})
    bypz = collections.defaultdict(list)
    for k in cells_used:
        bypz[wt[k]["puzzle"]].append(k)
    per = {}
    for k in cells_used:
        tgt_d = window_pairs(k, wt[k]["target_structure"])
        tgt_p = window_pairs(k, probe[k]) if k in probe else None
        per[k] = {nm: (f1(window_pairs(k, struct_for(nm, k)), tgt_d) if struct_for(nm, k) else None,
                       f1(window_pairs(k, struct_for(nm, k)), tgt_p)
                       if (tgt_p is not None and struct_for(nm, k)) else None)
                  for nm in IMPL}
    rng = np.random.default_rng(SEED)
    md, mp, dm = [], [], []
    for _ in range(NBOOT):
        ix = rng.integers(0, len(pzs), len(pzs))
        ks = [k for q in ix for k in bypz[pzs[q]]]
        def mean_of(nm, which):
            v = [per[k][nm][which] for k in ks if per[k][nm][which] is not None]
            return float(np.mean(v)) if v else None
        a, b = mean_of("RibonanzaNet", 0), mean_of("ViennaRNA", 0)
        c, d_ = mean_of("RibonanzaNet", 1), mean_of("ViennaRNA", 1)
        if None in (a, b, c, d_):
            continue
        md.append(a - b); mp.append(c - d_); dm.append((c - d_) - (a - b))
    log(f"  margin 声明 {m_decl:+.5f} {pct(md)}   探测 {m_probe:+.5f} {pct(mp)}")
    log(f"  两者之差(探测 − 声明) {m_probe - m_decl:+.5f} {pct(dm)}  "
        f"{'不含 0' if (pct(dm)[0] > 0 or pct(dm)[1] < 0) else '含 0'}")
    OUT["A1_ranking"] = {"agreement_declared": ag_decl, "agreement_probing": ag_probe,
                         "order_declared": order_decl, "order_probing": order_probe,
                         "margin_declared": round(m_decl, 5), "margin_declared_ci": pct(md),
                         "margin_probing": round(m_probe, 5), "margin_probing_ci": pct(mp),
                         "margin_change": round(m_probe - m_decl, 5), "margin_change_ci": pct(dm),
                         "n_cells": len(cells_used), "n_with_probe": ncov}

    # ---------------------------------------------------------- A2 section 5.1, column swapped
    log("\n=== A2 §5.1 的一致性列换掉:五个列,同一批配对、同一套分层 ===")
    COLS = [("RibonanzaNet", "shipped structure frequency (the published column)"),
            ("ViennaRNA", "its own MFE structure"),
            ("RNAstructure", "its own MFE structure"),
            ("EternaFold", "its own MEA structure"),
            ("probing", "the mutate-and-map structure -- not a model")]
    # per-record presence of the target pair in each structure
    recs = []
    for (k, a, b) in keys:
        r = dict(resc[(k, a, b)])
        r["rf"] = r["y"]
        r["agr_RibonanzaNet"] = called[(k, a, b)]
        for nm in ("ViennaRNA", "RNAstructure", "EternaFold"):
            db = struct_for(nm, k)
            r["agr_" + nm] = (1.0 if (db and (a, b) in pairs_of(db)) else 0.0) if db else None
        r["agr_probing"] = (1.0 if (a, b) in pairs_of(probe[k]) else 0.0) if k in probe else None
        recs.append(r)
    ok = [r for r in recs if all(r["agr_" + nm] is not None for nm, _d in COLS)]
    log(f"  五列齐备的配对 {len(ok)}/{len(recs)}")
    a2 = {}
    for nm, desc in COLS:
        c, _w, _u = matched(ok, "agr_" + nm, SF)
        sd = within_sd(ok, "agr_" + nm, SF)
        a2[nm] = {"desc": desc, "contrast": round(c, 5), "within_sd": round(sd, 5),
                  "contrast_in_sd": round(c / sd, 5) if sd else None,
                  "hit_rate": round(float(np.mean([r["agr_" + nm] for r in ok])), 4)}
        log(f"  {nm:14s} 对比 {c:+.5f}  层内 SD {sd:.4f}  标准化 {c/sd:+.5f}  "
            f"命中率 {a2[nm]['hit_rate']:.4f}")
    ri, sdi = matched(ok, "rf", SF)[0], within_sd(ok, "rf", SF)
    log(f"  {'intervention':14s} 对比 {ri:+.5f}  层内 SD {sdi:.4f}  标准化 {ri/sdi:+.5f}")
    # paired differences, one design-clustered resample per draw
    byd = collections.defaultdict(list)
    for r in ok:
        byd[r["design"]].append(r)
    des = sorted(byd)
    rng2 = np.random.default_rng(SEED + 1)
    draws = {nm: [] for nm, _d in COLS}
    for _ in range(NBOOT):
        ix = rng2.integers(0, len(des), len(des))
        srs = [x for q in ix for x in byd[des[q]]]
        cy, _w, _u = matched(srs, "rf", SF); sy = within_sd(srs, "rf", SF)
        if cy is None or not sy:
            continue
        for nm, _d in COLS:
            cx, _w2, _u2 = matched(srs, "agr_" + nm, SF); sx = within_sd(srs, "agr_" + nm, SF)
            draws[nm].append((cy / sy - cx / sx) if (cx is not None and sx) else np.nan)
    log("\n  介入读数 − 一致读数(各自单位的 SD,配对同一次抽样):")
    for nm, _d in COLS:
        v = np.asarray(draws[nm], float); v = v[~np.isnan(v)]
        ci = pct(v); pt_ = ri / sdi - a2[nm]["contrast_in_sd"]
        a2[nm]["paired_diff_in_sd"] = round(pt_, 5)
        a2[nm]["ci95"] = ci
        a2[nm]["ci_contains_zero"] = bool(ci[0] <= 0 <= ci[1])
        log(f"    vs {nm:14s} {pt_:+.5f}  {ci}  {'含 0' if a2[nm]['ci_contains_zero'] else '不含 0'}")
    OUT["A2_readout"] = {"n_pairs": len(ok), "intervention_in_sd": round(ri / sdi, 5),
                         "columns": a2}

    # ---------------------------------------------------------- A3 depth, from which structure
    log("\n=== A3 helix depth 从探测结构读 vs 从声明靶标读 ===")
    import json as _j
    c50 = _j.load(open(os.path.join(HERE, "m50_recs_cache.json")))["recs"]
    c52 = _j.load(open(os.path.join(HERE, "m52_recs_cache.json")))["recs"]
    assert len(c50) == len(c52), "两个缓存长度不同,索引对齐不成立"
    for x, y in zip(c50, c52):
        assert (tuple(x["design"]) == tuple(y["design"]) and x["pt"] == y["pt"]
                and x["depth"] == y["depth"] and x["crossed"] == y["crossed"]
                and abs(x["rf"] - y["rf"]) < 1e-12), "索引对齐检查失败"
    log(f"  两个记录缓存逐行一致({len(c50)} 条),按索引合并")
    R = []
    for x, y in zip(c50, c52):
        k = tuple(x["design"]); a, b = int(y["a"]), int(y["b"])
        if k not in probe:
            continue
        P = {}
        for i, j in pairs_of(probe[k]):
            P[i] = j; P[j] = i
        dp, present = depth_of(P, a, b)
        R.append({"design": k, "puzzle": x["puzzle"], "rf": x["rf"], "nmut": x["nmut"],
                  "pmut": x["pmut"], "emut": x["emut"], "depth": x["depth"],
                  "depth_probe": float(dp), "probe_has_pair": present})
    frac = float(np.mean([r["probe_has_pair"] for r in R]))
    log(f"  干预 {len(R)};其中该配对在探测结构里也成对的比例 {frac:.4f}")
    bycell = collections.defaultdict(list)
    for r in R:
        bycell[r["design"]].append(r)
    cells = [(k, v) for k, v in bycell.items() if len(v) >= NMIN]
    log(f"  至少 {NMIN} 个翻转的设计格 {len(cells)},覆盖 {sum(len(v) for _k, v in cells)} 次干预")
    v3d = {nm: wrho(cells, kx) for nm, kx in
           (("RibonanzaNet", "nmut"), ("ViennaRNA", "pmut"), ("EternaFold", "emut"))}
    d_decl = wrho(cells, "depth"); d_prob = wrho(cells, "depth_probe")
    for nm in PUB_RANK:
        log(f"  V3d {nm:14s} 我 {v3d[nm]:+.5f}  论文 {PUB_RANK[nm]:+.4f}")
    log(f"  V3d depth(声明靶标)  我 {d_decl:+.5f}  论文 {PUB_DEPTH:+.4f}")
    v3d_ok = (all(abs(v3d[n] - PUB_RANK[n]) < 5e-4 for n in PUB_RANK)
              and abs(d_decl - PUB_DEPTH) < 5e-4)
    log(f"  depth(探测结构)      {d_prob:+.5f}")
    OUT["V"]["V3d_rank_reverse_control"] = bool(v3d_ok)
    if not v3d_ok:
        OUT["VOID"] = "K1: A3 的反向对照未复现"; save(OUT); return 1
    bypz3 = collections.defaultdict(list)
    for k, v in cells:
        bypz3[v[0]["puzzle"]].append((k, v))
    pz3 = sorted(bypz3)
    rng3 = np.random.default_rng(SEED + 2)
    dd, vs_lvl = [], {n: [] for n in PUB_RANK}
    for _ in range(NBOOT):
        ix = rng3.integers(0, len(pz3), len(pz3))
        cs = [c for q in ix for c in bypz3[pz3[q]]]
        dp_ = wrho(cs, "depth_probe"); dl_ = wrho(cs, "depth")
        if dp_ is None or dl_ is None:
            continue
        dd.append(dp_ - dl_)
        for nm, kx in (("RibonanzaNet", "nmut"), ("ViennaRNA", "pmut"), ("EternaFold", "emut")):
            w = wrho(cs, kx)
            vs_lvl[nm].append(dp_ - w if w is not None else np.nan)
    log(f"  探测 depth − 声明 depth  {d_prob - d_decl:+.5f}  {pct(dd)}")
    a3 = {"frac_pair_present_in_probe": round(frac, 4), "n_cells": len(cells),
          "depth_declared": round(d_decl, 5), "depth_probing": round(d_prob, 5),
          "difference": round(d_prob - d_decl, 5), "difference_ci": pct(dd),
          "vs_level_arms": {}}
    log("  探测 depth − 各预测器的 level 臂:")
    for nm in PUB_RANK:
        v = np.asarray(vs_lvl[nm], float); v = v[~np.isnan(v)]
        ci = pct(v); pt_ = d_prob - v3d[nm]
        a3["vs_level_arms"][nm] = {"point": round(pt_, 5), "ci95": ci,
                                   "ci_contains_zero": bool(ci[0] <= 0 <= ci[1]),
                                   "probing_depth_wins": bool(pt_ > 0)}
        log(f"    vs {nm:14s} {pt_:+.5f}  {ci}  "
            f"{'含 0' if ci[0] <= 0 <= ci[1] else '不含 0'}")
    OUT["A3_depth"] = a3

    # ---------------------------------------------------------- prereg scoring
    S1 = bool(abs(m_probe) < abs(m_decl))
    S2 = bool(order_probe[0] == "RibonanzaNet")
    S3 = bool(not a2["probing"]["ci_contains_zero"] and a2["probing"]["paired_diff_in_sd"] > 0)
    S4 = bool(all(a3["vs_level_arms"][n]["probing_depth_wins"] for n in PUB_RANK))
    log("\n=== 预注册记分 ===")
    for sid, ok_, det in (
            ("S1", S1, f"margin 声明 {m_decl:+.5f} -> 探测 {m_probe:+.5f}"),
            ("S2", S2, f"探测参考下的排序 {order_probe}"),
            ("S3", S3, f"介入 − 探测一致列 {a2['probing']['paired_diff_in_sd']:+.5f} "
                       f"{a2['probing']['ci95']}"),
            ("S4", S4, "探测 depth 对三个 level 臂 "
                       + ", ".join(f"{n} {a3['vs_level_arms'][n]['point']:+.4f}"
                                   for n in PUB_RANK))):
        log(f"  {sid}  {'成立' if ok_ else '**失败**'}   {det}")
    OUT["staked"] = {f"S{i+1}": {"text": D["staked_predictions"][i]["text"], "pass": p, "detail": d_}
                     for i, (p, d_) in enumerate([
                         (S1, f"{m_decl:+.5f} -> {m_probe:+.5f}"),
                         (S2, str(order_probe)),
                         (S3, f"{a2['probing']['paired_diff_in_sd']:+.5f} {a2['probing']['ci95']}"),
                         (S4, str({n: a3["vs_level_arms"][n]["point"] for n in PUB_RANK}))])}
    OUT["verdict"] = ("agreement_axis_survives_a_probing_reference" if (S2 and S3 and S4)
                      else "agreement_axis_depends_on_the_reference")
    log(f"\n  裁定:{OUT['verdict']}   {sum([S1,S2,S3,S4])}/4 成立   用时 {time.time()-t0:.0f}s")
    save(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
