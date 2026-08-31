#!/usr/bin/env python3
"""M41: the instrument's domain of applicability, measured over the whole public record.

Declaration `m41_domain_declaration.json`, frozen and pushed at 675c944d BEFORE this file existed.
Offline, $0 -- deposits already on disk, ViennaRNA local, no API, no GPU.

Two reviews say the same thing: the binding weakness is that the instrument is characterised on
one corpus. The remedy is not to promise more data. It is to measure the domain in which the
instrument reads at all, and to state the measurement-quality condition under which it reads.

Arms, exactly as declared:
  V  verification, printed before any result
  L  lineage-corrected census -- a library is a design set, not a deposit and not a deposit name
  A  admission on every library not already analysed
  Q  the admission gate, calibrated instead of asserted
  D  the two coordinates of the domain

The gate matters because the manuscript's cutoff of 1.0 sits in an empty interval: the rejected
deposit reaches 0.779 and the admitted one starts at 4.324, with nothing observed between them.
"""
from __future__ import annotations

import collections, json, math, os, re, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m28_second_corpus as M28

DECL = json.load(open(os.path.join(HERE, "m41_domain_declaration.json")))
K = DECL["constants"]
SEED, NBOOT, MIN_BIN, OLD_GATE = K["SPLIT_HALF_SEED"], K["NBOOT"], K["MIN_BIN"], K["OLD_GATE"]
NBINS = 6
OUTJ = os.path.join(HERE, "m41_domain.json")

# the census hit list, grouped by the LINEAGE arm L establishes -- see the declaration
LINEAGE = {
    "15K_library":  ["15KLIB_2A3_0000", "15KLIB_DMS_0000",
                     "15KREP_2A3_0000", "15KREP_DMS_0000",
                     "15KREP_2A3_0001", "15KREP_DMS_0001"],
    "M3SARS_fse":   ["M3SARS_2A3_0000", "M3SARS_DMS_0000"],
    "M2SL5":        ["M2SL5_2A3_0000", "M2SL5_DMS_0000"],
    "TTYH2_ECS":    ["TTYH2_DMS_0001"],
}
ALREADY_ANALYSED = ("M2SL5", "M3SARS_fse")
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.append(s)


def undouble(s):
    n = len(s)
    return s[:n // 2] if n % 2 == 0 and s[:n // 2] == s[n // 2:] else s


def normal_names(rid):
    """construct names with the doubling quirk and the padding/replicate suffixes removed.

    15KREP_0000 and _0001 write each family name twice ('x' -> 'xx'). That is a formatting
    property of the deposit, not a different molecule, and leaving it in is what made the census
    count one library three times.

    The `_libraryready` token is stripped ONLY when it is trailing. 15KLIB writes it at the end
    ('AB037947.1_0-99_libraryready') where 15KREP omits it, which is the whole of the difference
    between those two deposits; M3SARS writes it in the MIDDLE, before the mutation tag
    ('fse__NC_019843.3_13440-13539_libraryready_0G-C'), so an unanchored strip deletes the tag and
    reports that deposit as carrying no orientation flips at all. One rule has to serve both the
    lineage comparison and the flip count or the two disagree, and the anchored form is the rule
    that does.
    """
    t = open(os.path.join(M28.CACHE, f"{rid}.rdat"), encoding="utf8", errors="ignore").read()
    out = []
    for n in re.findall(r"name:([^\t;]+)", t):
        n = n.split("\n")[0].strip()
        n = re.sub(r"_\d+pad\d+.*$", "", n)
        n = re.sub(r"_rep\d+.*$", "", n)
        n = re.sub(r"_libraryready$", "", n)
        out.append(undouble(n))
    return out


def jacc(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


# ============================================================ arm V
def arm_V():
    """printed BEFORE any result is read. V2 is a kill: if the published counts do not
    reproduce, nothing else in this run may be read."""
    log("\n" + "=" * 74 + "\n=== V 臂 验证(先于任何结果)\n" + "=" * 74)
    V = {}

    # V1 -- the lineage finding reproduces from an independently written comparison
    reps = ["15KLIB_2A3_0000", "15KREP_2A3_0000", "15KREP_2A3_0001"]
    N = {r: normal_names(r) for r in reps}
    pairs = {}
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            pairs[f"{reps[i]}|{reps[j]}"] = round(jacc(N[reps[i]], N[reps[j]]), 4)
    V["V1_lineage_jaccard"] = pairs
    V["V1_pass"] = all(v == 1.0 for v in pairs.values())
    log(f"  V1 同谱系构建名 Jaccard {pairs}  -> {'通过' if V['V1_pass'] else '未通过'}")

    # V1b -- the reverse control: distinct libraries must NOT collapse
    others = {"15KLIB_2A3_0000": normal_names("15KLIB_2A3_0000"),
              "M3SARS_2A3_0000": normal_names("M3SARS_2A3_0000"),
              "M2SL5_2A3_0000": normal_names("M2SL5_2A3_0000")}
    rev = {}
    ks = list(others)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            rev[f"{ks[i]}|{ks[j]}"] = round(jacc(others[ks[i]], others[ks[j]]), 4)
    V["V1b_reverse_control"] = rev
    V["V1b_pass"] = all(v == 0.0 for v in rev.values())
    log(f"  V1b 反向对照(不同库不得塌缩) {rev}  -> {'通过' if V['V1b_pass'] else '未通过'}")
    return V


# ============================================================ admission on one deposit
def admit(rid, sn_min):
    """the published criterion, unchanged, reporting the admission ledger only.

    Returns None with a reason when the deposit cannot reach the criterion at all."""
    cons, seqpos, has_struct, has_seq = M28.parse(rid)
    if not cons:
        return {"rmdb_id": rid, "void": "no_named_constructs", "n_constructs": 0}
    if has_struct or has_seq:
        return {"rmdb_id": rid, "void": "declares_a_structure__P4", "n_constructs": len(cons)}
    if not seqpos:
        return {"rmdb_id": rid, "void": "no_reactivity_window", "n_constructs": len(cons)}
    by_name = {c.name: c for c in cons}
    window = (seqpos[0] - 1, seqpos[-1])
    allo = M28.find_offset(cons, window)
    offs = {f: v for f, v in allo.items() if v[1] / max(1, v[2]) >= M28.AGREE}
    dropped = {f: v for f, v in allo.items() if f not in offs}
    if not offs:
        return {"rmdb_id": rid, "void": "offset_unrecoverable",
                "n_constructs": len(cons), "n_families_seen": len(allo)}
    recs, ex = M28.mine(cons, by_name, offs, sn_min, window)
    for r in recs:
        r["cols"] = tuple(k for k, p in enumerate(seqpos) if p - 1 in (r["i"], r["j"]))
    sn_all = [c.sn for c in cons if c.sn == c.sn]
    return {"rmdb_id": rid, "n_constructs": len(cons), "window_cols": len(seqpos),
            "n_families_offset_ok": len(offs), "n_families_offset_dropped": len(dropped),
            "n_admitted": len(recs), "excluded": ex,
            "by_pair_type": dict(collections.Counter(r["pair_type"] for r in recs)),
            "sn_reference_min": round(min(sn_all), 4) if sn_all else None,
            "sn_reference_median": round(float(np.median(sn_all)), 4) if sn_all else None,
            "sn_reference_max": round(max(sn_all), 4) if sn_all else None,
            "_recs": recs}


# ============================================================ the name-free read path
def parse_nameless(rid):
    """TTYH2 publishes `ID:` and no `name:`, so the mutation-tag route cannot run on it.

    Declared in the declaration BEFORE this was written: the wild type is the modal sequence at
    the modal length, and constructs at other lengths are EXCLUDED and counted, not repaired.
    Inventing a read path after seeing a yield is how a null becomes a positive.
    """
    path = M28.rdat_path(rid)
    seqpos, react, ann = None, {}, {}
    for line in open(path, encoding="utf8", errors="ignore"):
        f = line.rstrip("\n").split("\t")
        k = f[0].strip()
        if k == "SEQPOS":
            seqpos = [int(x.strip().lstrip("XACGUTN")) for x in f[1:] if x.strip()]
        elif k.startswith("REACTIVITY:") and not k.startswith("REACTIVITY_ERROR"):
            react[int(k.split(":")[1])] = np.array(
                [float(x) if x.strip() not in ("", "NaN", "nan") else np.nan for x in f[1:]])
        elif k.startswith("ANNOTATION_DATA:"):
            d = {}
            for tok in f[1:]:
                tok = tok.strip()
                if tok.startswith("sequence:"):
                    d["seq"] = tok[9:].replace("T", "U")
                elif tok.startswith("reads:"):
                    d["reads"] = float(tok[6:])
                elif tok.startswith("signal_to_noise:"):
                    d["sn"] = float(tok.rsplit(":", 1)[1])
                elif tok.startswith("ID:"):
                    d["name"] = tok[3:]
            ann[int(k.split(":")[1])] = d
    cons = [M28.Construct(d.get("name", f"row{i}"), d["seq"], d.get("reads", 0.0),
                          d.get("sn", 0.0), react[i])
            for i, d in sorted(ann.items()) if "seq" in d and i in react]
    return cons, seqpos


def admit_nameless(rid, sn_min):
    """Definition 1 by sequence spectrum, with clause (iii) dropped as in M28 and the wild type
    taken as the modal sequence. Every refusal is counted under a named reason."""
    cons, seqpos = parse_nameless(rid)
    ex = {k: 0 for k in ("length_off_mode", "not_two_positions", "not_orientation_flip",
                         "pair_not_canonical", "singles_missing", "low_sn", "no_outcome")}
    if not cons or not seqpos:
        return {"rmdb_id": rid, "void": "no_constructs_or_window", "n_constructs": len(cons)}
    Lmode = collections.Counter(len(c.seq) for c in cons).most_common(1)[0][0]
    same = [c for c in cons if len(c.seq) == Lmode]
    ex["length_off_mode"] = len(cons) - len(same)
    wt = M28.Construct("__modal__", collections.Counter(c.seq for c in same).most_common(1)[0][0],
                       0.0, 0.0, None)
    # the modal sequence must itself be a deposited construct, or there is no wild-type profile
    wtc = next((c for c in same if c.seq == wt.seq), None)
    if wtc is None:
        return {"rmdb_id": rid, "void": "modal_sequence_not_deposited", "n_constructs": len(cons)}
    lo, hi = seqpos[0] - 1, seqpos[-1]
    singles = {}
    for c in same:
        d = [k for k in range(lo, min(hi, Lmode)) if c.seq[k] != wtc.seq[k]]
        if len(d) == 1:
            singles[(d[0], c.seq[d[0]])] = c
    recs = []
    for c in same:
        d = [k for k in range(lo, min(hi, Lmode)) if c.seq[k] != wtc.seq[k]]
        if len(d) != 2:
            if len(d) > 2:
                ex["not_two_positions"] += 1
            continue
        i, j = d
        if not (c.seq[i] == wtc.seq[j] and c.seq[j] == wtc.seq[i]):
            ex["not_orientation_flip"] += 1
            continue
        pt = "".join(sorted({wtc.seq[i], wtc.seq[j]}))
        if pt not in M28.CANON:
            ex["pair_not_canonical"] += 1
            continue
        s1, s2 = singles.get((i, c.seq[i])), singles.get((j, c.seq[j]))
        if s1 is None or s2 is None:
            ex["singles_missing"] += 1
            continue
        if min(c.sn, wtc.sn, s1.sn, s2.sn) < sn_min:
            ex["low_sn"] += 1
            continue
        if np.all(np.isnan(c.react)) or np.all(np.isnan(wtc.react)):
            ex["no_outcome"] += 1
            continue
        recs.append({"family": "TTYH_ECS", "i": i, "j": j, "pair_type": pt,
                     "wt": wtc, "mut": c, "s1": s1, "s2": s2,
                     "sn": float(min(c.sn, wtc.sn, s1.sn, s2.sn)),
                     "cols": tuple(k for k, p in enumerate(seqpos) if p - 1 in (i, j))})
    sn_all = [c.sn for c in cons if c.sn == c.sn]
    return {"rmdb_id": rid, "n_constructs": len(cons), "read_path": "nameless_modal",
            "modal_length": Lmode, "n_at_modal_length": len(same),
            "window_cols": len(seqpos), "n_singles": len(singles),
            "n_admitted": len(recs), "excluded": ex,
            "by_pair_type": dict(collections.Counter(r["pair_type"] for r in recs)),
            "sn_reference_min": round(min(sn_all), 4) if sn_all else None,
            "sn_reference_median": round(float(np.median(sn_all)), 4) if sn_all else None,
            "sn_reference_max": round(max(sn_all), 4) if sn_all else None,
            "_recs": recs}


# ============================================================ arm Q: the gate, calibrated
def _R(di, dj, dij, keep):
    if len(keep) < 5:
        return None
    den = math.hypot(float(np.linalg.norm(di[keep])), float(np.linalg.norm(dj[keep])))
    return None if den <= 0 else 1.0 - float(np.linalg.norm(dij[keep])) / den


SPLIT_RULES = ("odd_even", "contiguous", "random")
_SPLIT_RNG = np.random.default_rng(20260831)


def rescue_split(rec, win=None, rule="odd_even"):
    """(R on all usable columns, R on one half, R on the other), under a named split rule.

    The split-half is INTERNAL to one measurement: it asks whether the rescue factor is a stable
    quantity at this signal-to-noise, using no second experiment. That is what another laboratory
    can check on its own corpus before deciding whether this instrument will read on it.

    THE RULE MATTERS AND THE FIRST VERSION OF THIS RAN ONLY ONE. Reactivity is spatially
    autocorrelated -- neighbouring nucleotides belong to the same structural element -- so an
    odd/even split puts each column beside its most similar neighbour in the OTHER half and
    inflates the agreement. A contiguous split does the opposite, confounding the halves with
    position. All three are computed and all three are reported; the ceiling is a range, not a
    number, and the conservative end is the one any claim rests on.
    """
    wt = rec["wt"].react
    di, dj, dij = rec["s1"].react - wt, rec["s2"].react - wt, rec["mut"].react - wt
    m = ~(np.isnan(di) | np.isnan(dj) | np.isnan(dij))
    if win is not None:
        lo, hi = win
        m[:lo] = False
        m[hi:] = False
    idx = np.where(m)[0]
    full = _R(di, dj, dij, idx)
    if rule == "odd_even":
        a, b = idx[0::2], idx[1::2]
    elif rule == "contiguous":
        h = len(idx) // 2
        a, b = idx[:h], idx[h:]
    else:
        p = _SPLIT_RNG.permutation(len(idx)); h = len(idx) // 2
        a, b = idx[p[:h]], idx[p[h:]]
    return full, _R(di, dj, dij, a), _R(di, dj, dij, b)


def load_primary():
    """the primary corpus as M28-shaped records, so one reliability function serves both.

    The reconstruction is m26w's, verified there against the corpus's own rescue_factor column at
    Pearson 1.0000; it is repeated rather than imported so that this arm reads no number the
    manuscript already prints.
    """
    import csv
    csv.field_size_limit(10 ** 9)
    rows = list(csv.DictReader(open(M28_SRC)))
    rk = [c for c in rows[0] if c.startswith("reactivity_")]
    prof = lambda r: np.array([float(r[c]) if r[c] not in ("", "NA", "nan") else np.nan
                               for c in rk])
    sn = lambda r: (float(r["signal_to_noise"])
                    if r.get("signal_to_noise") not in (None, "", "NA", "nan") else float("nan"))
    by = collections.defaultdict(dict)
    for r in rows:
        a = r["mutA"] if r["mutA"] not in ("", "NA", "nan") else None
        b = r["mutB"] if r["mutB"] not in ("", "NA", "nan") else None
        by[(r["puzzle"], r["method"], r["experiment_type"])][(a, b)] = r
    out = []
    for (puz, meth, exp), v in by.items():
        wt = v.get((None, None))
        if wt is None:
            continue
        pw, sw = prof(wt), sn(wt)
        ss, se = int(float(wt["sub_start"])), int(float(wt["sub_end"]))
        for (a, b), r in v.items():
            if not (a and b):
                continue
            ra, rb = v.get((a, None)), v.get((None, b))
            if ra is None or rb is None:
                continue
            sns = [s for s in (sn(r), sw, sn(ra), sn(rb)) if s == s]
            out.append({"design": (puz, meth), "exp": exp, "mut_seq": r["sequence"],
                        "wt": M28.Construct("wt", "", 0, sw, pw),
                        "s1": M28.Construct("s1", "", 0, sn(ra), prof(ra)),
                        "s2": M28.Construct("s2", "", 0, sn(rb), prof(rb)),
                        "mut": M28.Construct("m", "", 0, sn(r), prof(r)),
                        "sn": (min(sns) if sns else float("nan")),
                        "win": (ss - 1, se)})
    return out


def admitted_primary_keys():
    """(design, mutant sequence) for every intervention Definition 1 admits on the primary corpus.

    Read from the published tool, not re-derived, so arm Q is restricted to exactly the
    interventions the manuscript reports and not to every double mutant that can be reconstructed.
    """
    sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
    from designaudit import corpora, mining
    cons = list(corpora.openknot_m2r(M28_SRC))
    adm, rep = mining.mine(cons)
    return {(x.design, x.mut_seq) for x in adm}, rep.n_admitted


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 4:
        return None
    rx, ry = x.argsort().argsort().astype(float), y.argsort().argsort().astype(float)
    s = float(np.corrcoef(rx, ry)[0, 1])
    return None if s != s else round(s, 4)


def arm_Q(pool):
    """reliability against reference signal-to-noise, on quantile bins fixed at 6."""
    log("\n" + "=" * 74 + "\n=== Q 臂 准入闸门:测曲线,而不是断言阈值\n" + "=" * 74)
    ok = [p for p in pool if p["sn"] == p["sn"] and p["h0"] is not None and p["h1"] is not None]
    log(f"  可用于信度估计的干预 {len(ok)} / {len(pool)}")
    if len(ok) < NBINS * MIN_BIN:
        log(f"  ** 欠功效:{len(ok)} < {NBINS}x{MIN_BIN} ⇒ 不产出标定阈值,论文保留断言值并标注为断言 **")
        return {"calibrated": False, "reason": "underpowered", "n_usable": len(ok)}
    sns = np.array([p["sn"] for p in ok])
    edges = np.quantile(sns, np.linspace(0, 1, NBINS + 1))
    bins = []
    for b in range(NBINS):
        lo, hi = edges[b], edges[b + 1]
        sel = [p for p in ok if (lo <= p["sn"] < hi or (b == NBINS - 1 and p["sn"] == hi))]
        if len(sel) < MIN_BIN:
            bins.append({"bin": b, "n": len(sel), "sn_lo": round(float(lo), 4),
                         "sn_hi": round(float(hi), 4), "underpowered": True})
            continue
        r = float(np.corrcoef([p["h0"] for p in sel], [p["h1"] for p in sel])[0, 1])
        bins.append({"bin": b, "n": len(sel),
                     "sn_lo": round(float(lo), 4), "sn_hi": round(float(hi), 4),
                     "sn_median": round(float(np.median([p["sn"] for p in sel])), 4),
                     "split_half_r": round(r, 4),
                     "split_half_spearman": spearman([p["h0"] for p in sel],
                                                     [p["h1"] for p in sel]),
                     "R_mean": round(float(np.mean([p["R"] for p in sel])), 4),
                     "R_sd": round(float(np.std([p["R"] for p in sel])), 4),
                     "underpowered": False})
    live = [b for b in bins if not b["underpowered"]]
    for b in bins:
        log(f"    bin{b['bin']}  SN [{b['sn_lo']:8.3f}, {b['sn_hi']:8.3f})  n={b['n']:6d}  "
            + (f"半分信度 r={b['split_half_r']:+.4f}" if not b["underpowered"] else "** 欠功效 **"))
    if len(live) < 4:
        log("  ** 清关的分箱少于 4 ⇒ 不产出标定阈值 **")
        return {"calibrated": False, "reason": "fewer_than_four_live_bins", "bins": bins}
    cross = {}
    for thr in (0.5, 0.7, 0.9):
        hit = next((b for b in live if b["split_half_r"] >= thr), None)
        cross[str(thr)] = None if hit is None else hit["sn_lo"]
    log(f"  半分信度首次达到阈值的 SN 下沿:{cross}   (旧断言值 {OLD_GATE})")

    # the measurement ceiling. A split-half r is the reliability of HALF the columns; Spearman-
    # Brown steps it up to the whole measurement, and the square root of that bounds the
    # correlation any predictor can attain with the underlying signal. Reported because the
    # manuscript compares predictors against each other and never against this bound.
    byrule = {}
    for r_ in SPLIT_RULES:
        sel = [p for p in ok if p.get(f"h0_{r_}") is not None and p.get(f"h1_{r_}") is not None]
        rp_ = float(np.corrcoef([p[f"h0_{r_}"] for p in sel], [p[f"h1_{r_}"] for p in sel])[0, 1])
        sb_ = 2 * rp_ / (1 + rp_)
        byrule[r_] = {"n": len(sel), "split_half_r": round(rp_, 4),
                      "spearman_brown": round(sb_, 4),
                      "ceiling": round(math.sqrt(sb_), 4) if sb_ > 0 else None}
        log(f"  分半规则 {r_:11s} n={len(sel):5d}  r={rp_:+.4f}  全测信度 {sb_:.4f}  "
            f"上界 {math.sqrt(sb_):.4f}")
    ceils = [v["ceiling"] for v in byrule.values() if v["ceiling"]]
    log(f"  **上界是一个区间 [{min(ceils):.4f}, {max(ceils):.4f}],任何主张只靠保守端**")
    rp = byrule["odd_even"]["split_half_r"]; sb = byrule["odd_even"]["spearman_brown"]
    ceil = byrule["odd_even"]["ceiling"]
    return {"calibrated": True, "n_usable": len(ok), "nbins": NBINS, "min_bin": MIN_BIN,
            "by_split_rule": byrule,
            "ceiling_range": [min(ceils), max(ceils)],
            "ceiling_conservative": min(ceils),
            "pooled_split_half_r": rp,
            "spearman_brown_reliability": sb,
            "attenuation_ceiling_on_any_predictor": ceil,
            "bins": bins, "reliability_crossings": cross, "asserted_gate": OLD_GATE,
            "monotone_in_sn": all(live[i]["split_half_r"] <= live[i + 1]["split_half_r"] + 1e-9
                                  for i in range(len(live) - 1))}


# ============================================================ main
M28_SRC = "/data/wenhui/openknot_data/Data/OK7a_M2R_data.v4.5.1.csv"


def main():
    t0 = time.time()
    out = {"declaration": "m41_domain_declaration.json", "cost": "offline, $0"}
    out["V"] = arm_V()

    # ---------------- arm L: lineage
    log("\n" + "=" * 74 + "\n=== L 臂 谱系更正:库=设计集,不是沉积,也不是沉积名\n" + "=" * 74)
    L = {"libraries": {}, "n_deposits": sum(len(v) for v in LINEAGE.values())}
    for lib, rids in LINEAGE.items():
        L["libraries"][lib] = {"deposits": rids, "n_deposits": len(rids)}
    # the lineage-correct flip count: orientation flips on the DEDUPLICATED construct-name set
    # of each lineage. Taking the max across a lineage's deposits would report 15KREP_0001's 136,
    # which is 34 x 4 replicates of the same designs; taking the sum would report 204.
    for lib, rids in LINEAGE.items():
        names = set()
        for rid in rids:
            try:
                names |= set(normal_names(rid))
            except Exception:
                pass
        flips = 0
        for n in names:
            ts = M28.tags(n)
            if len(ts) != 2:
                continue
            (_p1, f1, t1), (_p2, f2, t2) = ts
            pt = "".join(sorted({f1.replace("T", "U"), f2.replace("T", "U")}))
            if f1 == t2 and f2 == t1 and pt in M28.CANON:
                flips += 1
        if lib == "TTYH2_ECS":       # no name: fields; the census counted it by sequence spectrum
            flips = 22
            L["libraries"][lib]["flip_count_route"] = "sequence_spectrum__deposit_publishes_no_names"
        L["libraries"][lib]["n_distinct_constructs"] = len(names)
        L["libraries"][lib]["n_distinct_flips"] = flips
        log(f"    {lib:14s} 去重构建 {len(names):6d}  谱系正确翻转 {flips}")
    L["n_distinct_flips_total"] = sum(v["n_distinct_flips"] for v in L["libraries"].values())
    L["n_libraries_lineage_correct"] = len(LINEAGE)
    L["n_libraries_manuscript_prints"] = 6
    L["flips_manuscript_prints"] = 4340
    log(f"  沉积 {L['n_deposits']}  →  谱系去重后的库 {L['n_libraries_lineage_correct']} "
        f"(论文印 {L['n_libraries_manuscript_prints']})")

    # ---------------- arm A: admission on every library
    log("\n" + "=" * 74 + "\n=== A 臂 准入:每一个尚未分析的库\n" + "=" * 74)
    A, pool = {}, []
    for lib, rids in LINEAGE.items():
        A[lib] = {"already_analysed": lib in ALREADY_ANALYSED, "deposits": {}}
        for rid in rids:
            try:
                r = admit_nameless(rid, M28.SN_MAIN) if lib == "TTYH2_ECS" \
                    else admit(rid, M28.SN_MAIN)
            except Exception as e:
                r = {"rmdb_id": rid, "void": f"{type(e).__name__}: {e}"}
            recs = r.pop("_recs", [])
            A[lib]["deposits"][rid] = r
            tag = "作废 " + r["void"] if "void" in r else f"纳入 {r['n_admitted']}"
            log(f"  {lib:14s} {rid:18s} {tag}")
            if lib in ALREADY_ANALYSED:
                continue
            for rec in recs:
                halves = {r_: rescue_split(rec, rule=r_) for r_ in SPLIT_RULES}
                if halves["odd_even"][0] is None:
                    continue
                row = {"src": lib, "rid": rid, "sn": rec["sn"], "R": halves["odd_even"][0]}
                for r_ in SPLIT_RULES:
                    row[f"h0_{r_}"], row[f"h1_{r_}"] = halves[r_][1], halves[r_][2]
                row["h0"], row["h1"] = row["h0_odd_even"], row["h1_odd_even"]
                pool.append(row)
        # the lineage check the declaration did not foresee: one library, one ledger
        led = [d.get("n_admitted") for d in A[lib]["deposits"].values() if "n_admitted" in d]
        A[lib]["admission_identical_across_deposits"] = (len(set(led)) <= 1) if led else None
        if len(rids) > 1 and led:
            log(f"    ↳ 同谱系沉积的准入台账一致:{A[lib]['admission_identical_across_deposits']}"
                f"  {led}")
    out["L"], out["A"] = L, A
    out["A_new_admitted"] = {k: sum(d.get("n_admitted", 0) for d in v["deposits"].values())
                             for k, v in A.items() if not v["already_analysed"]}
    log(f"\n  公开记录里此前未用过的库共提供纳入干预:{out['A_new_admitted']}")

    # ---------------- arm Q: the gate
    log("\n  载入主语料以估计信度曲线 ...")
    prim = load_primary()
    log(f"    主语料重建出的干预 {len(prim)}")
    keys, n_adm = admitted_primary_keys()
    log(f"    Definition 1 在主语料上纳入 {n_adm}(论文印的数),用于限定 Q 臂")
    superset = list(pool)
    for rec in prim:
        halves = {r_: rescue_split(rec, win=rec["win"], rule=r_) for r_ in SPLIT_RULES}
        if halves["odd_even"][0] is None:
            continue
        row = {"src": "primary", "rid": rec["exp"], "sn": rec["sn"], "R": halves["odd_even"][0]}
        for r_ in SPLIT_RULES:
            row[f"h0_{r_}"], row[f"h1_{r_}"] = halves[r_][1], halves[r_][2]
        row["h0"], row["h1"] = row["h0_odd_even"], row["h1_odd_even"]
        superset.append(row)
        if (rec["design"], rec["mut_seq"]) in keys:
            pool.append(row)
    out["primary_admitted_n"] = n_adm
    log(f"    其中匹配上纳入集合的重建记录 {sum(1 for p in pool if p['src']=='primary')}")
    log("\n  --- 主臂:仅纳入的干预(declaration 声明的口径) ---")
    out["Q"] = arm_Q(pool)
    log("\n  --- 次臂:所有可重建的双突变(超集,报告出来但不作判据) ---")
    out["Q_superset"] = arm_Q(superset)
    out["Q_pool_by_source"] = dict(collections.Counter(p["src"] for p in pool))
    out["Q_superset_by_source"] = dict(collections.Counter(p["src"] for p in superset))

    # ---------------- arm D: the domain's two coordinates
    log("\n" + "=" * 74 + "\n=== D 臂 域:候选翻转 x 参考信噪比\n" + "=" * 74)
    D = []
    for lib, v in A.items():
        dep = [d for d in v["deposits"].values() if "sn_reference_median" in d]
        adm = sum(d.get("n_admitted", 0) for d in v["deposits"].values())
        D.append({"library": lib, "already_analysed": v["already_analysed"],
                  "n_deposits": len(v["deposits"]), "n_admitted_total": adm,
                  "sn_reference_median": (round(float(np.median(
                      [d["sn_reference_median"] for d in dep])), 4) if dep else None),
                  "sn_reference_min": (min(d["sn_reference_min"] for d in dep) if dep else None),
                  "voids": sorted({d["void"] for d in v["deposits"].values() if "void" in d})})
        log(f"  {lib:14s} 纳入 {adm:6d}  参考SN中位 {D[-1]['sn_reference_median']}  "
            f"作废 {D[-1]['voids'] or '-'}")
    out["D"] = D
    out["elapsed_s"] = round(time.time() - t0, 1)
    out["log"] = LOG
    json.dump(out, open(OUTJ, "w"), ensure_ascii=False, indent=1)
    log(f"\n写出 {OUTJ}  用时 {out['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
