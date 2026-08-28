#!/usr/bin/env python3
"""M28: does the instrument transfer to natural RNA?

Declaration `m28_second_corpus_declaration.json`, frozen and pushed at 066f4ba1 BEFORE this file
existed. Offline, $0 -- RMDB public deposits streamed, ViennaRNA local, RibonanzaNet local weights
on a local GPU.

The sentence this paper is most easily attacked on is "this only holds on RNA that was designed".
The second corpus does not answer "is there more data"; it answers **whether the mining step and
the scoring step transfer to natural RNA**. M2SL5 and M3SARS are coronavirus 5'UTR stem-loops and
frameshift elements: natural molecules, independently deposited, sharing no molecule with OK7a.

Two things about these deposits change the criterion, and the declaration wrote both down first:

  * They declare no target structure (STRUCTURE is all dots, SEQUENCE all X). The OK7a admission
    clause "the two positions are a pair in the declared target structure" therefore cannot run.
    Dropping it is what makes this test *harder*, not easier: every position pair was mutated, so
    the model must predict WHICH flips rescue and which do not, rather than score already-known
    pairs. P4 asserts no structure is read anywhere in this file.

  * `depth` and `crossed` do not exist without a structure, so stratification is (family) alone.
    The retention numbers here are therefore NOT comparable word-for-word with OK7a's, and arm E
    is pre-judged underpowered: **an interval covering zero here may not be read as a negative.**

The outcome is the OK7a formula, unchanged, including its treatment of the mutated positions:
`R = 1 - ||d_ij|| / hypot(||d_i||, ||d_j||)` over the deposit's reactivity window, where OK7a
masks only NaNs and the window edges and does *not* excise the mutated columns (verified against
`m26w_rescue_provenance.py`). Because DMS reads only A and C, a flip changes the chemistry at
those two columns, so an excise-the-mutated-columns variant runs alongside as a declared
sensitivity and both are reported. Consistency with OK7a decides which one is primary.
"""
from __future__ import annotations

import argparse, collections, json, math, os, re, sys, time, urllib.request
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DETAIL = "https://rmdb.stanford.edu/detail/{}/"
CACHE = os.path.join(HERE, "rmdb_cache")
DEPOSITS = ("M2SL5_DMS_0000", "M2SL5_2A3_0000", "M3SARS_DMS_0000", "M3SARS_2A3_0000")

CANON = ("AU", "CG", "GU")
SN_MAIN = 1.0                    # P2: fixed once, never tuned afterwards
SN_SENS = (0.5, 2.0)
NBINS, NPERM = 5, 2000
DELTAS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 1.00)
SELF_MAX = 0.15                  # P3: the delta=0 self-check
AGREE = 0.99                     # a family's offset must be this constant or the family is dropped
MUT = re.compile(r"_(\d+)([ACGUT])-([ACGUT])")


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------- rdat parsing
class Construct:
    __slots__ = ("name", "seq", "reads", "sn", "react")

    def __init__(self, name, seq, reads, sn, react):
        self.name, self.seq, self.reads, self.sn, self.react = name, seq, reads, sn, react


def fetch(url, dest, timeout=600):
    if dest and os.path.exists(dest) and os.path.getsize(dest) > 0:
        return open(dest, "rb").read()
    req = urllib.request.Request(url, headers={"User-Agent": "rna-harness/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        b = r.read()
    if dest:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        open(dest, "wb").write(b)
    return b


def rdat_path(rid):
    p = os.path.join(CACHE, f"{rid}.rdat")
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return p
    html = fetch(DETAIL.format(rid), os.path.join(CACHE, f"{rid}.html")).decode("utf8", "ignore")
    m = re.search(r'href="(https://[^"]*?%s\.rdat)"' % re.escape(rid), html)
    if not m:
        raise RuntimeError(f"{rid}: detail page carries no rdat link")
    fetch(m.group(1), p)
    return p


def parse(rid):
    """constructs + the reactivity window, from the deposit's own annotations.

    P4 lives here: STRUCTURE and SEQUENCE are read only to ASSERT that the deposit declares
    neither, and their contents never leave this function.
    """
    path = rdat_path(rid)
    seqpos, react, ann, struct, refseq = None, {}, {}, None, None
    for line in open(path, encoding="utf8", errors="ignore"):
        f = line.rstrip("\n").split("\t")
        k = f[0].strip()
        if k == "SEQPOS":
            seqpos = [int(x.strip().lstrip("XACGUTN")) for x in f[1:] if x.strip()]
        elif k == "STRUCTURE":
            struct = f[1].strip() if len(f) > 1 else ""
        elif k == "SEQUENCE":
            refseq = f[1].strip() if len(f) > 1 else ""
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
                elif tok.startswith("name:"):
                    d["name"] = tok[5:]
            ann[int(k.split(":")[1])] = d

    declares_structure = bool(struct) and set(struct) != {"."}
    declares_sequence = bool(refseq) and set(refseq) != {"X"}
    cons = []
    for i, d in sorted(ann.items()):
        if "name" not in d or "seq" not in d or i not in react:
            continue
        cons.append(Construct(d["name"], d["seq"], d.get("reads", 0.0), d.get("sn", 0.0), react[i]))
    return cons, seqpos, declares_structure, declares_sequence


# ----------------------------------------------------------------------------- names
def tags(name):
    """the mutations a construct's NAME declares, deduplicated by position.

    Some deposits repeat `<id>_<pos><from>-<to>` inside one name and the id itself carries digits,
    so a raw findall reports a single mutant as a double. Two mutations at one position are
    impossible; a repeat is a naming quirk. This is the m27 fix, kept.
    """
    return sorted({m[0]: m for m in MUT.findall(name)}.values(), key=lambda m: int(m[0]))


def wt_name(name):
    """the wild type's name is the mutant's name with every mutation tag deleted.

    Verified on M2SL5: `SL5_SARS_CoV_2_0G-A_5pad6_w53barcode` minus `_0G-A` is exactly
    `SL5_SARS_CoV_2_5pad6_w53barcode`, which is the wild-type row. Family-prefix matching (what
    m27 used for counting) would NOT work here: the padding suffix survives on the wild type and
    is stripped from nothing, so wild type and mutant would land in different families.
    """
    return MUT.sub("", name)


def single_name(name, keep):
    """the same construct with only the `keep`-th of its two tags left in place."""
    ts = tags(name)
    drop = ts[1 - keep]
    return name.replace(f"_{drop[0]}{drop[1]}-{drop[2]}", "", 1)


def find_offset(cons, by_name, window):
    """position tags are 0-based inside the designed region; sequences carry padding.

    The offset is PER FAMILY, not per deposit. M2SL5's construct names say so out loud --
    `SL5_SARS_CoV_2_5pad6_...` carries five nucleotides of 5' pad and `SL5_MERS_GCadded_0pad0_...`
    carries none -- and the measured offsets are 31 and 26 accordingly. A single deposit-wide
    offset silently discarded two of this deposit's three families, which is why this is computed
    and asserted per family rather than once.

    Diffs are taken INSIDE the reactivity window only. Every construct here carries its own
    barcode hairpin 3' of the designed region, so a whole-sequence comparison reports a single
    mutant as differing at ~15 positions and no offset is ever recovered. This is the barcode that
    produced m27's second false negative, biting a third time; the OK7a criterion's "inside the
    design window" clause is what prevents it, and dropping the structure clause must not drop
    this one with it.

    Returns {family: (offset, n_agreeing, n_seen)}. A family whose offset is not constant to
    within `AGREE` is left out and reported, never fitted per construct.
    """
    lo, hi = window
    per = collections.defaultdict(collections.Counter)
    for c in cons:
        ts = tags(c.name)
        if len(ts) != 1:
            continue
        fam = wt_name(c.name)
        wt = by_name.get(fam)
        if wt is None or len(wt.seq) != len(c.seq):
            continue
        d = [k for k in range(lo, hi) if c.seq[k] != wt.seq[k]]
        if len(d) == 1:
            per[fam][d[0] - int(ts[0][0])] += 1
    out = {}
    for fam, seen in per.items():
        off, n = seen.most_common(1)[0]
        out[fam] = (off, n, sum(seen.values()))
    return out


# ----------------------------------------------------------------------------- arm A: mining
EXCLUSIONS = ("no_wild_type", "length_mismatch", "not_two_positions", "not_orientation_flip",
              "pair_not_canonical", "tag_position_mismatch", "singles_missing", "low_sn",
              "no_outcome")


def mine(cons, by_name, offsets, sn_min, window):
    """the OK7a criterion with exactly one clause dropped, and the drop is declared.

    Dropped: "the two positions form a pair in the declared target structure". These deposits
    declare no structure, and because the scan is exhaustive the clause is not doing selection
    work here -- every position pair was mutated.
    """
    ex = {k: 0 for k in EXCLUSIONS}
    out = []
    for c in cons:
        ts = tags(c.name)
        if len(ts) != 2:
            if len(ts) > 2:
                ex["not_two_positions"] += 1
            continue
        (p1, f1, t1), (p2, f2, t2) = ts
        if not (f1 == t2 and f2 == t1):
            ex["not_orientation_flip"] += 1
            continue
        pt = "".join(sorted({f1.replace("T", "U"), f2.replace("T", "U")}))
        if pt not in CANON:
            ex["pair_not_canonical"] += 1
            continue
        fam = wt_name(c.name)
        wt = by_name.get(fam)
        if wt is None or fam not in offsets:
            ex["no_wild_type"] += 1
            continue
        if len(wt.seq) != len(c.seq):
            ex["length_mismatch"] += 1
            continue
        offset = offsets[fam][0]
        i, j = int(p1) + offset, int(p2) + offset
        lo, hi = window
        d = [k for k in range(lo, hi) if c.seq[k] != wt.seq[k]]
        if sorted(d) != sorted([i, j]):
            ex["tag_position_mismatch"] += 1
            continue
        s1 = by_name.get(single_name(c.name, 0))
        s2 = by_name.get(single_name(c.name, 1))
        if s1 is None or s2 is None:
            ex["singles_missing"] += 1
            continue
        if min(c.sn, wt.sn, s1.sn, s2.sn) < sn_min:
            ex["low_sn"] += 1
            continue
        if np.all(np.isnan(c.react)) or np.all(np.isnan(wt.react)):
            ex["no_outcome"] += 1
            continue
        out.append({"family": fam, "i": i, "j": j, "pair_type": pt,
                    "wt": wt, "mut": c, "s1": s1, "s2": s2,
                    "sn": float(min(c.sn, wt.sn, s1.sn, s2.sn))})
    return out, ex


# ----------------------------------------------------------------------------- arm B: outcome
def rescue(rec, excise=False):
    """OK7a's formula, unchanged. `excise` is the declared sensitivity, not the primary."""
    wt = rec["wt"].react
    di, dj, dij = rec["s1"].react - wt, rec["s2"].react - wt, rec["mut"].react - wt
    m = ~(np.isnan(di) | np.isnan(dj) | np.isnan(dij))
    if excise:
        for col in rec.get("cols", ()):
            if 0 <= col < len(m):
                m[col] = False
    if m.sum() < 10:
        return None
    den = math.hypot(float(np.linalg.norm(di[m])), float(np.linalg.norm(dj[m])))
    if den <= 0:
        return None
    return 1.0 - float(np.linalg.norm(dij[m])) / den


def wire_check(rec):
    """K0's wiring half: a double that IS the wild type must score exactly 1.0."""
    wt = rec["wt"].react
    di, dj = rec["s1"].react - wt, rec["s2"].react - wt
    m = ~(np.isnan(di) | np.isnan(dj))
    den = math.hypot(float(np.linalg.norm(di[m])), float(np.linalg.norm(dj[m])))
    return None if den <= 0 else 1.0 - 0.0 / den


# ----------------------------------------------------------------------------- estimators
def within_slope(sid, x, y):
    by = collections.defaultdict(list)
    for k in range(len(sid)):
        by[sid[k]].append(k)
    sxy = sxx = 0.0
    n = 0
    for _s, ix in by.items():
        if len(ix) < 2:
            continue
        xa = x[ix] - x[ix].mean()
        ya = y[ix] - y[ix].mean()
        sxy += float((xa * ya).sum())
        sxx += float((xa * xa).sum())
        n += len(ix)
    return (sxy / sxx if sxx else None), n, len(by)


def slope_perm(sid, x, y, nperm=NPERM, seed=0):
    """rule 80: the OUTCOME is permuted inside the design, never the treatment."""
    obs, n, ns = within_slope(sid, x, y)
    if obs is None:
        return None, 1.0, 0, 0
    rng = np.random.default_rng(seed)
    by = collections.defaultdict(list)
    for k in range(len(sid)):
        by[sid[k]].append(k)
    ge = 0
    for _ in range(nperm):
        yp = y.copy()
        for _s, ix in by.items():
            a = np.asarray(ix)
            yp[a] = y[a[rng.permutation(len(a))]]
        s, _n, _k = within_slope(sid, x, yp)
        ge += (s is not None and abs(s) >= abs(obs))
    return float(obs), (ge + 1) / (nperm + 1), int(n), int(ns)


def calibration(sid, x, y, nbins=NBINS):
    edges = [float(np.quantile(x, q)) for q in np.linspace(0, 1, nbins + 1)[1:-1]]
    b = np.searchsorted(edges, x, side="right")
    cen = np.empty(len(y))
    by = collections.defaultdict(list)
    for k in range(len(sid)):
        by[sid[k]].append(k)
    for _s, ix in by.items():
        cen[ix] = y[ix] - y[ix].mean()
    rows = []
    for k in range(nbins):
        m = b == k
        rows.append({"bin": k, "n": int(m.sum()),
                     "pred": round(float(x[m].mean()), 4) if m.any() else None,
                     "rescue": round(float(y[m].mean()), 4) if m.any() else None,
                     "rescue_centred": round(float(cen[m].mean()), 4) if m.any() else None})
    seen = [r for r in rows if r["n"]]
    mono = all(seen[k]["rescue_centred"] <= seen[k + 1]["rescue_centred"]
               for k in range(len(seen) - 1))
    return rows, bool(mono)


def matched_contrast(sid, grp, y, nperm=NPERM, seed=0):
    """weighted CG-minus-AU inside strata, weight min(n_A, n_B); permutation on the outcome."""
    by = collections.defaultdict(list)
    for k in range(len(sid)):
        by[sid[k]].append(k)

    def est(vals):
        num = w = 0.0
        for _s, ix in by.items():
            a = [k for k in ix if grp[k] == 1]
            b = [k for k in ix if grp[k] == 0]
            if not a or not b:
                continue
            ww = min(len(a), len(b))
            num += ww * (vals[a].mean() - vals[b].mean())
            w += ww
        return (num / w if w else None), w

    obs, w = est(y)
    if obs is None:
        return None, 1.0, 0.0, None
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(nperm):
        yp = y.copy()
        for _s, ix in by.items():
            a = np.asarray(ix)
            yp[a] = y[a[rng.permutation(len(a))]]
        e, _w = est(yp)
        if e is not None:
            null.append(e)
    null = np.asarray(null)
    p = (float((np.abs(null) >= abs(obs)).sum()) + 1) / (len(null) + 1)
    return float(obs), float(p), float(w), float(null.std())


def ladder(sid, grp, y, deltas=DELTAS, ndraw=20, seed=0):
    """P3: built on the null. Relabel so the true difference is 0, then inject delta."""
    rng = np.random.default_rng(seed)
    by = collections.defaultdict(list)
    for k in range(len(sid)):
        by[sid[k]].append(k)
    out = {}
    for d in deltas:
        hit = 0
        for t in range(ndraw):
            g = grp.copy()
            for _s, ix in by.items():
                a = np.asarray(ix)
                g[a] = grp[a[rng.permutation(len(a))]]
            yy = y + d * (g == 1)
            e, p, _w, _sd = matched_contrast(sid, g, yy, nperm=200, seed=seed + 7919 * t + 13)
            hit += (e is not None and p <= 0.05)
        out[str(d)] = round(hit / ndraw, 3)
    return out


# ----------------------------------------------------------------------------- predictors
def vienna_scores(recs, log):
    import RNA
    out = np.full(len(recs), np.nan)
    for k, r in enumerate(recs):
        seq = r["mut"].seq
        fc = RNA.fold_compound(seq)
        fc.pf()
        bpp = fc.bpp()
        a, b = min(r["i"], r["j"]) + 1, max(r["i"], r["j"]) + 1     # bpp is 1-indexed
        try:
            out[k] = bpp[a][b]
        except Exception:                                          # noqa: BLE001
            out[k] = np.nan
        if (k + 1) % 200 == 0:
            log(f"    ViennaRNA {k+1}/{len(recs)}")
    return out


def neural_scores(recs, log):
    import torch
    import m26c_neural_scored as M26C
    model, dev, missing, unexpected = M26C.load_model()
    log(f"    RibonanzaNet 载入 dev={dev} missing={len(missing)} unexpected={len(unexpected)}")
    TOK = M26C.TOK
    uniq = sorted({r["mut"].seq for r in recs})
    log(f"    唯一突变体序列 {len(uniq)}")
    cache = {}
    with torch.no_grad():
        for k, s in enumerate(uniq):
            if any(ch not in TOK for ch in s):
                continue
            t = torch.tensor([[TOK[ch] for ch in s]], device=dev)
            lg = model.pair_logits(t)[0]
            cache[s] = torch.sigmoid(lg).float().cpu().numpy()
            if (k + 1) % 100 == 0:
                log(f"    RibonanzaNet {k+1}/{len(uniq)}")
    out = np.full(len(recs), np.nan)
    for k, r in enumerate(recs):
        m = cache.get(r["mut"].seq)
        if m is not None:
            out[k] = float(m[r["i"], r["j"]])
    return out


# ----------------------------------------------------------------------------- one deposit
def run_deposit(rid, args):
    t0 = time.time()
    log(f"\n{'='*74}\n=== {rid}\n{'='*74}")
    cons, seqpos, has_struct, has_seq = parse(rid)
    log(f"  构建体 {len(cons)}  反应性窗口 {len(seqpos) if seqpos else 0} 列"
        f"  (SEQPOS {seqpos[0]}..{seqpos[-1]})" if seqpos else "")
    # P4 -- the assertion, not a comment
    if has_struct or has_seq:
        return {"rmdb_id": rid, "void": "P4_violated__deposit_declares_a_structure"}
    log("  P4 通过:沉积不声明靶标结构,全程不读入任何结构")

    by_name = {c.name: c for c in cons}
    window = (seqpos[0] - 1, seqpos[-1])          # 0-based, half-open: the reactivity columns
    allo = find_offset(cons, by_name, window)
    log(f"  设计窗(0基,半开)= {window};窗外每条构建体带自己的条码,比对一律不出窗")
    offs, dropped = {}, {}
    for fam, (off, n, tot) in sorted(allo.items()):
        frac = n / max(1, tot)
        log(f"    族 {fam[:44]:46s} 偏移 {off:3d}  一致 {n}/{tot} = {frac:.3f}"
            f"{'' if frac >= AGREE else '   ** 不恒定 ⇒ 该族剔除,不逐条拟合 **'}")
        (offs if frac >= AGREE else dropped)[fam] = (off, n, tot)
    if not offs:
        log("  ** 没有一个族恢复出恒定偏移 ⇒ 作废 **")
        return {"rmdb_id": rid, "void": "offset_unrecoverable", "window": list(window)}

    # ---- arm A
    recs, ex = mine(cons, by_name, offs, SN_MAIN, window)
    for r in recs:
        r["cols"] = tuple(k for k, p in enumerate(seqpos) if p - 1 in (r["i"], r["j"]))
    log(f"\n  【A 臂 挖掘】SN>={SN_MAIN} 下纳入 {len(recs)}")
    for k, v in sorted(ex.items(), key=lambda kv: -kv[1]):
        if v:
            log(f"      排除 {k:26s} {v}")
    sens = {}
    for s in SN_SENS:
        rs, _e = mine(cons, by_name, offs, s, window)
        sens[str(s)] = len(rs)
    log(f"      P2 敏感性 SN>=0.5 -> {sens['0.5']}   SN>=2.0 -> {sens['2.0']}"
        f"   (**主臂只用 {SN_MAIN},事后不调**)")
    by_pt = dict(collections.Counter(r["pair_type"] for r in recs))
    log(f"      类型分布 {by_pt}   族 {len(set(r['family'] for r in recs))}")

    res = {"rmdb_id": rid, "n_constructs": len(cons), "window": len(seqpos),
           "offsets": {k: v[0] for k, v in offs.items()},
           "offsets_dropped": {k: v[0] for k, v in dropped.items()},
           "A": {"n_admitted": len(recs), "excluded": ex, "by_pair_type": by_pt,
                 "n_families": len(set(r["family"] for r in recs)), "sn_sensitivity": sens}}
    if len(recs) < 30:
        res["void"] = "K3_void_too_few"
        log(f"  ** K3 触发:纳入 {len(recs)} < 30 ⇒ 该沉积作废,不放宽判据去凑 **")
        return res

    # ---- arm B
    nan = float("nan")
    rf = np.array([(v if (v := rescue(r)) is not None else nan) for r in recs])
    rf_ex = np.array([(v if (v := rescue(r, excise=True)) is not None else nan) for r in recs])
    ok = ~np.isnan(rf)
    wire = [wire_check(r) for r in recs[:50]]
    wire = [w for w in wire if w is not None]
    res["B"] = {"n_scored": int(ok.sum()),
                "mean": round(float(np.nanmean(rf)), 4), "sd": round(float(np.nanstd(rf)), 4),
                "median": round(float(np.nanmedian(rf)), 4),
                "excise_mean": round(float(np.nanmean(rf_ex)), 4),
                "excise_corr": (round(float(np.corrcoef(rf[ok & ~np.isnan(rf_ex)],
                                                        rf_ex[ok & ~np.isnan(rf_ex)])[0, 1]), 4)
                                if (ok & ~np.isnan(rf_ex)).sum() > 3 else None),
                "wire_check_all_one": bool(wire and max(abs(w - 1.0) for w in wire) < 1e-12)}
    log(f"\n  【B 臂 结果量】n={int(ok.sum())}  均值 {res['B']['mean']}  SD {res['B']['sd']}"
        f"  中位 {res['B']['median']}")
    log(f"      接线自检(双突变==野生型 ⇒ 必须恰好 1.0):{res['B']['wire_check_all_one']}")
    log(f"      敏感性(挖掉突变列)均值 {res['B']['excise_mean']}  与主臂相关 {res['B']['excise_corr']}")
    if not res["B"]["wire_check_all_one"] or res["B"]["sd"] == 0:
        res["void"] = "K0_outcome_not_self_consistent"
        log("  ** K0 触发:结果量不自洽 ⇒ 作废,先修导出式 **")
        return res

    recs = [r for r, o in zip(recs, ok) if o]
    rf = rf[ok]
    sid = np.array([r["family"] for r in recs])
    log(f"      分层只有 (family):{len(set(sid))} 层 —— **留存数不可与 OK7a 逐字并列**")

    # ---- arm C (PRIMARY)
    log("\n  【C 臂 记分 —— 本实验主臂】")
    arms = {}
    for nm, fn in (("ViennaRNA", vienna_scores), ("RibonanzaNet", neural_scores)):
        try:
            x = fn(recs, log)
        except Exception as e:                                              # noqa: BLE001
            log(f"    {nm} 失败 {repr(e)[:110]}")
            arms[nm] = {"failed": repr(e)[:200]}
            continue
        good = ~np.isnan(x)
        if good.sum() < 30:
            arms[nm] = {"n": int(good.sum()), "too_few": True}
            continue
        s, p, n, ns = slope_perm(sid[good], x[good], rf[good], nperm=args.nperm)
        cal, mono = calibration(sid[good], x[good], rf[good])
        arms[nm] = {"n": int(good.sum()), "slope": None if s is None else round(s, 5),
                    "p_perm": round(p, 5), "n_strata": ns, "calibration": cal, "monotone": mono,
                    "pred_mean": round(float(np.nanmean(x)), 4),
                    "pred_sd": round(float(np.nanstd(x)), 4)}
        log(f"    {nm:14s} 层内技能斜率 {arms[nm]['slope']}  置换 p {arms[nm]['p_perm']}"
            f"  n={arms[nm]['n']}  层 {ns}  标定{'单调' if mono else '非单调'}")
    res["C"] = arms
    have = [a for a in arms.values() if a.get("slope") is not None]
    skilled = [a for a in have if a["slope"] > 0 and a["p_perm"] <= 0.05]
    res["K1_instrument_transfers"] = len(skilled) == 2 and len(have) == 2
    res["K2_no_transfer"] = len(have) == 2 and not skilled

    # ---- arm E (pre-judged underpowered)
    keep = np.array([r["pair_type"] in ("CG", "AU") for r in recs])
    if keep.sum() >= 30:
        grp = np.array([1 if r["pair_type"] == "CG" else 0 for r in recs])[keep]
        e, p, w, nsd = matched_contrast(sid[keep], grp, rf[keep], nperm=args.nperm)
        lad = ladder(sid[keep], grp, rf[keep], ndraw=args.ndraw)
        thr = next((float(d) for d in DELTAS if d > 0 and lad.get(str(d), 0) >= 0.8), None)
        res["E"] = {"contrast": None if e is None else round(e, 5), "p_perm": round(p, 5),
                    "weight": w, "null_sd": None if nsd is None else round(nsd, 5),
                    "n": int(keep.sum()), "ladder": lad, "delta0": lad.get("0.0"),
                    "detect_threshold": thr}
        log(f"\n  【E 臂 身份对比 —— 预判功率不足,任何含 0 的区间不许读成否定】")
        log(f"      G-C 减 A-U {res['E']['contrast']}  p {res['E']['p_perm']}  权重 {w}  n={int(keep.sum())}")
        log(f"      P3 梯级 delta=0 自检 {lad.get('0.0')} / {SELF_MAX}"
            f"   检出阈值 {thr}(OK7a 是 0.05)")
        if lad.get("0.0", 0) > SELF_MAX:
            res["E"]["void"] = "K5_ladder_self_check_failed"
            log("      ** K5 触发:梯级自检超标 ⇒ E 臂作废 **")
    else:
        res["E"] = {"skipped": f"CG+AU only {int(keep.sum())} < 30"}

    res["records"] = [{"family": r["family"], "i": r["i"], "j": r["j"],
                       "pair_type": r["pair_type"], "rescue": round(float(v), 6),
                       "sn": round(r["sn"], 3)} for r, v in zip(recs, rf)]
    res["seconds"] = round(time.time() - t0, 1)
    log(f"\n  {rid} 用时 {res['seconds']}s")
    return res


# ----------------------------------------------------------------------------- arm D
def two_probes(results):
    """P5: DMS and 2A3 are the SAME interventions measured twice. Never pooled."""
    out = {}
    for fam in ("M2SL5", "M3SARS"):
        a = next((r for r in results if r["rmdb_id"].startswith(fam) and "DMS" in r["rmdb_id"]), None)
        b = next((r for r in results if r["rmdb_id"].startswith(fam) and "2A3" in r["rmdb_id"]), None)
        if not a or not b or "records" not in a or "records" not in b:
            out[fam] = {"skipped": "one probe void or missing"}
            continue
        ka = {(r["family"], r["i"], r["j"]): r["rescue"] for r in a["records"]}
        kb = {(r["family"], r["i"], r["j"]): r["rescue"] for r in b["records"]}
        sh = sorted(set(ka) & set(kb))
        if len(sh) < 10:
            out[fam] = {"n_shared": len(sh), "skipped": "fewer than 10 shared interventions"}
            continue
        x = np.array([ka[k] for k in sh])
        y = np.array([kb[k] for k in sh])
        rx, ry = x.argsort().argsort(), y.argsort().argsort()
        sl = {}
        for nm in ("ViennaRNA", "RibonanzaNet"):
            sl[nm] = {"DMS": a.get("C", {}).get(nm, {}).get("slope"),
                      "2A3": b.get("C", {}).get(nm, {}).get("slope")}
        out[fam] = {"n_shared": len(sh),
                    "pearson": round(float(np.corrcoef(x, y)[0, 1]), 4),
                    "spearman": round(float(np.corrcoef(rx, ry)[0, 1]), 4),
                    "slopes": sl,
                    "same_sign": {nm: (None if None in (v["DMS"], v["2A3"])
                                       else (v["DMS"] > 0) == (v["2A3"] > 0))
                                  for nm, v in sl.items()}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deposits", default=",".join(DEPOSITS))
    ap.add_argument("--nperm", type=int, default=NPERM)
    ap.add_argument("--ndraw", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(HERE, "m28_second_corpus.json"))
    a = ap.parse_args()
    t0 = time.time()
    log("=== M28:挖掘与记分两步能不能迁移到天然 RNA ===")
    log("  声明 m28_second_corpus_declaration.json,冻结于 066f4ba1,先于本文件。")
    log("  离线 $0。**主臂是 C(仪器还读不读得出东西),不是 E。**")
    log(f"  押注已先写:A 臂 M2SL5>=130、M3SARS_DMS<=100;B 臂两探针 Spearman>=0.5;")
    log(f"  C 臂 ViennaRNA 斜率正且 p<=0.05、RibonanzaNet 斜率更大;E 臂区间含 0;阈值>=0.15\n")

    results = []
    for rid in a.deposits.split(","):
        try:
            results.append(run_deposit(rid.strip(), a))
        except Exception as e:                                              # noqa: BLE001
            log(f"  {rid} 出错 {repr(e)[:200]}")
            results.append({"rmdb_id": rid.strip(), "error": repr(e)[:300]})

    log(f"\n{'='*74}\n【D 臂 两个探针是同一批干预的两次测量 —— 分开报,不合并】\n{'='*74}")
    d = two_probes(results)
    for fam, v in d.items():
        log(f"  {fam}: {json.dumps(v, ensure_ascii=False)}")

    live = [r for r in results if "void" not in r and "error" not in r]
    k1 = [r["rmdb_id"] for r in live if r.get("K1_instrument_transfers")]
    k2 = [r["rmdb_id"] for r in live if r.get("K2_no_transfer")]
    log(f"\n{'='*74}\n【裁定】\n{'='*74}")
    log(f"  可判沉积 {len(live)}/{len(results)}")
    log(f"  K1 两个模型都有技能(仪器迁移成立):{k1 if k1 else '无'}")
    log(f"  K2 两个模型都没技能(限制原样保留):{k2 if k2 else '无'}")
    log("  **K4 提醒:E 臂只能得出『在这里测不出来』,不许拿它否证 OK7a。**")

    payload = {"deposits": results, "D_two_probes": d,
               "K1_deposits": k1, "K2_deposits": k2,
               "sn_threshold": SN_MAIN, "seconds": round(time.time() - t0, 1)}
    with open(a.out, "w", encoding="utf8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    log(f"\n写出 {a.out}  总用时 {payload['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
