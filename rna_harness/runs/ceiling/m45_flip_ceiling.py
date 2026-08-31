#!/usr/bin/env python3
"""M45: the ceiling on how many compensatory orientation flips the full released corpus could hold.

Declaration `m45_flip_ceiling_declaration.json`, frozen and pushed at 84daee1c BEFORE this file
existed. Offline, $0.

W1: every number here is an UPPER BOUND on candidates. It may not be written as interventions, as
usable data, or as breadth being solved. Definition 1 additionally requires the two positions to be
a DESIGNED PAIR in a declared target structure, and this file carries no structure at all.
"""
from __future__ import annotations
import collections, csv, json, os, re, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "16")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
DECL = json.load(open(os.path.join(HERE, "m45_flip_ceiling_declaration.json")))
FULL = "/data/tmp/ribonanza/train_data.csv"
OK7A = "/data/wenhui/openknot_data/Data/OK7a_M2R_data.v4.5.1.csv"
OUTJ = os.path.join(HERE, "m45_flip_ceiling.json")
WC = frozenset([frozenset("AU"), frozenset("GC"), frozenset("GU")])
BUCKET_CAP = 20000          # a bucket above this VOIDS the run; it is never silently skipped
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


# --------------------------------------------------------------------------- the detector
def comp_key(s):
    """Composition, which a transposition preserves exactly. In the blocking key it is what stops
    a shared adapter from collapsing a whole library into one all-pairs bucket."""
    return (s.count("A"), s.count("C"), s.count("G"), s.count("U"))


def find_flips(seqs, cap=BUCKET_CAP):
    """seqs: iterable of unique sequences of any lengths, all from ONE dataset.

    Returns (candidates, transpositions, voided_buckets).
      transpositions -- unordered pairs at Hamming exactly 2 with identical composition
      candidates     -- the subset whose exchanged unordered character set is a WC or wobble pair

    Pigeonhole: two mismatches cannot touch all three contiguous thirds, so a pair is guaranteed to
    share at least one third verbatim. Blocking on (length, third index, third content, composition)
    therefore cannot miss a pair. Every bucket is verified; none is capped away.
    """
    by = collections.defaultdict(list)
    for s in seqs:
        n = len(s); a, b = n // 3, 2 * n // 3
        c = comp_key(s)
        for t, part in ((0, s[:a]), (1, s[a:b]), (2, s[b:])):
            by[(n, t, part, c)].append(s)
    trans, cand, voided = set(), set(), []
    for k, group in by.items():
        g = sorted(set(group))
        if len(g) < 2:
            continue
        if len(g) > cap:
            voided.append((k[0], k[1], len(g)))
            continue
        arr = np.frombuffer("".join(g).encode(), dtype=np.uint8).reshape(len(g), k[0])
        for i in range(len(g) - 1):
            d = (arr[i + 1:] != arr[i]).sum(1)
            for j in np.nonzero(d == 2)[0]:
                a_, b_ = g[i], g[i + 1 + int(j)]
                if a_ > b_:
                    a_, b_ = b_, a_
                if (a_, b_) in trans:
                    continue
                pos = [p for p in range(len(a_)) if a_[p] != b_[p]]
                # a transposition: the two exchanged characters must cross over
                if a_[pos[0]] != b_[pos[1]] or a_[pos[1]] != b_[pos[0]]:
                    continue
                trans.add((a_, b_))
                if frozenset((a_[pos[0]], a_[pos[1]])) in WC:
                    cand.add((a_, b_))
    return cand, trans, voided


BRUTE_MAX = 2500


def brute(seqs):
    """All-pairs, no blocking, no composition short-cut in the OUTER loop -- the point is that it
    shares no reasoning with find_flips. It is O(n^2) in Python and is therefore only ever called on
    a set small enough to permit it; being handed a larger one is a programming error, not a reason
    to sample, because a sampled brute force cannot certify completeness of anything."""
    g = sorted(set(seqs))
    if len(g) > BRUTE_MAX:
        raise ValueError(f"brute() given {len(g)} sequences; cap is {BRUTE_MAX}")
    out = set()
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            a_, b_ = g[i], g[j]
            if len(a_) != len(b_) or comp_key(a_) != comp_key(b_):
                continue
            pos = [p for p in range(len(a_)) if a_[p] != b_[p]]
            if len(pos) != 2:
                continue
            if a_[pos[0]] != b_[pos[1]] or a_[pos[1]] != b_[pos[0]]:
                continue
            if frozenset((a_[pos[0]], a_[pos[1]])) in WC:
                out.add((a_, b_))
    return out


def flip_once(s, rng):
    """Plant one valid orientation flip: pick two positions whose characters form a WC/wobble pair
    and exchange them. Returns None if the sequence offers none."""
    idx = list(range(len(s)))
    rng.shuffle(idx)
    for i in idx[:400]:
        for j in idx[:400]:
            if i < j and s[i] != s[j] and frozenset((s[i], s[j])) in WC:
                t = list(s); t[i], t[j] = t[j], t[i]
                return "".join(t)
    return None



# --------------------------------------------------------------------------- construct anatomy
def flanks(seqs, L):
    """Longest constant prefix and suffix across same-length sequences of one library. These
    constructs are 5' leader + design region + 3' tail, and the leader and tail carry nothing
    anyone designed, so they are cut before anything is compared. Measured, never assumed."""
    S = [s for s in seqs if len(s) == L]
    if len(S) < 2:
        return 0, 0
    A = np.frombuffer("".join(S).encode(), dtype=np.uint8).reshape(len(S), L)
    var = (A != A[0]).any(0)
    if not var.any():
        return L, 0
    return int(np.argmax(var)), int(np.argmax(var[::-1]))


def load_ok7a():
    """This paper's own corpus, split into wild types and DOUBLE mutants.

    Definition 1's interventions are the DOUBLE mutants. A first version of this run built the
    reference set from every mutation-tagged row and V3 read 0.3279, because two thirds of the
    tagged rows are single mutants at Hamming distance 1 and a flip detector is right to refuse
    them. An earlier version compared WHOLE CONSTRUCTS and read 0.0000: a wild type and its own
    double mutant differ at 47 to 61 positions over the full construct and at exactly 2 inside the
    design window, because the constructs carry different flanking sequence.
    """
    csv.field_size_limit(10 ** 9)
    NA = ("", "NA", "nan")
    wt, dbl = {}, []
    for d in csv.DictReader(open(OK7A)):
        if not d.get("sequence"):
            continue
        cell = (d["puzzle"], d["method"])
        a, b = d["mutA"], d["mutB"]
        if a in NA and b in NA:
            wt[cell] = d
        elif a not in NA and b not in NA:
            dbl.append((cell, d))
    return wt, dbl


def design_region(d):
    ss, se = int(float(d["sub_start"])), int(float(d["sub_end"]))
    return d["sequence"][ss - 1:se]


# --------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    out = {"declaration": "m45_flip_ceiling_declaration.json", "cost": "offline, $0",
           "wording_rules": DECL["wording_rules"], "not_a_test": DECL["not_a_test"]}

    log("=== V 臂 验证(先于任何计数)")
    V = {}
    rng = __import__("random").Random(20260902)

    wt, dbl = load_ok7a()
    regions, known = set(), set()
    for cell, d in dbl:
        w = wt.get(cell)
        if not w:
            continue
        a, b = design_region(w), design_region(d)
        regions.add(a); regions.add(b)
        if a != b and len(a) == len(b):
            known.add((a, b) if a < b else (b, a))
    ours, tr_ours, _v = find_flips(regions)
    hit = known & ours
    frac = len(hit) / len(known) if known else 0.0
    log(f"  V3 本文语料:野生型格 {len(wt)}  双突变体 {len(dbl)}  已发表配对 {len(known)}")
    log(f"     探测器换位 {len(tr_ours)}  候选 {len(ours)}  命中 {len(hit)}  召回 {frac:.4f}"
        f"  -> {'通过' if frac >= 0.90 else '**未通过**'}")
    V["V3"] = {"known_published_double_mutant_pairs": len(known), "transpositions": len(tr_ours),
               "candidates": len(ours), "recovered": len(hit), "recall": round(frac, 4),
               "pass": bool(frac >= 0.90),
               "transpositions_equal_known": bool(len(tr_ours) == len(known)),
               "misses_are": ("transpositions of non-pairing character sets -- (A,C), (C,U), "
                              "(A,G) -- which the Watson-Crick/wobble filter is right to refuse"),
               "caught": ("whole-construct comparison read 0.0000; counting single mutants as "
                          "reference pairs read 0.3279")}

    sub = []
    for a_, b_ in sorted(known)[:600]:
        sub += [a_, b_]
    for x in sorted(regions):
        if len(set(sub)) >= BRUTE_MAX:
            break
        sub.append(x)
    sub = set(sorted(set(sub))[:BRUTE_MAX])
    bf = brute(sub); blk, _t4, _v4 = find_flips(sub)
    V["V4"] = {"subset": len(sub), "brute_force": len(bf), "blocked": len(blk),
               "identical": bool(bf == blk), "nonvacuous": bool(len(bf) > 0)}
    log(f"  V4 子集 {len(sub)}:穷举 {len(bf)} 对,分块 {len(blk)} 对 -> "
        f"{'一致' if bf == blk else '**不一致**'}{'' if bf else '  ** 但都是 0,检验为空 **'}")

    base = sorted(regions)[:3000]
    planted, want = list(base), set()
    for s in base[:200]:
        t = flip_once(s, rng)
        if t and t not in set(base):
            planted.append(t)
            want.add((s, t) if s < t else (t, s))
    got, _t2, _v2 = find_flips(set(planted))
    V["V2"] = {"planted": len(want), "recovered": len(want & got),
               "pass": bool(len(want) > 0 and (want & got) == want)}
    log(f"  V2 植入 {len(want)} 个真实取向翻转,找回 {len(want & got)}"
        f"  -> {'通过' if V['V2']['pass'] else '**未通过**'}")

    z = [len(find_flips({"AAAAAAAAAA"})[0]), len(find_flips({"ACGUACGUAC"})[0]),
         len(find_flips({"AAAACCCCGG", "AAAACCCCGU"})[0])]
    V["V1"] = {"singleton": z[0], "single_seq": z[1], "hamming_one_pair": z[2],
               "pass": bool(sum(z) == 0)}
    log(f"  V1 零点 {z} -> {'通过' if V['V1']['pass'] else '**未通过**'}")

    V["pass"] = bool(V["V1"]["pass"] and V["V2"]["pass"] and V["V3"]["pass"]
                     and V["V4"]["identical"] and V["V4"]["nonvacuous"])
    out["V"] = V
    if not V["pass"]:
        log("\n  ** V 臂未通过 ⇒ 按 kill 作废,不报告任何上界 **")
        out["void"] = "V_arm"
        json.dump(out, open(OUTJ, "w"), ensure_ascii=False, indent=1); return 1

    # E5 calibration: the fraction of candidates that survives Definition 1, measured on the one
    # corpus where Definition 1 CAN be applied. A ceiling elsewhere must be read through it.
    all_ok = set()
    for d in csv.DictReader(open(OK7A)):
        if d.get("sequence"):
            all_ok.add(design_region(d))
    cal, cal_tr, _ = find_flips(all_ok)
    out["E5_calibration"] = {
        "own_corpus_design_regions": len(all_ok), "candidates_on_own_corpus": len(cal),
        "transpositions_on_own_corpus": len(cal_tr), "admissible_under_definition_1": 4317,
        "conversion_rate": round(4317 / len(cal), 5) if cal else None}
    log(f"\n=== E5 标定:本文语料全部设计区 {len(all_ok):,} 条 → 候选 {len(cal):,},"
        f"Definition 1 纳入 4317,转化率 {out['E5_calibration']['conversion_rate']}")

    log("\n=== 载入完整语料")
    by_lib = collections.defaultdict(set)
    sn_ok = set(); nrow = 0
    with open(FULL, newline="") as fh:
        r = csv.reader(fh); hdr = next(r)
        ci = {c: k for k, c in enumerate(hdr)}
        iseq, ilib, isn = ci["sequence"], ci["dataset_name"], ci["SN_filter"]
        for row in r:
            nrow += 1
            by_lib[row[ilib]].add(row[iseq])
            if row[isn] == "1":
                sn_ok.add(row[iseq])
    uniq = {s for v in by_lib.values() for s in v}
    log(f"  行 {nrow:,}  唯一序列 {len(uniq):,}  dataset_name 取值 {len(by_lib)}"
        f"  SN_filter==1 {len(sn_ok):,}  ({time.time()-t0:.0f}s)")
    out["n_rows"] = nrow; out["n_unique"] = len(uniq)
    out["n_dataset_name_values"] = len(by_lib)

    coll = collections.defaultdict(set)
    for lib, v in by_lib.items():
        coll[re.sub(r"_?(2A3|DMS)_?", "_", lib).strip("_")] |= v
    out["W4_library_count"] = {
        "dataset_name_values": len(by_lib), "after_collapsing_chemistry": len(coll),
        "m42_recorded": 23,
        "resolution": ("the file carries one dataset_name per library PER CHEMISTRY; collapsing "
                       "the 2A3/DMS token gives the library count M42 recorded. Both numbers are "
                       "correct for what each counts.")}
    log(f"  W4 折叠探针化学后 {len(coll)} 个库(M42 记录 23)")

    log("\n=== 构建体解剖:每个库的恒定两翼(量出来的)")
    anat = {}
    for lib, seqs in sorted(coll.items()):
        L = collections.Counter(len(s) for s in seqs).most_common(1)[0][0]
        pre, suf = flanks(seqs, L)
        anat[lib] = {"modal_length": L, "const_prefix": pre, "const_suffix": suf,
                     "design_region_length": L - pre - suf, "n_unique": len(seqs)}
    out["construct_anatomy"] = anat
    log(f"  恒定前缀 {dict(collections.Counter(v['const_prefix'] for v in anat.values()))}"
        f"   恒定后缀 {dict(collections.Counter(v['const_suffix'] for v in anat.values()))}")

    log("\n=== E1 上界:逐库(设计区),并与全长比对以检查两翼里有没有条形码")
    per, cand_all, tr_all, voided, cand_full = {}, set(), set(), [], set()
    for lib, seqs in sorted(coll.items(), key=lambda x: -len(x[1])):
        a = anat[lib]
        reg = {s[a["const_prefix"]:len(s) - a["const_suffix"]] for s in seqs
               if len(s) > a["const_prefix"] + a["const_suffix"]}
        c, tr, v = find_flips(reg)
        cf, _tf, vf = find_flips(seqs)
        per[lib] = {"n_unique": len(seqs), "n_regions": len(reg),
                    "modal_length": a["modal_length"],
                    "design_region_length": a["design_region_length"],
                    "transpositions": len(tr), "candidates": len(c),
                    "candidates_full_length": len(cf)}
        cand_all |= c; tr_all |= tr; cand_full |= cf; voided += v + vf
        log(f"    {lib[:50]:50s} 唯一 {len(seqs):7,}  设计区 {a['design_region_length']:3d}"
            f"  换位 {len(tr):5,}  候选 {len(c):5,}  (全长 {len(cf):5,})  {time.time()-t0:.0f}s")
    if voided:
        log(f"\n  ** {len(voided)} 个桶超过 {BUCKET_CAP} ⇒ 按 kill 作废 **")
        out["void"] = "bucket_over_cap"; out["voided_buckets"] = voided[:50]
        json.dump(out, open(OUTJ, "w"), ensure_ascii=False, indent=1); return 1
    out["per_library"] = per
    agree = (sum(v["candidates"] for v in per.values())
             == sum(v["candidates_full_length"] for v in per.values()))
    out["flank_cut_is_inert"] = bool(agree)
    log(f"\n  设计区与全长给出同一个数:{'是 —— 两翼里没有条形码' if agree else '**否**'}")

    contributing = {l: v["candidates"] for l, v in per.items() if v["candidates"]}
    out["E1"] = {"candidates_total": len(cand_all), "transpositions_total": len(tr_all),
                 "wc_filter_removed": len(tr_all) - len(cand_all),
                 "wc_filter_removed_frac": round(1 - len(cand_all) / len(tr_all), 4)
                 if tr_all else None,
                 "expected_admissible_at_calibrated_rate":
                     round(len(cand_all) * (out["E5_calibration"]["conversion_rate"] or 0), 1)}
    out["E2"] = {"n_contributing_libraries": len(contributing), "n_libraries": len(coll),
                 "by_library": dict(sorted(contributing.items(), key=lambda x: -x[1]))}
    log(f"\n  候选合计 {len(cand_all):,}   换位合计 {len(tr_all):,}")
    log(f"  有贡献的库 {len(contributing)} / {len(coll)}:{out['E2']['by_library']}")
    log(f"  按本文语料的转化率折算,期望可纳入 "
        f"{out['E1']['expected_admissible_at_calibrated_rate']}(本文现有 4170)")

    sn_reg = set()
    for lib, seqs in coll.items():
        a = anat[lib]
        for s in seqs:
            if s in sn_ok and len(s) > a["const_prefix"] + a["const_suffix"]:
                sn_reg.add(s[a["const_prefix"]:len(s) - a["const_suffix"]])
    both = {(a, b) for a, b in cand_all if a in sn_reg and b in sn_reg}
    out["E3"] = {"both_members_sn_pass": len(both),
                 "frac": round(len(both) / len(cand_all), 4) if cand_all else None}
    log(f"\n=== E3 两端都过 SN_filter:{len(both)} / {len(cand_all)}")

    overlap = {(a, b) for a, b in cand_all if a in all_ok or b in all_ok}
    out["E4"] = {"touching_this_papers_corpus": len(overlap),
                 "net_new": len(cand_all) - len(overlap)}
    log(f"=== E4 与本文语料交叠 {len(overlap)};净新增 {len(cand_all)-len(overlap)}")

    st = {"1_ceiling_exceeds_4170": bool(len(cand_all) > 4170),
          "2_more_than_four_libraries": bool(len(contributing) > 4),
          "3_wc_filter_removes_majority": bool(out["E1"]["wc_filter_removed_frac"] is not None
                                               and out["E1"]["wc_filter_removed_frac"] > 0.5),
          "4_sn_retains_at_least_20pc": bool(out["E3"]["frac"] is not None
                                             and out["E3"]["frac"] >= 0.20)}
    out["staked"] = st; out["staked_text"] = DECL["staked"]
    log("\n=== 预注册记分")
    for k, v in st.items():
        log(f"  {k:34s} {'成立' if v else '**失败**'}")

    out["elapsed_s"] = round(time.time() - t0, 1); out["log"] = LOG
    json.dump(out, open(OUTJ, "w"), ensure_ascii=False, indent=1)
    log(f"\n写出 {OUTJ}  用时 {out['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
