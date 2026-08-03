#!/usr/bin/env python3
"""The first time anything in this project is compared against a MEASUREMENT rather than another model.

THE BOUNDARY THIS ATTACKS. Every ruler built here is validated against other predictors. The pseudoknot
criterion is checked by pKiss/ProbKnot/ShapeKnots and then by RNAPKplex; the co-transcriptional criterion is
checked by Kinfold. All of those are models. The one place physical reality has ever entered is the natural
structures used as positives, and even those arrive as database annotations. "Judgment-missing" ends at the
judgment of judgments, and that can only be physics.

Co-transcriptional SHAPE-seq measures chemical reactivity at every nucleotide AT EVERY TRANSCRIPT LENGTH --
which is exactly the object our criterion is made of. trap_fraction counts prefixes whose MFE contains a pair
the target does not have. Here the prefixes were actually synthesised and actually probed.

THE TARGET COMES FROM THE MEASUREMENT, NOT FROM US. Folding the full-length sequence with our own model to
get the reference would make the whole thing circular. Instead the reference is produced by RNAstructure's
Fold with the full-length experimental reactivity as a restraint -- a different codebase with a different
parameter set, driven by the data.

PRE-REGISTERED, written before any RDAT was scored:

    At prefixes our criterion calls TRAPS, the measurement should agree BETTER with MFE(prefix) than with the
    target restricted to that prefix. That is the physical content of the word "trap": the molecule really is
    sitting in something that is not the final structure.

    At NON-trap prefixes the two structures are much closer, so the same difference should be smaller. The
    contrast between those two groups is the test; the trap group alone could be an artefact of MFE simply
    fitting reactivity better than any fixed reference does.

    Test 1 (within traps):   paired AUROC_mfe vs AUROC_tgt, Wilcoxon.
    Test 2 (the real one):   difference-of-differences, trap vs non-trap, with standard errors clustered on
                             the RNA, because 71 transcript lengths of one molecule are not 71 molecules.

GATES DECLARED IN ADVANCE, because a small-n null here would mean nothing and it would be very easy to
present one as a finding:

    fewer than 6 distinct RNAs usable   -> report 样本不足, NOT a null
    fewer than 50 trap prefixes         -> report 样本不足, NOT a null
    AUROC_mfe ~ 0.5 overall             -> the oracle itself fails against measurement, and THAT is the
                                           headline, ahead of anything about the criterion

WHAT THIS CANNOT DO, stated here rather than in a footnote. It validates the ORACLE -- prefix MFE -- against
measurement on NATURAL RNAs. It does not validate the criterion on OUR DESIGNS, because nobody has ever
probed one of our sequences and nobody will without wet-lab work. The claim on offer is "the quantity our
criterion counts is visible in real transcription", not "our designs behave as we say".
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)

RNASTR = "/home/ubuntu/miniconda3/envs/rnastr/bin"
DATAPATH = "/home/ubuntu/miniconda3/envs/rnastr/share/rnastructure/data_tables"
MANIFEST = "https://rmdb.stanford.edu/manifest.json"
DETAIL = "https://rmdb.stanford.edu/detail/{}/"
MIN_LENGTHS = 30          # a construct needs this many measured transcript lengths to be a 2D dataset
MIN_RNAS = 6              # declared gate
MIN_TRAPS = 50            # declared gate
CACHE = os.path.join(HERE, "rmdb_cache")


def fetch(url: str, dest: str = None, timeout: int = 180):
    if dest and os.path.exists(dest) and os.path.getsize(dest) > 0:
        return open(dest, "rb").read()
    req = urllib.request.Request(url, headers={"User-Agent": "rna-harness/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        b = r.read()
    if dest:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        open(dest, "wb").write(b)
    return b


def parse_rdat(text: str):
    """Full sequence plus, for every measured transcript length, that length's own prefix and reactivity.

    The prefix is taken from the file's own annotation rather than sliced out of the full sequence, so a
    construct whose intermediates are not simple prefixes cannot silently be treated as if they were -- it is
    checked below and dropped instead.
    """
    lines = text.splitlines()
    seq = next((l.split("\t", 1)[1].strip() for l in lines if l.startswith("SEQUENCE")), None)
    if not seq:
        return None
    ann, data = {}, {}
    for l in lines:
        if l.startswith("DATA_ANNOTATION"):
            i = int(l.split("\t")[0].split(":")[1])
            m = re.search(r"ID:Length(\d+)", l)
            sq = re.search(r"sequence:([ACGU]+)", l)
            ann[i] = (int(m.group(1)) if m else None,
                      "REACTIVITY" if "REACTIVITY" in l else "OTHER",
                      sq.group(1) if sq else None)
        elif l.startswith("DATA:"):
            i = int(l.split("\t")[0].split(":")[1])
            data[i] = [x for x in l.split("\t")[1:] if x != ""]
    out = {}
    for i, vals in data.items():
        k, kind, pre = ann.get(i, (None, None, None))
        if kind != "REACTIVITY" or k is None or not pre:
            continue
        r = [float(x) if x not in ("", "NA", "nan") else None for x in vals]
        out[k] = {"prefix": pre, "reactivity": r}
    return {"sequence": seq, "lengths": out}


def shape_target(seq: str, react):
    """Reference structure from RNAstructure driven by the FULL-LENGTH experimental reactivity.

    A different codebase and a different parameter set from the one being tested, and the data -- not our
    model -- picks the structure. Returns None if the tool did not run, never a guess.
    """
    with tempfile.TemporaryDirectory() as d:
        sf, qf, cf, bf = (os.path.join(d, x) for x in ("sh.txt", "q.seq", "o.ct", "o.db"))
        with open(sf, "w") as f:
            for i, v in enumerate(react, 1):
                f.write(f"{i}\t{v if v is not None else -999}\n")
        open(qf, "w").write(f";\nx\n{seq}1\n")
        env = dict(os.environ, DATAPATH=DATAPATH)
        try:
            subprocess.run([f"{RNASTR}/Fold", qf, cf, "-sh", sf, "-mfe"], env=env,
                           capture_output=True, timeout=600, check=True)
            subprocess.run([f"{RNASTR}/ct2dot", cf, "1", bf], env=env,
                           capture_output=True, timeout=120, check=True)
        except Exception:                                                      # noqa: BLE001
            return None
        # ct2dot writes header, SEQUENCE, then structure. Concatenating every non-header line silently
        # produces a string of twice the right length, which then fails the length check and is reported as
        # "the tool did not run" -- a tool failure that never happened, and the exact confusion this file
        # is elsewhere careful about. Take only lines that are structure characters.
        db = [l.strip() for l in open(bf).read().splitlines()
              if l.strip() and set(l.strip()) <= set(".()")]
    s = "".join(db)
    return s if len(s) == len(seq) else None


def auroc(labels, scores):
    """P(score of a positive > score of a negative), ties counted as half. None if a class is empty."""
    pos = [s for l, s in zip(labels, scores) if l]
    neg = [s for l, s in zip(labels, scores) if not l]
    if not pos or not neg:
        return None
    tot = 0.0
    for p in pos:
        for n in neg:
            tot += 1.0 if p > n else (0.5 if p == n else 0.0)
    return tot / (len(pos) * len(neg))


def score_construct(rid: str, rec, min_lengths: int):
    is_dms = "_DMS_" in rid
    from rna_harness.benchmarks.kinetic_suite import pair_set
    seq, L = rec["sequence"], rec["lengths"]
    full = max(L) if L else 0
    if len(L) < min_lengths or full != len(seq):
        return {"rmdb_id": rid, "skipped": f"only {len(L)} lengths / full={full} vs seq={len(seq)}"}
    # every intermediate must genuinely be a 5' prefix of the full construct, or the object being measured
    # is not the object our criterion is defined on
    bad = [k for k, v in L.items() if not seq.startswith(v["prefix"])]
    if bad:
        return {"rmdb_id": rid, "skipped": f"{len(bad)} intermediates are not 5' prefixes"}

    target = shape_target(seq, L[full]["reactivity"])
    if not target:
        return {"rmdb_id": rid, "skipped": "RNAstructure did not produce a reference"}
    tgt_pairs = pair_set(target)

    import RNA
    rows = []
    for k in sorted(L):
        if k == full:
            continue                                        # the reference was derived from this one
        pre, rv = L[k]["prefix"], L[k]["reactivity"]
        n = min(len(pre), len(rv))
        idx = [i for i in range(n) if rv[i] is not None]
        if len(idx) < 15:
            continue
        mfe = RNA.fold(pre)[0]
        mp = pair_set(mfe)
        tp = {(i, j) for i, j in tgt_pairs if j < len(pre)}
        paired_m = {i for p in mp for i in p}
        paired_t = {i for p in tp for i in p}
        sc = [rv[i] for i in idx]
        a_m = auroc([i not in paired_m for i in idx], sc)    # unpaired should be MORE reactive
        a_t = auroc([i not in paired_t for i in idx], sc)
        if a_m is None or a_t is None:
            continue
        # THE FOCUSED TEST, and the reason it exists is that the AUROC comparison above turned out to be
        # rigged. Near the end of transcription a prefix IS nearly the whole molecule, so comparing
        # "MFE(prefix)" against "the SHAPE-derived reference truncated" degenerates into comparing MFE
        # against a SHAPE-directed fold ON SHAPE DATA -- which the SHAPE-directed fold wins by construction.
        # The artefact grows exactly where the supposed effect appeared, so that comparison cannot answer
        # the question at all.
        #
        # This one is symmetric. Our criterion's claim is about SPECIFIC off-target pairs. Take the positions
        # that (a) are in an off-target pair of MFE(prefix) and (b) are UNPAIRED in the truncated reference.
        # Compare their reactivity against the OTHER positions that are also unpaired in the reference. Both
        # groups are reference-unpaired, so how well the reference fits cancels; the only thing that differs
        # is whether OUR model says a pair is sitting there. If the trap is real, those positions must be
        # LESS reactive.
        off = mp - tgt_pairs
        off_pos = {i for p in off for i in p} - paired_t
        ctrl_pos = {i for i in idx if i not in paired_t} - off_pos
        focus = None
        if len(off_pos & set(idx)) >= 3 and len(ctrl_pos) >= 3:
            ov = [rv[i] for i in sorted(off_pos & set(idx))]
            cv = [rv[i] for i in sorted(ctrl_pos)]
            # WITHIN NUCLEOTIDE TYPE, because the two groups are not compositionally alike: the off-target
            # group ran 39% A / 45% C against the control's 75% A / 15% C, and DMS attacks A-N1 and C-N3 at
            # different rates. Comparing across that difference measures chemistry. Restricted to A/C for DMS
            # for the same reason -- a G or U carrying no DMS signal is not evidence of pairing.
            bases = ("A", "C") if is_dms else ("A", "C", "G", "U")
            per_base = {}
            for b in bases:
                o = [rv[i] for i in sorted(off_pos & set(idx)) if pre[i] == b]
                c = [rv[i] for i in sorted(ctrl_pos) if pre[i] == b]
                if len(o) >= 3 and len(c) >= 3:
                    a = auroc([True] * len(o) + [False] * len(c), [-x for x in o + c])
                    if a is not None:
                        per_base[b] = {"auroc": a, "n_off": len(o), "n_ctrl": len(c)}
            focus = {"n_off": len(ov), "n_ctrl": len(cv),
                     "auroc_less_reactive": auroc([True] * len(ov) + [False] * len(cv),
                                                  [-x for x in ov + cv]),
                     "per_base": per_base}
        rows.append({"k": k, "trap": bool(off), "n_off_pairs": len(off),
                     "auroc_mfe": a_m, "auroc_tgt": a_t, "n_scored": len(idx), "focus": focus})
    return {"rmdb_id": rid, "n": len(seq), "target": target, "n_lengths": len(rows), "rows": rows}


def report(R, out):
    import numpy as np
    from scipy.stats import mannwhitneyu, t as tdist, wilcoxon
    ok = [r for r in R if not r.get("skipped") and r.get("rows")]
    sk = [r for r in R if r.get("skipped")]
    print(f"\n=== 与【测量】对照 ===")
    print(f"  可用构建体 {len(ok)};跳过 {len(sk)}")
    for r in sk[:8]:
        print(f"    跳过 {r['rmdb_id']:22s} {r['skipped']}")
    allr = [(r["rmdb_id"], x) for r in ok for x in r["rows"]]
    traps = [(i, x) for i, x in allr if x["trap"]]
    print(f"  转录长度总数 {len(allr)},其中判据判为【陷阱】的 {len(traps)}")

    if len(ok) < MIN_RNAS or len(traps) < MIN_TRAPS:
        print(f"\n  !! 未达跑前声明的门槛(≥{MIN_RNAS} 个 RNA 且 ≥{MIN_TRAPS} 个陷阱前缀)")
        print(f"     报为【样本不足】,不报为 null —— 这是跑之前就写死的。")
        json.dump({"gate": "insufficient", "n_rnas": len(ok), "n_traps": len(traps),
                   "rows": R}, open(out, "w"), ensure_ascii=False, indent=1)
        return

    am = np.array([x["auroc_mfe"] for _, x in allr])
    at = np.array([x["auroc_tgt"] for _, x in allr])
    print(f"\n  预测与测量的一致性(AUROC,0.5 = 无信息):")
    print(f"    MFE(前缀)   {am.mean():.3f}      靶标截断到该长度   {at.mean():.3f}")
    if am.mean() < 0.55:
        print("    !! 预言机本身对不上测量 —— 这才是头条,判据的事排在后面")

    tm = np.array([x["auroc_mfe"] for _, x in traps])
    tt = np.array([x["auroc_tgt"] for _, x in traps])
    d_trap = tm - tt
    p1 = float(wilcoxon(tm, tt, alternative="greater")[1])
    print(f"\n  [检验1] 陷阱前缀内配对:MFE 比靶标截断更贴合测量?")
    print(f"    差 {d_trap.mean():+.3f}   更贴合 {int((d_trap>0).sum())}/{len(d_trap)}   Wilcoxon p={p1:.3g}")

    nont = [(i, x) for i, x in allr if not x["trap"]]
    d_non = np.array([x["auroc_mfe"] - x["auroc_tgt"] for _, x in nont])
    print(f"\n  [检验2 —— 决定性的] 陷阱 vs 非陷阱的【差之差】:")
    print(f"    陷阱   {d_trap.mean():+.3f} (n={len(d_trap)})")
    print(f"    非陷阱 {d_non.mean():+.3f} (n={len(d_non)})")
    p2 = float(mannwhitneyu(d_trap, d_non, alternative="greater")[1]) if len(d_non) >= 5 else None
    print(f"    差之差 {d_trap.mean()-d_non.mean():+.3f}   Mann-Whitney p={p2:.3g}" if p2 is not None
          else "    非陷阱样本太少,不做这个检验")

    # clustered by RNA: 71 transcript lengths of one molecule are not 71 molecules
    gid = sorted({i for i, _ in allr})
    per = []
    for g in gid:
        dt = [x["auroc_mfe"] - x["auroc_tgt"] for i, x in traps if i == g]
        dn = [x["auroc_mfe"] - x["auroc_tgt"] for i, x in nont if i == g]
        if dt and dn:
            per.append((g, float(np.mean(dt)), float(np.mean(dn))))
    print(f"\n  按 RNA 聚类({len(per)} 个 RNA 同时有陷阱与非陷阱前缀):")
    for g, a, b in per:
        print(f"    {g:22s} 陷阱 {a:+.3f}   非陷阱 {b:+.3f}   差 {a-b:+.3f}")
    p3 = None
    if len(per) >= 6:
        A = np.array([a for _, a, _ in per]); B = np.array([b for _, _, b in per])
        p3 = float(wilcoxon(A, B, alternative="greater")[1])
        print(f"    RNA 层面配对 Wilcoxon p={p3:.3g}   (n={len(per)} 个 RNA,这是诚实的自由度)")
    else:
        print(f"    只有 {len(per)} 个 RNA 两侧都有 —— 不做 RNA 层面检验,不足以支撑结论")

    # --- the AUROC comparison is confounded; show the confound rather than reporting its output ----------
    print("\n  [检验1/2 的设计缺陷,复核时才发现,写在这里而不是脚注]")
    print("    按转录进度分段看 MFE 减 靶标:")
    band = []
    for lo, hi in ((0.0, 0.4), (0.4, 0.7), (0.7, 1.0)):
        g = [x for r in ok for x in r["rows"]
             if x["trap"] and lo <= x["k"] / r["n"] < hi]
        if len(g) < 20:
            continue
        dm = np.array([x["auroc_mfe"] - x["auroc_tgt"] for x in g])
        band.append({"lo": lo, "hi": hi, "n": len(g), "delta": float(dm.mean())})
        print(f"      [{lo},{hi})  n={len(g):>4}  {dm.mean():+.3f}")
    print("    优势在【接近全长时最大、在早期为零】—— 而接近全长时前缀几乎就是整条分子,")
    print("    于是这一比退化成【MFE 对 SHAPE 定向折叠,在 SHAPE 数据上比】,后者按构造必胜。")
    print("    伪影出现的位置正好是所谓效应出现的位置,所以这两个检验回答不了这个问题。")

    # --- the focused test: symmetric, so the reference's fit quality cancels -----------------------------
    # A DEPOSIT IS NOT A MOLECULE. Fourteen of these are point variants of one glycine riboswitch, so a
    # test with n=24 would be claiming four times the independence it has.
    mol = lambda rid: re.match(r"[A-Z0-9]+", rid).group(0)                     # noqa: E731
    fo = [(i, x["focus"]) for i, x in traps if x.get("focus")]
    print(f"\n  [检验3 —— 对称的那个] 判据说【有配对】而参照说【无配对】的位点,反应性是否更低?")
    print(f"    可用前缀 {len(fo)}(两组都是参照未配对,参照拟合得好不好因此抵消)")
    p4 = None
    if len(fo) >= MIN_TRAPS:
        au = np.array([f["auroc_less_reactive"] for _, f in fo])
        p4 = float(wilcoxon(au - 0.5, alternative="greater")[1])
        print(f"    AUROC(更低反应性) 均值 {au.mean():.3f}  中位 {np.median(au):.3f}  "
              f">0.5 的比例 {(au > 0.5).mean():.1%}   Wilcoxon vs 0.5  p={p4:.3g}")
        # within-nucleotide-type, aggregated to the MOLECULE -- the only honest unit here
        pb = {}
        for i, f in fo:
            for b, v in (f.get("per_base") or {}).items():
                pb.setdefault(mol(i), {}).setdefault(b, []).append(v["auroc"])
        pr = []
        for g in sorted(pb):
            bym = {b: float(np.mean(v)) for b, v in pb[g].items() if len(v) >= 10}
            if bym:
                pr.append((g, float(np.mean(list(bym.values()))),
                           {b: {"auroc": v, "n": len(pb[g][b])} for b, v in bym.items()}))
        print(f"    同碱基内部比、聚合到【分子】(变体不算独立分子):")
        for g, m, bym in pr:
            print(f"      {g:10s} {m:.3f}   "
                  + "  ".join(f"{b} {v['auroc']:.3f}(n={v['n']})" for b, v in sorted(bym.items())))
        above = sum(1 for _, m, _ in pr if m > 0.5)
        print(f"    {above}/{len(pr)} 个分子 >0.5   均值 {np.mean([m for _, m, _ in pr]):.3f}")
        p5 = float(wilcoxon(np.array([m for _, m, _ in pr]) - 0.5,
                            alternative="greater")[1]) if len(pr) >= 6 else None
        if p5 is not None:
            print(f"    分子层面 Wilcoxon vs 0.5  p={p5:.3g}  (n={len(pr)} 个独立分子)")
            print(f"    注意:n={len(pr)} 时单侧 Wilcoxon 全正的最小可能 p 是 {2.0**-len(pr):.4f} "
                  f"—— 证据已到这个样本量的上限,再小的 p 拿不到")
        p4 = p5 if p5 is not None else p4
    else:
        print(f"    可用前缀不足 {MIN_TRAPS},不下结论")

    print(f"\n  判据在天然 RNA 上把 {len(traps)}/{len(allr)} = {len(traps)/len(allr):.1%} "
          f"的转录中间体判成【陷阱】 —— 这个数字与上面的设计缺陷无关,独立成立。")

    verdict = (p4 is not None and p4 < 0.05)
    print(f"\n  结论以【检验3】为准(检验1/2 的设计被自己的分段检查否掉了)-> "
          f"{'通过 —— 判据说的那些配对,在真实转录里确实压低了反应性' if verdict else '未通过 —— 照实报'}")
    json.dump({"gate": "ok", "n_rnas": len(ok), "n_lengths": len(allr), "n_traps": len(traps),
               "auroc_mfe": float(am.mean()), "auroc_tgt": float(at.mean()),
               "trap_delta": float(d_trap.mean()), "nontrap_delta": float(d_non.mean()),
               "p_within_traps": p1, "p_diff_in_diff": p2, "p_clustered_by_rna": p3,
               "design_flaw_bands": band, "n_focus": len(fo), "p_focused": p4,
               "trap_rate_on_natives": len(traps) / len(allr),
               "per_rna": [{"rmdb_id": g, "trap": a, "nontrap": b} for g, a, b in per],
               "focused_per_molecule": [{"molecule": g, "auroc": m, "per_base": bym}
                                        for g, m, bym in (pr if 'pr' in dir() else [])],
               "passes": bool(verdict), "rows": R}, open(out, "w"), ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=24, help="how many distinct constructs to score")
    ap.add_argument("--min-lengths", type=int, default=MIN_LENGTHS)
    ap.add_argument("--out", default=os.path.join(HERE, "experiment_check.json"))
    a = ap.parse_args()
    if not os.path.exists(f"{RNASTR}/Fold"):
        print(f"!! 找不到 {RNASTR}/Fold —— 靶标必须由独立代码库从测量导出,不用我们自己的模型顶替。")
        return

    man = json.loads(fetch(MANIFEST, os.path.join(CACHE, "manifest.json")).decode())
    co = [r for r in man
          if re.search(r"cotranscription|co-transcription",
                       " ".join(str(r.get(k, "")) for k in ("name", "pub_title", "rmdb_id")), re.I)
          and r.get("sequence_len") and r["data_points"] / r["sequence_len"] >= a.min_lengths]
    # one deposit per distinct construct: replicates of one molecule are not independent RNAs
    seen, picks = set(), []
    for r in sorted(co, key=lambda r: -r["data_points"]):
        key = re.sub(r"[,\s]*(1x|2x|4x)?Biotin.*|_rep\d.*", "", r["name"]).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        picks.append(r)
    picks = picks[:a.max]
    print(f"RMDB 共转录条目 {len(co)} 条沉积 -> {len(picks)} 个不同构建体(重复样本不算独立 RNA)\n"
          f"  靶标由 RNAstructure + 全长实测反应性导出(独立代码库、独立参数,数据说了算)\n"
          f"  跑前门槛:≥{MIN_RNAS} 个 RNA 且 ≥{MIN_TRAPS} 个陷阱前缀,否则报样本不足\n", flush=True)

    R, t0 = [], time.time()
    for n, r in enumerate(picks, 1):
        rid = r["rmdb_id"]
        try:
            html = fetch(DETAIL.format(rid), os.path.join(CACHE, f"{rid}.html")).decode("utf8", "ignore")
            m = re.search(r'href="(https://[^"]*?%s\.rdat)"' % re.escape(rid), html)
            if not m:
                R.append({"rmdb_id": rid, "skipped": "详情页没有 rdat 链接"})
                continue
            txt = fetch(m.group(1), os.path.join(CACHE, f"{rid}.rdat")).decode("utf8", "ignore")
            rec = parse_rdat(txt)
            R.append(score_construct(rid, rec, a.min_lengths) if rec
                     else {"rmdb_id": rid, "skipped": "rdat 解析失败"})
        except Exception as e:                                                 # noqa: BLE001
            R.append({"rmdb_id": rid, "skipped": f"抓取/打分出错 {repr(e)[:80]}"})
        print(f"    {n}/{len(picks)}  {rid:24s} {(time.time()-t0)/60:.1f} min", flush=True)
    report(R, a.out)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
