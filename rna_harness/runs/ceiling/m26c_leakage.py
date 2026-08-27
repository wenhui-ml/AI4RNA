#!/usr/bin/env python3
"""M26c leakage ladder L1/L1b/L2/L3 -- declared in m26c_neural_scored_declaration.json (c22b1ddd).

Not a new experiment. These are probes the M26c declaration wrote down as `P0_leakage_ladder` and
required to run, and which that run reported as deferred because the training sequences were not
on disk. They are on the network, without credentials: RibonanzaNet's training libraries were
deposited in RMDB by the same lab, under names that say what they are
(DasLabBigLib_OneMil_OpenKnot_Round_2_train, ...openknot3_splitA, ...).

The bound this can and cannot establish, stated before it runs: these deposits are Ribonanza's
SOURCE LIBRARIES, not a manifest of what went into the 2024-05 checkpoint. A zero here licenses
"no overlap with the training sources we could obtain", never "no leakage".

Streams; writes no rdat to disk (/data is at 96%).
"""
from __future__ import annotations

import argparse, collections, csv, gzip, io, json, os, re, sys, time, urllib.request

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "rna-harness/1.0"}
SRC = "/data/wenhui/openknot_data/Data/OK7a_M2R_data.v4.5.1.csv"
DETAIL = "https://rmdb.stanford.edu/detail/{}/"

# Ribonanza's own source libraries, by RMDB id. Names from the manifest, quoted in the audit.
TRAIN_DEPOSITS = [
    "OK2TRN_DMS_0000", "OK2TRN_2A3_0000",        # OpenKnot Round 2 train
    "OK2TST_DMS_0000", "OK2TST_2A3_0000",        # Round 2 test
    "OK3TST_DMS_0000", "OK3TST_DMS_0001", "OK3TST_DMS_0002", "OK3TST_DMS_0003",
    "M2PK90_DMS_0000",                            # single-nt mutants, OpenKnot
    "M2RFPK_DMS_0000", "M2RFPK_DMS_0001",         # Rfam pseudoknots
    "15KLIB_DMS_0000", "15KREP_DMS_0000", "15KREP_DMS_0001",
]
K = 20            # k-mer length for the near-neighbour index


def rdat_url(rid):
    html = urllib.request.urlopen(
        urllib.request.Request(DETAIL.format(rid), headers=UA), timeout=180).read().decode("utf8", "ignore")
    m = re.search(r'href="(https://[^"]*?%s\.rdat)"' % re.escape(rid), html)
    return m.group(1) if m else None


def stream_sequences(url, log):
    """pull ANNOTATION_DATA sequences out of an rdat without ever writing it down."""
    out = []
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=1800) as r:
        buf = b""
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for l in lines:
                if l.startswith(b"ANNOTATION_DATA") or l.startswith(b"DATA_ANNOTATION"):
                    m = re.search(rb"sequence:([ACGUTN]+)", l)
                    if m:
                        out.append(m.group(1).decode().replace("T", "U"))
    return out


def ok7a_sequences():
    """the 4475 unique sequences M26c scored, plus each one's declared design window."""
    csv.field_size_limit(10 ** 9)
    seqs, windows = {}, {}
    for d in csv.DictReader(open(SRC)):
        try:
            ss, se = int(float(d["sub_start"])), int(float(d["sub_end"]))
        except (ValueError, KeyError):
            continue
        s = d["sequence"]
        seqs[s] = True
        windows[s] = (ss - 1, se)
    return list(seqs), windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "m26c_leakage.json"))
    a = ap.parse_args()
    LOG = []

    def log(s=""):
        print(s, flush=True); LOG.append(s)

    t0 = time.time()
    log("=== M26c 泄漏阶梯 L1/L1b/L2/L3 ===")
    log("  声明 c22b1ddd 的 P0_leakage_ladder。不落盘,离线 API $0。\n")

    ours, win = ok7a_sequences()
    log(f"OK7a 唯一序列 {len(ours)},长度 {min(map(len,ours))}~{max(map(len,ours))}")
    our_win = {s: s[win[s][0]:win[s][1]] for s in ours}
    log(f"  设计窗内子串唯一数 {len(set(our_win.values()))}\n")

    train_full, train_by_dep = set(), {}
    log("=== 取 Ribonanza 源库(RMDB,同一实验室的原始沉积)===")
    for rid in TRAIN_DEPOSITS:
        try:
            u = rdat_url(rid)
            if not u:
                log(f"  {rid:20s} 无 rdat 链接,跳过"); train_by_dep[rid] = None; continue
            s = stream_sequences(u, log)
            train_by_dep[rid] = len(s)
            train_full.update(s)
            log(f"  {rid:20s} {len(s):8d} 条  累计唯一 {len(train_full):9d}  {time.time()-t0:5.0f}s")
        except Exception as e:                                                 # noqa: BLE001
            log(f"  {rid:20s} 取用失败 {repr(e)[:70]}"); train_by_dep[rid] = None
    OUT = {"n_ours": len(ours), "n_train_unique": len(train_full),
           "per_deposit": train_by_dep}
    if not train_full:
        OUT["verdict"] = "unavailable_no_training_sequences"
        json.dump(OUT, open(a.out, "w"), ensure_ascii=False, indent=1)
        log("\n**一条训练序列都没取到 ⇒ L1/L2 记为 unavailable,只有 L4 兜底。**"); return 1

    # ---------------- L1 exact, whole construct
    hit1 = [s for s in ours if s in train_full]
    log(f"\n=== L1 整条精确重合 ===\n  {len(hit1)} / {len(ours)}")
    OUT["L1_exact_full"] = len(hit1)

    # ---------------- L1b exact, design window only (the meaningful comparison)
    train_sub = set()
    for s in train_full:
        train_sub.add(s)
    # index training substrings of our window lengths, so a shared design in different flanks is caught
    wl = collections.Counter(len(v) for v in our_win.values())
    log(f"\n=== L1b 设计窗内精确重合(剥掉共享侧翼)===\n  窗长分布 {dict(wl)}")
    sub_index = set()
    for s in train_full:
        for L in wl:
            if len(s) >= L:
                for st in range(0, len(s) - L + 1):
                    sub_index.add(s[st:st + L])
        if len(sub_index) > 60_000_000:
            log("  子串索引超过 6e7,停止扩展并照报(这是覆盖上限,不是零)")
            break
    hit1b = [s for s in ours if our_win[s] in sub_index]
    log(f"  {len(hit1b)} / {len(ours)}   (训练子串索引 {len(sub_index)})")
    OUT["L1b_exact_window"] = len(hit1b)
    OUT["L1b_index_size"] = len(sub_index)

    # ---------------- L2 near neighbour by k-mer candidate generation
    log(f"\n=== L2 最大序列同一性(k={K} 倒排索引找候选)===")
    inv = collections.defaultdict(list)
    tl = list(train_full)
    for i, s in enumerate(tl):
        for st in range(0, max(1, len(s) - K + 1), K // 2):
            inv[s[st:st + K]].append(i)
    log(f"  索引 {len(inv)} 个 k-mer")
    best = []
    for n, s in enumerate(ours):
        w = our_win[s]
        cand = collections.Counter()
        for st in range(0, max(1, len(w) - K + 1)):
            for i in inv.get(w[st:st + K], ()):
                cand[i] += 1
        top = [i for i, _c in cand.most_common(40)]
        b = 0.0
        for i in top:
            t = tl[i]
            for st in range(0, max(1, len(t) - len(w) + 1)):
                seg = t[st:st + len(w)]
                if len(seg) != len(w):
                    continue
                ident = sum(x == y for x, y in zip(w, seg)) / len(w)
                if ident > b:
                    b = ident
        best.append(b)
        if (n + 1) % 1000 == 0:
            log(f"    {n+1}/{len(ours)}  {time.time()-t0:.0f}s")
    best = np.array(best)
    OUT["L2"] = {"mean": round(float(best.mean()), 4), "median": round(float(np.median(best)), 4),
                 "max": round(float(best.max()), 4),
                 "frac_gt_0.8": round(float((best > 0.8).mean()), 4),
                 "frac_gt_0.9": round(float((best > 0.9).mean()), 4),
                 "n_eq_1": int((best >= 0.999).sum())}
    log(f"  最大同一性  均值 {best.mean():.4f}  中位 {np.median(best):.4f}  最大 {best.max():.4f}")
    log(f"  >0.8 的比例 {100*(best>0.8).mean():.2f}%   >0.9 {100*(best>0.9).mean():.2f}%   "
        f"=1.0 的条数 {int((best>=0.999).sum())}")

    # ---------------- verdict, with the bound written in
    k0 = (OUT["L1_exact_full"] == 0 and OUT["L1b_exact_window"] == 0
          and OUT["L2"]["frac_gt_0.8"] <= 0.05)
    OUT["K0_leakage_suspected"] = bool(not k0)
    OUT["verdict"] = ("no_overlap_with_obtainable_training_sources" if k0
                      else "leakage_suspected")
    log(f"\n=== 裁定: {OUT['verdict']} ===")
    log("  **界:这些沉积是 Ribonanza 的源库,不是 2024-05 检查点的训练清单。**")
    log("  **允许写的上限是『与我们能取到的训练来源无重合』,不是『没有泄漏』。**")
    if not k0:
        log("  **K0 触发:M26c 的 N2/N4 不许解读为跨模型类证据,泄漏进正文而非附录。**")
    json.dump(OUT, open(a.out, "w"), ensure_ascii=False, indent=1)
    open(os.path.join(HERE, "m26c_leakage.log"), "w").write("\n".join(LOG) + "\n")
    log(f"\n写出 {a.out}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
