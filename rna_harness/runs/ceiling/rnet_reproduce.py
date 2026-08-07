#!/usr/bin/env python3
"""Can we reproduce RibonanzaNet's published predictions ourselves? Acceptance test before any integration.

WHY THIS COMES FIRST. Today measured that RibonanzaNet beats our thermodynamic panel at every point we could
find: within-target correlation with wet-lab score +0.364 against our +0.155; inside the top stratum where
selection actually happens ours is flat while RNet still orders at +0.463; inside the stratum our
certificate calls not_confirmed, ours reaches +0.163 and RNet +0.328. Everywhere our signal knows something,
RNet knows more. The honest response to that is to use it, not to keep defending the panel.

But every one of those numbers came from a COLUMN IN SOMEONE ELSE'S CSV. We have never run the model. Until
we can reproduce that column we do not actually know what beat us, we cannot apply it to a target that is
not in OpenKnot, and we cannot tell a real advantage from an artifact of how they post-processed. Running
the model is therefore not the integration -- it is the precondition for being allowed to talk about it.

WHAT MAKES A CLEAN ACCEPTANCE TEST POSSIBLE. The release publishes `RNet_structure`, the predicted
dot-bracket itself, for all 34,750 rows -- not merely the derived F1. So the test is not "do we get a
similar score" but "do we output the same structure", which is far sharper: a mismatch in tokenisation,
checkpoint, symmetrisation or thresholding shows up immediately instead of hiding inside an aggregate.

MODEL. RibonanzaNet-SS.pt from kaggle shujun717/ribonanzanet-weights (MIT). 11.3 M parameters, 9 layers,
ninp 256, pairwise dimension 64, triangular attention off. The secondary-structure head is a single
Linear(64 -> 1) over the final pairwise features (`ct_predictor` in the checkpoint), so the whole model is
the published backbone plus one layer -- there is nothing hidden to guess at.

PRE-REGISTERED ACCEPTANCE, written before the model was run. n = 300 designs sampled by stride across the
whole length range so 240-mers are represented in proportion.

    R1  Median base-pair F1 between OUR predicted structure and their published `RNet_structure` >= 0.90.
        Below that we have not reproduced the model and must not use it, quote it, or claim to have been
        beaten by it -- we would only know we were beaten by something.
    R2  Spearman between our derived `F1_crossed_pair` and their published `RNet_F1_crossed_pair` >= 0.90
        over the same designs. R1 checks the structure, R2 checks the quantity today's comparisons actually
        used.
    R3  A THRESHOLD SWEEP IS REPORTED, not a single number. Today's own audit found a 14.1% headline that
        turned out to be almost entirely its threshold's doing; the same discipline applies here, where
        pair calls come from thresholding a sigmoid.

    KILL: R1 fails -> we do not have the model, and the integration plan stops here until we do.

TIMING IS PART OF THE RESULT. Whether the full 27,362-design rescoring is a CPU job or needs a GPU is an
empirical question about an O(L^2) pairwise stack at L up to 240, not something to estimate. Single-sequence
forward passes are timed at both ends of the length range and extrapolated, and the extrapolation is
reported so the decision to stay on CPU or move to slurm is made on a measurement.

COST: $0. Local CPU, no API.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics as st
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RNET_SRC = "/data/wenhui/RibonanzaNet"
WEIGHTS = "/data/wenhui/rnet_weights/RibonanzaNet-SS.pt"
DATA = "/data/wenhui/openknot_data/Data/OpenKnotBench_data.v4.5.1.csv"
sys.path.insert(0, REPO)
sys.path.insert(0, RNET_SRC)

TOKENS = {nt: i for i, nt in enumerate("ACGU")}
TOKENS["P"] = 4


def build_model():
    import torch
    import torch.nn as nn
    import yaml
    from Network import RibonanzaNet

    cfg = yaml.safe_load(open(os.path.join(RNET_SRC, "configs", "pairwise.yaml")))
    cfg = types.SimpleNamespace(**cfg)
    cfg.dropout = 0.0
    model = RibonanzaNet(cfg)
    model.ct_predictor = nn.Linear(64, 1)
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not [k for k in missing if "ct_predictor" in k or "encoder.weight" == k], \
        f"checkpoint does not fit the model we built: missing {missing[:6]}"
    print(f"  载入 {WEIGHTS}:缺失 {len(missing)} 个键,多余 {len(unexpected)} 个")
    model.eval()
    return model


def pairwise_logits(model, seq):
    """Backbone forward that returns the pairwise stack, then the SS head. Mirrors RibonanzaNet.forward
    up to the point where it collapses to a sequence-level decoder, which is not what we want here."""
    import torch
    src = torch.tensor([[TOKENS[c] for c in seq]], dtype=torch.long)
    with torch.no_grad():
        mask = torch.ones_like(src)          # every position is real; the layers require an explicit mask
        x = model.encoder(src).reshape(1, len(seq), -1)
        pf = model.outer_product_mean(x)
        pf = pf + model.pos_encoder(src)
        for layer in model.transformer_encoder:
            x, pf = layer(x, pf, mask)
        pf = pf + pf.permute(0, 2, 1, 3)          # symmetrise: a pair is (i,j) and (j,i)
        return model.ct_predictor(pf).squeeze(-1).squeeze(0)


def pairs_from_logits(logits, thresh=0.5, min_loop=3):
    """Greedy maximum-probability matching over the sigmoid map: each base takes at most one partner."""
    import torch
    p = torch.sigmoid(logits)
    L = p.shape[0]
    cand = []
    for i in range(L):
        for j in range(i + min_loop + 1, L):
            v = float(p[i, j])
            if v >= thresh:
                cand.append((v, i, j))
    cand.sort(reverse=True)
    used, out = set(), set()
    for v, i, j in cand:
        if i in used or j in used:
            continue
        used.add(i); used.add(j)
        out.add((i, j))
    return out


def f1(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    tp = len(a & b)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(a), tp / len(b)
    return 2 * prec * rec / (prec + rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(HERE, "rnet_reproduce.json"))
    a = ap.parse_args()

    import json
    from m2r_pairwise import spearman

    # Parsed here rather than imported from the harness: this script runs in the RibonanzaNet venv, and
    # importing rna_harness pulls in ViennaRNA bindings that venv does not have. Dot-bracket parsing is
    # four lines and duplicating it is cheaper than making the model env depend on the folding stack.
    _OPEN = {"(": ")", "[": "]", "{": "}", "<": ">"}
    _CLOSE = {v: k for k, v in _OPEN.items()}

    def pk_pairs(struct):
        stacks = {o: [] for o in _OPEN}
        out = set()
        for i, c in enumerate(struct):
            if c in _OPEN:
                stacks[c].append(i)
            elif c in _CLOSE and stacks[_CLOSE[c]]:
                out.add((stacks[_CLOSE[c]].pop(), i))
        return out

    def crossing(pairs):
        """A pair (i,j) is CROSSED iff some other pair straddles exactly one of its ends -- the release's
        own definition. Computed WITHIN a structure, so the predicted set and the target set each supply
        their own; intersecting the prediction with the target first (which an earlier version of this
        script did) pins precision at 1.0 and silently turns F1 into recall."""
        out = set()
        for (i, j) in pairs:
            for (m, n) in pairs:
                if (m < i < n < j) or (i < m < j < n):
                    out.add((i, j))
                    break
        return out

    csv.field_size_limit(10 ** 9)
    want = ("id", "SN_filter", "puzzle", "method", "design_sequence", "target_structure",
            "RNet_structure", "RNet_F1", "RNet_F1_crossed_pair")
    rows = []
    with open(DATA, newline="") as fh:
        r = csv.reader(fh)
        idx = {c: i for i, c in enumerate(next(r)) if c in want}
        for x in r:
            d = {c: x[i] for c, i in idx.items()}
            if d["SN_filter"] != "1" or not d["RNet_structure"]:
                continue
            s, t, rs = d["design_sequence"], d["target_structure"], d["RNet_structure"]
            if not s or set(s) - set("ACGU") or len(s) != len(t) or len(rs) != len(s):
                continue
            try:
                d["rnet_f1c"] = float(d["RNet_F1_crossed_pair"])
            except (TypeError, ValueError):
                continue
            rows.append(d)
    step = max(1, len(rows) // a.n)
    sample = rows[::step][:a.n]
    print(f"可比对的行 {len(rows)};按步长取样 {len(sample)} 条,"
          f"长度 {min(len(d['design_sequence']) for d in sample)}-"
          f"{max(len(d['design_sequence']) for d in sample)} nt\n")

    print("=== 载入模型 ===")
    model = build_model()

    print("\n=== 计时(决定 CPU 够不够,不靠估计)===")
    import torch
    for L in (100, 240):
        s = "ACGU" * (L // 4) + "A" * (L % 4)
        t0 = time.time()
        pairwise_logits(model, s)
        dt = time.time() - t0
        print(f"  L={L:3d}  单条前向 {dt:.2f}s  -> 27,362 条单进程 {dt*27362/3600:.1f} 小时;"
              f"  16 进程 {dt*27362/3600/16:.1f} 小时")

    print(f"\n=== 复现比对({len(sample)} 条)===")
    recs, t0 = [], time.time()
    for n, d in enumerate(sample, 1):
        seq, tgt, theirs = d["design_sequence"], d["target_structure"], d["RNet_structure"]
        lg = pairwise_logits(model, seq)
        ours = pairs_from_logits(lg)
        their_pairs = pk_pairs(theirs)
        tp = pk_pairs(tgt)
        recs.append({"id": d["id"], "n": len(seq),
                     "f1_vs_theirs": f1(ours, their_pairs),
                     "our_f1_all": f1(ours, tp),
                     "our_f1_cross": f1(crossing(ours), crossing(tp)),
                     "their_f1_cross": d["rnet_f1c"]})
        if n % 60 == 0 or n == len(sample):
            print(f"    {n}/{len(sample)}  {(time.time()-t0)/60:.1f} min", flush=True)

    v = [r["f1_vs_theirs"] for r in recs]
    print(f"\n  R1 我们的结构 vs 他们发布的 RNet_structure:")
    print(f"     碱基对 F1 中位 {st.median(v):.3f},均值 {st.mean(v):.3f},"
          f">=0.9 的占 {sum(1 for x in v if x >= 0.9)/len(v):.1%}")
    r1 = st.median(v) >= 0.90
    print(f"     {'通过' if r1 else 'KILL —— 我们没有复现出这个模型,不得使用也不得引用'}")

    ours_c = [r["our_f1_cross"] for r in recs]
    theirs_c = [r["their_f1_cross"] for r in recs]
    rho = spearman(ours_c, theirs_c)
    r2 = (rho or 0) >= 0.90
    print(f"\n  R2 交叉配对 F1 的一致性:Spearman {rho:+.3f}   {'通过' if r2 else '未通过'}")

    print(f"\n  R3 阈值扫描(今天已经吃过一次单一阈值的亏)")
    for th in (0.3, 0.4, 0.5, 0.6, 0.7):
        vv = []
        for d in sample[::max(1, len(sample)//60)]:
            lg = pairwise_logits(model, d["design_sequence"])
            vv.append(f1(pairs_from_logits(lg, th), pk_pairs(d["RNet_structure"])))
        print(f"     阈值 {th:.1f}  与他们的结构 F1 中位 {st.median(vv):.3f}(n={len(vv)})")

    json.dump({"n": len(recs), "R1_median_f1": st.median(v), "R1": r1,
               "R2_spearman": rho, "R2": r2, "recs": recs},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
