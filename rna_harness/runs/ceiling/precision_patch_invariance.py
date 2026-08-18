#!/usr/bin/env python3
"""Did adding the precision term to `_knot_recall` change any verdict? Checked against the PREVIOUS code.

WHY THIS EXISTS RATHER THAN A CLAIM THAT THE PATCH IS ADDITIVE. "I only added keys" is exactly the kind of
statement that is true until it is not: `_knot_recall` is read by the ladder, by the screen, by the tie-break
and by the pool dump, and its `per_oracle` values feed the `mean_recall >= 0.1` screen. The project's own rule
after CERTIFY_POOL_DUMP was that behaviour invariance gets VERIFIED, not asserted, and that a single
before/after pair cannot tell an effect from the wall-clock ladder's own jitter. So this compares the two
implementations on IDENTICAL inputs, where there is no jitter at all:

    I1  the previous `_knot_recall`, loaded from git (HEAD's certify.py), against the current one, on the same
        (sequence, target) pairs drawn from the dumped pools. Every shared key must be identical -- the four
        the verdict reads (`per_oracle`, `n_cross`, `confirmed_by`, `knot_confirmed`) and nothing excused.
    I2  the screen quantity the ladder actually thresholds -- mean of the non-None per-oracle recalls -- must
        be identical, and the >= 0.1 decision must be identical, on every pair.
    I3  the deterministic verdict path end to end: `certify.py --sequence` re-checks a GIVEN sequence with no
        search, so it is reproducible by construction (folding is deterministic). Every certificate field
        must match except the three that are wall-clock derived (`generated_utc`, `wait_s`, `cost_usd` --
        cost.py bills elapsed vCPU-seconds), and both runs must see the same DESIRNA_CMD or the provenance
        line differs for a reason unrelated to the patch. Both of those were mistakes in the first version
        of this check, which reported a failure that was its own.

    Failure of any of these means the patch is not additive and must be reverted, not explained.

The new keys are printed as a summary at the end, because the point of keeping them is that they are worth
looking at: the panel over-predicts crossing pairs by a wide margin and the shipped criterion cannot see it.

COST: $0. Local folds only; ~3 oracle calls per sequence, run once per implementation.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import statistics as st
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

SHARED = ("per_oracle", "n_cross", "confirmed_by", "knot_confirmed")


def load_old_impl(ref: str):
    """Import HEAD's certify.py under a throw-away module name so both versions coexist in one process."""
    import importlib.util
    src = subprocess.run(["git", "show", f"{ref}:rna_harness/cognitive/certify.py"],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    d = tempfile.mkdtemp(prefix="oldcertify_")
    pkg = os.path.join(d, "oldpkg")
    os.makedirs(pkg)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    p = os.path.join(pkg, "certify_old.py")
    # the module is loaded standalone, so every relative import has to become absolute: `..` is the package
    # root, `.` is rna_harness.cognitive. Order matters -- rewrite `..` before `.` or the second pass eats
    # the first one's output.
    open(p, "w").write(src.replace("from ..", "from rna_harness.")
                          .replace("from .", "from rna_harness.cognitive."))
    spec = importlib.util.spec_from_file_location("certify_old", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["certify_old"] = m          # dataclasses resolves annotations via sys.modules, so register first
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="HEAD", help="git ref holding the pre-patch certify.py")
    ap.add_argument("--n", type=int, default=60, help="sequence/target pairs for I1+I2")
    ap.add_argument("--n-cert", type=int, default=3, help="pairs for the end-to-end I3 check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "precision_patch_invariance.json"))
    a = ap.parse_args()

    from rna_harness.cognitive.certify import _knot_recall as new_kr
    old = load_old_impl(a.ref)
    print(f"旧实现取自 git {a.ref};新实现取自工作区")

    R = json.load(open(os.path.join(HERE, "leaderboard_replay.json")))
    T = {p: v["target"] for p, v in R["targets"].items()}
    import hashlib
    pairs = []
    for p, t in sorted(T.items()):
        f = os.path.join(HERE, "pool_dump", hashlib.sha1(t.encode()).hexdigest()[:16] + ".jsonl")
        if not os.path.exists(f):
            continue
        for line in open(f):
            try:
                pairs.append((p, json.loads(line)["seq"], t))
            except Exception:                                                  # noqa: BLE001
                continue
    rng = random.Random(a.seed)
    rng.shuffle(pairs)
    pairs = pairs[:a.n]
    print(f"抽 {len(pairs)} 对 (靶标, 序列) 做逐键比对\n")

    bad1, bad2, rows = [], [], []
    for puz, seq, tgt in pairs:
        o, n = old._knot_recall(seq, tgt), new_kr(seq, tgt)
        for k in SHARED:
            if o[k] != n[k]:
                bad1.append((puz, k))
        vo = [x for x in o["per_oracle"].values() if x is not None]
        vn = [x for x in n["per_oracle"].values() if x is not None]
        so = (sum(vo) / len(vo)) if vo else None
        sn = (sum(vn) / len(vn)) if vn else None
        if so != sn or (so is not None and (so >= 0.1) != (sn >= 0.1)):
            bad2.append(puz)
        pv = [x for x in n["per_oracle_precision"].values() if x is not None]
        fv = [x for x in n["per_oracle_f1"].values() if x is not None]
        nv = [x for x in n["n_pred_cross"].values() if x is not None]
        rows.append({"puzzle": puz, "recall": so,
                     "precision": (sum(pv) / len(pv)) if pv else None,
                     "f1": (sum(fv) / len(fv)) if fv else None,
                     "n_pred_cross": (sum(nv) / len(nv)) if nv else None,
                     "n_cross": n["n_cross"], "knot_confirmed": n["knot_confirmed"]})

    print(f"=== I1 四个被判据读取的键,逐对比对 ===\n  不一致 {len(bad1)} 处 {bad1[:6]}"
          + ("  ✓" if not bad1 else "  ✗"))
    print(f"=== I2 阶梯真正阈值的那个量(非 None 召回的均值)与 >=0.1 的判定 ===\n"
          f"  不一致 {len(bad2)} 对 {bad2[:6]}" + ("  ✓" if not bad2 else "  ✗"))

    # ---- I3 end-to-end on the deterministic --sequence path -------------------------------------------
    print(f"\n=== I3 端到端(--sequence,不搜索,折叠确定,故可逐字节比) ===")
    bad3 = []
    for puz, seq, tgt in pairs[:a.n_cert]:
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = "0"
        # The version line records whether DesiRNA is reachable, so both runs must see the same solver or the
        # certificates differ on provenance for a reason that has nothing to do with the patch. Found by the
        # first version of this check reporting a failure that was entirely its own.
        env.setdefault("DESIRNA_CMD", "/home/ubuntu/miniconda3/envs/rna/bin/python "
                                      "/data/wenhui/DesiRNA/DesiRNA.py")
        cmd = [sys.executable, "-m", "rna_harness.cognitive.certify", tgt, "--sequence", seq, "--json"]
        cur = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env).stdout
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["git", "worktree", "add", "--detach", tmp, a.ref],
                           cwd=REPO, capture_output=True, text=True, check=True)
            try:
                prev = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, env=env).stdout
            finally:
                subprocess.run(["git", "worktree", "remove", "--force", tmp],
                               cwd=REPO, capture_output=True, text=True)
        def norm(s):
            try:
                d = json.loads(s)
            except Exception:                                                  # noqa: BLE001
                return s
            # Three fields are wall-clock derived and cannot be equal across two runs: the timestamp, the
            # measured wait, and cost_usd, which cost.py bills from elapsed vCPU-seconds. Dropping them is
            # not an excuse -- they are the ONLY fields a re-run of the same deterministic path may move,
            # and the ladder is not entered at all in --sequence mode.
            for k in ("generated_utc", "wait_s", "cost_usd"):
                d.pop(k, None)
            return json.dumps(d, sort_keys=True)
        same = norm(cur) == norm(prev)
        print(f"  {puz}  证书一致: {'✓' if same else '✗'}")
        if not same:
            bad3.append(puz)

    ok = not (bad1 or bad2 or bad3)
    print(f"\n=== 判定 ===\n  {'行为不变,补丁是纯附加的。' if ok else '补丁改变了行为 —— 应当回滚,而不是解释。'}")

    prec = [r["precision"] for r in rows if r["precision"] is not None]
    npc = [r["n_pred_cross"] for r in rows if r["n_pred_cross"] is not None]
    ncr = [r["n_cross"] for r in rows]
    print(f"\n=== 新记录的量(这才是留着它的理由)===")
    print(f"  面板预测的交叉配对数 中位 {st.median(npc):.1f}  对 目标真实交叉配对数 中位 {st.median(ncr):.0f}"
          f"  → 过预测约 {st.median(npc)/max(1,st.median(ncr)):.1f} 倍")
    print(f"  精确度 中位 {st.median(prec):.3f};出厂判据只看召回,看不见这一项")

    json.dump({"invariant": ok, "i1_mismatches": bad1, "i2_mismatches": bad2, "i3_mismatches": bad3,
               "n_pairs": len(pairs), "ref": a.ref,
               "precision_median": st.median(prec) if prec else None,
               "n_pred_cross_median": st.median(npc) if npc else None,
               "n_cross_median": st.median(ncr) if ncr else None,
               "rows": rows}, open(a.out, "w"), indent=1)
    print(f"\n写出 {a.out}")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
