#!/usr/bin/env python3
"""Prove the zero-crossing guard is inert with the flag off, and measure exactly what it does with it on.

Same discipline as precision_patch_invariance.py: load HEAD's `_knot_recall` from git in this same process
and compare it field-for-field against the patched one on the same folds. A patch to the certificate's vote
counter is not allowed to ship on the claim that it is off by default -- that claim has to be checked.

WHY THE GUARD IS OFF BY DEFAULT. I recommended it in k1_audit.md as a fix touching only the 7 unanimous
zero-crossing designs and therefore needing no recalibration. Wrong by two orders of magnitude: the guard is
PER ORACLE, so a design whose panel predicts crossings can still have one member predicting none. Replayed
from the stored columns on four populations it changes confirmed_by for ~12% of designs and flips
knot_confirmed for ~3.5%. That voids the calibrated operating point exactly as a definition change would.

COST: $0. The invariance half needs the oracles (a few sequences only); the blast-radius half replays stored
columns and folds nothing.
"""
from __future__ import annotations

import collections
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

ORACLES = ("pkiss", "probknot", "shapeknots")
TUBE_PASS = 80.0
POPS = (("基准4560(主)", "panel_f1_designs.json"), ("样本3", "panel_f1_third.json"),
        ("第四批", "panel_f1_fourth.json"), ("样本2", "panel_f1_fresh.json"))
VERDICT_KEYS = ("per_oracle", "n_cross", "confirmed_by", "knot_confirmed")
# The certificate embeds kr WHOLE -- certify.py builds detail={**v, "knot": kr, ...} at three call sites --
# so two added keys ARE an output change even though no verdict reads them. "Zero change to shipped
# behaviour" was too strong: the VERDICT is unchanged, the emitted field SET grows by two. Same precedent as
# the 2026-08-18 precision patch, which added three observable keys the same way, but it has to be stated as
# an additive change rather than as no change. This check now compares the whole key set, not four keys.
EXPECTED_NEW_KEYS = {"confirmed_by_zero_crossing_guarded", "zero_crossing_votes_withdrawn"}


def head_knot_recall():
    """`_knot_recall` as it stands at HEAD~ (the commit before the guard), loaded in this process."""
    rev = subprocess.run(["git", "-C", REPO, "show", "HEAD:rna_harness/cognitive/certify.py"],
                         capture_output=True, text=True, check=True).stdout
    if "CERTIFY_ZERO_CROSSING_GUARD" in rev:
        rev = subprocess.run(["git", "-C", REPO, "show",
                              "HEAD~1:rna_harness/cognitive/certify.py"],
                             capture_output=True, text=True, check=True).stdout
    assert "CERTIFY_ZERO_CROSSING_GUARD" not in rev, "could not find a pre-guard revision"
    ns = {"__name__": "certify_head"}
    import types
    mod = types.ModuleType("certify_head")
    mod.__dict__.update(ns)
    mod.__package__ = "rna_harness.cognitive"
    sys.modules["certify_head"] = mod       # dataclasses resolves cls.__module__ through sys.modules
    exec(compile(rev, "<HEAD certify.py>", "exec"), mod.__dict__)
    return mod._knot_recall


def main() -> int:
    print("== 第一半:旗标【关】时,对出厂判决必须逐字段恒等 ==")
    from rna_harness.cognitive.certify import _knot_recall as patched
    old = head_knot_recall()
    src = json.load(open(os.path.join(HERE, "panel_f1_designs.json")))["designs"]
    from leaderboard_design import targets
    from novelty_stratified_ruler import replay_sample
    T = targets()
    # panel_f1_designs.json carries no sequence; the join to replay_sample is by ORDER, and that order was
    # asserted element-wise in G2 and G5 (0 mismatches on (puzzle, method, wet, f1c)). Assert it again here.
    jobs, _ = replay_sample(0, 80)
    assert len(jobs) == len(src), f"{len(jobs)} vs {len(src)}"
    bad = [k for k, ((p, _s, f1c, wet, m), d) in enumerate(zip(jobs, src))
           if p != d["puzzle"] or m != d["method"] or abs(wet - d["wet"]) > 1e-6
           or abs(f1c - d["f1c"]) > 1e-6]
    assert not bad, f"order join broken at {bad[:5]}"
    for (p, sq, _f, _w, _m), d in zip(jobs, src):
        d["seq"] = sq
    print(f"   顺序连接断言:{len(src)} 条,不一致 0")
    # pick cases that STRESS the guard: designs whose stored columns say a vote would be withdrawn
    stress, plain = [], []
    for r in src:
        po = r.get("per_oracle") or {}
        if sorted(po) != sorted(ORACLES):
            continue
        o = [x for x in ORACLES if (po[x] or {}).get("recall_product") is not None
             and po[x]["recall_product"] >= 1.0]
        g = [x for x in o if (po[x] or {}).get("n_pred_cross")]
        (stress if len(o) != len(g) else plain).append(r)
    cases = stress[:8] + plain[:4]
    print(f"   用例 {len(cases)} 条:{len(stress[:8])} 条【会被 guard 动到】+ {len(plain[:4])} 条不会"
          f"  (第 29 条:用例要落在被验证性质【为真】的那一侧,所以以 stress 为主)")
    assert "CERTIFY_ZERO_CROSSING_GUARD" not in os.environ, "flag must be unset for this half"
    mism, checked = [], 0
    for r in cases:
        a = old(r["seq"], T[r["puzzle"]]["target"])
        b = patched(r["seq"], T[r["puzzle"]]["target"])
        checked += 1
        for k in VERDICT_KEYS:
            if a.get(k) != b.get(k):
                mism.append((r["puzzle"], k, a.get(k), b.get(k)))
        # every key the old version had must be present AND equal; the new keys must be exactly the two
        # declared ones, so a third slipping in is a failure rather than a curiosity
        for k in a:
            if a[k] != b.get(k):
                mism.append((r["puzzle"], f"shared:{k}", a[k], b.get(k)))
        extra = set(b) - set(a)
        if extra != EXPECTED_NEW_KEYS:
            mism.append((r["puzzle"], "keyset", sorted(EXPECTED_NEW_KEYS), sorted(extra)))
    print(f"   出厂判决四个键 + 全部共有键 + 键集合,比对 {checked} 条:不一致 {len(mism)}")
    print(f"   证书字段集合:【加法式】变更 —— 新增恰好 {sorted(EXPECTED_NEW_KEYS)},无键丢失、无共有键改值。")
    print(f"   注:certify.py 用 detail={{**v, 'knot': kr, ...}} 把 kr 整体嵌进证书,所以这两个键【会出现在证书里】。")
    print(f"   所以正确说法是『判决零变化 + 证书字段加二』,不是『零变化』。")
    if mism:
        for m in mism[:5]:
            print(f"     {m}")
        print("   FAIL: 旗标关闭时不恒等,不许提交")
        return 1
    b0 = patched(cases[0]["seq"], T[cases[0]["puzzle"]]["target"])
    print(f"   新增的两个【观测列】存在且不被任何判决读取:"
          f"confirmed_by_zero_crossing_guarded={b0['confirmed_by_zero_crossing_guarded']}, "
          f"zero_crossing_votes_withdrawn={b0['zero_crossing_votes_withdrawn']}")

    print("\n== 第二半:旗标【开】时到底动了多少(从已存列回放,不折叠) ==")
    print(f"   {'人群':16}{'confirmed_by 变':>14}{'撤票数':>8}{'判决 True→False':>16}{'被剥离那批的试管通过率':>16}")
    out = {}
    for nm, fn in POPS:
        d = json.load(open(os.path.join(HERE, fn)))["designs"]
        n = cb = vt = fl = 0
        wets = []
        for r in d:
            po = r.get("per_oracle") or {}
            if sorted(po) != sorted(ORACLES):
                continue
            n += 1
            o = [x for x in ORACLES if (po[x] or {}).get("recall_product") is not None
                 and po[x]["recall_product"] >= 1.0]
            g = [x for x in o if (po[x] or {}).get("n_pred_cross")]
            vt += len(o) - len(g)
            if len(o) != len(g):
                cb += 1
                if o and not g:
                    fl += 1
                    wets.append(r["wet"])
        pr = sum(1 for w in wets if w >= TUBE_PASS) / len(wets) if wets else None
        out[fn] = {"n": n, "confirmed_by_changed": cb, "frac_changed": round(cb / n, 4),
                   "votes_withdrawn": vt, "verdict_flipped": fl, "frac_flipped": round(fl / n, 4),
                   "stripped_tube_pass": None if pr is None else round(pr, 4)}
        print(f"   {nm:16}{cb:8d} ({cb/n*100:5.2f}%){vt:8d}{fl:8d} ({fl/n*100:5.2f}%)"
              f"{('%.3f' % pr) if pr is not None else ' n/a ':>16}")
    print("\n   => 这【不是】一次 0.27% 的小改。k1_audit.md 里『只动那 7 条、不需重新校准』的前提被推翻。")
    print("      出厂的 84.0% 灵敏度 / 0.4% 误收 与 分档灵敏度 0.432/0.636/0.785 都会失效,")
    print("      所以旗标默认关闭,开启前必须先重跑那次校准。")
    print("   => 方向是对的(被剥离那批试管通过率约 0.34~0.47,而 confirmed 全体约 0.80),但方向对不等于可以上线。")

    json.dump({"flag_off_verdict_identical": True, "certificate_field_set": "additive: +2 observable keys",
               "not_zero_change": "certify.py embeds kr whole via detail={**v, 'knot': kr}; the VERDICT is "
                                  "unchanged but the emitted field set grows by two",
               "keys_compared": list(VERDICT_KEYS) + ["all shared keys", "full key set"],
               "n_cases": checked, "mismatches": 0, "blast_radius": out,
               "default": "off", "env_flag": "CERTIFY_ZERO_CROSSING_GUARD=1",
               "may_not_ship_until": "the 84.0%/0.4% operating point and the per-grade sensitivities "
                                     "are re-measured on the guarded column",
               "premise_retracted": "k1_audit.md said this touches only 7 designs and needs no "
                                    "recalibration; measured it changes confirmed_by for ~12% and flips "
                                    "knot_confirmed for ~3.5%"},
              open(os.path.join(HERE, "zero_crossing_guard_invariance.json"), "w"),
              indent=1, ensure_ascii=False)
    print("\n   -> zero_crossing_guard_invariance.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
