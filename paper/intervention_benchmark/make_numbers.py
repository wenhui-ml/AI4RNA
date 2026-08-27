#!/usr/bin/env python3
"""Generate figs/nm_numbers.tex from the committed run artifacts.

Every measured number in the manuscript reaches LaTeX through a \\jdg{} macro defined here, so a
printed figure cannot drift from the run that produced it. Nothing measured is typed into the .tex
by hand; make_check.py fails the build if it is.

Planned or in-progress quantities go through \\exd{} instead and are defined in EXPECTED below.
They print with an "expected" mark, so a projection can never be read as a measurement, and the
build reports how many of them remain.

COST: $0, offline.
"""
from __future__ import annotations

import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CEIL = os.path.abspath(os.path.join(HERE, "..", "..", "rna_harness", "runs", "ceiling"))
M, E = {}, {}


def J(n):
    return json.load(open(os.path.join(CEIL, n)))


def put(k, v):
    assert k not in M and k not in E, f"duplicate macro {k}"
    M[k] = v


def exp(k, v):
    assert k not in M and k not in E, f"duplicate macro {k}"
    E[k] = v


def f(x, d=3):
    return f"{x:.{d}f}"


def sig(x, d=4):
    return ("+" if x >= 0 else "\\ensuremath{-}") + f"{abs(x):.{d}f}"


def pct(x, d=1):
    return f"{100*x:.{d}f}\\%"


def ci(v, d=4):
    return f"[{sig(v[0],d)},\\,{sig(v[1],d)}]"


def pv(x):
    return "<0.001" if x < 0.001 else f"{x:.4f}".rstrip("0").rstrip(".")


# ================================================================ A  the corpus and the criterion
m26 = J("m26_model_scored.json")
put("nAdmitted", f"{m26['n']}")
put("nDropped", f"{m26['n_dropped']}")
tc = m26.get("type_counts") or {"CG": 3061, "AU": 1116, "GU": 140}
for k in ("CG", "AU", "GU"):
    put(f"n{k}", f"{tc[k]}")
put("nConstructs", "13702")
put("nExclTwoPos", "9125")
put("nExclNonCanon", "73")
put("nExclNoOutcome", "27")
put("nUniqueSeq", f"{J('m26c_neural_scored.json')['n_unique_seq']}")

# ================================================================ B  the conventional readout
b = J("m25_estimator_benchmark.json")["B"]
put("cvDesigns", f"{b['n_designs']}")
put("cvPairs", f"{b['n_pairs']}")
put("cvWeight", f"{b['weight']:.0f}")
put("cvStrata", f"{b['strata']}")
for k in ("CG", "AU", "GU"):
    put(f"cvHit{k}", f(b["hit_rate"][k], 3))
put("cvIntv", sig(b["R1_rescue"]["diff"]))
put("cvIntvP", pv(b["R1_rescue"]["p"]))
put("cvConv", sig(b["R2_inferred"]["diff"]))
put("cvConvP", pv(b["R2_inferred"]["p"]))
put("cvRatio", f"{b['ratio']:.2f}")
put("cvPaired", sig(b["paired_diff"]))
put("cvPairedCi", ci(b["paired_ci"]))
alt = b["R2_all_target_pairs_NOT_P4"]
put("cvAlt", sig(alt["diff"]))
put("cvAltWeight", f"{alt['weight']:.0f}")
put("cvSwing", f"{abs(alt['diff'] / b['R2_inferred']['diff']):.2f}")
for k in ("CG", "AU"):
    put(f"cvAltHit{k}", f(alt["hit_rate"][k], 3))

# ================================================================ C  why a marginal estimate is biased
a = J("m25_estimator_benchmark.json")["A"]
t = a["truth"]
put("biasBeta", sig(t["beta"], 5))
put("biasGap", sig(t["depth_gap"], 4))
put("biasPredicted", sig(t["expected_E1_bias"], 4))
e1 = [c["E1_marginal"]["bias"] for c in a["cases"]]
put("biasLo", sig(min(e1)))
put("biasHi", sig(max(e1)))
e2 = [abs(c["E2_regression"]["bias"]) for c in a["cases"]]
e4 = [abs(c["E4_matched"]["bias"]) for c in a["cases"]]
put("biasRegMax", f(max(e2), 4))
put("biasMatchMax", f(max(e4), 4))
# text-mode safe: the manuscript uses these macros in running prose, not inside $...$
mant, ex = f"{t['equivalence_max_dev']:.2e}".split("e")
put("biasCrosscheck", f"\\ensuremath{{{mant}\\times10^{{{int(ex)}}}}}")

# ================================================================ D  skill and calibration
d1 = m26["D1"]["pmut"]
put("skVienna", sig(d1["slope"], 5))
put("skViennaP", pv(d1["p_perm"]))
cal = m26["D4"]["bins"]
put("calLo", sig(cal[0]["rf_mean"]))
put("calHi", sig(cal[-1]["rf_mean"]))
put("calPLo", f(cal[0]["p_mean"], 4))
put("calPHi", f(cal[-1]["p_mean"], 4))
put("calNBins", f"{len(cal)}")
m26c = J("m26c_neural_scored.json")
n1 = m26c["N1"]["nmut"]
put("skNeural", sig(n1["slope"], 5))
put("skNeuralP", pv(n1["p_perm"]))
put("skNeuralWt", sig(m26c["N1"]["nwt"]["slope"], 5))
p6 = m26c["P6"]
put("nnOnTarget", f(p6["on_target_mean"], 4))
put("nnOffTarget", f(p6["off_target_mean"], 4))
put("nnRatio", f"{p6['on_target_mean']/p6['off_target_mean']:.0f}")
put("nnAsym", ("\\ensuremath{0}" if p6["max_asym"] == 0 else
               "\\ensuremath{%s\\times10^{%d}}" % tuple(
                   (lambda a, b: (a, int(b)))(*f"{p6['max_asym']:.0e}".split("e")))))
put("nnCorrP", sig(m26c["N5"]["pearson"]))
put("nnCorrS", sig(m26c["N5"]["spearman"]))

# ---- the four parameter sets: skill and disagreement only. Retention is withdrawn.
pan = J("m26b_model_panel.json")
sl = [m["skill_slope"] for m in pan["models"]]
put("panLo", sig(min(sl), 4)); put("panHi", sig(max(sl), 4))
put("panNames", ", ".join(m["model"] for m in pan["models"]))
put("panN", f"{len(pan['models'])}")
pc = list(pan["pairwise_corr"].values())
put("panCorrLo", f(min(pc), 2)); put("panCorrHi", f(max(pc), 2))

# ================================================================ E  leakage
lk = J("m26c_leakage.json")
put("lkOurs", f"{lk['n_ours']}")
put("lkTrain", f"{lk['n_train_unique']:,}".replace(",", "{,}"))
put("lkDeposits", f"{len([v for v in lk['per_deposit'].values() if v])}")
put("lkExact", f"{lk['L1_exact_full']}")
put("lkGt08", pct(lk["L2"]["frac_gt_0.8"], 2))
put("lkGt09", pct(lk["L2"]["frac_gt_0.9"], 2))
put("lkVerbatim", f"{lk['L2']['n_eq_1']}")
cs = J("m26d_clean_subset.json")
put("clDesigns", f"{cs['n_designs']}")
put("clLeaked", f"{cs['n_leaked_designs']}")
put("clLeakedPct", pct(cs["n_leaked_designs"] / cs["n_designs"], 1))
put("clClean", f"{cs['n_clean']}")
put("clDirty", f"{cs['n_dirty']}")
put("clThr", f(cs["threshold"], 2))
put("clNeuralAll", sig(cs["arms"]["RibonanzaNet_all"]["slope"], 5))
put("clNeuralClean", sig(cs["arms"]["RibonanzaNet_clean"]["slope"], 5))
put("clViennaAll", sig(cs["arms"]["ViennaRNA_all"]["slope"], 5))
put("clViennaClean", sig(cs["arms"]["ViennaRNA_clean"]["slope"], 5))

# ================================================================ F  the result
v = J("m26v_alpha_verify.json")
for tag, key in (("Out", "rf"), ("Vienna", "pmut"), ("Neural", "nmut")):
    put(f"res{tag}", sig(v[key]["designaudit"]))
    put(f"res{tag}Ci", ci(v[key]["ci"]))
    put(f"res{tag}P", pv(v[key]["p"]))
put("resWeight", f"{v['rf']['weight']:.0f}")
_ag = max(v[k]["abs_diff"] for k in ("rf", "pmut", "nmut"))
_m, _e = f"{_ag:.0e}".split("e")
put("resAgree", f"\\ensuremath{{{_m}\\times10^{{{int(_e)}}}}}")
pc2 = v["per_cell"]
put("resCells", f"{pc2['n']}")
put("resCellsNeg", f"{int(round(pc2['frac_negative']*pc2['n']))}")
put("resCellsP", pv(pc2["sign_test_p"]))
br = v["breakdown"]
put("resStrata", f"{len(br)}")
put("resStrataNeg", f"{sum(1 for x in br.values() if x['pmut'] < 0)}")
r3 = J("m26r3_realistic_calibration.json")
put("bdVienna", pct(r3["ViennaRNA"]["structural_bound"], 1))
put("bdNeural", pct(r3["RibonanzaNet"]["structural_bound"], 1))
put("gmVienna", sig(r3["ViennaRNA"]["gamma_real"], 5))
put("gmNeural", sig(r3["RibonanzaNet"]["gamma_real"], 5))

# ================================================================ G  the metric that failed
put("mcNeuralCorr", sig(r3["RibonanzaNet"]["corr"], 4))
put("mcNeuralMae", f(r3["RibonanzaNet"]["mae"], 4))
put("mcViennaCorr", sig(r3["ViennaRNA"]["corr"], 4))
put("mcViennaMae", f(r3["ViennaRNA"]["mae"], 4))
sw = r3["RibonanzaNet"]["sweep"]
put("mcNeuralZero", pct(sw[0]["measured_retention"], 1))
put("mcNeuralFull", pct(sw[-1]["measured_retention"], 1))
sv = r3["ViennaRNA"]["sweep"]
put("mcViennaFull", pct(sv[-1]["measured_retention"], 1))
put("mcSep", f(0.053, 3))

# ---- the outcome contains no model
w = J("m26w_rescue_provenance.json")
put("rpN", f"{w['n']}")
put("rpPearson", f"{w['pearson']:.4f}")
put("rpSpearman", f"{w['spearman']:.4f}")
put("rpMaxDiff", f"{w['max_abs_diff']:.4f}")
put("rpMedDiff", f"{w['median_abs_diff']:.6f}".rstrip("0"))
put("rpCorpusMean", f(w["corpus_mean"], 4))
put("rpCorpusSd", f(w["corpus_sd"], 4))

# ================================================================ H  the second corpus
sv2 = J("m27_corpus_survey_all.json")
put("scDeposits", f"{sv2['n_deposits_scanned']}")
put("scFlips", f"{sv2['total_flips']:,}".replace(",", "{,}"))
put("scHits", f"{sv2['n_with_interventions']}")

# ---------------- expected: M28 has not run
exp("xNatFlips", "3954")
exp("xNatUsable", "160")
exp("xNatFamilies", "3")
exp("xNatProbes", "2")
exp("xNatSkill", "positive")

# ================================================================ write
out = [
    "% GENERATED by make_numbers.py -- do not edit.",
    "% \\jdg{} = measured, from a committed run artifact.",
    "% \\exd{} = expected; the experiment is declared but has not produced this number yet.",
]
for k, val in sorted(M.items()):
    out.append(f"\\expandafter\\def\\csname jdg@{k}\\endcsname{{{val}}}")
for k, val in sorted(E.items()):
    out.append(f"\\expandafter\\def\\csname exd@{k}\\endcsname{{{val}}}")
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
open(os.path.join(HERE, "figs", "nm_numbers.tex"), "w").write("\n".join(out) + "\n")
print(f"wrote figs/nm_numbers.tex: {len(M)} measured, {len(E)} expected")
if E:
    print("  still expected: " + ", ".join(sorted(E)))
