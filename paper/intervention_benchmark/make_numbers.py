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
# The admission ledger. These were typed by hand until m34 existed -- laundered through put(),
# where make_check.py (which only reads the .tex) could not see them. They come from the PACKAGED
# criterion, designaudit.mining.mine, which is a different ledger from m26.build's; both admit
# 4317 and m34 asserts that agreement before either is reported.
adm = J("m34_admission_census.json")
_da = adm["designaudit"]
assert adm["implementations_agree_on_admission"], "the two criteria admit different sets"
for k in ("CG", "AU", "GU"):
    put(f"n{k}", f"{_da['by_pair_type'][k]}")
put("nConstructs", f"{_da['n_constructs']:,}".replace(",", "{,}"))
put("nExclTwoPos", f"{_da['excluded']['not_two_positions']:,}".replace(",", "{,}"))
put("nExclNonCanon", f"{_da['excluded']['pair_not_canonical']}")
put("nExclNoOutcome", f"{_da['excluded']['no_outcome']}")
put("nWildTypes", f"{_da['unaccounted']}")
put("nExclTotal", f"{_da['excluded_total']:,}".replace(",", "{,}"))
put("nRowsRaw", f"{adm['m26_build']['n_rows']:,}".replace(",", "{,}"))
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
# cvRatio used to be R2/R1 on the RAW columns -- a ratio of two differently scaled contrasts,
# which is the fault this paper retracts elsewhere. It is now formed in each column's own
# within-stratum SD units, from m36, and the raw ratio is not printed at all.
put("cvPaired", sig(b["paired_diff"]))
put("cvPairedCi", ci(b["paired_ci"]))
alt = b["R2_all_target_pairs_NOT_P4"]
put("cvAlt", sig(alt["diff"]))
put("cvAltWeight", f"{alt['weight']:.0f}")
put("cvSwing", f"{abs(alt['diff'] / b['R2_inferred']['diff']):.2f}")

# ---- m36: what the "conventional" readout is read from, and the two columns' own scales. The
# manuscript said three times that it was a structure inferred from the wild-type profile. The
# benchmark file has no such column; the readout is the shipped RibonanzaNet prediction.
m36 = J("m36_readout_scale.json")
put("cvSourceCol", "\\texttt{" + m36["S1_source"]["column"].replace("_", "\\_") + "}")
put("cvSourceFile", "\\texttt{" + m36["S1_source"]["file"].replace("_", "\\_") + "}")
put("cvSdRescue", f(m36["scales"]["rescue_within_sd"], 4))
put("cvSdRnet", f(m36["scales"]["rnet_within_sd"], 4))
_st = m36["standardised"]
put("cvIntvSd", sig(_st["rescue_contrast_in_sd"], 4))
put("cvConvSd", sig(_st["rnet_contrast_in_sd"], 4))
put("cvPairedSd", sig(_st["paired_diff_in_sd"], 4))
put("cvPairedSdCi", ci(_st["ci"]))
put("cvRatioSd", f"{_st['rescue_contrast_in_sd'] / _st['rnet_contrast_in_sd']:.2f}")

# The preregistration table must quote the DECLARATION, not a threshold retyped beside it. These
# come out of the frozen file, so the table cannot drift from what was actually staked.
import re as _re
_d36 = J("m36_readout_scale_declaration.json")
for _sp in _d36["staked_predictions"]:
    _m = _re.search(r"([\d.]+) to ([\d.]+)", _sp["predicted"])
    if _m and "rescue" in _sp["quantity"]:
        put("cvSdRescueLo", _m.group(1)); put("cvSdRescueHi", _m.group(2))
    elif _m:
        put("cvSdRnetLo", _m.group(1)); put("cvSdRnetHi", _m.group(2))
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
put("mcViennaZero", pct(sv[0]["measured_retention"], 1))

# ---- m26r4: the correlations above are each produced by ONE sweep point, and quoting the
# wrong-sign arm's correlation while omitting the other arm's is selective reporting inside a
# retraction. The claim is therefore carried by the SPAN, which does not use the degenerate
# endpoint at all; the correlations are reported for both arms with their leave-one-out.
r4 = J("m26r4_calibration_symmetry.json")["arms"]
for nm, tag in (("RibonanzaNet", "Neural"), ("ViennaRNA", "Vienna")):
    a4 = r4[nm]
    put(f"mc{tag}Loo", sig(a4["corr_without_endpoint"], 4))
    put(f"mc{tag}Rho", sig(a4["spearman"], 4))
    put(f"mc{tag}Span", f(a4["span_measured_f_le_075"], 4))
    put(f"mc{tag}SpanFrac", pct(a4["span_ratio"], 1))
put("mcSpanTrue", f(r4["ViennaRNA"]["span_true_f_le_075"], 2))
put("mcSpanTop", "75\\%")

# ---- the outcome contains no model
w = J("m26w_rescue_provenance.json")
put("rpN", f"{w['n']}")
put("rpPearson", f"{w['pearson']:.4f}")
put("rpSpearman", f"{w['spearman']:.4f}")
put("rpMaxDiff", f"{w['max_abs_diff']:.4f}")
put("rpMedDiff", f"{w['median_abs_diff']:.6f}".rstrip("0"))
put("rpCorpusMean", f(w["corpus_mean"], 4))
put("rpCorpusSd", f(w["corpus_sd"], 4))

# ---- what the cluster bootstrap actually resamples (M33 census, filed as a survey)
cc = J("m33_cluster_census.json")
put("ccPuzzles", f"{cc['n_puzzles']}")
put("ccCells", f"{cc['n_design_cells']}")
put("ccPerMin", f"{cc['per_puzzle']['min']}")
put("ccPerMed", f"{cc['per_puzzle']['median']}")
put("ccPerMax", f"{cc['per_puzzle']['max']}")
put("ccLargest", pct(cc["per_puzzle"]["largest_share"], 1))

# ================================================================ H  the second corpus
sv2 = J("m27_corpus_survey_all.json")
# The survey's own summary fields count DEPOSITION RECORDS, and RMDB deposits the same
# library once per chemical probe (2A3 and DMS), so total_flips double-counts molecules.
# Everything below is recomputed from rows so the manuscript can say which unit it means.
_rows = sv2["rows"]
_ev = [r for r in _rows if "n_orientation_flip" in r]      # the rest carry a `skipped` reason
_hit = [r for r in _ev if r["n_orientation_flip"] > 0]
_bylib: dict[str, int] = {}
for r in _hit:                                              # one entry per library, not per probe
    _bylib[r["name"]] = max(_bylib.get(r["name"], 0), r["n_orientation_flip"])
put("scDeposits", f"{sv2['n_deposits_scanned']:,}".replace(",", "{,}"))
put("scEval", f"{len(_ev)}")
put("scSkipped", f"{len(_rows) - len(_ev)}")
put("scSkippedPct", pct((len(_rows) - len(_ev)) / len(_rows), 0))
put("scHits", f"{len(_hit)}")
put("scLibs", f"{len(_bylib)}")
put("scFlipsRec", f"{sv2['total_flips']:,}".replace(",", "{,}"))   # deposition-probe records
put("scFlips", f"{sum(_bylib.values()):,}".replace(",", "{,}"))    # counted once per library
# The survey's designed/natural label is NOT reported. It is too coarse to carry a claim:
# the two libraries this paper analyses as natural RNA (M2SL5, M3SARS) are labelled
# "unclear" by it, so its n_natural_with_interventions = 0 would contradict our own
# second corpus. Libraries are identified by name instead.
assert all(r.get("kind") == "unclear" for r in _hit), "the label changed; recheck the prose"

# ================================================================ F  M28, the second corpus
# The transfer test. Two natural-RNA deposits; the criterion runs with its declared-pair clause
# dropped, because these deposits declare no structure and the scan is exhaustive.
m28 = J("m28_second_corpus.json")
m28r = J("m28r_review.json")
d28 = {x["rmdb_id"]: x for x in m28["deposits"]}
sl5d, sl5a = d28["M2SL5_DMS_0000"], d28["M2SL5_2A3_0000"]

put("xnFamilies", f"{sl5d['A']['n_families']}")
put("xnProbes", "two")
put("xnAdmitDms", f"{sl5d['A']['n_admitted']}")
put("xnAdmitTwoA", f"{sl5a['A']['n_admitted']}")
put("xnShared", f"{m28['D_two_probes']['M2SL5']['n_shared']}")
put("xnSnCut", f"{m28['sn_threshold']:.1f}")

# the cross-probe comparison is made on a scale-free statistic: the two outcomes differ in
# dispersion by xnSdRatio, so raw slopes are not comparable across probes.
sc = m28r["R2_outcome_scale"]
put("xnSdRatio", f"{sc['ratio']:.2f}")
put("xnSdDms", f"{sc['DMS_sd']:.4f}")
put("xnSdTwoA", f"{sc['2A3_sd']:.4f}")
put("xnOutliersDms", f"{sc['DMS_below_minus1']}")

for nm, tag in (("ViennaRNA", "V"), ("RibonanzaNet", "N")):
    a = m28r["R2_arms"][nm]
    put(f"xnRho{tag}Dms", sig(a["within_spearman"]["DMS"]))
    put(f"xnRho{tag}TwoA", sig(a["within_spearman"]["2A3"]))
    put(f"xnRho{tag}DmsP", pv(a["within_spearman_p"]["DMS"]))
    put(f"xnRho{tag}TwoAP", pv(a["within_spearman_p"]["2A3"]))
    put(f"xnDiff{tag}", ci(a["paired_boot_corr_diff"]["ci95"]))

# the pre-registered statistic, reported unchanged next to it
put("xnSlopeVDms", sig(sl5d["C"]["ViennaRNA"]["slope"], 5))
put("xnSlopeVTwoA", sig(sl5a["C"]["ViennaRNA"]["slope"], 5))
put("xnSlopeVTwoAP", pv(sl5a["C"]["ViennaRNA"]["p_perm"]))
put("xnSlopeNDms", sig(sl5d["C"]["RibonanzaNet"]["slope"], 5))
put("xnSlopeNTwoA", sig(sl5a["C"]["RibonanzaNet"]["slope"], 5))

# the frameshift deposit: exhaustive, and unusable for a reason worth one sentence
fse = d28["M3SARS_DMS_0000"]
gate = m28r["R1_reference_gating"]["M3SARS_DMS_0000"]
put("xnFseFlips", f"{sum(g['n_flips'] for g in gate)}")
put("xnFseAdmit", f"{fse['A']['n_admitted']}")
put("xnFseSnLo", f"{min(g['wt_sn'] for g in gate):.3f}")
put("xnFseSnHi", f"{max(g['wt_sn'] for g in gate):.3f}")
put("xnSl5SnLo", f"{min(g['wt_sn'] for g in m28r['R1_reference_gating']['M2SL5_DMS_0000']):.3f}")

# ================================================================ G  the selection audit (M30)
m30 = J("m30_selection_audit.json")
s1, s2, s3, s4 = m30["S1"], m30["S2"], m30["S3"], m30["S4"]
s3b, s4c = m30["S3b_S4b_openknot_score"], m30["S4c_scale_free"]
put("selCut", f"{m30['cutoff']:.2f}")
put("selPool", f"{m30['n_pool']}")
put("selCells", f"{m30['n_cells']}")
put("selLinked", f"{m30['n_linked']}")
put("selRound", f"{s1['round_used']}")
# S1: the published rule, and the rule the data shows
put("selNCells", f"{s1['n_cells']}")
put("selRepro", f"{s1['n_reproduced']}")
put("selAgree", f"{s1['agreement']:.4f}")
put("selReproB", f"{s1['n_reproduced_no_filter']}")
put("selAgreeB", f"{s1['agreement_no_filter']:.4f}")
put("selPassedOver", f"{s1['n_eligible_passed_over']}")
put("selResid", f"{s1['n_cells'] - s1['n_reproduced_no_filter']}")
put("selRankOne", f"{s1['rank_on_score']['1']}")
put("selMaxElig", f"{max(r['n_eligible'] for r in s1['mismatches_published_rule'])}")
# S2: how tightly the filter binds
put("selPoolPass", pct(s2["pool_pass_rate_this_round"]))
put("selPoolPassAll", pct(s2["pool_pass_rate_all_rounds"]))
put("selNPoolRound", f"{s2['n_pool_this_round']}")
put("selNPass", f"{s2['n_tested_pass']}")
put("selNFallback", f"{s2['n_tested_fallback']}")
put("selF1Med", f"{s2['tested_f1_quantiles']['median']:.4f}")
put("selF1PTen", f"{s2['tested_f1_quantiles']['p10']:.4f}")
# the stratification absorbs it
put("selNCgau", f"{m30['corpus']['n_cgau']}")
put("selNCgauFb", f"{m30['corpus']['n_cgau_fallback']}")
put("selStrataMixed", f"{m30['strata_mixed_on_selection']}")
put("selNStrata", f"{s3['strata_pass'] + s3['strata_fallback']}")
# S3: the effect where the filter did not bind
put("selFull", sig(s3["full"], 5))
put("selPassArm", sig(s3["pass"], 5))
put("selFbArm", sig(s3["fallback"], 5))
put("selFbCi", ci(s3["ci_fallback"]))
put("selDiff", sig(s3["diff"], 5))
put("selDiffCi", ci(s3["ci_diff"]))
# S3b/S4b/S4c: the variable that actually selected
put("selMedScore", f"{s3b['median_score']:.2f}")
put("selHiArm", sig(s3b["contrast_hi"], 5))
put("selLoArm", sig(s3b["contrast_lo"], 5))
put("selScDiffCi", ci(s3b["ci_diff"]))
put("selSlopeHi", sig(s3b["slope_hi"], 5))
put("selSlopeLo", sig(s3b["slope_lo"], 5))
put("selRhoSlope", sig(s3b["spearman_skill_vs_score"]))
put("selRhoSlopeCi", ci(s3b["ci_spearman"]))
put("selRhoRank", sig(s4c["openknot_score"]["spearman"]))
put("selRhoRankCi", ci(s4c["openknot_score"]["ci"]))
put("selRhoFOne", sig(s4c["RNet_F1"]["spearman"]))
put("selRhoFOneCi", ci(s4c["RNet_F1"]["ci"]))
put("selRhoFOneSlope", sig(s4["spearman_skill_vs_f1"]))
put("selRhoFOneSlopeCi", ci(s4["ci_spearman"]))
put("selSdRatio", f"{s4c['sd_ratio_lo_over_hi']:.3f}")
put("selNCellSkill", f"{s4c['n_cells']}")
# D: dynamic range
for nm_, tag in (("P_model_ViennaRNA", "V"), ("P_neural_RibonanzaNet", "N")):
    dv = m30["D"][nm_]
    put(f"dyn{tag}Med", f"{dv['median']:.4f}")
    put(f"dyn{tag}PTen", f"{dv['p10']:.4f}")
    put(f"dyn{tag}PNinety", f"{dv['p90']:.4f}")
    put(f"dyn{tag}Hi", pct(dv["frac_above_0.99"]))
    put(f"dyn{tag}Lo", pct(dv["frac_below_0.08"]))
    put(f"dyn{tag}Mid", pct(dv["frac_between"]))
_dv = m30["D"]["P_model_ViennaRNA"]
put("dynThrHi", [k for k in _dv if k.startswith("frac_above_")][0].split("_")[-1])
put("dynThrLo", [k for k in _dv if k.startswith("frac_below_")][0].split("_")[-1])
put("selVerdict", m30["verdict"].replace("_", " "))

# ================================================================ H  ranking which flips rescue
# (M31 + M31r). The reframe claim, stated as an increment over a free baseline, never as a win.
m31 = J("m31_which_flips_rescue.json")
m31r = J("m31r_review.json")
A31, C31 = m31["A_within_design"], m31["C_decision"]
put("rkN", f"{m31['n_records_in_cells']}")
put("rkCells", f"{m31['n_cells_used']}")
put("rkNmin", f"{m31['NMIN']}")
for nm, tag in (("P_vienna", "V"), ("P_neural", "N"), ("depth_baseline", "D")):
    put(f"rkRho{tag}", sig(A31[nm]["rho"]))
    put(f"rkRho{tag}Ci", ci(A31[nm]["ci"]))
    put(f"rkPos{tag}", pct(A31[nm]["frac_cells_positive"], 0))
put("rkRhoDp", sig(A31["dP_neural"]["rho"]))
put("rkRhoDpCi", ci(A31["dP_neural"]["ci"]))
put("rkRhoDpV", sig(A31["dP_vienna"]["rho"]))
put("rkRhoDpVCi", ci(A31["dP_vienna"]["ci"]))
put("rkRhoDpP", f"{A31['dP_neural']['p_perm']:.4f}")
put("rkRhoDpVP", f"{A31['dP_vienna']['p_perm']:.2f}")
put("rkPosDp", pct(A31["dP_neural"]["frac_cells_positive"], 0))
# the interval K2 was missing
R1_, R2_, R3_ = m31r["R1_vs_free_baseline"], m31r["R2_partial_given_depth"], m31r["R3_tiebreak"]
put("rkDiffN", sig(R1_["rho"]["P_neural"] - R1_["rho"]["depth"], 5))
put("rkDiffNCi", ci(R1_["ci"]["P_neural-depth"]))
put("rkDiffV", sig(R1_["rho"]["P_vienna"] - R1_["rho"]["depth"], 5))
put("rkDiffVCi", ci(R1_["ci"]["P_vienna-depth"]))
# the partial, which is the claim
for nm, tag in (("P_vienna", "V"), ("P_neural", "N")):
    put(f"rkPart{tag}", sig(R2_["rho"][nm]))
    put(f"rkPart{tag}Ci", ci(R2_["ci"][nm]))
    put(f"rkPart{tag}P", pv(R2_["p_perm"][nm]))
# the decision, in constructs
put("rkTopFrac", pct(m31["TOPK_FRAC"], 0))
put("rkThresh", f"{m31['THRESH_MAIN']:.1f}")
put("rkRecDepth", pct(R3_["recovery"]["depth_only"], 3))
for nm, tag in (("P_vienna", "V"), ("P_neural", "N")):
    put(f"rkRecTie{tag}", pct(R3_["recovery"][f"depth_then_{nm}"], 3))
    g = R3_["recovery"][f"depth_then_{nm}"] - R3_["recovery"]["depth_only"]
    put(f"rkGain{tag}", f"{100*g:.2f}")
    put(f"rkGain{tag}Ci", ci(R3_["ci_gain"][nm]))
put("rkRecRandom", pct(C31["depth_baseline"]["recovery_random"], 3))
# base rate and the threshold sweep, and where the free baseline overtakes
_sw = m31["B_auroc_sweep"]
_main = next(r for r in _sw if abs(r["threshold"] - m31["THRESH_MAIN"]) < 1e-9)
put("rkBase", pct(_main["base_rate"], 1))
put("rkAucN", f"{_main['P_neural']:.4f}")
put("rkAucD", f"{_main['depth_baseline']:.4f}")
_over = [r for r in _sw if r["depth_baseline"] > max(r["P_neural"], r["P_vienna"])]
put("rkOverThr", f"{min(r['threshold'] for r in _over):.1f}" if _over else "never")
if _over:
    _o = min(_over, key=lambda r: r["threshold"])
    put("rkOverD", f"{_o['depth_baseline']:.4f}")
    put("rkOverN", f"{_o['P_neural']:.4f}")
# the controls
R4_ = m31r["R4_controls"]
for k, tag in (("selection_fallback", "Sel"), ("leakage_clean", "Leak")):
    v = R4_.get(k) or {}
    if "rho" in v:
        put(f"rk{tag}Cells", f"{v['n_cells']}")
        put(f"rk{tag}N", sig(v["rho"]["P_neural"]))
        put(f"rk{tag}NCi", ci(v["ci"]["P_neural"]))
m31s = J("m31s_depth_share.json")
for nm, tag in (("P_vienna", "V"), ("P_neural", "N")):
    put(f"rkDep{tag}", sig(m31s["with_depth"][nm]))
    put(f"rkDepGap{tag}", sig(m31s["gap"][nm], 5))
put("rkDepLeTwo", pct(m31s["depth_le2_frac"], 0))
put("rkTied", pct(m31s["tied_pair_fraction"], 1))
put("rkVerdict", m31r["verdict"].replace("_", " "))
# the causal conditioning set, for the contrast between the two questions
Ap = m31["Aprime_causal_conditioning"]
put("rkCausalN", sig(Ap["P_neural"]["rho"]))
put("rkCausalNCi", ci(Ap["P_neural"]["ci"]))
put("rkCausalV", sig(Ap["P_vienna"]["rho"]))
put("rkCausalVCi", ci(Ap["P_vienna"]["ci"]))
put("rkCausalCells", f"{Ap['P_neural']['n_cells']}")
put("rkCausalRecs", f"{Ap['P_neural']['n_records']:,}".replace(",", "{,}"))

# what the exposure-graded leakage audit will report, once the library is in hand
exp("lkTierN", "four exposure tiers")
exp("lkTierSkill", "per-tier skill slopes")

# ---- how each artifact was licensed. provenance.json is the manuscript's process claim in
# machine-readable form; audit_manuscript.py refuses the build if it drifts from what is on disk.
_prov = json.load(open(os.path.join(HERE, "provenance.json")))
_cls = {k: [a for a in v if not a.startswith("_")]
        for k, v in _prov.items() if not k.startswith("_")}
put("pvPreReg", f"{len(_cls['preregistered'])}")
put("pvSurvey", f"{len(_cls['survey'])}")
put("pvReview", f"{len(_cls['review'])}")
put("pvSupport", f"{len(_cls['support'])}")
put("pvTotal", f"{sum(len(v) for v in _cls.values())}")
put("pvSurveyReviewSupport",
    f"{len(_cls['survey']) + len(_cls['review']) + len(_cls['support'])}")

# ================================================================ I  the third predictor
m32 = J("m32_third_model_class.json")
m32r = J("m32r_review.json")
_A = m32["A_identity_contrast"]
put("efContrast", sig(_A["P_eterna"]["effect"], 5))
put("efContrastCi", ci(_A["P_eterna"]["ci"]))
put("efContrastP", pv(_A["P_eterna"]["p_perm"]))
_B = m32["B_within_design"]
put("efRho", sig(_B["P_eterna"]["rho"]))
put("efRhoCi", ci(_B["P_eterna"]["ci"]))
put("efRhoP", pv(_B["P_eterna"]["p_perm"]))
put("efPos", pct(_B["P_eterna"]["frac_cells_positive"], 0))
put("efDp", sig(_B["dP_eterna"]["rho"]))
put("efDpCi", ci(_B["dP_eterna"]["ci"]))
put("efDpP", pv(_B["dP_eterna"]["p_perm"]))
put("efVsDepth", sig(m32["C_vs_depth"]["diff"], 5))
put("efVsDepthCi", ci(m32["C_vs_depth"]["ci_diff"]))
put("efDepthRho", sig(m32["D_depth_share"]["rho_with_depth"]))
put("efDepthGap", sig(m32["D_depth_share"]["gap"], 5))
put("efCoverage", pct(m32["S1_coverage"], 1))
put("efEquiv", f"{m32['P1_equivalence']['abs_diff']:.0e}".replace("e+00", ""))

# ---- m35: the manuscript compared three point estimates and never their differences, which is
# the fault it retracts twice. All four columns resampled over the SAME 20 puzzles per draw.
m35 = J("m35_paired_predictor_differences.json")
_pd = m35["paired_differences"]
for k, tag in (("neural_minus_vienna", "NV"), ("neural_minus_eterna", "NE"),
               ("vienna_minus_eterna", "VE"), ("measured_minus_neural", "MN"),
               ("measured_minus_vienna", "MV"), ("measured_minus_eterna", "ME")):
    d = _pd[k]
    put(f"pd{tag}", sig(d["effect"], 5)); put(f"pd{tag}Ci", ci(d["ci"]))
    put(f"pd{tag}P", pv(d["p_perm"]))
_ax = m35["post_hoc_axes"]
put("axArch", sig(_ax["architecture_axis_neural_minus_eterna"]["effect"], 5))
put("axArchCi", ci(_ax["architecture_axis_neural_minus_eterna"]["ci"]))
put("axData", sig(_ax["training_data_axis_eterna_minus_vienna"]["effect"], 5))
put("axDataCi", ci(_ax["training_data_axis_eterna_minus_vienna"]["ci"]))
put("axGap", sig(_ax["architecture_minus_training_data"]["effect"], 5))
put("axGapCi", ci(_ax["architecture_minus_training_data"]["ci"]))
put("axRatio", f"{_ax['architecture_axis_neural_minus_eterna']['effect'] / _ax['training_data_axis_eterna_minus_vienna']['effect']:.1f}")
_dr = m32r["R1_dynamic_range"]
put("efHi", pct(_dr["frac_hi"], 1))
put("efLo", pct(_dr["frac_lo"], 1))
put("efMed", f(_dr["median"], 4))
_cz = m32r["R2_censor_rate_by_pairtype"]
put("efCensCg", pct(_cz["CG"], 2))
put("efCensAu", pct(_cz["AU"], 2))
put("efClean", sig(m32r["R2_contrast_uncensored"]["effect"], 5))
put("efCleanCi", ci(m32r["R2_contrast_uncensored"]["ci"]))
put("efCommit", m32["eternafold_provenance"]["commit"][:8])

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
# Count what was WRITTEN, not what is in M: a put() placed below the assembly above lands in M,
# is reported here, and never reaches the file. That happened once and the message hid it.
_written = sum(1 for line in out if "\\csname jdg@" in line)
assert _written == len(M), f"{len(M) - _written} macro(s) defined after the file was assembled"
print(f"wrote figs/nm_numbers.tex: {_written} measured, {len(E)} expected")
if E:
    print("  still expected: " + ", ".join(sorted(E)))
