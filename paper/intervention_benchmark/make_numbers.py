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

import collections, math, json, os, sys

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
    # \allowbreak after the thin space: an interval is otherwise one unbreakable atom, and a
    # line holding two of them overran the text block by 15pt with nowhere to break.
    return f"[{sig(v[0],d)},\\,\\allowbreak {sig(v[1],d)}]"


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
put("nWildTypes", f"{_da['n_wild_type']}")
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
# a retention of minus eight hundred percent does not have a tenth-of-a-percent; the sweep
# endpoint is printed at the precision the number can carry.
put("mcViennaFull", pct(sv[-1]["measured_retention"], 0))
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
# The library count and the flip count come from M41's lineage arm, NOT from the survey's own
# deposit `name` field. Keying on that field counted one 15,000-construct design set three times,
# because RMDB carries it under three deposit names -- and the third of those is four replicates
# of it, so its 136 flips are 34 x 4. M41 establishes the design set by construct-name identity
# (lineage Jaccard 1.0000 within, 0.0000 between) and counts flips on the deduplicated set.
m41 = J("m41_domain.json")
_L = m41["L"]
assert _L["n_libraries_manuscript_prints"] == 6 and _L["flips_manuscript_prints"] == 4340, \
    "M41 no longer records the values this correction replaces; the erratum text would be stale"
assert m41["V"]["V1_pass"] and m41["V"]["V1b_pass"], \
    "M41's lineage verification did not pass; the corrected census may not be printed"
put("scDeposits", f"{sv2['n_deposits_scanned']:,}".replace(",", "{,}"))
put("scEval", f"{len(_ev)}")
put("scSkipped", f"{len(_rows) - len(_ev)}")
put("scSkippedPct", pct((len(_rows) - len(_ev)) / len(_rows), 0))
put("scHits", f"{len(_hit)}")
put("scLibs", f"{_L['n_libraries_lineage_correct']}")
put("scFlipsRec", f"{sv2['total_flips']:,}".replace(",", "{,}"))   # deposition-probe records
put("scFlips", f"{_L['n_distinct_flips_total']:,}".replace(",", "{,}"))   # once per DESIGN SET
put("scLibsDeposit", f"{len(set(r['name'] for r in _hit))}")       # what the deposit names say
put("scBiggest", f"{max(v['n_distinct_flips'] for v in _L['libraries'].values()):,}".replace(",", "{,}"))
put("scBiggestPct", pct(max(v["n_distinct_flips"] for v in _L["libraries"].values())
                        / _L["n_distinct_flips_total"], 1))

# arm A: what the whole remaining public record yields once the criterion runs, not a pattern count
_A = m41["A_new_admitted"]
put("scNewLibs", f"{len(_A)}")

# arm Q: the admission gate, calibrated, and the measurement ceiling it implies
_Q = m41["Q"]
put("qcAsserted", f"{_Q['asserted_gate']:.1f}")
# The ceiling depends on how the reactivity columns are split, and the first version of this
# reported one rule as if it were the answer. Odd/even puts each column beside its most similar
# neighbour in the other half and inflates the agreement; contiguous confounds the halves with
# position. All three are printed and every claim rests on the conservative end.
_cr = _Q["ceiling_range"]
put("qcCeilLo", f"{min(_cr):.4f}")
put("qcCeilHi", f"{max(_cr):.4f}")
put("qcCeiling", f"{_Q['ceiling_conservative']:.4f}")
for _r, _lab in (("odd_even", "Oe"), ("contiguous", "Cg"), ("random", "Rd")):
    put(f"qcR{_lab}", f"{_Q['by_split_rule'][_r]['split_half_r']:+.4f}")
    put(f"qcCeil{_lab}", f"{_Q['by_split_rule'][_r]['ceiling']:.4f}")
put("qcQn", f"{_Q['n_usable']}")
put("qcCross07", f"{_Q['reliability_crossings']['0.7']:.2f}")
put("qcLoBin", f"{min(b['split_half_r'] for b in _Q['bins'] if not b['underpowered']):.4f}")
put("qcHiBin", f"{max(b['split_half_r'] for b in _Q['bins'] if not b['underpowered']):.4f}")
put("qcNbins", f"{_Q['nbins']}")
# the lineage agreements the census correction rests on, and the two reliability thresholds the
# crossing table is read at -- both fixed in the declaration, neither chosen after seeing a curve
_jin = sorted(set(m41["V"]["V1_lineage_jaccard"].values()))
_jout = sorted(set(m41["V"]["V1b_reverse_control"].values()))
assert _jin == [1.0] and _jout == [0.0], f"lineage agreements changed: {_jin} / {_jout}"
put("scLinIn", f"{_jin[0]:.4f}")
put("scLinOut", f"{_jout[0]:.4f}")
_thr = sorted(float(k) for k in _Q["reliability_crossings"])
put("qcThrA", f"{_thr[1]:.1f}")
put("qcThrB", f"{_thr[2]:.1f}")
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
    # C8's far end, and the statistic it has to be read against. The interval permits a probe
    # effect as large as the whole rank agreement being measured, so it bounds nothing.
    put(f"xnBd{tag}", f"{max(abs(x) for x in a['paired_boot_corr_diff']['ci95']):+.4f}")
    put(f"xnRhoMax{tag}", f"{max(a['within_spearman'].values()):+.4f}")

# the pre-registered statistic, reported unchanged next to it
put("xnSlopeVDms", sig(sl5d["C"]["ViennaRNA"]["slope"], 5))
put("xnSlopeVTwoA", sig(sl5a["C"]["ViennaRNA"]["slope"], 5))
put("xnSlopeVTwoAP", pv(sl5a["C"]["ViennaRNA"]["p_perm"]))
put("xnSlopeNDms", sig(sl5d["C"]["RibonanzaNet"]["slope"], 5))
put("xnSlopeNTwoA", sig(sl5a["C"]["RibonanzaNet"]["slope"], 5))

# the frameshift deposit: exhaustive, and unusable for a reason worth one sentence
fse = d28["M3SARS_DMS_0000"]
fse2 = d28["M3SARS_2A3_0000"]
gate = m28r["R1_reference_gating"]["M3SARS_DMS_0000"]
put("xnFseFlips", f"{sum(g['n_flips'] for g in gate)}")
put("xnFseAdmit", f"{fse['A']['n_admitted']}")
# the same LIBRARY under the other probe, at the same cutoff. Printing the DMS zero alone said
# that a poorly measured corpus yields nothing; the library's other deposit yields this.
put("xnFseAdmitTwoA", f"{fse2['A']['n_admitted']}")
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
# disattenuated against M41's measurement ceiling: an observed rank correlation divided by the
# largest one this outcome permits. Both series are attenuated by the same factor, so the ratio
# between them is unchanged and the correction cannot manufacture a separation.
_ceil = m41["Q"]["ceiling_conservative"]        # the conservative end, never the permissive one
put("rkRhoDAtt", sig(A31["depth_baseline"]["rho"] / _ceil))
put("rkRhoNAtt", sig(A31["P_neural"]["rho"] / _ceil))
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
# lkTierN / lkTierSkill were PROJECTIONS of this audit. M39 delivered it, so the projections are
# gone rather than kept beside the measurement they were standing in for.

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
# Only the predictor-against-predictor differences are emitted with an interval and a p-value:
# those three are inside one unit system. The three measurement-against-predictor differences are
# raw subtractions across two scales, which section 5.4 now withdraws as licensing quantities; MN
# survives as a bare point estimate because the manuscript prints it as the thing being corrected,
# and MV and ME are not emitted at all rather than left computed and unshown.
for k, tag in (("neural_minus_vienna", "NV"), ("neural_minus_eterna", "NE"),
               ("vienna_minus_eterna", "VE")):
    d = _pd[k]
    put(f"pd{tag}", sig(d["effect"], 5)); put(f"pd{tag}Ci", ci(d["ci"]))
    put(f"pd{tag}P", pv(d["p_perm"]))
put("pdMN", sig(_pd["measured_minus_neural"]["effect"], 5))
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

# ================================================================ the channel ledger (M37)
# The organising table. Every price in it is published elsewhere in this manuscript; what the
# table adds is the axis they all lie on. Generated, so a row cannot drift from its run.
# C6 is priced in THIS paper's units: printing the companion's mediated fraction here would be
# printing exactly the class of quantity this paper retracts.
# C2's price is measured in the companion study, which owns that channel in full. NM prints the
# two contrasts themselves -- not a mediated fraction, which is the class of quantity this paper
# retracts -- so the row is readable here without importing the estimator that failed.
_m20 = J("m20_designer_confounding.json")["A2"]
put("medMarginal", sig(_m20["marginal"], 4))
put("medMatched", sig(_m20["matched"], 4))

# ---------------------------------------------------------------- M38: the purpose-built comparator
# riboSNitch-style mutation-effect scores, computed from the same folding model the paper already
# scores, on the same interventions in the same cells by the same statistic. Declaration frozen
# before the code. The direction was declared NEGATIVE before any number existed, so a positive
# rho here means the declaration was right.
_m38 = J("m38_ribosnitch_baseline.json")
_a38, _b38 = _m38["A_within_design"], _m38["B_vs_free_depth"]
for _k, _tag in (("d_bppm", "rsBppm"), ("d_mfe", "rsMfe"), ("d_ens", "rsEns"),
                 ("d_bppm_local", "rsLocal")):
    put(_tag, sig(_a38[_k]["rho"], 4))
    put(_tag + "Ci", ci(_a38[_k]["ci"], 4))
    put(_tag + "Pos", f"{_a38[_k]['frac_cells_positive']:.2f}")
    put(_tag + "Vs", sig(_b38[_k]["diff"], 4))
    put(_tag + "VsCi", ci(_b38[_k]["ci"], 4))
put("rsN", f"{_m38['n_records_in_cells']}")
put("rsCells", f"{_m38['n_cells']}")
put("rsNbeat", f"{sum(v['beats_depth'] for v in _b38.values())}")
put("rsNarms", f"{len(_b38)}")
put("rsIdent", f"{_m38['D_verification']['identity_max']:.0f}")
_ob = _m38["D_verification"]["obweight_max"]
put("rsObw", "$" + ("%.0e" % _ob).replace("e-0", "e-").replace("e-", "\\times 10^{-") + "}$")

# ---------------------------------------------------------------- M39: C7 against the real corpus
# The leakage audit re-run against the competition's own training file instead of RMDB deposits.
# The file is a SUBSET, so every count is a LOWER BOUND -- declaration K3 makes that a wording
# rule, and no sentence here may say a design is absent from the training set.
_m39 = J("m39_exposure_tiers.json")
put("lkTrainReal", f"{_m39['n_train_unique']:,}".replace(",", ","))
put("lkLibReal", f"{_m39['n_train_libraries']}")
put("lkExpDesigns", f"{_m39['E1']['n_exposed_designs']}")
put("lkExpFrac", f"{100*_m39['E1']['frac']:.1f}\\%")
put("lkProxyDesigns", f"{_m39['E1']['m26d_proxy_count']}")
put("lkProxyFrac", f"{100*_m39['E1']['m26d_proxy_frac']:.1f}\\%")
put("lkVerbatimT3", f"{_m39['E2']['tiers']['T3 >=0.999 逐字']}")
put("lkCleanN", f"{_m39['n_clean']}")
put("lkExpN", f"{_m39['n_dirty']}")
_a39 = _m39["E3_arms"]
put("lkNeuralClean", sig(_a39["RibonanzaNet_clean"]["slope"], 4))
put("lkNeuralExp", sig(_a39["RibonanzaNet_exposed"]["slope"], 4))
put("lkViennaClean", sig(_a39["ViennaRNA_clean"]["slope"], 4))
put("lkViennaExp", sig(_a39["ViennaRNA_exposed"]["slope"], 4))
put("lkNeuralDiffCi", ci(_m39["E3_difference_ci"]["RibonanzaNet"]["clean_minus_exposed_ci"], 4))
put("lkViennaDiffCi", ci(_m39["E3_difference_ci"]["ViennaRNA"]["clean_minus_exposed_ci"], 4))

# the staked bracket is read out of the frozen declaration, never retyped: make_check.py refuses
# a hand-typed decimal in the body and it is right to, because a retyped stake can drift from the
# declaration it claims to quote.
import re as _re
_d39 = json.load(open(os.path.join(CEIL, "m39_exposure_tiers_declaration.json")))
_lo, _hi = _re.search(r"\[([\d.]+),\s*([\d.]+)\]", _d39["staked"]["4"]).groups()
put("lkStakeLo", _lo); put("lkStakeHi", _hi)

# ---- M49: the criterion that moved C8 off "empty", applied to C7, and an interval on the
# asymmetry C7's reading rests on. Two of its four stakes failed and the failures are the result:
# the difference in drops does not resolve, so neither the memorisation reading nor the
# difficulty reading is excluded, which is what forces C7 to unresolved alongside the reach ratio.
_m49 = J("m49_c7_did.json")
_d49 = _m49["A3_did"]
put("didPoint", sig(_d49["point"], 4))
put("didCi", ci(_d49["ci95"], 4))
put("didVienna", sig(_d49["vienna_drop"], 4))
put("didNeural", sig(_d49["neural_drop"], 4))
put("didPermP", f"{_m49['A4_permutation']['p_one_sided']:.3f}")
put("didPermCentre", sig(_m49["A4_permutation"]["null_median"], 4))
put("didPermN", f"{_m49['A4_permutation']['n_perm']}")
put("lkNeuralExpCi", ci(_m49["A5_exposed_arm"]["RibonanzaNet"]["ci95"], 4))
put("lkViennaExpCi", ci(_m49["A5_exposed_arm"]["ViennaRNA"]["ci95"], 4))
put("chReachSeven", f"{_m49['A6_reach']['C7']['ratio']:.2f}")
# ---- the permutation floor. A permutation p cannot fall below 1/(1+B), and B is a property of
# the code that produced the p-values, not of the manuscript, so it is READ OUT of that code
# rather than typed. The paragraph in Methods claimed a Bonferroni correction at thirty tests was
# satisfiable by the values reported; against this paper's own gate of 0.005 it is not, because
# the floor sits above the corrected threshold. That claim is withdrawn there and these are the
# three numbers the withdrawal rests on.
_src26 = open(os.path.join(CEIL, "m26_model_scored.py")).read()
_B = int(_re.search(r"NBINS,\s*NPERM,\s*NPLACEBO\s*=\s*\d+,\s*(\d+)", _src26).group(1))
_gate, _ntests = 0.005, 30
# W10 asked for B and the seed in the Methods body rather than only inside the code. B is not one
# number: the permutation count and the bootstrap count differ, and the review runs raised the
# bootstrap count. All three are read out of source for the same reason the floor is.
_Bboot26 = int(_re.search(r"def cluster_ci\([^)]*nb=(\d+)", _src26).group(1))
_src51 = open(os.path.join(CEIL, "m51_level_vs_change.py")).read()
_Bboot51 = int(_re.search(r"NMIN,\s*NBOOT,\s*SEED\s*=\s*\d+,\s*(\d+)", _src51).group(1))
_seed26 = int(_re.search(r'add_argument\("--seed",\s*type=int,\s*default=(\d+)\)', _src26).group(1))
put("permB", f"{_B:,}".replace(",", "{,}"))
put("bootBprimary", f"{_Bboot26}")
put("bootBreview", f"{_Bboot51:,}".replace(",", "{,}"))
put("seedPrimary", f"{_seed26}")
put("permFloor", f"{1/(1+_B):.5f}")
put("permBonf", f"{_gate/_ntests:.6f}")
put("permBneed", f"{math.ceil(_ntests/_gate)-1:,}".replace(",", ","))
assert 1/(1+_B) > _gate/_ntests, (
    "the permutation floor now clears the corrected threshold; the Methods paragraph saying a "
    "corrected p-value is unattainable here is stale")

put("chReachEight", f"{_m49['A6_reach']['C8']['ratio']:.2f}")

# ---- M52: the Discussion said the honest bound on Assumption 1 was "the sensitivity analysis
# already reported". There was none: Section 5.2 reports a controlled-truth benchmark and says in
# the same subsection that a benchmark built from a confounder we chose is not one. This is the
# bound. It is closed-form because eq:est is linear in the outcome, so an omitted U shifts the
# estimate by exactly beta_u * g_u, a product invariant to rescaling U.
_m52 = J("m52_sensitivity.json")
assert all(_m52["V"].values()), "M52 verification did not pass"
_s1, _s3, _s4 = _m52["A1_multiplier"], _m52["A3_observed_covariates"], _m52["A4"]
put("senK", f"{_s1['K1']:.2f}")
put("senKCi", ci(_s1["ci95"], 2))
put("senBench", sig(abs(_s1["benchmark_bias"]), 4))
put("senDiag", sig(_m52["A2_diagonal"]["required_beta_eq_g"], 3))
put("senDiagSd", f"{_m52['A2_diagonal']['in_outcome_within_sd']:.2f}")
put("senNcov", f"{len(_m52['covariates'])}")
put("senMaxOne", sig(_s3["max_abs_bias"], 4))
put("senArgOne", _s3["argmax"])
put("senMarginCi", ci(_s3["paired_margin"]["ci95"], 4))
put("senMaxAll", sig(_s4["max_abs_bias"], 4))
put("senMaxAllPct", pct(_s4["max_abs_bias"] / _s3["alpha"], 0))
put("senNdeg", f"{len(_s4['dropped_no_within_variance'])}")
_srcM52 = open(os.path.join(CEIL, "m52_sensitivity.py")).read()
_nrand = int(_re.search(r"SEED, NBOOT, NRAND = \d+, \d+, (\d+)", _srcM52).group(1))
put("senNrand", f"{_nrand:,}".replace(",", "{,}"))
# S2 failed and the manuscript has to say so: the multiplier's interval reaches below one, because
# the lower end of the contrast's OWN interval is about the size of the bias placement would have
# caused. That sentence is only true while this stays true.
assert not _s1["ci_excludes_one"], (
    "M52's multiplier interval now excludes 1; the sentence saying a confounder no stronger than "
    "placement cannot be ruled out at the lower end is stale")
assert _s3["max_abs_bias"] < _s3["alpha"] and _s4["max_abs_bias"] < _s3["alpha"], (
    "an observed covariate now reaches the contrast; W3 of the M52 declaration binds")



# ---- M50: section 5.4 differenced four columns that are not on one scale. Two scales exist and
# they answer different questions; the bridge between them is g = rho * SD_R/SD_p, so the slope
# route is the SD route attenuated by rho. Both are reported, because the run found that the
# comparison RESOLVES on one and does not on the other, and saying only one would be choosing.
_m50 = J("m50_units.json")
_sd, _a2, _a3, _a4 = (_m50["A1_within_sd"], _m50["A2_standardised"],
                      _m50["A3_paired_standardised_difference"], _m50["A4_slope_route"])
for _k, _tag in (("rf", "Meas"), ("nmut", "Neural"), ("emut", "Etf"), ("pmut", "Vienna")):
    put("sd" + _tag, f(_sd[_k], 4))
    put("std" + _tag, sig(_a2[_k]["standardised_contrast"], 4))
for _k, _tag in (("nmut", "MN"), ("emut", "ME"), ("pmut", "MV")):
    put("std" + _tag, sig(_a3[_k]["point"], 4)); put("std" + _tag + "Ci", ci(_a3[_k]["ci95"], 4))
    put("sl" + _tag, sig(_a4[_k]["point"], 4)); put("sl" + _tag + "Ci", ci(_a4[_k]["ci95"], 4))
put("convNeural", sig(_a4["nmut"]["converted_contrast"], 4))
put("rhoNeural", sig(_m50["A5_bridge"]["nmut"]["rho_within"], 4))
put("rhoVienna", sig(_m50["A5_bridge"]["pmut"]["rho_within"], 4))

# ---- M51: section 5.6 chose the level arm over the change arm because it was larger and said in
# print that it put no interval on the difference. Here is the interval, and the comparison the
# section is actually about -- both arms against the free integer -- on both arms rather than one.
_m51 = J("m51_level_vs_change.json")
_lv, _vd = _m51["A2_level_minus_change"], _m51["A3_vs_depth"]
for _k, _tag in (("RibonanzaNet", "Neural"), ("ViennaRNA", "Vienna"), ("EternaFold", "Etf")):
    put("lvc" + _tag, sig(_lv[_k]["difference"], 4))
    put("lvc" + _tag + "Ci", ci(_lv[_k]["ci95"], 4))
    put("chgDep" + _tag + "Ci", ci(_vd[_k + "_change"]["ci95"], 4))
    put("lvlDep" + _tag + "Ci", ci(_vd[_k + "_level"]["ci95"], 4))
put("lvcNres", f"{_m51['n_resolving']}")
put("lvcNarms", f"{len(_lv)}")
# The section's claim is that neither arm beats a free integer. That is only sayable if every
# level arm covers zero against depth AND every change arm resolves on the losing side; if either
# half stopped holding, the sentence stating it would be a different sentence.
assert all(_vd[k + "_level"]["ci_contains_zero"] for k in _lv), \
    "a level arm now separates from the free depth baseline; the sentence saying none does is stale"
assert all((not _vd[k + "_change"]["ci_contains_zero"]) and _vd[k + "_change"]["difference"] < 0
           for k in _lv), \
    "a change arm no longer resolves on the losing side against depth; the sentence is stale"
# The whole point of the section is that these two routes give different answers on the same
# comparison. If that ever stopped being true, the paragraph explaining why both are reported
# would be describing something that is not there.
assert _a3["nmut"]["ci_contains_zero"] and not _a4["nmut"]["ci_contains_zero"], (
    "the two unit routes no longer disagree about whether measurement and transformer resolve; "
    "the paragraph in section 5.4 that reports both because they disagree is stale")
assert not _a3["emut"]["ci_contains_zero"] and not _a3["pmut"]["ci_contains_zero"], (
    "a nearest-neighbour predictor is no longer separated from the measurement on the SD scale")
# The reclassification is not a matter of taste: C7's interval reaches FURTHER, relative to the
# quantity it is about, than the interval that moved C8. If that ever stopped holding, the row
# below would be asserting something the run does not support.
assert _m49["A6_reach"]["C7"]["ratio"] >= _m49["A6_reach"]["C8"]["ratio"], (
    "C7 no longer reaches as far as C8 relative to its own quantity; the reclassification that "
    "the ledger row states is no longer forced by the criterion it cites")
assert _d49["ci_contains_zero"], (
    "the difference-in-differences now resolves; the manuscript sentence saying the asymmetry is "
    "unresolved is stale and the memorisation reading can no longer be left open")
assert _m49["A5_exposed_arm"]["ViennaRNA"]["ci_contains_zero"], (
    "the thermodynamic control now resolves from zero on the exposed subset; the sentence saying "
    "it does not is stale")

# ------------------------------------------------- M42: the same audit on the FULL released corpus
# M39 searched the competition's QUICK_START slice and the manuscript called every count from it a
# lower bound, PREDICTING that the full file would raise exposure. It does not: the exposed design
# count, the exposed intervention count and both skill slopes are unchanged to every digit, and one
# wild type moves from near-verbatim into verbatim. The prediction is recorded as failed here rather
# than reworded, and M39's frozen declaration is not edited.
_m42 = J("m42_full_corpus.json")
put("lkFullTrain", f"{_m42['n_train_unique_full']:,}".replace(",", "{,}"))
put("lkFullRows", f"{_m42['n_train_rows_full']:,}".replace(",", "{,}"))
put("lkFullLib", f"{_m42['n_train_libraries_full']}")
put("lkFullGrow", f"{_m42['n_train_unique_full']/_m39['n_train_unique']:.1f}$\\times$")
put("lkFullExp", f"{_m42['E1']['n_exposed_designs']}")
put("lkFullFrac", f"{100*_m42['E1']['frac']:.1f}\\%")
_T3KEY = [k for k in _m42["E2"]["tiers"] if k.startswith("T3 ")][0]
put("lkFullVerbatim", f"{_m42['E2']['tiers'][_T3KEY]}")
put("lkFullMoved", f"{_m42['E2']['n_designs_moved_up']}")
put("lkFullShuf", sig(_m42["V"]["shuffle_max"], 4))
_a42 = _m42["E3_arms"]
put("lkFullNeuralClean", sig(_a42["RibonanzaNet_clean"]["slope"], 4))
put("lkFullViennaClean", sig(_a42["ViennaRNA_clean"]["slope"], 4))
put("lkFullCleanN", f"{_m42['n_clean']}")
put("lkFullExpN", f"{_m42['n_dirty']}")

# The containment check is what licenses calling this the SAME search enlarged, rather than a second
# unrelated one: every QUICK_START sequence is inside the full file, and restricting the full file
# back to QUICK_START returns M39's published count.
assert _m42["containment_ok"], "QUICK_START is not nested inside the full corpus"
assert _m42["V"]["quick_restricted_exposed"] == _m39["E1"]["n_exposed_designs"], "V3 kill check"
assert _m42["n_clean"] == _m39["n_clean"] and _m42["n_dirty"] == _m39["n_dirty"]
for _k in ("RibonanzaNet_clean", "ViennaRNA_clean"):
    assert abs(_a42[_k]["slope"] - _a39[_k]["slope"]) < 1e-9, _k

# M40v: the two verification steps (equivalent form, reverse control) M38 and M39 had not done.
# Read here so the manuscript's process sentence covers it and the audit's converse check can see it.
_m40 = J("m40v_verify_m38_m39.json")
put("vKendallD", sig(_m40["M38_B_equivalent_form"]["rows"]["d_bppm"]["kendall_tau_b"], 4))
put("vKendallDep", sig(_m40["M38_B_equivalent_form"]["rows"]["depth_baseline"]["kendall_tau_b"], 4))
put("vBackMax", f"{max(abs(v['delta']) for v in _m40['M38_A_reverse_control']['drift'].values()):.0e}")
put("vIdxGap", sig(_m40["M39_A_reverse_control"]["max_identity_abs_diff"], 3))

_led = J("m37_channel_ledger.json")

put("chUnpriced", f"{_led['check3']['n_unpriced_named']}")
put("chEps", sig(_led["check2_identity"]["residual_epsilon"], 4))
put("chDrift", f"{len(_led['check1_macro_vs_artifact_drift'])}")
# check 4: what the field's standard remedy -- a fairer split -- would and would not close.
# The counts come from the ledger artifact's own classification, never from prose.
put("chSplit", f"{_led['check4_split']['n_split_closable']}")
put("chNoSplit", f"{_led['check4_split']['n_not_split_closable']}")
put("chSplitCarry", f"{len(_led['check4_split']['split_closable_and_carrying'])}")
put("chSplitId", ", ".join(_led["check4_split"]["split_closable"]) or "none")

# The C6 bound used to print |alpha*gamma/total| -- a ratio of a mediated effect to the total
# contrast, which IS the mediated fraction this paper withdraws, printed inside the very table
# whose caption says the column cannot be summed. It is reported in CONTRAST UNITS now.
put("bdViennaAbs", sig(abs(r3["ViennaRNA"]["alpha"] * r3["ViennaRNA"]["gamma_real"]), 4))

_ROWS = [
    ("C1", "the model being scored", "its own predicted structure, used as the reference",
     r"\jdg{cvIntvSd} against \jdg{cvConvSd}\,SD", "carries"),
    ("C2", "the design programs", "where in a helix they put the pair",
     r"\jdg{medMarginal} $\to$ \jdg{medMatched}", "carries"),
    ("C3", "whoever wrote the target", "helix depth, read off the declared structure",
     r"\jdg{rkRhoD} against \jdg{rkRhoN}", "carries"),
    ("C4", "the corpus builder", "the rule that chose which designs were made",
     r"\jdg{selRhoRank} \jdg{selRhoRankCi}", "against"),
    ("C5", "the model's authors", "the decomposition, and separately the training data",
     r"\jdg{axArch} \jdg{axArchCi}\newline \jdg{axData} \jdg{axDataCi}", "carries"),
    ("C6", "the design objective", "the folding model's own energy, which every program optimised",
     r"\jdg{resVienna} \jdg{resViennaCi}; at most \jdg{bdViennaAbs} of the contrast can run "
     r"through it", "empty"),
    ("C7", "the training set", "window overlap with the model's own training file",
     r"\jdg{lkNeuralClean} unexposed against \jdg{lkNeuralExp} exposed \jdg{lkNeuralDiffCi}; "
     r"the interval reaches \jdg{chReachSeven} of the quantity, so it excludes nothing", "unresolved"),
    ("C8", "the depositors", "which probe chemistry the outcome was read under",
     r"\jdg{xnDiffV}; the interval reaches \jdg{xnBdV} against a rank agreement of "
     r"\jdg{xnRhoMaxV}, so it excludes nothing", "unresolved"),
]
# One list, checked against the other. m37 owns which macros price each channel. The ledger also
# verifies macros the table does not print (an interval's midpoint, a swing quoted in prose), so
# containment is the wrong relation -- what must never happen is a row pricing a channel with
# macros the ledger does not stand behind at all, which is what a retired artifact looks like.
_ledrows = {c["id"]: set(c["price_macros"]) for c in _led["channels"]}
_base = lambda s: s[:-2] if s.endswith("Ci") else s
for _cid, _who, _cov, _price, _v in _ROWS:
    _used = {_base(m) for m in _re.findall(r"\\jdg\{(\w+)\}", _price)}
    _want = {_base(m) for m in _ledrows.get(_cid, set())}
    if _want and not (_used & _want):
        raise SystemExit(f"ledger drift: table row {_cid} prices with {sorted(_used)}, none of "
                         f"which the ledger verifies ({sorted(_want)}). That is what a retired "
                         f"artifact looks like; fix the row or the ledger, not this check.")
_verd = collections.Counter(_v for *_x, _v in _ROWS)
put("chPriced", f"{len(_ROWS)}")
put("chCarry", f"{_verd['carries'] + _verd['against']}")
put("chEmpty", f"{_verd['empty']}")
put("chUnres", f"{_verd['unresolved']}")
assert _verd["carries"] + _verd["against"] == _led["check3"]["n_carrying"], (
    "the table and the ledger disagree about how many channels carry the number")
assert len(_ROWS) == _led["check3"]["n_priced"], "the table prices a different number of channels"

_MARK = {"carries": r"\textbf{carries}", "against": r"\textbf{carries}, against the reported effect",
         "empty": r"\emph{empty}", "unresolved": r"\emph{unresolved} --- not empty"}
_t = [r"% GENERATED by make_numbers.py -- do not edit.",
      # the last column held intervals and ran 91pt past the text block as an "l"; it is a
      # paragraph column now, and column 3 gives back the width it needs.
      r"\begin{tabular}{@{}llp{0.26\textwidth}p{0.32\textwidth}@{}}", r"\toprule",
      r"& who chose the covariate & what it is & what it is worth \\", r"\midrule"]
for _cid, _who, _cov, _price, _v in _ROWS:
    _t.append(_cid + " & " + _who + " & " + _cov + " & " + _price + r" \\")
    _t.append(r"& \multicolumn{3}{@{}l@{}}{\footnotesize " + _MARK[_v] + r"} \\[3pt]")
_t += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(HERE, "figs", "tab_channels.tex"), "w").write("\n".join(_t) + "\n")

# ================================================================ H  appendix tables
# The appendix held a preregistration record and seven lines about the command line, and
# everything a reader might want to check for themselves -- the two admission ledgers, what the
# bootstrap resamples, the comparator's five arms, the exposure tiers -- was compressed into a
# sentence in the main text or into a figure. These are that material at full width. They are
# GENERATED, like tab_channels: no number below is typed.

def _tab(path, colspec, head, rows, note=None):
    t = [r"% GENERATED by make_numbers.py -- do not edit.",
         r"\begin{tabular}{" + colspec + "}", r"\toprule"]
    t += [h + r" \\" for h in head]
    t.append(r"\midrule")
    for r in rows:
        t.append(r"\midrule" if r == "-" else r + r" \\")
    t += [r"\bottomrule", r"\end{tabular}"]
    if note:
        t.append(r"\\[3pt] {\footnotesize " + note + "}")
    open(os.path.join(HERE, "figs", path), "w").write("\n".join(t) + "\n")


_n = lambda v: f"{v:,}".replace(",", "{,}")
_mb = adm["m26_build"]
assert _da["ledger_closes"] and _da["unaccounted_are_wild_types"], \
    "the admission ledger no longer closes; the appendix table says in print that it does"
# The two ledgers get their own rows rather than sharing one. They count different objects --
# constructs against released rows -- and a shared row would assert an equivalence that the
# whole point of printing both is to deny.
_tab("tab_admission.tex", r"@{}lrr@{}",
     [r"& \texttt{designaudit.mine} & \texttt{m26.build}",
      r"& the packaged criterion & the paper's script",
      r"& unit: construct & unit: released row"],
     [f"read & {_n(_da['n_constructs'])} & {_n(_mb['n_rows'])}",
      f"wild types, never candidates & {_da['n_wild_type']} & ---",
      "-",
      f"refused, other than two positions changed & {_n(_da['excluded']['not_two_positions'])} & ---",
      f"refused, wild-type pair not canonical & {_da['excluded']['pair_not_canonical']} & ---",
      f"refused, no measured outcome & {_da['excluded']['no_outcome']} & ---",
      f"refused, no outcome or signal-to-noise & --- & {_n(_mb['excluded']['no_outcome_or_sn'])}",
      f"refused, non-canonical pair & --- & {_mb['excluded']['non_canonical_pair']}",
      "-",
      f"\\textbf{{admitted}} & \\textbf{{{_da['n_admitted']}}} & \\textbf{{{_mb['n_admitted']}}}",
      f"\\quad on C--G pairs & {_da['by_pair_type']['CG']} & ---",
      f"\\quad on A--U pairs & {_da['by_pair_type']['AU']} & ---",
      f"\\quad on G--U pairs & {_da['by_pair_type']['GU']} & ---"],
     note=r"First column: $" + f"{_da['n_admitted']}+{_da['excluded_total']}+{_da['n_wild_type']}"
          f"={_da['n_constructs']}$" + r". The identity is asserted in \texttt{m34} before this "
          r"table is written, so a ledger that stops closing fails the build.")

_cc = J("m33_cluster_census.json")
_pp, _pc = _cc["per_puzzle"], _cc["per_design_cell"]
put("stPuzMin", f"{_pp['min']}"); put("stPuzMed", f"{_pp['median']}"); put("stPuzMax", f"{_pp['max']}")
put("stShare", pct(_pp["largest_share"]))
put("stCellMin", f"{_pc['min']}"); put("stCellMed", f"{_pc['median']}"); put("stCellMax", f"{_pc['max']}")
_tab("tab_strata.tex", r"@{}lr@{}", [r"what the intervals resample & count"],
     [f"interventions & {_cc['n_interventions']}",
      f"targets, the unit the bootstrap resamples & {_cc['n_puzzles']}",
      f"design cells, the stratum the estimator matches within & {_cc['n_design_cells']}",
      "-",
      f"interventions per target, min / median / max & {_pp['min']} / {_pp['median']} / {_pp['max']}",
      f"largest target's share of the corpus & {pct(_pp['largest_share'])}",
      f"interventions per design cell, min / median / max & {_pc['min']} / {_pc['median']} / {_pc['max']}"])

# ---- W9: one table saying what every number is computed on. The counts differ across analyses
# for stated reasons, and a reader who cannot track them cannot check anything. Every cell here
# comes from the artifact of the run that used it, so a restriction that changes fails the build
# rather than going unnoticed.
_m52n = J("m52_sensitivity.json")
_m51n = J("m51_level_vs_change.json")
_m44n = J("m44_decision.json")
_m31n = J("m31_which_flips_rescue.json")
_m31rn = J("m31r_review.json")
_m30n = J("m30_selection_audit.json")
_m25B = J("m25_estimator_benchmark.json")["B"]
_m39n = J("m39_exposure_tiers.json")
_vv = J("m26v_alpha_verify.json")
_ap31 = _m31n["Aprime_causal_conditioning"]["P_neural"]
_lk31 = _m31rn["R4_controls"]["leakage_clean"]
assert _m39n["n_clean"] + _m39n["n_dirty"] == _cc["n_interventions"], "the leakage split is not exhaustive"
assert _da["by_pair_type"]["CG"] + _da["by_pair_type"]["AU"] == _m30n["corpus"]["n_cgau"], (
    "the C-G plus A-U count no longer equals the two-arm corpus")
put("naStrataBoth", f"{_m52n['n_strata_both_arms']}")
_NMIN51 = int(_re.search(r"NMIN,\s*NBOOT,\s*SEED\s*=\s*(\d+)",
                        open(os.path.join(CEIL, "m51_level_vs_change.py")).read()).group(1))
_tab("tab_naccount.tex",
     r"@{}p{0.30\textwidth}lr p{0.36\textwidth}@{}",
     [r"& & count & why it is not \jdg{nAdmitted}"],
     [f"constructs in the released file & \\S\\ref{{sec:corpus}} & {_da['n_constructs']:,} & ---".replace(",", "{,}"),
      f"\\textbf{{admitted interventions}} & \\S\\ref{{sec:mining}} & \\textbf{{{_da['n_admitted']}}} "
      f"& the criterion",
      f"\\quad design cells & & {_cc['n_design_cells']} & ---",
      f"\\quad targets, the bootstrap unit & & {_cc['n_puzzles']} & ---",
      "-",
      f"entering the C--G vs A--U contrast & \\S\\ref{{sec:estimation}} & {_m30n['corpus']['n_cgau']} "
      f"& G--U ({_da['by_pair_type']['GU']}) is a third arm, not compared",
      f"strata supplying both arms & \\S\\ref{{sec:estimation}} & {_m52n['n_strata_both_arms']} "
      f"& a stratum with one arm contributes nothing",
      f"matched weight $\\sum_s \\min(n_A,n_B)$ & \\S\\ref{{sec:estimation}} & {_vv['rf']['weight']:.0f} "
      f"& the effective size of the contrast",
      f"design cells in the per-cell sign test & \\S\\ref{{sec:result}} & {_vv['per_cell']['n']} "
      f"& cells holding both arms",
      "-",
      f"both readouts defined & \\S\\ref{{sec:conventional}} & {_m25B['n_pairs']} "
      f"& designs with no called structure",
      f"\\quad its strata & & {_m25B['strata']} & ---",
      f"selection audit & \\S\\ref{{sec:selection}} & {_m30n['corpus']['n_cgau']} & the same two arms",
      f"leakage split, unexposed & \\S\\ref{{sec:leakage}} & {_m39n['n_clean']} & exhaustive with the next row",
      f"leakage split, exposed & \\S\\ref{{sec:leakage}} & {_m39n['n_dirty']} & ---",
      "-",
      f"design cells with $\\ge$\\,{_NMIN51} flips & \\S\\ref{{sec:ranking}} & {_m51n['n_cells']} "
      f"& covering all {_m51n['n_covered']} interventions",
      f"\\quad surviving the leakage cut & & {_lk31['n_cells']} & {_lk31['n_leaked']} cells matched training structures",
      f"\\quad under the causal conditioning set & & {_ap31['n_cells']} "
      f"& a finer partition, so fewer reach the flip floor; {_ap31['n_records']} interventions",
      f"cells with both outcomes present & \\S\\ref{{sec:reorder}} & {_m44n['n_cells_scored']} "
      f"& a cell all-rescue or all-not cannot be scored"],
     note=r"Every count is read from the artifact of the run that used it. The three "
          r"restrictions that are not arithmetic --- the two-arm corpus, the flip floor and the "
          r"leakage cut --- are stated where they are applied.")

_rs = J("m38_ribosnitch_baseline.json")
_ra, _rb, _rv = _rs["A_within_design"], _rs["B_vs_free_depth"], _rs["D_verification"]
put("rsIdentMax", f"{_rv['identity_max']:.0f}")
put("rsNIdent", f"{_rv['n_identity']}")
put("rsCellsPosB", pct(_ra["d_bppm"]["frac_cells_positive"]))
put("rsCellsPosD", pct(_ra["depth_baseline"]["frac_cells_positive"]))
put("rsWindow", f"{_rs['constants']['WINDOW']}")
_RSN = [("d_bppm", "pairing-matrix distance"),
        ("d_bppm_local", r"\quad inside a $\pm\jdg{rsWindow}$\,nt window"),
        ("d_mfe", "MFE base-pair distance"),
        ("d_ens", r"$|\Delta G_{\mathrm{ens}}|$ change"),
        ("depth_baseline", r"\textbf{helix depth}, free")]
# p_perm is at the permutation floor for all five arms, so it is a sentence in the note rather
# than a column of identical values.
_tab("tab_ribosnitch.tex", r"@{}lccccc@{}",
     [r"score & $\rho$ & 95\% CI & median & cells & paired difference",
      r"& & & cell $\rho$ & $\rho>0$ & against depth, 95\% CI"],
     [f"{lab} & {sig(_ra[k]['rho'])} & {ci(_ra[k]['ci'])} & {sig(_ra[k]['median_cell_rho'])} & "
      f"{pct(_ra[k]['frac_cells_positive'])} & "
      + (f"{sig(_rb[k]['diff'])} {ci(_rb[k]['ci'])}" if k in _rb else "---")
      for k, lab in _RSN],
     note=r"$p_{\mathrm{perm}}<0.001$, the floor of "
          + f"{_rs['constants']['NPERM']}" + r" permutations, on all five arms: every score reads "
          r"the intervention. The last column is what the section is about, and none of it "
          r"crosses zero from below. Every arm shares the corpus, the strata and the seed "
          r"with the ranking table of \S\ref{sec:ranking}.")

_ex = J("m39_exposure_tiers.json")
_e2, _ar = _ex["E2"], _ex["E3_arms"]
put("exTrainN", _n(_ex["n_train_unique"]))
put("exLibs", f"{_ex['n_train_libraries']}")
put("exShufMax", f"{_ex['V_verification']['shuffle_max']:.4f}")
put("exShufMed", f"{_ex['V_verification']['shuffle_median']:.4f}")
# the declared window length, read out of the erratum the artifact records rather than
# out of the declaration, so the manuscript quotes what the run says it got wrong
put("exDeclLen", _re.search(r"assumed a (\d+) nt", _ex["declaration_erratum"]).group(1))
# one macro, not two: a macro whose value is exactly 100 makes every '100\\%' in the
# manuscript look like a hand-typed measurement to audit_manuscript.py
put("exLenRange", f"{min(_ex['query_lengths'])}--{max(_ex['query_lengths'])}")
_tr = {k.split()[0]: v for k, v in _e2["tiers"].items()}
_TIER = [("T0", r"below the threshold, unexposed"), ("T1", r"$[0.8,\,0.9)$"),
         ("T2", r"$[0.9,\,0.999)$"), ("T3", r"$\geq 0.999$, verbatim")]
_tab("tab_exposure.tex", r"@{}lrr@{}",
     [r"identity of a design's window to its nearest training sequence & designs & share"],
     [f"{t}, {d} & {_tr[t]} & {pct(_tr[t]/_ex['n_designs'], 1)}" for t, d in _TIER]
     + ["-"]
     + [f"\\quad exposed designs traced to \\texttt{{{k.replace('_', chr(92) + '_')}}} & {v} & "
        for k, v in sorted(_e2["libraries"].items(), key=lambda kv: -kv[1])],
     note=f"Of {_ex['n_designs']} design cells, against {_n(_ex['n_train_unique'])} unique "
          f"sequences in {_ex['n_train_libraries']} libraries of the model's own training file. "
          r"Every count is a LOWER bound: a design absent from this file may still be present in "
          r"the part of the corpus the release does not contain.")

_ARM = [("all", "whole corpus"), ("clean", "unexposed"), ("exposed", "exposed")]
_tab("tab_exposure_arms.tex", r"@{}llrrr@{}",
     [r"predictor & arm & interventions & skill slope & $p$"],
     [f"{m if a == 'all' else ''} & {lab} & {_ar[m + '_' + a]['n']} & "
      f"{sig(_ar[m + '_' + a]['slope'])} & {pv(_ar[m + '_' + a]['slope_p'])}"
      for m in ("RibonanzaNet", "ViennaRNA") for a, lab in _ARM],
     note=r"The thermodynamic rows are the control that makes the split readable: that model was "
          r"trained on nothing, and it falls on the exposed arm the same way.")

_vf = J("m40v_verify_m38_m39.json")
put("vrDriftMax", f"{max(abs(v['delta']) for v in _vf['M38_A_reverse_control']['drift'].values()):.0f}")
put("vrEquivMax", f"{_vf['M39_B_equivalent_form']['max_abs_diff']:.0f}")
put("vrKmerN", _n(_vf["M39_A_reverse_control"]["n_subset"]))
put("vrKmerExp", f"{_vf['M39_A_reverse_control']['exhaustive_exposed']}")
put("vrSpearMax", f"{max(abs(r['spearman_dot_recomputed'] - r['m38_committed']) for r in _vf['M38_B_equivalent_form']['rows'].values()):.0f}")


# ================================================ M43: a fourth implementation, and the reorder test
# The peer question was whether the intervention readout REORDERS the predictors relative to the
# structure benchmark. It does not, and the honest report of that is the tau plus the one swap's
# interval -- not a claim that the two agree, and not a claim that they disagree.
#
# The distinction the arm turns on, and that the declaration got wrong about EternaFold (recorded
# there as an erratum, not edited away): an IMPLEMENTATION is an independent codebase; a PARAMETER
# VARIANT is the same engine reading a different file. Conflating them would let a paper claim six
# predictors while scoring one program six times.
_m43 = J("m43_reorder.json")
_ag, _rs = _m43["R_agree"], _m43["R_resp"]
_IMPL4 = ("ViennaRNA", "EternaFold", "RNAstructure", "RibonanzaNet")
for _n, _k in zip(_IMPL4, ("Vie", "Etf", "Rst", "Rnet")):
    put(f"reAgree{_k}", f"{_ag[_n]:.4f}")
    put(f"reResp{_k}", sig(_rs[_n]["rho"], 4))
    put(f"reRespCi{_k}", ci(_rs[_n]["ci"], 4))
put("reNimpl", f"{len(_IMPL4)}")
put("reTau", sig(_m43["E2"]["kendall_tau"], 4))
put("reNswap", f"{len(_m43['E2']['swaps'])}")
put("reNresolved", f"{_m43['E2']['n_resolved_swaps']}")
_sw = _m43["E2"]["swaps"][0]
put("reSwapA", _sw["pair"][0]); put("reSwapB", _sw["pair"][1])
put("reSwapAgree", sig(_sw["agree_diff"], 4)); put("reSwapResp", sig(_sw["resp_diff"], 4))
put("reSwapCi", ci(_sw["dd_ci"], 4))
put("reSpanImpl", sig(_m43["span"]["implementations"], 4))
put("reSpanParam", sig(_m43["span"]["parameter_variants"], 4))
put("reSpanRatio", f"{_m43['span']['implementations']/_m43['span']['parameter_variants']:.1f}$\\times$")
put("reNvar", f"{len(_m43['E3_parameter_variants'])}")
put("reVarNames", ", ".join(sorted(_m43["E3_parameter_variants"])))
put("reMfeAgree", f"{_m43['V']['V2_mfe_agreement_vienna_vs_rnastructure']:.4f}")
put("reNcells", f"{_m43['n_cells']}")
# the incumbent margin this paper keeps quoting, computed once here so no two places can disagree
put("reIncMargin", sig(_ag["RibonanzaNet"] - _ag["ViennaRNA"], 3))
put("reIncMarginWorst", sig(_ag["RibonanzaNet"] - _ag["RNAstructure"], 5))
assert all(_m43["coverage"][c]["missing"] == 0 for c in _m43["coverage"]), "M43 coverage gap"

# ------------- M43s: the scorecard M43 did not write, and the pairwise matrix stake 1 needed
# M43 froze four staked predictions and its artifact carries staked:null. Three grade from the
# published run; the fourth needs the implementation-by-implementation agreement matrix, of which
# M43 computed exactly one cell.
_m43s = J("m43s_score.json")
assert _m43s["checks"]["pass"], "M43s checks did not pass; its scorecard may not be printed"
_MAT = _m43s["pairwise_median_f1"]
_ab = {"ViennaRNA": "Vie", "EternaFold": "Etf", "RNAstructure": "Rst", "RibonanzaNet": "Rnet"}
for _a in _ab:
    for _b in _ab:
        if _a != _b:
            put(f"pw{_ab[_a]}{_ab[_b]}", f"{_MAT[f'{_a}|{_b}']:.4f}")
put("pwNN", f"{_MAT['ViennaRNA|RNAstructure']:.4f}")
put("pwNNvsNeuralLo", f"{min(_MAT['ViennaRNA|RibonanzaNet'], _MAT['RNAstructure|RibonanzaNet']):.4f}")
put("pwNNvsNeuralHi", f"{max(_MAT['ViennaRNA|RibonanzaNet'], _MAT['RNAstructure|RibonanzaNet']):.4f}")
put("pwNfail", f"{_m43s['n_failed']}")
put("pwNstake", f"{len(_m43s['staked'])}")
# EternaFold is the separator: same decomposition as the thermodynamic pair, same data as the
# neural one, and it lands between them in structure space rather than beside either.
assert _MAT["ViennaRNA|RNAstructure"] > _MAT["EternaFold|ViennaRNA"] > _MAT["EternaFold|RibonanzaNet"], \
    "EternaFold no longer sits between the two families; the sentence in sec:reorder is stale"
put("pwEtfNN", f"{min(_MAT['EternaFold|ViennaRNA'], _MAT['EternaFold|RNAstructure']):.4f}")
put("pwEtfNeural", f"{_MAT['EternaFold|RibonanzaNet']:.4f}")
# the near-miss that made the reverse control worth having: the published full-length V2 against the
# window-restricted cell of the same pair. Read out, never retyped, because a retyped near-miss is
# exactly the kind of number that drifts.
_rv = _m43s["checks"]["reverse_v2_full_length"]
put("pwV2Full", f"{_rv['ours']:.4f}")
put("pwV2Win", f"{_m43s['checks']['window_restricted_same_pair']:.4f}")
put("pwV2Gap", f"{abs(_m43s['checks']['window_restricted_same_pair'] - _rv['ours']):.4f}")

# ============================================ M44: what that margin buys at a fixed tube budget
# The translation the instrument exists to make. Recovery at a fixed budget, every difference from
# the FREE depth baseline carrying an interval, and the incumbent margin converted into constructs.
_m44 = J("m44_decision.json")
_E1 = _m44["E1"]
put("dcNcells", f"{_m44['n_cells_scored']}")
put("dcNrescue", f"{_m44['E2']['n_rescuing_interventions_total']}")
for _f, _k in (("0.1", "Ten"), ("0.25", "Qtr"), ("0.5", "Half")):
    put(f"dcRand{_k}", f"{_E1[_f]['random']['recovery']:.4f}")
    put(f"dcDepth{_k}", f"{_E1[_f]['depth']['recovery']:.4f}")
    put(f"dcOracle{_k}", f"{_E1[_f]['oracle']['recovery']:.4f}")
    for _n, _kk in zip(_IMPL4, ("Vie", "Etf", "Rst", "Rnet")):
        put(f"dc{_kk}{_k}", f"{_E1[_f][_n]['recovery']:.4f}")
        put(f"dc{_kk}Vs{_k}", sig(_E1[_f][_n]["minus_depth"], 4))
        put(f"dc{_kk}VsCi{_k}", ci(_E1[_f][_n]["minus_depth_ci"], 4))
    put(f"dcStack{_k}", f"{_E1[_f]['depth_then_RibonanzaNet']['recovery']:.4f}")
    put(f"dcStackVs{_k}", sig(_E1[_f]["depth_then_RibonanzaNet"]["minus_depth"], 4))
    put(f"dcStackVsCi{_k}", ci(_E1[_f]["depth_then_RibonanzaNet"]["minus_depth_ci"], 4))
# the headline translation: the incumbent's biggest margin, in tubes
_pair = [r for r in _m44["E2"]["pairs"]
         if r["leader"] == "RibonanzaNet" and r["trailing"] == "ViennaRNA" and r["budget"] == 0.25][0]
# M43 and M44 arrive at this margin from different artifacts; they must agree, and printing it
# to five places advertised a precision neither carries.
assert abs(_pair["incumbent_margin_f1"] - (_ag["RibonanzaNet"] - _ag["ViennaRNA"])) < 5e-6, (
    "M43 and M44 disagree about the incumbent margin")
put("dcTransF1", sig(_pair["incumbent_margin_f1"], 3))
put("dcTransRec", sig(_pair["decision_margin_recovery"], 5))
put("dcTransN", f"{abs(_pair['extra_rescuing_interventions']):.0f}")
put("dcTransPct", f"{100*_pair['extra_rescuing_interventions']/_m44['E2']['n_rescuing_interventions_total']:.1f}\\%")
assert abs(_pair["incumbent_margin_f1"] - (_ag["RibonanzaNet"] - _ag["ViennaRNA"])) < 1e-9
# how many of the six ordered pairs change sign between the two scales
_flip = [r for r in _m44["E2"]["pairs"] if r["budget"] == 0.25 and r["decision_margin_recovery"] < 0]
put("dcNflip", f"{len(_flip)}")
put("dcNpair", f"{len([r for r in _m44['E2']['pairs'] if r['budget'] == 0.25])}")
_h = _m44["E3_held_out"]
put("dcHoldIn", f"{_h['in_sample_mean']:.4f}"); put("dcHoldOut", f"{_h['held_out_mean']:.4f}")
put("dcHoldShrink", sig(_h["shrinkage"], 4)); put("dcHoldN", f"{_h['n_splits']}")
put("dcHoldTop", max(_h["rules_picked"], key=_h["rules_picked"].get).replace("_", r"\_"))
put("dcHoldTopN", f"{max(_h['rules_picked'].values())}")
put("dcHoldNrule", f"{len(_h['rules_picked'])}")
# M44's staked criterion 2 quotes two thresholds. They are read out of the frozen declaration rather
# than retyped into the body, the same rule lkStakeLo/Hi follow.
_d44 = json.load(open(os.path.join(CEIL, "m44_decision_declaration.json")))
_m = _re.search(r"more than ([\d.]+) in mean structure-agreement F1 and by less than ([\d.]+)",
                _d44["staked"]["2"])
put("dcStakeAgree", _m.group(1)); put("dcStakeRec", _m.group(2))

# ============================ M44v: the review that had to exist before any of the above is printed
# depth is an INTEGER, so it ties, and a stable sort breaks ties on the CSV's row order. Everything
# above would be partly a statement about how the file was written if that order carried anything.
_m44v = J("m44v_verify.json")
assert _m44v["S3_reverse_control"]["pass"], "M44v cannot reproduce M44"
assert _m44v["S2_equivalent_form"]["pass"] and _m44v["S4_identities"]["pass"]
put("tvBack", "$" + f"{_m44v['S3_reverse_control']['max_abs_diff']:.0e}".replace("e-0", r"\times 10^{-") + "}$")
put("tvBackN", f"{len(_m44v['S3_reverse_control']['cells'])}")
put("tvShift", sig(_m44v["S1_max_abs_shift"], 4))
_t25 = _m44v["S1_tie_rule"]["0.25"]
put("tvDepthQtr", sig(_t25["depth"]["delta"], 4))
put("tvDepthHalf", sig(_m44v["S1_tie_rule"]["0.5"]["depth"]["delta"], 4))
put("tvDepthTen", sig(_m44v["S1_tie_rule"]["0.1"]["depth"]["delta"], 4))
put("tvNzero", f"{sum(1 for f in _m44v['S1_tie_rule'] for r in _m44v['S1_tie_rule'][f] if r != 'depth' and _m44v['S1_tie_rule'][f][r]['delta'] == 0)}")
put("tvNother", f"{sum(1 for f in _m44v['S1_tie_rule'] for r in _m44v['S1_tie_rule'][f] if r != 'depth')}")
# the free share, at every budget and under both tie rules -- the bare 25% number is a cherry-pick
_e4 = _m44v["E4_by_budget"]
for _f, _k in (("0.1", "Ten"), ("0.25", "Qtr"), ("0.5", "Half")):
    put(f"tvShare{_k}", f"{100*_e4[f'{_f}|stable']['fraction']:.1f}\\%")
    put(f"tvShareR{_k}", f"{100*_e4[f'{_f}|random']['fraction']:.1f}\\%")
    put(f"tvBest{_k}", _e4[f"{_f}|stable"]["best_implementation"])
_shares = [_e4[k]["fraction"] for k in _e4]
put("tvShareLo", f"{100*min(_shares):.1f}\\%"); put("tvShareHi", f"{100*max(_shares):.1f}\\%")
# staked 2's missing interval
_g = _m44v["staked2_missing_interval"]
for _f, _k in (("0.1", "Ten"), ("0.25", "Qtr"), ("0.5", "Half")):
    put(f"tvGap{_k}", sig(_g[_f]["point"], 4)); put(f"tvGapCi{_k}", ci(_g[_f]["ci"], 4))
put("tvGapNres", f"{sum(1 for f in _g if _g[f]['excludes_zero'])}")
assert _g["0.1"]["point"] < 0 < _g["0.5"]["point"], "the inversion this paragraph reports is gone"
assert _g["0.1"]["excludes_zero"], "the 10% inversion no longer excludes zero"

# ============================= M46: the inversion, hardened against the choices that produced it
# The headline of Reading 2 rested on 20 targets, one threshold, one predictor pair and three budget
# points. This arm asks whether it survives each of those choices being made differently.
_m46 = J("m46_inversion.json")
assert _m46["V_reverse_control"]["pass"], "M46 does not reproduce M44; nothing here bears on it"
_lo = str(min(_m46["E3_budget_sweep"]["points"], key=lambda p: p["budget"])["budget"])
_L = _m46["E1_leave_one_target_out"]
put("ivLooLo", sig(_L["0.1"]["min"], 4)); put("ivLooHi", sig(_L["0.1"]["max"], 4))
put("ivLooN", f"{_L['0.1']['n_drops']}")
assert all(_L[f]["sign_stable"] for f in _L), "staked 1 has failed; W2 requires withdrawal"
put("ivLooStable", f"{sum(1 for f in _L if _L[f]['sign_stable'])}")
put("ivNbudget", f"{len(_L)}")
_E2 = _m46["E2_per_target"]
put("ivTgtNeg", f"{_E2['n_negative']}"); put("ivTgtN", f"{_E2['n_targets']}")
put("ivTgtPos", f"{_E2['n_positive']}"); put("ivTgtTie", f"{_E2['n_zero']}")
_S = _m46["E3_budget_sweep"]
put("ivSweepN", f"{_S['n_points']}")
put("ivSweepLo", pct(min(p["budget"] for p in _S["points"]), 0))
put("ivSweepHi", pct(max(p["budget"] for p in _S["points"]), 0))
put("ivGapLo", sig(min(_S["points"], key=lambda p: p["budget"])["gap"], 4))
put("ivGapHi", sig(max(_S["points"], key=lambda p: p["budget"])["gap"], 4))
put("ivNcross", f"{_S['n_sign_changes']}")
put("ivCrossA", pct(_S["crossing_between"][0], 1)); put("ivCrossB", pct(_S["crossing_between"][1], 1))
_P = _m46["E5_other_pairs"]
put("ivPairInv", f"{_P['n_inverting']}"); put("ivPairN", f"{_P['n_pairs']}")
_T = _m46["E4_threshold_sweep"]
put("ivThrInv", f"{sum(1 for v in _T.values() if v.get('inverts'))}")
put("ivThrN", f"{len(_T)}")
put("ivThrFail", ", ".join(k for k, v in _T.items() if not v.get("inverts")))
put("ivThrFailN", f"{[v for k, v in _T.items() if not v.get('inverts')][0]['n_scorable']}")
# THE SHAPE OF THE FINDING, and it is not the one the first draft asserted. Two of the inverting
# pairs do not involve the neural model at all, so "the neural model's advantage lives in the tail"
# is false as written. What is true is stronger: a SINGLE TOTAL ORDER explains all six pairs, and it
# is the incumbent benchmark's own ordering.
#
# Family membership is taken from M43s's measured pairwise agreement, not from a chosen F1 gap:
# ViennaRNA and RNAstructure agree at 0.9710 and every other pair at 0.909 or less, so they are one
# family and nothing else is. The prediction tested here is then: every CROSS-family pair inverts
# with the high budget favouring the model that ranks higher on structure agreement, and the one
# WITHIN-family pair does not invert. Both halves are asserted, so the sentence cannot go stale.
_SAME_FAMILY = 0.95
_nfam = _ninv = 0
for _k, _v in _P["pairs"].items():
    _a, _b = _k.split("|")
    _same = _MAT[f"{_a}|{_b}"] >= _SAME_FAMILY
    if _same:
        _nfam += 1
        assert not _v["inverts"], f"{_k} is one family and now inverts"
    else:
        assert _v["inverts"], f"{_k} crosses families and no longer inverts"
        assert (_v["high"] > 0) == (_ag[_a] > _ag[_b]), f"{_k}: the high budget no longer favours agreement"
        _ninv += 1
put("ivCrossFam", f"{_ninv}"); put("ivSameFam", f"{_nfam}")
put("ivTailPairs", f"{_ninv}")

# ================================ M45: the ceiling on what any public corpus could ever supply
# The breadth question, answered by measurement instead of by argument. W1 of the declaration: every
# number here is an UPPER BOUND on CANDIDATES and may not be written as interventions.
_m45 = J("m45_flip_ceiling.json")
assert _m45["V"]["pass"] and _m45["flank_cut_is_inert"], "M45 checks did not pass"
put("dmSeq", f"{_m45['n_unique']:,}".replace(",", "{,}"))
put("dmRows", f"{_m45['n_rows']:,}".replace(",", "{,}"))
put("dmLib", f"{_m45['W4_library_count']['after_collapsing_chemistry']}")
put("dmDsName", f"{_m45['W4_library_count']['dataset_name_values']}")
put("dmCand", f"{_m45['E1']['candidates_total']}")
put("dmContrib", f"{_m45['E2']['n_contributing_libraries']}")
put("dmTop", f"{max(_m45['E2']['by_library'].values())}")
put("dmTopName", max(_m45["E2"]["by_library"], key=_m45["E2"]["by_library"].get).replace("_", r"\_"))
put("dmNetNew", f"{_m45['E4']['net_new']}")
put("dmSnFrac", pct(_m45["E3"]["frac"], 1))
# the conversion measured where Definition 1 can be applied at all
_c45 = _m45["E5_calibration"]
put("dmOwnReg", f"{_c45['own_corpus_design_regions']:,}".replace(",", "{,}"))
put("dmOwnCand", f"{_c45['candidates_on_own_corpus']:,}".replace(",", "{,}"))
put("dmConv", pct(_c45["conversion_rate"], 2))
put("dmExpected", f"{_m45['E1']['expected_admissible_at_calibrated_rate']:.0f}")
# construct anatomy, identical across every library -- measured, not assumed
_pre = {v["const_prefix"] for v in _m45["construct_anatomy"].values()}
_suf = {v["const_suffix"] for v in _m45["construct_anatomy"].values()}
assert len(_pre) == 1 and len(_suf) == 1, "the construct anatomy is not uniform any more"
put("dmPre", f"{_pre.pop()}"); put("dmSuf", f"{_suf.pop()}")
# the detector's own standing
_v45 = _m45["V"]
put("dmRecall", f"{_v45['V3']['recall']:.4f}")
put("dmKnown", f"{_v45['V3']['known_published_double_mutant_pairs']:,}".replace(",", "{,}"))
put("dmBrute", f"{_v45['V4']['brute_force']:,}".replace(",", "{,}"))
put("dmPlanted", f"{_v45['V2']['planted']}")
assert _v45["V3"]["transpositions_equal_known"], "V3's internal identity no longer holds"
# how far the largest public corpus is from this paper's own yield
# dmRatio was a ratio of two DIFFERENT layers -- admitted interventions over candidate pairs --
# and the sentence that printed it has been withdrawn. Kept out of the file rather than
# silently left defined, because an unused macro is an invitation to reuse a bad comparison.
# the single-nucleotide-mutant library is named in the text, so its count is read out rather than
# asserted: single mutants of DIFFERENT parents are not near neighbours of each other.
_snm = [v["candidates"] for k, v in _m45["per_library"].items() if "Single_nt_mutants" in k]
assert _snm, "the single-nucleotide-mutant library is no longer in the corpus"
put("dmCandZero", f"{max(_snm)}")

# ============================ M47: formal equivalence, replacing "the interval covers zero"
# The margin comes out of a rule frozen before the code and touching only the free baseline and the
# corpus geometry, so it cannot have been chosen to suit the differences it judges.
_m47 = J("m47_equivalence.json")
assert _m47["V_reverse_control"]["pass"], "M47 does not reproduce M44"
_D = _m47["delta_tube"]
assert "void" not in _D, "delta_tube was voided by its own range check"
put("eqDelta", sig(_D["delta"], 4))
put("eqSlope", sig(_D["slope_per_unit_budget"], 4))
put("eqMedCell", f"{_D['median_cell_size']:.0f}")
put("eqOneTube", pct(_D["one_construct_is_this_much_budget"], 0))
_E = _m47["E1_E3_E4"]
_S47 = _m47["summary"]
put("eqNcomp", f"{_S47['n_implementation_comparisons']}")
put("eqNeq", f"{_S47['n_equivalent_at_delta_tube']}")
put("eqNfail", f"{_S47['n_implementation_comparisons'] - _S47['n_equivalent_at_delta_tube']}")
put("eqSelfDs", f"{_S47['depth_self_delta_star']:.5f}")
# the three that fail, named, with the direction that makes the failure meaningful
_fail = sorted(((k, v) for k, v in _E.items()
                if k.split("|")[1] in _IMPL4 and not v["equivalent_at_delta_tube"]),
               key=lambda kv: -kv[1]["delta_star"])
put("eqFailN", f"{len(_fail)}")
put("eqFailBudget", pct(float(_fail[0][0].split("|")[0]), 0) if _fail else "---")
put("eqFailNames", ", ".join(k.split("|")[1] for k, _ in _fail))
put("eqFailDsHi", sig(_fail[0][1]["delta_star"], 4) if _fail else "---")
put("eqFailDsLo", sig(_fail[-1][1]["delta_star"], 4) if _fail else "---")
put("eqFailWorst", sig(min(v["point"] for _, v in _fail), 4) if _fail else "---")
assert _fail and all(v["point"] < 0 for _, v in _fail), \
    "the failing arms are no longer on the worse side; the sentence about direction is stale"
assert len({k.split("|")[0] for k, _ in _fail}) == 1, "the failures now span more than one budget"
# the stacked arms: statistically resolved, and still under one construct
_stack = {k: v for k, v in _E.items() if k.split("|")[1].startswith("depth_then_")}
put("eqStackMaxDs", sig(max(v["delta_star"] for v in _stack.values()), 4))
assert all(v["equivalent_at_delta_tube"] for v in _stack.values()), \
    "a stacked arm now exceeds one construct; the sentence saying none does is stale"
_best = _E["0.25|depth_then_RibonanzaNet"]
put("eqBestPt", sig(_best["point"], 4)); put("eqBestDs", sig(_best["delta_star"], 4))
put("eqBestCi", ci(_best["ci90"], 4))

# ================================ M48: is a window-level outcome ground truth for a pair prediction
_m48 = J("m48_locality.json")
assert _m48["V1_reproduces_column"]["pass"] and _m48["V2_position_check"]["pass"]
_A48 = _m48["E1_window_agreement"]; _H48 = _m48["E2_headline_by_window"]
for _w, _k in (("5", "Five"), ("10", "Ten"), ("15", "Fifteen"), ("25", "TwentyFive")):
    put(f"lcRho{_k}", sig(_A48[_w]["spearman_vs_full"], 4))
    put(f"lcDepth{_k}", f"{_H48[_w]['depth@0.25']:.4f}")
    put(f"lcGap{_k}", sig(_H48[_w]["RNet-Vie@0.1"], 4))
put("lcDepthFull", f"{_H48['None']['depth@0.25']:.4f}")
put("lcGapFull", sig(_H48["None"]["RNet-Vie@0.1"], 4))
put("lcNwin", f"{len(_A48)}")
put("lcRhoLo", sig(min(v["spearman_vs_full"] for k, v in _A48.items() if k != "None"), 4))
_live = [v for v in _H48.values() if not v.get("underpowered")]
put("lcGapMax", sig(max(v["RNet-Vie@0.1"] for v in _live), 4))
put("lcAdvMax", sig(max(v[f"{n}@{f}"] for v in _live for n in _IMPL4 for f in (0.1, 0.25, 0.5)), 4))
assert all(v["RNet-Vie@0.1"] < 0 for v in _live), "the inversion no longer holds at every window"
put("lcV1Pearson", f"{_m48['V1_reproduces_column']['pearson']:.6f}")

# ---- the empirical null the failed check produced
_N0 = _m48["E0_noop_null"]
put("noopN", f"{_N0['n']}")
put("noopMean", f"{_N0['mean']:.4f}"); put("noopMed", f"{_N0['median']:.4f}")
put("noopSd", f"{_N0['sd']:.4f}")
put("noopLo", sig(_N0["min"], 4)); put("noopHi", sig(_N0["max"], 4))
put("noopFP", pct(_N0["frac_above_rescue_threshold"], 1))
put("noopThr", f"{_N0['rescue_threshold']}")
assert _N0["none_admitted"], "a no-op control has entered the admitted set"
# the rate at which the position check fired BEFORE the no-ops were diagnosed out of it: read from
# the two counts rather than retyped from the log, so it cannot drift from them.
_v2 = _m48["V2_position_check"]
put("noopFired", pct(_N0["n"] / (_N0["n"] + _v2["n_checked"]), 2))
put("noopKill", pct(0.01, 0))
# the staked range for the margin, read out of M47's frozen declaration
_d47 = json.load(open(os.path.join(CEIL, "m47_equivalence_declaration.json")))
_mm = _re.search(r"between ([\d.]+) and ([\d.]+)", _d47["staked"]["4"])
put("eqStakeLo", _mm.group(1)); put("eqStakeHi", _mm.group(2))
_m2 = _re.search(r"below ([\d.]+)", _d47["staked"]["2"])
put("eqStakeSelf", _m2.group(1))

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
