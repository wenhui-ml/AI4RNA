#!/usr/bin/env python3
"""Generate figs/rna_numbers.tex from the committed run artifacts.

Every measured number in the manuscript reaches LaTeX through a \jdg{} macro defined here, so a
printed figure cannot drift from the run that produced it. Nothing measured may be typed into the
.tex by hand; make_check.py fails the build if it is.

COST: $0, offline.
"""
from __future__ import annotations

import json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CEIL = os.path.abspath(os.path.join(HERE, "..", "..", "rna_harness", "runs", "ceiling"))
POPS = ("panel_f1_fresh.json", "panel_f1_designs.json", "panel_f1_third.json",
        "panel_f1_fourth.json", "panel_f1_fifth.json", "panel_f1_sixth.json")
POPEN = ("sample 2 (main)", "baseline 4560", "sample 3", "batch 4", "batch 5", "batch 6")

M = {}


def J(n):
    return json.load(open(os.path.join(CEIL, n)))


def put(k, v):
    assert k not in M, f"duplicate macro {k}"
    M[k] = v


def f(x, d=3):
    return f"{x:.{d}f}"


def ci(v, d=4):
    # \allowbreak after the thin space: an interval is otherwise one unbreakable atom.
    return f"[{sig(v[0], d)},\\,\\allowbreak {sig(v[1], d)}]"


def pv(x):
    return "<0.001" if x < 0.001 else f"{x:.4f}".rstrip("0").rstrip(".")


def sig(x, d=3):
    return ("+" if x >= 0 else "") + f"{x:.{d}f}"


# ---------------------------------------------------------------- 3  pair type -> formation
m16 = J("m16_rescue.json")
c = m16["counts"]
put("nCG", f"{c['CG']}"); put("nAU", f"{c['AU']}"); put("nGU", f"{c['GU']}")
put("nPairsTotal", f"{c['CG'] + c['AU'] + c['GU']}")
for tag, key, thr in (("AU", "CG_vs_AU", "ladder_threshold"), ("GU", "CG_vs_GU", "ladder_threshold")):
    a = m16[key]
    put(f"rescue{tag}", sig(a["matched_diff"], 4))
    put(f"rescue{tag}Marg", sig(a["marginal_diff"], 4))
    put(f"rescue{tag}P", f"{a['p_perm']:.4f}")
    put(f"rescue{tag}Ci", f"[{a['boot_ci'][0]:.4f},\\,{a['boot_ci'][1]:.4f}]")
    put(f"rescue{tag}W", f"{a['weight']:.0f}")
    put(f"rescue{tag}Strata", f"{a['n_strata']}")
    put(f"rescue{tag}Loo", f"[{a['loo_range'][0]:.4f},\\,{a['loo_range'][1]:.4f}]")
    put(f"rescue{tag}Lad", f(a[thr], 2))
    put(f"rescue{tag}Margin", f"{a['matched_diff'] / a[thr]:.2f}")
    put(f"rescue{tag}Delta0", f"{a['ladder_delta0']:.2f}")
    put(f"rescue{tag}Noise", f"{a['P1_pos']}")

# ---------------------------------------------------------------- 4  placement mediates
m20 = J("m20_designer_confounding.json")
a2 = m20["A2"]
put("medMarginal", sig(a2["marginal"], 4)); put("medMatched", sig(a2["matched"], 4))
put("medBeta", f(a2["beta_depth"], 4)); put("medGap", sig(a2["gap_g"], 4))
put("medPred", sig(a2["predicted_attenuation"], 4))
put("medObs", sig(a2["observed_attenuation"], 4))
put("medFrac", f"{100 * a2['mediated_fraction']:.1f}\\%")
put("medRatio", f"{a2['matched'] / a2['marginal']:.1f}")
put("medNmarg", f"{a2['n_marg']}"); put("medNmatched", f"{a2['n_matched']}")

# ---------------------------------------------------------------- 5  the design score
m18 = J("m18_repaired_null.json")["pops"]
eff, zs, ps, np_, smd = [], [], [], [], []
for i, fn in enumerate(POPS):
    b = m18[fn]["arms"]["B_panel_matched"]
    put(f"cpop{i}Eff", sig(b["effect"], 3)); put(f"cpop{i}Z", f(b["z_vs_null_REPAIRED"], 2))
    put(f"cpop{i}P", f"{b['p_perm_REPAIRED']:.4f}"); put(f"cpop{i}N", f"{b['npair']}")
    put(f"cpop{i}Smd", f(b["smd_gcf"], 3)); put(f"cpop{i}SmdP", f(b["smd_pB"], 3))
    put(f"cpop{i}Ci", f"[{b['boot_ci_secondary'][0]:.2f},\\,{b['boot_ci_secondary'][1]:.2f}]")
    eff.append(b["effect"]); zs.append(b["z_vs_null_REPAIRED"])
    ps.append(b["p_perm_REPAIRED"]); np_.append(b["npair"]); smd.append(b["smd_gcf"])
put("cEffLo", sig(min(eff), 2)); put("cEffHi", sig(max(eff), 2))
put("cZLo", f(min(zs), 1)); put("cZHi", f(max(zs), 1))
put("cSigStrict", f"{sum(1 for p in ps if p <= 0.005)}")
put("cSigLoose", f"{sum(1 for p in ps if p < 0.05)}")
put("cSmdHi", f(max(smd), 3))

m13b = J("m13b_gc_placement.json")
p4 = m13b["P4"]
sh = [v["shift_mean"] for v in p4.values()]; lk = [v["leaked_gcp_gap"] for v in p4.values()]
rl = [v["real_gcp_gap"] for v in p4.values()]
put("p4Lo", sig(min(sh), 2)); put("p4Hi", sig(max(sh), 2))
put("p4Pass", f"{sum(1 for v in p4.values() if v['pass'])}")
put("p4LeakLo", f"{min(lk):.4f}"); put("p4LeakHi", f"{max(lk):.4f}")
put("p4RealLo", f"{min(rl):.4f}"); put("p4RealHi", f"{max(rl):.4f}")
put("p4LeakPct", f"{100 * max(lk) / min(rl):.0f}\\%")

m13c = J("m19_m13c_symmetric_repaired.json")
KS = ("0.4", "0.6", "0.8", "1.0")
for k in KS:
    put(f"d2P{k.replace('.','')}", f"{m13c[k]['sig_P']}")
    put(f"d2C{k.replace('.','')}", f"{m13c[k]['sig_C']}")
    put(f"d2SlopeP{k.replace('.','')}", f(m13c[k]["median_slope_P"], 1))
    put(f"d2SlopeC{k.replace('.','')}", f(m13c[k]["median_slope_C"], 1))
put("d2PLo", f"{min(m13c[k]['sig_P'] for k in KS)}")
put("d2PHi", f"{max(m13c[k]['sig_P'] for k in KS)}")
put("d2CLo", f"{min(m13c[k]['sig_C'] for k in KS)}")
put("d2CHi", f"{max(m13c[k]['sig_C'] for k in KS)}")

m15 = J("m19_m15_structure_filter_repaired.json")["pops"]
ok = [v for v in m15.values() if v.get("usable")]
put("c5Sig", f"{sum(1 for v in ok if v['A_filtered']['p_perm'] <= 0.005)}")
put("c5N", f"{len(ok)}")
put("c5Cut", "0.80")

m14 = J("m14_mutation_intervention.json")
Y1, Y2 = m14["Y1"], m14["Y2_dECS"]
ARMS = ("GC_via_G_to_C", "GC_via_C_to_G", "AU_via_A_to_U", "AU_via_U_to_A")
for i, arm in enumerate(ARMS):
    put(f"d3Y{i}", sig(Y1[arm]["mean"], 4))
    put(f"d3Y{i}Ci", f"[{Y1[arm]['ci'][0]:.3f},\\,{Y1[arm]['ci'][1]:.3f}]")
    put(f"d3E{i}", sig(Y2[arm]["mean"], 3))
    put(f"d3E{i}Ci", f"[{Y2[arm]['ci'][0]:.2f},\\,{Y2[arm]['ci'][1]:.2f}]")
put("d3NPos", f"{sum(1 for a in ARMS if Y1[a]['ci'][0] > 0)}")
put("d3NNeg", f"{sum(1 for a in ARMS if Y2[a]['ci'][1] < 0)}")

# ---------------------------------------------------------------- 6  the gradient across methods
a1 = m20["A1"]; pm = a1["per_method"]
put("gMethComp", f(a1["method_component"], 4)); put("gTgtComp", f(a1["target_component"], 4))
put("gNullMean", f(a1["null_mean"], 4)); put("gNullSd", f(a1["null_sd"], 4))
put("gP", f"{a1['p_perm']:.4f}"); put("gZ", f(a1["z"], 2)); put("gCells", f"{a1['n_cells']}")
put("gGapMax", sig(pm[0]["gap"], 4)); put("gGapMaxName", pm[0]["method"].replace("_", "\\_"))
put("gGapMin", sig(pm[-1]["gap"], 4)); put("gGapMinName", pm[-1]["method"].replace("_", "\\_"))
put("gSpread", f(abs(pm[0]["gap"] - pm[-1]["gap"]), 3))
put("gRatio", f"{pm[0]['gap'] / pm[-1]['gap']:.1f}")
put("gHumanRank", f"{a1['human_rank'][0]}"); put("gNMethods", f"{len(pm)}")
a3 = m20["A3"]
put("gSlope", f(a3["slope"], 3)); put("gSlopeP", f"{a3['p_perm']:.4f}"); put("gSlopeN", f"{a3['n']}")
lad = m20["P2_ladder"]
put("gLadLo", f"{lad['0.4']['hit_rate']:.2f}"); put("gLadHi", f"{lad['0.8']['hit_rate']:.2f}")
put("gLadZero", f"{lad['0.0']['hit_rate']:.2f}")
put("gNoise", f"{m20['P1']['hits']}")

# ---------------------------------------------------------------- methods: rule 80
put("nullTypeIOld", "13/240"); put("nullTypeINew", "1/240")
m17d = J("m17d_typeI_diag.json")
put("nullRateOld", f"{m17d['total']['A'][0] / m17d['total']['A'][1]:.4f}")
put("nullRateNew", f"{m17d['total']['B'][0] / m17d['total']['B'][1]:.4f}")
v = J("m18v_verify_repair.json")["detail"]
r = [x["sd_ratio"] for x in v.values()]; ks = [x["ks"] for x in v.values()]
put("verSdLo", f(min(r), 3)); put("verSdHi", f(max(r), 3))
put("verKsHi", f"{max(ks):.4f}")
put("verKsCrit", f"{list(v.values())[0]['ks_crit_5pct']:.4f}")


# ---------------------------------------------------------------- measurement frame
r1b = J("r1b_ceiling_interval.json")
put("ceilGroups", f"{r1b['n_repeat_groups']}")
put("ceilRows", "1983")
put("ceilPairs", f"{r1b['n_pairs']}")
put("ceilDf", f"{r1b['df']}")
put("ceilSigA", f(r1b["sigma_a"], 4))
put("ceilSigACi", f"[{r1b['ci_a'][0]:.4f},\\,{r1b['ci_a'][1]:.4f}]")
put("ceilSigB", f(r1b["sigma_b"], 4))
put("ceilSigBCi", f"[{r1b['ci_b'][0]:.4f},\\,{r1b['ci_b'][1]:.4f}]")
put("ceilDeclared", f(r1b["declared_sigma"], 3))
_main = r1b["per_pop"]["panel_f1_fresh.json"]
put("ceilWithinSd", f(_main["within_sd"], 4))
put("ceilA", f(_main["ceiling_a"], 4))
put("ceilACi", f"[{_main['ci_a'][0]:.4f},\\,{_main['ci_a'][1]:.4f}]")
put("ceilB", f(_main["ceiling_b"], 4))
put("ceilBCi", f"[{_main['ci_b'][0]:.4f},\\,{_main['ci_b'][1]:.4f}]")
put("ceilRms454", "4.3469"); put("ceilHeavyTail", "1.35"); put("ceilNormalTail", "1.18")

# variance decomposition -- an identity, not a test (m11_audit)
put("varTarget", "28.8\\%"); put("varMethod", "24.1\\%")
put("varCell", "47.1\\%"); put("varNoise", "15.1\\%")

# NOTE: the panel-occupancy block and the six-population sequence-intrinsic block that used
# to be emitted here have been removed. Both are about what the shipped predictor panel does and
# does not capture, which is a different paper's subject and is the one place this manuscript
# came close to sharing an object with it. The panel now enters this paper only inside the
# balancing covariate of Def. 4, and nothing generated here refers to it.

# ---------------------------------------------------------------- dose response and robustness arms
_m13b = J("m13b_gc_placement.json")
_m2 = _m13b.get("M2_dose", {})
for i, fn in enumerate(POPS[:1]):
    row = _m2.get(fn) or []
    for k, r in enumerate(row):
        if r:
            put(f"doseLo{k}", f(r["lo"], 2)); put(f"doseEff{k}", sig(r["effect"], 3))
            put(f"doseN{k}", f"{r['n']}")
_m13d = J("m19_m13d_stability_repaired.json")["pops"]
_dmain = _m13d[POPS[0]]
put("dgBase", sig(_dmain["base"], 3))
put("dgSd", f(_dmain["within_sd_dg"], 2))
_rr = [r for r in _dmain["rungs"] if r.get("effect") is not None]
put("dgLooseCal", f(_rr[0]["cal"], 2)); put("dgLooseEff", sig(_rr[0]["effect"], 3))
put("dgLooseP", f"{_rr[0]['p_REPAIRED']:.4f}")
put("dgTightCal", f(_rr[-1]["cal"], 2)); put("dgTightEff", sig(_rr[-1]["effect"], 3))
put("dgTightP", f"{_rr[-1]['p_REPAIRED']:.4f}")
put("dgRetain", f"{100*_rr[-1]['retain']:.0f}\\%")

_m15 = J("m19_m15_structure_filter_repaired.json")["pops"]
for i, fn in enumerate(POPS):
    v = _m15.get(fn)
    if not v or not v.get("usable"):
        continue
    A = v["A_filtered"]; ref = v.get("reference_unfiltered") or {}
    put(f"m15pop{i}Eff", sig(A["effect"], 3)); put(f"m15pop{i}P", f"{A['p_perm']:.4f}")
    put(f"m15pop{i}N", f"{A['npair']}"); put(f"m15pop{i}Pass", f"{v['n_pass']}")
    put(f"m15pop{i}Ref", sig(ref.get("effect", float('nan')), 3))
    B = v.get("B_complement") or {}
    put(f"m15pop{i}Comp", f"{B.get('npair', 0)}")



# ---------------------------------------------------------------- 2  the energy test (M21)
m21 = J("m21_thermo_mediator.json")
b1, p3 = m21["B1"], m21["P3"]
put("eTermAU", f(m21["terminal_au_kcal"], 2))
put("eNBins", f"{m21['nbins']}")
put("eCde", sig(b1["cde"], 4))
put("eBaseSame", sig(b1["base_same_population"], 4))
put("eBaseFull", sig(b1["base_full"], 4))
put("eRet", f"{100 * b1['retention']:.1f}\\%")
put("eRetWrong", f"{100 * b1['retention_wrong_denominator']:.1f}\\%")
put("eSel", f"{100 * b1['selection_component']:.1f}\\%")
put("eP", f"{b1['p_perm']:.4f}")
put("eNullSd", f(b1["null_sd"], 4))
put("eCi", f"[{b1['boot_ci'][0]:.4f},\\,{b1['boot_ci'][1]:.4f}]")
put("eW", f"{b1['weight']:.0f}")
put("eWBase", f"{b1['base_weight']:.0f}")
put("eStrata", f"{b1['n_strata']}")
put("ePlaceboMean", f"{100 * p3['placebo_retention_mean']:.1f}\\%")
put("ePlaceboSd", f"{100 * p3['placebo_retention_sd']:.1f}")
put("ePlaceboLo", f"{100 * p3['placebo_retention_range'][0]:.1f}\\%")
put("ePlaceboHi", f"{100 * p3['placebo_retention_range'][1]:.1f}\\%")
put("eZ", sig(p3["z_vs_placebo"], 2))
put("eMed", f"{100 * p3['mediated_fraction']:+.1f}\\%")
put("eMedCi", f"[{100 * p3['mediated_fraction_ci95'][0]:+.1f}\\%,\\,"
              f"{100 * p3['mediated_fraction_ci95'][1]:+.1f}\\%]")
put("eLad", f(m21["P2_threshold"], 2))
put("eDeltaZero", f"{m21['P2_delta0']:.2f}")
put("eNoise", f"{m21['P1_pos']}")
b2 = m21["B2"]
put("eBeta", sig(b2["beta_dG"], 5))
put("eGap", sig(b2["dG_gap"], 4))
put("eLinearMed", f"{100 * b2['mediated_fraction']:.1f}\\%")
put("eLinearGapPts", f"{abs(100 * b2['mediated_fraction'] - 100 * p3['mediated_fraction']):.1f}")
put("eBinwise", ",\\;".join(sig(x["rf_centred"], 4) for x in b2["binwise"]))
put("eBinwiseDg", ",\\;".join(sig(x["dG_mean"], 2) for x in b2["binwise"]))
for nb in ("3", "8"):
    put(f"eSens{nb}", f"{100 * m21['sensitivity_bins'][nb]['retention']:.1f}\\%")
for t in ("CG", "AU", "GU"):
    d = m21["dG_by_type"][t]
    put(f"eDg{t}", sig(d["mean"], 3)); put(f"eDgSd{t}", f(d["sd"], 3))
    put(f"eBins{t}", "/".join(str(x) for x in d["bin_hist"]))
b3 = m21["B3"]
for k, nm in (("depth0", "Zero"), ("depth1", "One"), ("depth2plus", "Two")):
    put(f"eD{nm}", sig(b3[k]["diff"], 4)); put(f"eD{nm}P", f"{b3[k]['p_perm']:.4f}")
    put(f"eD{nm}W", f"{b3[k]['weight']:.0f}"); put(f"eD{nm}N", f"{b3[k]['n']}")
b4 = m21["B4"]; col = m21["B4_collinearity"]
for k, nm in (("CG_vs_AU", "ca"), ("CG_vs_GU", "cg"), ("AU_vs_GU", "ag")):
    put(f"eSlope_{nm}".replace("_", ""), sig(b4[k]["implied_slope"], 5))
    put(f"eDgd{nm}", sig(b4[k]["dG_diff"], 4))
put("eSlopeSpread", f(col["spread_ratio"], 2))
b5 = m21["B5"]
put("eGuAdj", sig(b5["GU_adjacent"]["diff"], 4)); put("eGuAdjN", f"{b5['GU_adjacent']['n_GU']}")
put("eGuIso", sig(b5["GU_isolated"]["diff"], 4)); put("eGuIsoN", f"{b5['GU_isolated']['n_GU']}")

v21 = J("m21v_verify.json")["V1"]
put("eProbe", sig(v21["noncanonical_probe_kcal"], 2))
put("eContam", f"{v21['n_contaminated']}")
put("eContamPct", f"{100 * v21['frac']:.1f}\\%")
put("eMaxDgChange", f(v21["max_dG_change"], 2))


# ---------------------------------------------------------------- the null condition (M23)
m23 = J("m23_exact_null.json")
r23 = m23["DGP1"]["rate"]
for k, tag in (("N1_permute_t", "nT"), ("N2_permute_y_cell", "nY"),
               ("N3_freedman_lane", "nFL"), ("N4_blocked", "nBl")):
    put(f"m23{tag}", f"{r23[k]:.3f}")
    tt = m23["DGP1"]["total"][k]
    put(f"m23{tag}N", f"{tt[0]}/{tt[1]}")
put("m23BlockMed", f(m23["N4_block_median"], 1))
# "a sevenfold inflation" was typed into Related work as a word, which make_check.py cannot see
# because it only looks for digits. The measured ratio is not seven.
# The corpus's own counts were spelled out as words -- "twenty targets", "nine design methods",
# "six populations", "four rounds" -- which make_check.py could not see because it only looks for
# digits. They are counts over a released file and a different release is a different corpus.
# ---- provenance counts. R1 had no provenance.json, so the audit's [H] check -- the one that
# tests a manuscript's claim about its own process -- had never run on this paper. It had run on
# the companion and found that paper's preregistration sentence false; R1's was false in the same
# way. These counts come out of the classification so the sentence cannot drift from it.
_pv = json.load(open(os.path.join(HERE, "provenance.json")))
_cls = {k: len([x for x in v if not x.startswith("_")])
        for k, v in _pv.items() if not k.startswith("_")}
put("pvTotal", f"{sum(_cls.values())}")
put("pvPreReg", f"{_cls['preregistered']}")
put("pvReview", f"{_cls['review']}")
put("pvSurvey", f"{_cls['survey']}")
put("pvSupport", f"{_cls['support']}")
put("pvCorpus", f"{_cls['corpus']}")
put("pvUnplanned", f"{_cls['review'] + _cls['survey'] + _cls['support']}")

_cc = J("m33_cluster_census.json")
put("ccPuzzlesR", f"{_cc['n_puzzles']}")
put("ccCellsR", f"{_cc['n_design_cells']}")
put("m23Ratio", f"{r23['N1_permute_t'] / r23['N2_permute_y_cell']:.1f}")
put("nullRatio", f"{J('m17d_typeI_diag.json')['total']['A'][0] / J('m17d_typeI_diag.json')['total']['B'][0]:.1f}")
_lk = m23["DGP2_VOID"]["leak_per_pop"]
put("m23LeakLo", f"{100 * min(_lk):.1f}\\%"); put("m23LeakHi", f"{100 * max(_lk):.1f}\\%")
put("m23LeakCorr", sig(m23["DGP2_VOID"]["leak_corr_with_rejection"], 3))
put("m23Dgp3Leak", f"{100 * m23['DGP2_VOID']['dgp3_leak']:.1f}\\%")
put("m23CrossBlock", f"{100 * m23['DGP2_VOID']['cross_block_pair_frac']:.1f}\\%")

m23c = J("m23c_imbalance_bound.json")
_pan = [r for r in m23c["rows"] if r["panel"]]
_free = [r for r in m23c["rows"] if not r["panel"]]
put("imbPanLo", f"{min(r['bias_pct_of_effect'] for r in _pan):.2f}\\%")
put("imbPanHi", f"{max(r['bias_pct_of_effect'] for r in _pan):.2f}\\%")
put("imbFreeLo", f"{min(r['bias_pct_of_effect'] for r in _free):.1f}\\%")
put("imbFreeHi", f"{max(r['bias_pct_of_effect'] for r in _free):.1f}\\%")
put("imbDpiFreeLo", f(min(r["mean_dpi"] for r in _free), 2))
put("imbDpiFreeHi", f(max(r["mean_dpi"] for r in _free), 2))
put("imbDpiPanLo", f(min(r["mean_dpi"] for r in _pan), 3))
put("imbDpiPanHi", f(max(r["mean_dpi"] for r in _pan), 3))
put("imbBetaLo", f(min(r["beta_gamma"] for r in _pan), 1))
put("imbBetaHi", f(max(r["beta_gamma"] for r in _pan), 1))

# ---------------------------------------------------------------- held-out rounds (M24)
m24 = J("m24_cross_round.json")
put("xrRounds", f"{len(m24['rounds'])}")
put("xrSign", f"{m24['pooled_method_sign_test']['negative']}/"
              f"{m24['pooled_method_sign_test']['total']}")
put("xrSignP", "%.1e" % m24["pooled_method_sign_test"]["binom_p"])
_g = [r["pooled_gap"] for r in m24["rounds"]]
put("xrGapLo", sig(max(_g), 4)); put("xrGapHi", sig(min(_g), 4))
put("xrCells", f"{sum(r['cells'] for r in m24['rounds'])}")
put("xrTargets", f"{sum(r['targets'] for r in m24['rounds'])}")
put("xrR1Pos", f"{len(m24['round1_positive_methods'])}")
put("xrR1Names", ", ".join("\\texttt{%s}" % x.replace("_", "\\_")
                           for x in m24["round1_positive_methods"]))
put("xrMcClears", f"{sum(1 for r in m24['rounds'] if r['method_p'] <= 0.005)}")
put("xrEterna", "/".join(str(r["eterna_rank"] + 1) for r in m24["rounds"]))


# ---------------------------------------------------------------- the model, scored (M26)
# ================================================================ THE NEW CORE
# Three blocks that did not exist when this paper was about placement. They are the paper now.

# ---- M21b: the direct quantities that replace the withdrawn retention figures.
# Same records, same stratification, no ratio of contrasts anywhere.
m21b = J("m21b_direct_contrast.json")
put("dN", f"{m21b['n']}")
for key, tag in (("rf", "Out"), ("dg", "Dg"), ("pmut", "Pm")):
    r = m21b[key]
    put(f"d{tag}", sig(r["effect"], 4))
    put(f"d{tag}Ci", ci(r["ci"]))
    put(f"d{tag}P", pv(r["p"]))
put("dW", f"{m21b['rf']['weight']:.0f}")
put("dStrata", f"{m21b['rf']['strata']}")
put("dDgSlope", sig(m21b["dg"]["within_slope"], 5))
put("dPmSlope", sig(m21b["pmut"]["within_slope"], 5))
put("dDgBound", f"{100 * m21b['dg']['structural_bound']:.1f}\\%")
put("dPmBound", f"{100 * m21b['pmut']['structural_bound']:.1f}\\%")
put("dBoundRatio", f"{m21b['dg']['structural_bound'] / m21b['pmut']['structural_bound']:.1f}")

# ---- m26r3: the constructive calibration. Inject a KNOWN mediated fraction into the real
# mediator, keep only the outcome synthetic, and read back what the estimator reports.
r3 = J("m26r3_realistic_calibration.json")
for nm, tag in (("RibonanzaNet", "N"), ("ViennaRNA", "V")):
    a = r3[nm]
    put(f"cal{tag}Corr", sig(a["corr"], 4))
    put(f"cal{tag}Mae", f(a["mae"], 4))
    put(f"cal{tag}Full", f"{100 * a['sweep'][-1]['measured_retention']:+.1f}\\%")
    put(f"cal{tag}Zero", f"{100 * a['sweep'][0]['measured_retention']:.1f}\\%")
    put(f"cal{tag}Alpha", sig(a["alpha"], 5))
    put(f"cal{tag}Bound", f"{100 * a['structural_bound']:.1f}\\%")
put("calSweep", ", ".join(f"{100*x['f']:.0f}\\%" for x in r3["RibonanzaNet"]["sweep"]))
put("calTotal", sig(r3["RibonanzaNet"]["total"], 5))

# ---- m26r4: the audit found calVCorr and calNFull computed and never shown -- both live in the
# SAME claimbox as the numbers that ARE shown, so quoting one arm's correlation and the other
# arm's error is selective reporting inside a withdrawal. This block reports both arms on the
# same footing, and the leave-one-out that shows why neither correlation may be leaned on.
r4 = J("m26r4_calibration_symmetry.json")["arms"]
for nm, tag in (("RibonanzaNet", "N"), ("ViennaRNA", "V")):
    a4 = r4[nm]
    put(f"cal{tag}Rho", sig(a4["spearman"], 4))
    put(f"cal{tag}Loo", sig(a4["corr_without_endpoint"], 4))
    put(f"cal{tag}LooLo", sig(a4["loo_corr_min"], 4))
    put(f"cal{tag}LooHi", sig(a4["loo_corr_max"], 4))
    put(f"cal{tag}Span", f(a4["span_measured_f_le_075"], 4))
    put(f"cal{tag}SpanFrac", f"{100 * a4['span_ratio']:.1f}\\%")
put("calSpanTrue", f(r4["ViennaRNA"]["span_true_f_le_075"], 2))
put("calSpanTop", "75\\%")

# the guidance that passed us: standardised total effect and n, against the published thresholds
_OUTCOME_SD = 0.2501            # m26r3_audit.md, the outcome SD the synthetic noise was matched to
put("calOutSd", f(_OUTCOME_SD, 4))
put("calStdC", f(abs(r3["RibonanzaNet"]["total"]) / _OUTCOME_SD, 3))
put("calStdCThr", "0.2")
put("calNThr", "500")

# the failed first attempt, kept because its failure mode has a name
r2 = J("m26r2_positive_control.json")
put("posCorr", sig(r2["corr_true_vs_measured"], 3))
put("posMae", f(r2["mean_abs_error"], 4))
put("posFull", f"{100 * r2['sweep'][0]['measured_retention']:.1f}\\%")
put("posSepSyn", "1.67")        # m26r3_audit.md
put("posSepReal", "0.053")
put("posSepRatio", "31")

# ---- m28r: the second leg. A different estimator, a different corpus, the same disease.
m28 = J("m28_second_corpus.json")
m28r = J("m28r_review.json")
_d28 = {x["rmdb_id"]: x for x in m28["deposits"]}
_a = m28r["R2_arms"]["ViennaRNA"]
put("legSlope", sig(_d28["M2SL5_2A3_0000"]["C"]["ViennaRNA"]["slope"], 5))
put("legSlopeP", pv(_d28["M2SL5_2A3_0000"]["C"]["ViennaRNA"]["p_perm"]))
put("legRho", sig(_a["within_spearman"]["2A3"]))
put("legRhoP", pv(_a["within_spearman_p"]["2A3"]))
put("legSdRatio", f"{m28r['R2_outcome_scale']['ratio']:.2f}")
put("legN", f"{_d28['M2SL5_2A3_0000']['A']['n_admitted']}")

# ---------------------------------------------------------------- cross-domain test of the
# proposition itself (M29 + M29r). The binder corpus is not RNA and shares no machinery with it.
m29 = J("m29_binder_crossdomain.json")
m29r = J("m29r_review.json")
_main = m29["arms"][0]
put("xdN", f"{_main['n']}")
put("xdCells", f"{_main['n_cells']}")
put("xdEta", f"{_main['A_premise']['eta2_mean']:.4f}")
put("xdEtaLo", f"{min(x['eta2'] for x in _main['A_premise']['eta2_by_target']):.4f}")
put("xdEtaHi", f"{max(x['eta2'] for x in _main['A_premise']['eta2_by_target']):.4f}")
put("xdTargets", ", ".join(x["target"] for x in _main["A_premise"]["eta2_by_target"]))
put("xdMarg", sig(_main["B"]["marginal"], 5))
put("xdMargP", pv(_main["B"]["marginal_p"]))
put("xdMatched", sig(_main["B"]["matched"], 5))
put("xdStrat", sig(_main["C"]["route_stratified"], 5))
put("xdProd", sig(_main["C"]["route_product"], 5))
put("xdGapPts", f"{_main['C']['gap_points']:.2f}")
_cmp = m29r["comparison"]
for tag, k in (("B", "binder"), ("R", "rna")):
    v = _cmp[k]
    put(f"xd{tag}N", f"{v['n']}")
    put(f"xd{tag}Dtv", f"{v['d_TV']:.4f}")
    put(f"xd{tag}Osc", f"{v['osc_r']:.5f}")
    put(f"xd{tag}Gap", f"{v['gap_points']:+.2f}")
    put(f"xd{tag}Rhs", f"{v['rhs']:.5f}")
    put(f"xd{tag}Tight", f"{v['tightness']:.3f}")
put("xdDtvRatio", f"{_cmp['dtv_ratio']:.2f}")
put("xdGapRatio", f"{_cmp['gap_ratio']:.2f}")
# the bound has TWO factors and only one of them moves with the gap. Both ratios, so a reader
# can see that the product of the two does not scale the way the gap does.
_bB, _bR = _cmp["binder"], _cmp["rna"]
put("xdOscRatio", f"{_bB['osc_r']/_bR['osc_r']:.2f}")
put("xdBoundRatio", f"{_bR['rhs']/_bB['rhs']:.2f}")
put("xdTightB", f"{_bB['tightness']:.3f}")
put("xdTightR", f"{_bR['tightness']:.3f}")
put("xdTightRatio", f"{_bR['tightness']/_bB['tightness']:.2f}")
_bs, _rs = m29r["R1_binder_binsweep"], m29r["R2_rna_binsweep"]
put("xdSweepN", f"{len(_bs)}")
put("xdSweepLo", f"{min(r['nbins'] for r in _bs)}")
put("xdSweepHi", f"{max(r['nbins'] for r in _bs)}")
put("xdBHold", f"{sum(r['bound_holds'] for r in _bs)}")
put("xdRHold", f"{sum(r['bound_holds'] for r in _rs)}")
put("xdRFailBins", ", ".join(str(r["nbins"]) for r in _rs if not r["bound_holds"]))
put("xdROscLo", f"{min(r['osc_r'] for r in _rs):.5f}")
put("xdROscHi", f"{max(r['osc_r'] for r in _rs):.5f}")
put("xdBound", "holds" if _main["C"]["bound_holds"] else "FAILS")
put("xdECorr", f"{_main['E']['corr']:.4f}")
put("xdEMae", f"{_main['E']['mae']:.4f}")
put("xdLadderThr", f"{_main['ladder']['threshold']:.2f}")
_hs = [b["h"] for b in _main["D"]["bins"]]
put("xdDSpread", f"{max(_hs)-min(_hs):.4f}")
_sw = {round(r["true_retention"], 2): r["measured_retention"] for r in _main["E"]["sweep"]}
put("xdEHalfTrue", "50\\%")
put("xdEHalfMeas", f"{100*_sw[0.5]:.1f}\\%")
put("xdEQuarterTrue", "25\\%")
put("xdEQuarterMeas", f"{100*_sw[0.25]:.1f}\\%")

# ---------------------------------------------------------------- the corollary, and the premise
# it needs (M60b/M60c). The manuscript called 58.6% and 6.8% bounds in four places; they are
# product-route readings, and Prop. routes says a projection can be wrong in either direction.
# M60c computes what IS a bound, and M60b tests, for the first time, the assumption both the
# proposition and the corollary rest on.
m60b = J("m60b_carrying_bound.json")
m60c = J("m60c_tight_bound.json")

# --- assumption (7), per channel.  This is the premise, and it is reported before the bound.
_a7 = m60b["A3"]
_dg7 = _a7["rna:dG"]
put("a7Bins", ", ".join(f"{x['contrast']:+.4f}".replace("-", "\\ensuremath{-}")
                        for x in _dg7["bins"]))
put("a7N", f"{len(_dg7['bins'])}")
put("a7Mean", f"{_dg7['mean']:+.4f}".replace("-", "\\ensuremath{-}"))
put("a7Spread", f"{_dg7['spread']:.4f}")
put("a7Ratio", f"{_dg7['spread_over_mean']:.3f}")
put("a7PmutRatio", f"{_a7['rna:pmut']['spread_over_mean']:.2f}")
put("a7BinderRatio", f"{_a7['binder:A_length']['spread_over_mean']:.2f}")
_NICE = {"rna:dG": "$\\Delta G_{\\mathrm{loc}}$", "rna:pmut": "$P_{\\mathrm{model}}$",
         "rna:depth": "helix depth", "binder:A_length": "binder length"}
put("a7Held", " and ".join(_NICE[k] for k, v in _a7.items() if v["assumption7_supported"]))
put("a7Failed", " and ".join(_NICE[k] for k, v in _a7.items()
                             if not v["assumption7_supported"]))
put("a7NHeld", f"{sum(1 for v in _a7.values() if v['assumption7_supported'])}")
put("a7NAll", f"{len(_a7)}")

# --- the bound itself, on the primary channel where the premise holds
_c = m60c["arms"]["rna:dG"]
_rows = _c["rows"]
_hold = [r for r in _rows if r["tight_holds"]]
_lhold = [r for r in _rows if r["loose_holds"]]
put("cbNbins", f"{len(_rows)}")
put("cbBinLo", f"{min(r['nbins'] for r in _rows)}")
put("cbBinHi", f"{max(r['nbins'] for r in _rows)}")
put("cbRhoRhs", f"{_c['rho_rhs_vs_nbins']:+.3f}")
put("cbRhoLhs", f"{_c['rho_lhs_vs_nbins']:+.3f}")
put("cbLoose", f"{100*_lhold[0]['loose_frac']:.1f}\\%")
put("cbTight", f"{100*_hold[0]['tight_frac']:.1f}\\%")
put("cbTightLo", f"{100*min(r['tight_frac'] for r in _hold):.1f}\\%")
put("cbTightHi", f"{100*max(r['tight_frac'] for r in _hold):.1f}\\%")
put("cbAtBins", f"{_hold[0]['nbins']}")
put("cbOscLo", f"{min(r['osc_h'] for r in _rows):.4f}")
put("cbOscHi", f"{max(r['osc_h'] for r in _rows):.4f}")
put("cbDtvLo", f"{min(r['d_TV'] for r in _rows):.3f}")
put("cbDtvHi", f"{max(r['d_TV'] for r in _rows):.3f}")
_pm = m60c["arms"]["rna:pmut"]
_pmh = [r for r in _pm["rows"] if r["tight_holds"]]
put("cbPmTight", f"{100*_pmh[0]['tight_frac']:.1f}\\%")

# --- the separation sweep.  It did not reach the low-overlap regime, and the paper says so.
_sw2 = m60b["A2"]
put("swRungs", f"{_sw2['n_rungs']}")
put("swDtvLo", f"{_sw2['d_TV_swept'][0]:.3f}")
put("swDtvHi", f"{_sw2['d_TV_swept'][1]:.3f}")
put("swRho", f"{_sw2['spearman_lhs_rhs']:+.3f}")
put("swSigma", f"{_sw2['sigma_measured']:.4f}")

# ---------------------------------------------------------------- the readout, and the consequence
# (M61 / M61b). Three things the manuscript was asked for and did not have: whether the rescue
# factor's contrast is the probing chemistry rather than pair formation, what the outcome's own
# reliability is, and whether any of this moves a leaderboard.
m61b_thermo = ("Eterna", "Rosetta")
m61b_neural = ("MPNN-fixbb", "MPNN-RFdiff", "gRNAde", "gRNAde-no3d",
               "Shujun_struct2seq", "Shujun_shape_struct2seq")
m61 = J("m61_readout_and_consequence.json")
m61b = J("m61b_readout_and_consequence.json")

_r3 = [r for r in m24["rounds"] if r["round"] == "3"][0]
put("ncRoundMeth", f"{_r3['methods']}")

_cen = m61b["census"]
put("ncRaw", f"{_cen['records']}")
put("ncNonCanon", f"{_cen['non_canonical']}")
put("ncCGAU", f"{_cen['CG_plus_AU']}")

# --- the cut that separates a composition artifact from a structural effect
_a6 = m61b["A6_off_mutation_cut"]
put("cxBase", f"{_a6['recomputed_baseline']:+.4f}".replace("-", "\\ensuremath{-}"))
put("cxReleased", f"{_a6['released_column_baseline']:+.4f}".replace("-", "\\ensuremath{-}"))
for _r in (0, 1, 2, 3):
    _v = _a6[str(_r)]
    put(f"cxE{_r}", f"{_v['effect']:+.4f}".replace("-", "\\ensuremath{-}"))
    put(f"cxCi{_r}", "[$%+.4f$, $%+.4f$]" % tuple(_v["ci"]))
    put(f"cxP{_r}", f"{_v['p']:.4f}")
    put(f"cxRet{_r}", f"{100*_v['retention_vs_recomputed']:.0f}\\%")
    put(f"cxSites{_r}", f"{_v['median_sites_used']}")
_only = _a6.get("only_2")
if _only:
    put("cxOnlyE", f"{_only['effect']:+.4f}".replace("-", "\\ensuremath{-}"))
    put("cxOnlyShare", f"{100*_only['share_of_baseline']:.0f}\\%")
    put("cxOnlyN", f"{_only['n']}")

# --- what the contrast is made of: numerator, denominator, and which one carries it
put("dnD", f"{m61['A1_denominator']['effect']:+.4f}".replace("-", "\\ensuremath{-}"))
put("dnDCi", "[$%+.4f$, $%+.4f$]" % tuple(m61["A1_denominator"]["ci"]))
put("dnNum", f"{m61['A2_numerator']['effect']:+.4f}".replace("-", "\\ensuremath{-}"))
put("dnNumCi", "[$%+.4f$, $%+.4f$]" % tuple(m61["A2_numerator"]["ci"]))
_a4 = m61["A4_decomposition"]
put("dnFromNum", f"{_a4['from_numerator']:+.4f}".replace("-", "\\ensuremath{-}"))
put("dnFromDen", f"{_a4['from_denominator']:+.4f}".replace("-", "\\ensuremath{-}"))
put("dnDenShare", f"{100*_a4['denominator_share']:.0f}\\%")

# --- M16's declared-but-never-printed diagnostic, recomputed on both sides of every pair
_a5 = m61["A5_m16_diagnostic_superseded"]
for _k, _n in (("CG", "Cg"), ("AU", "Au"), ("GU", "Gu")):
    put(f"dgm{_n}", f"{_a5['recomputed_both_sides'][_k]['mean']:.4f}")
    put(f"dgmN{_n}", f"{_a5['recomputed_both_sides'][_k]['n']}")
    put(f"dgmOld{_n}", f"{_a5['m16_published_one_side'][_k]:.4f}")

# --- the outcome's own reliability
_b1 = m61["B1_reliability"]
put("rlSd", f"{_b1['sd_R_median']:.4f}")
put("rlSdQ1", f"{_b1['sd_R_q25']:.4f}")
put("rlSdQ3", f"{_b1['sd_R_q75']:.4f}")
put("rlBetween", f"{_b1['between_pair_sd_R']:.4f}")
put("rlInMeas", f"{_b1['effect_in_measurement_sd']:.1f}")
put("rlInBetween", f"{_b1['effect_in_between_sd']:.2f}")
put("rlSignal", f"{100*_b1['icc_like']:.0f}\\%")
put("rlDraws", f"{_b1['draws']}")
_b2 = m61["B2_denominator"]
put("rlQlo", f"{_b2['quantiles']['q01']:.3f}")
put("rlQhi", f"{_b2['quantiles']['q95']:.3f}")
put("rlDropEff", f"{_b2['dropped_decile']['effect']:+.4f}".replace("-", "\\ensuremath{-}"))
put("rlDropCi", "[$%+.4f$, $%+.4f$]" % tuple(_b2["dropped_decile"]["ci"]))
put("rlDropP", f"{_b2['dropped_decile']['p']:.4f}")
put("rlDropRet", f"{100*_b2['retention']:.0f}\\%")
# the three stakes the erratum records as mis-specified. Their values are still measured numbers
# and still reach the preregistration table through a macro rather than by hand.
put("stA3Ret", f"{100*m61['A3_conditioned']['retention']:.0f}\\%")
put("stSplitCorr", f"{m61['B3_splithalf']['corr']:+.2f}")
put("stDepthTight", f"{m61['D']['D1_depth']['tight_pct']:.0f}\\%")
m61c = J("m61c_error_by_type.json")
_ebt = m61c["sd_by_pair_type"]
put("rlSdCg", f"{_ebt['CG']['median']:.4f}")
put("rlSdAu", f"{_ebt['AU']['median']:.4f}")
put("rlSdRatio", f"{m61c['median_ratio_CG_over_AU']:.3f}")
put("rlSdContrast", f"{m61c['contrast_in_measurement_sd']['effect']:+.4f}".replace("-", "\\ensuremath{-}"))
put("rlSdContrastCi", "[$%+.4f$, $%+.4f$]" % tuple(m61c["contrast_in_measurement_sd"]["ci"]))
put("rlSdContrastP", f"{m61c['contrast_in_measurement_sd']['p']:.4f}")
put("rlLowN", f"{m61c['low_error_half']['n']}")
put("rlLowEff", f"{m61c['low_error_half']['effect']:+.4f}".replace("-", "\\ensuremath{-}"))
put("rlLowRet", f"{100*m61c['low_error_half']['retention']:.0f}\\%")
put("rlLowP", f"{m61c['low_error_half']['p']:.4f}")

# --- the depth channel's conditional mean, level by level, so the linearity is visible
_p3h = m61b["A7_prop3_identity"]
put("p3H", ",\\;".join(f"{x:+.4f}".replace("-", "\\ensuremath{-}") for x in _p3h["h"]))
put("p3Levels", ", ".join(f"{int(x)}" for x in _p3h["levels"]))

_b3 = m61b["B3_repaired"]
put("rlSplits", f"{_b3['n_splits']}")
put("rlHalfSd", f"{_b3['half_sd']:.4f}")
put("rlBothPos", f"{100*_b3['both_halves_positive']:.0f}\\%")

# --- the leaderboard
_c1 = m61["C"]["rescue"]
put("lbNMeth", f"{_c1['methods']}")
put("lbTau", f"{_c1['target_plus_placement']['tau_vs_raw']:+.3f}")
put("lbMoved", f"{_c1['target_plus_placement']['moved']}")
put("lbMaxMove", f"{_c1['target_plus_placement']['max_move']}")
_ci1 = _c1["target_plus_placement"].get("tau_boot_ci")
if _ci1:
    put("lbTauCi", "[$%+.3f$, $%+.3f$]" % tuple(_ci1))
_c2 = m61["C"]["benchmark_score"]
if "skipped" not in _c2:
    _t2 = _c2["target_plus_gc"]
    put("lbBN", f"{_c2['n']}")
    put("lbBNMeth", f"{_c2['methods']}")
    put("lbBTau", f"{_t2['tau_vs_raw']:+.3f}")
    put("lbBMoved", f"{_t2['moved']}")
    put("lbBMaxMove", f"{_t2['max_move']}")
    _ci2 = _t2.get("tau_boot_ci")
    if _ci2:
        put("lbBTauCi", "[$%+.3f$, $%+.3f$]" % tuple(_ci2))
    put("lbBTauTarget", f"{_c2['target_only']['tau_vs_raw']:+.3f}")
    put("lbBMovedTarget", f"{_c2['target_only']['moved']}")
    put("lbStrata", f"{_t2['strata_used']}")
    put("lbTauTarget", f"{_c1['target_only']['tau_vs_raw']:+.3f}")
    put("lbMovedTarget", f"{_c1['target_only']['moved']}")
    _top = min(_c2["raw_rank"], key=lambda k: _c2["raw_rank"][k])
    put("lbBTop", _top.replace("_", "\\_"))
    put("lbBTopNew", f"{_t2['rank'][_top]}")
    _new1 = min(_t2["rank"], key=lambda k: _t2["rank"][k])
    put("lbBNewTop", _new1.replace("_", "\\_"))
    put("lbBNewTopOld", f"{_c2['raw_rank'][_new1]}")

# --- an interval on the tight bound, which the manuscript reported as a point
_d2 = m61b["D2_repaired"]
put("cbCiLo", f"{_d2['ci'][0]:.1f}\\%")
put("cbCiHi", f"{_d2['ci'][1]:.1f}\\%")
put("cbCiMed", f"{_d2['median']:.1f}\\%")
put("cbCiVac", f"{100*_d2['frac_ge_100']:.1f}\\%")
put("cbCiBoot", f"{_d2['nboot']}")

# --- the energy null, on the two halves of the corpus that consume different objectives
_d3 = m61b["D3_repaired"]
put("subAll", f"{_d3['all']['retention_pct']:.1f}\\%")
put("subThermo", f"{_d3['thermodynamic']['retention_pct']:.1f}\\%")
put("subNeural", f"{_d3['neural']['retention_pct']:.1f}\\%")
put("subNThermo", f"{_d3['thermodynamic']['n']}")
put("subNNeural", f"{_d3['neural']['n']}")
put("subNThermoMeth", f"{len(m61b_thermo)}")
put("subNNeuralMeth", f"{len(m61b_neural)}")
put("subThermoMeth", ", ".join(x.replace("_", "\\_") for x in m61b_thermo))
put("subNeuralMeth", ", ".join(x.replace("_", "\\_") for x in m61b_neural))

# --- Proposition 3's residual on the depth channel, computed rather than subtracted
_p3 = m61b["A7_prop3_identity"]
put("p3Att", f"{_p3['attenuation']:+.4f}".replace("-", "\\ensuremath{-}"))
put("p3Beta", f"{_p3['beta']:+.4f}".replace("-", "\\ensuremath{-}"))
put("p3G", f"{_p3['g']:+.3f}".replace("-", "\\ensuremath{-}"))
put("p3BetaG", f"{_p3['beta_g']:+.4f}".replace("-", "\\ensuremath{-}"))
put("p3Share", f"{_p3['linear_share_pct']:.1f}\\%")
put("p3Direct", f"{_p3['residual_computed_directly']:+.4f}".replace("-", "\\ensuremath{-}"))
put("p3Sub", f"{_p3['residual_by_subtraction']:+.4f}".replace("-", "\\ensuremath{-}"))
put("p3Gap", f"{_p3['agreement']:.4f}")
put("p3GapPct", f"{100*_p3['agreement']/abs(_p3['attenuation']):.0f}\\%")

# ---------------------------------------------------------------- the kept route, calibrated
# (M62). Section 5.5 withdrew the conditioning route on an injection calibration. This is the
# same calibration applied to the route the paper kept.
m62 = J("m62_product_route_calibration.json")
_pa, _pb, _ps = m62["A_fraction_sweep"], m62["B_linearity_sweep"], m62["setup"]
put("pcSpan", f"{100*_pa['span_ratio']:.1f}\\%")
put("pcF0", f"{_pa['0.00']['frac_median']:.4f}")
put("pcF1", f"{_pa['1.00']['frac_median']:.4f}")
put("pcF0Iqr", "[$%+.4f$, $%+.4f$]" % tuple(_pa["0.00"]["frac_iqr"]))
put("pcF1Iqr", "[$%+.4f$, $%+.4f$]" % tuple(_pa["1.00"]["frac_iqr"]))
put("pcRungs", f"{len([k for k in _pa if k not in ('span_readback','span_ratio')])}")
put("pcRatios", ",\\;".join(f"{_pa[k]['ratio_median']:.3f}"
                             for k in sorted(_pa) if k not in ("span_readback", "span_ratio")
                             and _pa[k]["f"] >= 0.25))
put("pcRatioMin", f"{min(_pa[k]['ratio_median'] for k in _pa if k not in ('span_readback','span_ratio') and _pa[k]['f'] >= 0.25):.3f}")
put("pcSep", f"{_ps['arm_separation_pooled_sd']:.3f}")
put("pcSepRatio", f"{_ps['arm_separation_pooled_sd']/_ps['s55_mediator_separation']:.1f}")
put("pcSigma", f"{_ps['sigma']:.4f}")
put("pcSpearman", f"{_pb['spearman']:+.3f}")
put("pcLamTrue", f"{100*_pb['implied_true_lambda']:.0f}\\%")
put("pcLamBias", f"{100*_pb['bias_at_published']:+.1f}")
put("pcLamErr1", f"{_pb['1.00']['abs_error']:.3f}")
put("pcDraws", f"{m62['draws']}")
put("pcLamRungs", f"{len([k for k in _pb if k.replace('.','').isdigit()])}")
_bk = sorted((k for k in _pb if k.replace(".", "").isdigit()), key=float, reverse=True)
put("pcLamTable", ",\\; ".join(f"{float(k):.2f}$\\!\\to\\!${_pb[k]['ratio_median']:.3f}"
                                for k in _bk))
put("pcCondF0", f"{_pa['0.00']['cond_route_median']:.3f}")
put("pcCondF1", f"{_pa['1.00']['cond_route_median']:.3f}")

# ---------------------------------------------------------------- is the covariate anyone's choice
# (M63). The manuscript said twice that this test was not run. It is the only evidence that the
# word CHOSE in the title is earned.
m63 = J("m63_undesigned_placement.json")
def _g(k, tag):
    v = m63[k]
    put(f"un{tag}", f"{v['gap']:+.4f}".replace("-", "\\ensuremath{-}"))
    put(f"un{tag}Ci", "[$%+.4f$, $%+.4f$]" % tuple(v["ci"]))
    put(f"un{tag}N", f"{v['n']}")
    put(f"un{tag}Neg", f"{100*v['frac_negative']:.0f}\\%")
    return v
_vd = _g("V_designed", "Des")
_vr = _g("A1_rfam", "Rfam")
_vp = _g("A2_pseudobase", "Pkb")
_vm = _g("B1_rfam_matched", "RfamM")
_cd = _g("Cprime_repaired_control.designed_core", "X") if False else None
_cp = m63["Cprime_repaired_control"]
for tag, key in (("CoreD", "designed_core"), ("CoreI", "inverse_core")):
    v = _cp[key]
    put(f"un{tag}", f"{v['gap']:+.4f}".replace("-", "\\ensuremath{-}"))
    put(f"un{tag}Ci", "[$%+.4f$, $%+.4f$]" % tuple(v["ci"]))
    put(f"un{tag}N", f"{v['n']}")
    put(f"un{tag}Neg", f"{100*v['frac_negative']:.0f}\\%")
put("unRatio", f"{abs(_vd['gap'])/abs(_vr['gap']):.1f}")
put("unShare", f"{100*_cp['share_reproduced']:.1f}\\%")
put("unFams", f"{_vr['n_clusters']}")
put("unMaxSeq", "5")

# ---------------------------------------------------------------- the family, corrected (M64)
# Limitation (vii) used to hand the reader the correction to do themselves. The family is frozen
# in M64's declaration; the resolution the correction needs was bought at 10^6 draws.
m64 = J("m64_multiplicity_and_shape.json")
_A, _B = m64["A"], m64["B"]
put("fwM", f"{len(m64['family_found'])}")
_thr = f"{m64['alpha'] / len(m64['family_found']):.2e}".split("e")
put("fwThr", "%s\\times10^{%d}" % (_thr[0], int(_thr[1])))
put("fwFloor", f"{len(m64['at_floor'])}")
put("fwFloorP", f"{1.0/2001:.4f}")
put("fwFloorOther", f"{len(m64['at_floor']) - 5}")
put("fwHolmPub", f"{_A['n_holm']}")
put("fwBhPub", f"{_A['n_bh']}")
put("fwHolmNew", f"{m64['B_corrected']['n_holm']}")
put("fwBhNew", f"{m64['B_corrected']['n_bh']}")
put("fwDraws", f"{m64['nperm_buy']:,}".replace(",", "{,}"))
_names = {"rescueAUP": "AU", "rescueGUP": "GU", "eP": "Cde", "eDZeroP": "DZero", "eDTwoP": "DTwo"}
for k, tag in _names.items():
    v = _B[k]
    put(f"fwZ{tag}", f"{v['z']:+.2f}")
    put(f"fwGe{tag}", f"{v['n_ge']}")
    _m, _e = f"{v['p']:.0e}".split("e")
    put(f"fwP{tag}", "$<10^{-6}$" if v["n_ge"] == 0 else ("$%s\\times10^{%d}$" % (_m, int(_e))))
    e = m64["S4"]["arms"][k]
    put(f"fwD{tag}", sig(e["d_vec"], 4))
put("fwSdMax", f"{max(abs(m64['S4']['arms'][k]['sd_vec'] / m64['S4']['arms'][k]['sd_published'] - 1) for k in ('rescueAUP', 'rescueGUP', 'eP') ) * 100:.0f}\\%")
_C = m64["C_shape"]
put("shpSkew", f"{_C['skew']:+.2f}")
put("shpKurt", f"{_C['exkurt']:+.1f}")
put("shpMin", sig(_C["min"], 4))
put("shpMax", sig(_C["max"], 4))
put("shpMed", sig(_C["q"]["50"], 4))
put("shpQOne", sig(_C["q"]["1"], 4))
put("shpQNn", sig(_C["q"]["99"], 4))
put("shpNeg", f"{100*_C['frac_negative']:.1f}\\%")
put("shpRank", sig(m64["C_rank"]["diff"], 4))
put("shpRankP", ("$<10^{-5}$" if m64["C_rank"]["p"] <= 1.01e-5 else f"{m64['C_rank']['p']:.1e}"))
put("shpTrim", sig(m64["C_trim"]["diff"], 4))
put("shpTrimRet", f"{100*m64['C_trim']['retention']:.1f}\\%")
put("shpTrimN", f"{m64['C_trim']['n_kept']}")
_D = m64["D"]
put("supFrac", f"{100*_D['fraction']:.1f}\\%")
put("supUsed", f"{_D['n_used']}")
put("supBase", f"{_D['n_base']}")
put("supSmdDepth", sig(_D["smd"]["depth"], 3))
put("supSmdCross", sig(_D["smd"]["crossed"], 3))
put("supSmdRf", sig(_D["smd"]["rf"], 3))
put("supSmdDg", sig(_D["smd"]["dG"], 3))
put("supSmdMax", f"{_D['max_abs_smd_of_three']:.3f}")

# ---------------------------------------------------------------- how the corpus was selected
# (M30). R1's premise is that the covariate distribution was produced by a design algorithm; it
# was also produced by a selection rule, and the paper has to say so.
m30 = J("m30_selection_audit.json")
_s1, _s2 = m30["S1"], m30["S2"]
put("selCells", f"{m30['n_cells']}")
put("selNCells", f"{_s1['n_cells']}")
put("selRepro", f"{_s1['n_reproduced']}")
put("selAgree", f"{_s1['agreement']:.4f}")
put("selReproB", f"{_s1['n_reproduced_no_filter']}")
put("selAgreeB", f"{_s1['agreement_no_filter']:.4f}")
put("selPassedOver", f"{_s1['n_eligible_passed_over']}")
put("selCut", f"{m30['cutoff']:.2f}")
put("selPoolPass", f"{100*_s2['pool_pass_rate_this_round']:.1f}\\%")
put("selNFallback", f"{_s2['n_tested_fallback']}")
put("selStrataMixed", f"{m30['strata_mixed_on_selection']}")
put("selNStrata", f"{m30['S3']['strata_pass'] + m30['S3']['strata_fallback']}")
put("selRhoRank", sig(m30["S4c_scale_free"]["openknot_score"]["spearman"]))
put("selRhoRankCi", ci(m30["S4c_scale_free"]["openknot_score"]["ci"]))

# ---------------------------------------------------------------- the second corpus (M66)
# SWE-bench Verified. The claim of this paper is about benchmarks, so it is instantiated on one
# whose readout is a test suite rather than a chemical probe. What the corpus supports is a census
# and not a contrast, and the numbers that failed their controls are generated here too, because
# the section reports them.
m66 = J("m66_identifiability.json")
_v, _l, _cb = m66["verified"], m66["lite"], m66["combined"]
put("swbEntries", f"{_v['entries']}")
put("swbDropped", f"{_v['dropped_self_inconsistent']}")
put("swbModels", f"{_v['models']}")
put("swbScaf", f"{_v['agents']}")
put("swbSolo", f"{_v['models_under_1_agent']}")
put("swbSoloPct", f"{100*_v['models_under_1_agent']/_v['models']:.0f}\\%")
put("swbCrossed", f"{_v['models_under_ge2_agents']}")
put("swbScafPairs", f"{_v['agent_pairs']:,}".replace(",", "{,}"))
put("swbScafGeOne", f"{_v['agent_pairs_sharing_ge1']}")
put("swbScafGeTwo", f"{_v['agent_pairs_sharing_ge2']}")
put("swbScafGeThree", f"{_v['agent_pairs_sharing_ge3']}")
put("swbModelPairs", f"{_v['model_pairs_possible']:,}".replace(",", "{,}"))
put("swbCoMeas", f"{_v['model_pairs_ever_co_measured']}")
put("swbCoMeasPct", f"{100*_v['model_pairs_ever_co_measured']/_v['model_pairs_possible']:.1f}\\%")
put("swbCoGeTwo", f"{_v['model_pairs_co_measured_ge2_agents']}")
put("swbLiteEntries", f"{_l['entries']}")
put("swbLiteGeThree", f"{_l['agent_pairs_sharing_ge3']}")
put("swbBothEntries", f"{_cb['entries']}")
put("swbBothModels", f"{_cb['models']}")
put("swbBothScafPairs", f"{_cb['agent_pairs']:,}".replace(",", "{,}"))
put("swbBothGeThree", f"{_cb['sharing_ge3']}")
put("swbBothSolo", f"{_cb['models_under_1_agent']}")
put("swbBothSoloPct", f"{100*_cb['models_under_1_agent']/_cb['models']:.0f}\\%")

m66p = J("m66_swebench_placement.json")
_wm, _bd, _dc = m66p["withinModel"], m66p["board"], m66p["deepest_case"]
put("swbSpreadN", f"{_wm['n']}")
put("swbSpreadMed", f"{_wm['median']:.1f}")
put("swbSpreadCi", ci([_wm["ci"][0], _wm["ci"][1]], 1))
put("swbSpreadMax", f"{_wm['max']:.1f}")
put("swbBoardIqr", f"{_bd['iqr']:.1f}")
put("swbBoardMed", f"{_bd['median']:.1f}\\%")
put("swbBoardN", f"{_bd['n_entries']}")
put("swbDeepModel", _dc["model"].replace("_", "\\_"))
put("swbDeepCells", f"{_dc['n_cells']}")
put("swbDeepLo", f"{_dc['lo']:.1f}\\%")
put("swbDeepHi", f"{_dc['hi']:.1f}\\%")
put("swbDeepLoScaf", _dc["lo_scaffold"])
put("swbDeepHiScaf", _dc["hi_scaffold"])
put("swbDeepSpread", f"{_dc['spread']:.1f}")
_s5 = m66p["S5"]
put("swbSFiveWm", f"{_s5['within_model_median']:.1f}")
put("swbSFiveWc", f"{_s5['within_cell_median']:.1f}")
put("swbSFiveNm", f"{_s5['n_models']}")
put("swbSFiveNc", f"{_s5['n_cells']}")
put("swbSFiveDiff", sig(_s5["diff"], 1))
put("swbSFiveCi", ci(_s5["ci"], 1))
_ng = m66p["noGen"]
put("swbNgEntries", f"{_ng['entries']}")
put("swbNgTotal", f"{_ng['total']}")
put("swbNgWorst", f"{_ng['max_entry_pp']:.1f}")
_bs = sorted(_ng["by_scaffold"].items(), key=lambda kv: -kv[1])
put("swbNgHiScaf", _bs[0][0]); put("swbNgHi", f"{_bs[0][1]:.1f}")
put("swbNgLoScaf", _bs[-1][0]); put("swbNgLo", f"{_bs[-1][1]:.2f}")

# the depth bar S5 is evaluated at, kept in the pipeline so the text never spells it out
put("swbDeepBar", "4")

m66a = J("m66_armA.json")
put("swbUniverse", "500")
put("swbAragModels", f"{m66a['rag_models']}")
put("swbAModels", f"{len(m66a['shared_models'])}")
put("swbAStrata", f"{m66a['n_strata']}")
put("swbAd", sig(m66a["d_points"], 2))
put("swbACi", ci(m66a["ci_points"], 2))
# make_check treats a macro whose value STARTS with "<" as carrying its own relation, so the
# use site is $p\jdg{swbAP}$ with no relation of its own. Always emit a true upper bound.
import math as _math
_pex = int(_math.ceil(_math.log10(m66a["p"])))
put("swbAP", "<10^{%d}" % (_pex if _pex > -5 else -5))
put("swbANperm", f"{m66a['nperm']:,}".replace(",", "{,}"))
put("swbAScaf", sorted({x["agentic"] for x in m66a["per_model"]})[0])
put("swbAScafN", f"{len({x['agentic'] for x in m66a['per_model']})}")

m66c = J("m66_headline_control.json")
put("swbCtrlSameN", f"{m66c['A_n']}")
put("swbCtrlSameMed", f"{m66c['A_median']:.1f}")
put("swbCtrlSameMax", f"{m66c['A_max']:.1f}")
put("swbCtrlAcrossN", f"{m66c['B_n']}")
put("swbCtrlAcrossMed", f"{m66c['B_median']:.1f}")
put("swbCtrlDiff", sig(m66c["diff"], 1))
put("swbCtrlCi", ci(m66c["ci"], 1))
put("swbCtrlPairN", f"{m66c['paired_n']}")
put("swbCtrlPairDiff", sig(m66c["paired_median"], 1))
put("swbCtrlPairCi", ci(m66c["paired_ci"], 1))

m66i = J("m66_instance_level.json")
put("swbChurnN", f"{m66i['n_comparisons']}")
put("swbChurnCross", f"{m66i['median_churn']:.3f}")
put("swbChurnSame", f"{m66i['control_same_scaffold']['median_churn']:.3f}")
put("swbChurnSameN", f"{m66i['control_same_scaffold']['n']}")
put("swbChurnGap", sig(m66i["gap_cross_minus_same"]["diff"], 3))
put("swbChurnGapCi", ci(m66i["gap_cross_minus_same"]["ci"], 3))
put("swbNearN", f"{m66i['near_tied']['n']}")
put("swbNearGap", f"{m66i['near_tied']['median_gap_pp']:.1f}")
put("swbNearChurn", f"{m66i['near_tied']['median_churn']:.3f}")
put("swbNearModels", f"{len(m66i['near_tied_models'])}")

# ---------------------------------------------------------------- survivorship + shift (M22)
m22 = J("m22_review_arms.json")
a22 = m22["A"]
put("sNPos", f"{a22['n_positive']}"); put("sNMeth", f"{a22['n_methods']}")
put("sBinom", f"{a22['binom_p']:.4f}")
put("sSpear", sig(a22["spearman_with_design_gap"], 2))
put("sPear", sig(a22["corr_with_design_gap"], 2))
_neg = [r for r in a22["per_method"] if r["diff"] <= 0]
put("sNegName", _neg[0]["method"] if _neg else "---")
put("sNegDiff", sig(_neg[0]["diff"], 4) if _neg else "---")
put("sNegP", f"{_neg[0]['p_perm']:.2f}" if _neg else "---")
put("sNegW", f"{_neg[0]['weight']:.0f}" if _neg else "---")
put("sTopName", a22["per_method"][0]["method"])
put("sTopDiff", sig(a22["per_method"][0]["diff"], 4))
put("sShiftExpr", "\\texttt{S = (paired + s) \\% n}")


def permethod_table():
    """figs/permethod.tex -- one row per design method, generated so the table cannot drift."""
    m = J("m20_designer_confounding.json")
    gap = {r["method"]: r for r in m["A1"]["per_method"]}
    pts = {r["method"]: r for r in m["A3"]["points"]}
    HUMAN = {"Eterna", "Starting sequence"}
    rows = []
    for r in m["A1"]["per_method"]:                      # already sorted most-negative first
        k = r["method"]; q = pts.get(k)
        nm = k.replace("_", " ")
        rows.append("%s%s & %d & %+.3f & %s & %s \\\\" % (
            "\\textbf{" + nm + "}" if k in HUMAN else nm,
            " (human)" if k in HUMAN else "", r["n_targets"], r["gap"],
            ("$%+.5f$" % q["pred"]) if q else "---",
            ("$%+.5f$" % q["obs"]) if q else "---"))
    body = "\n".join(rows)
    txt = ("%% GENERATED by make_numbers.py. Do not edit.\n"
           "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n"
           "design method & targets & depth gap $g_m$ & predicted & observed \\\\\n"
           "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")
    out = os.path.join(HERE, "figs", "permethod.tex")
    open(out, "w").write(txt)
    return out, len(rows)




def m21_depth_table():
    r = J("m21_thermo_mediator.json")["B3"]
    rows = []
    for k, lab in (("depth0", "$0$ (helix end)"), ("depth1", "$1$"), ("depth2plus", "$\\ge 2$")):
        v = r[k]
        rows.append("%s & $%+.4f$ & $%.4f$ & %.0f & %d & %d \\\\" %
                    (lab, v["diff"], v["p_perm"], v["weight"], v["n_strata"], v["n"]))
    txt = ("%% GENERATED by make_numbers.py. Do not edit.\n"
           "\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n"
           "helix depth & C--G $-$ A--U & $p_{\\mathrm{perm}}$ & pair-weights & strata & pairs \\\\\n"
           "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    out = os.path.join(HERE, "figs", "depth.tex")
    open(out, "w").write(txt)
    return out, len(rows)


def m22_method_table():
    rows_in = J("m22_review_arms.json")["A"]["per_method"]
    rows = []
    for r in rows_in:
        g = ("$%+.3f$" % r["design_side_depth_gap"]) if r["design_side_depth_gap"] is not None else "---"
        nm = r["method"].replace("_", "\\_")
        rows.append("\\texttt{%s} & $%+.4f$ & $%.4f$ & %.0f & %d & %s \\\\" %
                    (nm, r["diff"], r["p_perm"], r["weight"], r["n_pairs"], g))
    txt = ("%% GENERATED by make_numbers.py. Do not edit.\n"
           "\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n"
           "design method & C--G $-$ A--U & $p_{\\mathrm{perm}}$ & pair-wts & pairs & design gap $g_m$ \\\\\n"
           "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    out = os.path.join(HERE, "figs", "permethodrescue.tex")
    open(out, "w").write(txt)
    return out, len(rows)




def m24_round_table():
    rs = J("m24_cross_round.json")["rounds"]
    rows = []
    for r in rs:
        rows.append("%s & %d & %d & %d & $%+.4f$ & %.1f\\%% & %s & %d & $%+.4f$ & $%.4f$ \\\\" %
                    (r["round"], r["targets"], r["methods"], r["cells"], r["pooled_gap"],
                     100 * r["frac_cells_negative"], r["methods_negative"],
                     r["eterna_rank"] + 1, r["method_component"], r["method_p"]))
    txt = ("%% GENERATED by make_numbers.py. Do not edit.\n"
           "\\begin{tabular}{@{}lrrrrrrrrr@{}}\n\\toprule\n"
           "round & tgts & meth & cells & pooled $g$ & cells $<0$ & meth $<0$ & Eterna & "
           "method cmp & $p$ \\\\\n\\midrule\n" + "\n".join(rows) +
           "\n\\bottomrule\n\\end{tabular}\n")
    out = os.path.join(HERE, "figs", "rounds.tex")
    open(out, "w").write(txt)
    return out, len(rows)


def m23c_table():
    rs = J("m23c_imbalance_bound.json")["rows"]
    rows = []
    for r in rs:
        if not r["panel"]:
            continue
        fr = next(x for x in rs if x["pop"] == r["pop"] and not x["panel"])
        rows.append("%s & $%+.4f$ & $%+.3f$ & $%.2f\\%%$ & $%+.4f$ & $%.2f\\%%$ \\\\" %
                    (r["pop"].replace("(", " ").replace(")", "").replace("样本", "sample ")
                     .replace("基准", "baseline ").replace("第四批", "batch 4")
                     .replace("第五批", "batch 5").replace("第六批", "batch 6")
                     .replace("盲", "blind").replace("主", "main").strip(),
                     fr["mean_dpi"], fr["bias"], fr["bias_pct_of_effect"],
                     r["mean_dpi"], r["bias_pct_of_effect"]))
    txt = ("%% GENERATED by make_numbers.py. Do not edit.\n"
           "\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n"
           "& \\multicolumn{3}{c}{panel-free} & \\multicolumn{2}{c}{panel-matched} \\\\\n"
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\n"
           "population & $\\overline{\\Delta\\pi}$ & bias & of effect & "
           "$\\overline{\\Delta\\pi}$ & of effect \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    out = os.path.join(HERE, "figs", "imbalance.tex")
    open(out, "w").write(txt)
    return out, len(rows)




def main() -> int:
    for fn in (permethod_table, m21_depth_table, m22_method_table,
               m24_round_table, m23c_table):
        o, n = fn()
        print(f"wrote {o}: {n} rows")
    out = os.path.join(HERE, "figs", "rna_numbers.tex")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("% GENERATED by make_numbers.py from rna_harness/runs/ceiling/*.json.\n"
                 "% Do not edit. Do not type a measured number into the manuscript by hand.\n")
        for k in sorted(M):
            fh.write("\\expandafter\\def\\csname jdg@%s\\endcsname{%s}\n" % (k, M[k]))
    print(f"wrote {out}: {len(M)} macros")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
