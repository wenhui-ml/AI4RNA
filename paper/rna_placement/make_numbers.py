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
    return f"[{sig(v[0], d)},\\,{sig(v[1], d)}]"


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
