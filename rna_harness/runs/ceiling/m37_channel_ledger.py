#!/usr/bin/env python3
"""M37: the channel ledger -- who chose the covariate that carries each number, and what it costs.

A SUPPORT artifact. It estimates nothing. Every price below is already published in one of the two
manuscripts; this file collects them from their own committed JSONs, checks each against the macro
the manuscript actually prints, checks the identities that must close, and asks whether the list is
exhaustive. It exists because the two papers were organised by INSTRUMENT and by ESTIMATOR, and
every one of their findings is in fact an instance of one statement:

    on a corpus built by design programs and scored by a predictor, the number a benchmark reports
    is carried by a covariate SOMEBODY CHOSE, not by the property the number is named after.

Three checks, in the order the project's own rule requires -- verify, then judge:

  (1) 对拍 / cross-check. Every price is read from the run artifact AND from the \\jdg{} macro the
      manuscript prints, and the two must agree. A disagreement means the manuscript has drifted
      from its own run, which no amount of restructuring would fix.
  (2) identity. The placement channel is a decomposition and must close:
      observed attenuation == marginal - matched, and mediated fraction == predicted / observed.
  (3) exhaustiveness. Every chooser either has a price or is listed as UNPRICED. A thesis that says
      "name and price every chooser" is refuted by one chooser it cannot name.

COST: $0, offline. Reads committed JSON and the two generated macro files. No estimation.
"""
from __future__ import annotations

import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPERS = "/data/wenhui/openclaude_rna/paper"
NUMS = {"NM": os.path.join(PAPERS, "intervention_benchmark/figs/nm_numbers.tex"),
        "R1": os.path.join(PAPERS, "rna_placement/figs/rna_numbers.tex")}
LOG = []


def log(s=""):
    print(s, flush=True); LOG.append(s)


def J(fn):
    return json.load(open(os.path.join(HERE, fn)))


def macros(path):
    return dict(re.findall(r"\\csname\s+jdg@(\w+)\\endcsname\{(.*?)\}\s*$",
                           open(path).read(), re.M))


def num(s):
    """The numeric value a printed macro carries, or None if it is not a bare number."""
    t = (s.replace("\\ensuremath{-}", "-").replace("\\%", "").replace("\\,", "")
          .replace("$", "").replace("{", "").replace("}", "").strip())
    try:
        return float(t)
    except ValueError:
        return None


def main() -> int:
    M = {k: macros(v) for k, v in NUMS.items()}
    OUT = {"_what_this_is": "a ledger of priced covariate-choice channels; estimates nothing"}
    log("=== M37:通道账本 —— 每个数是被谁选的协变量扛着的 ===\n")

    m20 = J("m20_designer_confounding.json")["A2"]
    m21, m21p = J("m21_thermo_mediator.json")["B1"], J("m21_thermo_mediator.json")["P3"]
    m31 = J("m31_which_flips_rescue.json")["A_within_design"]
    m30 = J("m30_selection_audit.json")["S4c_scale_free"]["openknot_score"]
    m26d = J("m26d_clean_subset.json")["arms"]
    m35 = J("m35_paired_predictor_differences.json")
    m36 = J("m36_readout_scale.json")
    m25 = J("m25_estimator_benchmark.json")["B"]
    m28r = J("m28r_review.json")["R2_arms"]["ViennaRNA"]

    # ---- the ledger. Each row: who chose it, what it is worth, where the number lives, and the
    # macro the manuscript prints so the two can be compared.
    CH = [
        dict(id="C1", who="the model being scored (its structure IS the reference column)",
             paper="NM", carries=True,
             price=f"{m36['standardised']['rescue_contrast_in_sd']:+.4f} SD measured against "
                   f"{m36['standardised']['rnet_contrast_in_sd']:+.4f} SD read as agreement",
             checks=[("cvIntvSd", m36["standardised"]["rescue_contrast_in_sd"]),
                     ("cvConvSd", m36["standardised"]["rnet_contrast_in_sd"]),
                     ("cvSwing", abs(m25["R2_all_target_pairs_NOT_P4"]["diff"]
                                     / m25["R2_inferred"]["diff"]))]),
        dict(id="C2", who="the design programs, through where they put the pair",
             paper="R1", carries=True,
             price=f"marginal {m20['marginal']:+.4f} against matched {m20['matched']:+.4f}; "
                   f"{100 * m20['mediated_fraction']:.1f}% of the gap predicted by beta*g",
             checks=[("medMarginal", m20["marginal"]), ("medMatched", m20["matched"]),
                     ("medObs", m20["observed_attenuation"]),
                     ("medPred", m20["predicted_attenuation"])]),
        dict(id="C3", who="whoever wrote the declared target, through helix depth",
             paper="NM", carries=True,
             price=f"depth alone {m31['depth_baseline']['rho']:+.4f} against the transformer's "
                   f"{m31['P_neural']['rho']:+.4f}",
             checks=[("rkRhoD", m31["depth_baseline"]["rho"]),
                     ("rkRhoN", m31["P_neural"]["rho"])]),
        dict(id="C4", who="the corpus builder, through the rule that selected the designs",
             paper="NM", carries=True, direction="against",
             price=f"skill against the selecting score, Spearman {m30['spearman']:+.4f} "
                   f"{m30['ci']}",
             checks=[("selRhoRank", m30["spearman"])]),
        dict(id="C5", who="the model's authors, through decomposition and through training data",
             paper="NM", carries=True,
             price=f"decomposition {m35['post_hoc_axes']['architecture_axis_neural_minus_eterna']['effect']:+.5f}, "
                   f"training data {m35['post_hoc_axes']['training_data_axis_eterna_minus_vienna']['effect']:+.5f}",
             checks=[("axArch", m35["post_hoc_axes"]["architecture_axis_neural_minus_eterna"]["effect"]),
                     ("axData", m35["post_hoc_axes"]["training_data_axis_eterna_minus_vienna"]["effect"])]),
        dict(id="C6", who="the objective the design loop was told to optimise (the model's own energy)",
             paper="R1", carries=False,
             price=f"retention {100 * m21['retention']:.1f}%, mediated "
                   f"{100 * m21p['mediated_fraction']:+.1f}% "
                   f"{[round(100 * x, 1) for x in m21p['mediated_fraction_ci95']]}, "
                   f"placebo z {m21p['z_vs_placebo']:+.2f}",
             checks=[("eRet", 100 * m21["retention"]), ("eZ", m21p["z_vs_placebo"])]),
        dict(id="C7", who="the model's training set, through window overlap",
             paper="NM", carries=False,
             price=f"skill slope {m26d['RibonanzaNet_all']['slope']:+.5f} against "
                   f"{m26d['RibonanzaNet_clean']['slope']:+.5f} on the excised subset",
             checks=[("clNeuralAll", m26d["RibonanzaNet_all"]["slope"]),
                     ("clNeuralClean", m26d["RibonanzaNet_clean"]["slope"])]),
        dict(id="C8", who="the probe chemistry the depositors chose",
             paper="NM", carries=False,
             price=f"paired rank difference across the two chemistries "
                   f"{m28r['paired_boot_corr_diff']['ci95']}, covering zero",
             checks=[]),
    ]
    UNPRICED = [
        dict(id="U1", who="the experimenters, through WHICH pairs were given a compensatory mutant",
             why="the corpus supplies the mutants; nothing in it says why those pairs and not "
                 "others, and no arm of either paper prices it. Named in NM's Discussion and "
                 "unpriced there."),
        dict(id="U2", who="the released checkpoint's pseudo-label stage",
             why="a sequence can have been seen without its experimental label having been seen; "
                 "an audit built from deposited measurements cannot separate that case. Named in "
                 "NM's leakage scopebox and outside C7's reach."),
    ]

    # ---- check (1): every price must equal the macro the manuscript prints
    log("--- 检查 1:账本上的价格 vs 论文实际印出来的宏(对拍)---")
    drift = []
    for c in CH:
        for mac, val in c["checks"]:
            got = None
            for pk, mm in M.items():
                if mac in mm:
                    got = num(mm[mac]); src = pk
            if got is None:
                drift.append((c["id"], mac, "macro not defined in either paper", val)); continue
            if abs(got - val) > max(1e-3, abs(val) * 0.02):
                drift.append((c["id"], mac, got, val))
    if drift:
        for d in drift:
            log(f"  !! {d[0]} {d[1]}: 论文印 {d[2]}  账本读 {d[3]}")
    else:
        log(f"  {sum(len(c['checks']) for c in CH)} 个价格全部与论文印出来的宏一致")
    OUT["check1_macro_vs_artifact_drift"] = [list(map(str, d)) for d in drift]

    # ---- check (2): the one channel that is a decomposition must close as an identity
    log("\n--- 检查 2:placement 通道是分解,必须作为恒等式闭合 ---")
    att = m20["marginal"] - m20["matched"]
    d_att = abs(att - m20["observed_attenuation"])
    frac = m20["predicted_attenuation"] / m20["observed_attenuation"]
    d_frac = abs(frac - m20["mediated_fraction"])
    log(f"  marginal - matched = {att:+.5f};记录的 observed_attenuation = "
        f"{m20['observed_attenuation']:+.5f};差 {d_att:.2e}")
    log(f"  predicted / observed = {frac:.5f};记录的 mediated_fraction = "
        f"{m20['mediated_fraction']:.5f};差 {d_frac:.2e}")
    OUT["check2_identity"] = {"attenuation_closes": bool(d_att < 1e-4),
                              "fraction_closes": bool(d_frac < 1e-4),
                              "residual_epsilon": round(m20["observed_attenuation"]
                                                        - m20["predicted_attenuation"], 5)}
    log(f"  恒等式闭合:{OUT['check2_identity']['attenuation_closes']} / "
        f"{OUT['check2_identity']['fraction_closes']};"
        f"未闭合的残差 eps = {OUT['check2_identity']['residual_epsilon']:+.5f}")

    # ---- check (3): is the list exhaustive, and is the empty channel really empty
    log("\n--- 检查 3:穷尽性,以及那条空通道是不是真的空 ---")
    empty_ok = (m21p["mediated_fraction_ci95"][0] <= 0 <= m21p["mediated_fraction_ci95"][1]
                and abs(m21p["z_vs_placebo"]) < 2)
    log(f"  C6(模型自己的能量):中介区间含 0 且安慰剂 z 不显著 = {empty_ok}")
    log(f"  已定价通道 {len(CH)},其中承载 {sum(c['carries'] for c in CH)},"
        f"不承载 {sum(not c['carries'] for c in CH)}")
    log(f"  **未定价的选择者 {len(UNPRICED)}**,必须在论文里点名:")
    for u in UNPRICED:
        log(f"    {u['id']}  {u['who']}")
    OUT["check3"] = {"empty_channel_is_empty": bool(empty_ok),
                     "n_priced": len(CH), "n_carrying": sum(c["carries"] for c in CH),
                     "n_unpriced_named": len(UNPRICED)}

    log("\n=== 账本 ===")
    for c in CH:
        mark = "承载" if c["carries"] else "不承载"
        if c.get("direction") == "against":
            mark = "反向承载"
        log(f"  {c['id']}  [{c['paper']}] {mark:6s} {c['who']}")
        log(f"        {c['price']}")
    OUT["channels"] = [{k: v for k, v in c.items() if k != "checks"} for c in CH]
    OUT["unpriced"] = UNPRICED

    ok = not drift and OUT["check2_identity"]["attenuation_closes"] and empty_ok
    OUT["verdict"] = ("ledger_verified_the_unification_is_supported"
                      if ok else "LEDGER_FAILED_do_not_restructure_on_it")
    log(f"\n  {OUT['verdict']}")
    if not ok:
        log("  !! 统一还没有被验证,不许拿它去改论文结构")
    OUT["log"] = LOG
    json.dump(OUT, open(os.path.join(HERE, "m37_channel_ledger.json"), "w"),
              indent=2, ensure_ascii=False)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
