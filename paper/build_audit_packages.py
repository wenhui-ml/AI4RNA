#!/usr/bin/env python3
"""Build the input packages for the independent third-party audit system.

Two folders, one per manuscript. Everything here is GENERATED from the repository, so a package
cannot drift from the paper it describes: every artifact carries a sha256 taken at build time and
every quoted sentence is sliced out of the .tex that is actually compiled.

WHAT THIS PACKAGE IS NOT. It is not an answer sheet. The audit system's own fairness rule is that
the auditor derives its supervision plan itself and recomputes the load-bearing claim with an
INDEPENDENT tool, not by running the audited party's script and not by trusting self-reported
numbers. So the result-track material here is a SPECIFICATION of each estimand -- enough to
reimplement it from the public CSV -- and never our implementation of it. Our numbers appear in
exactly one place, `claims.json:printed_value`, and they appear there as the thing to be checked,
flagged `trust: "DO_NOT_TRUST_RECOMPUTE_INDEPENDENTLY"`.

COST: $0, offline.
"""
from __future__ import annotations

import hashlib, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CEIL = os.path.join(REPO, "rna_harness", "runs", "ceiling")
M2R = "/data/wenhui/openknot_data/Data/OK7a_M2R_data.v4.5.1.csv"
BENCH = "/data/wenhui/openknot_data/Data/OpenKnotBench_data.v4.5.1.csv"


def sha(path, cap=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def stat(path):
    if not os.path.exists(path):
        return {"path": path, "exists": False}
    return {"path": os.path.relpath(path, REPO) if path.startswith(REPO) else path,
            "exists": True, "bytes": os.path.getsize(path), "sha256": sha(path)}


def tracked(path):
    try:
        r = subprocess.run(["git", "ls-files", "--error-unmatch", path], cwd=REPO,
                           capture_output=True)
        return r.returncode == 0
    except Exception:
        return None


def macros(paper_dir, name):
    """every \\jdg{} the manuscript can print, with the value it prints"""
    p = os.path.join(paper_dir, "figs", name)
    out = {}
    for m in re.finditer(r"\\csname\s+jdg@(\w+)\\endcsname\{(.*)\}", open(p).read()):
        out[m.group(1)] = m.group(2)
    return out


# a quoted sentence has to read as it reads on paper. The generic command-stripper below turns
# "T \in \{A,B\}" into "T {A,B}", which changes the statement, so the operators a claim actually
# depends on are mapped to text FIRST.
MATH = [(r"\\mathbb\s*E", "E"), (r"\\operatorname\{osc\}", "osc"), (r"\\in\b", " in "),
        (r"\\le\b", " <= "), (r"\\ge\b", " >= "), (r"\\neq\b", " != "), (r"\\mid\b", " | "),
        (r"\\sum\b", "sum"), (r"\\pm\b", "+/-"), (r"\\cdot\b", "*"), (r"\\to\b", " -> "),
        (r"\\alpha\b", "alpha"), (r"\\beta\b", "beta"), (r"\\Delta\b", "Delta"),
        (r"\\delta\b", "delta"), (r"\\sigma\b", "sigma"), (r"\\rho\b", "rho"),
        (r"\\mathrm\{([^{}]*)\}", r"\1"), (r"\\left|\\right", ""),
        (r"\\d_\{\\mathrm\{TV\}\}", "d_TV"), (r"\\varepsilon\b", "eps"),
        (r"\\{", "{"), (r"\\}", "}"), (r"\\quad|\\qquad", " ")]


def expand(text, mac):
    """substitute \\jdg{x} by its printed value so a quoted sentence reads as it reads on paper"""
    def rep(m):
        return mac.get(m.group(1), "??" + m.group(1) + "??")
    t = re.sub(r"\\jdg\{(\w+)\}", rep, text)
    t = re.sub(r"\\(emph|textbf|texttt)\{([^{}]*)\}", r"\2", t)
    t = re.sub(r"\\ensuremath\{-\}", "-", t)
    for pat, rep in MATH:
        t = re.sub(pat, rep, t)
    t = t.replace("\\,", "").replace("\\%", "%").replace("$\\times$", "x").replace("\\times", "x")
    t = re.sub(r"\\ref\{([^}]*)\}", r"[\1]", t)
    t = re.sub(r"\\S(?=\[)", "section ", t)
    t = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\[a-zA-Z]+", "", t)
    t = t.replace("---", " -- ").replace("~", " ")
    return re.sub(r"\s+", " ", t).strip()


def slice_between(tex, start, end):
    i = tex.index(start); j = tex.index(end, i)
    return tex[i + len(start):j]


def write(folder, name, obj, raw=False):
    os.makedirs(folder, exist_ok=True)
    p = os.path.join(folder, name)
    with open(p, "w", encoding="utf-8") as f:
        if raw:
            f.write(obj)
        else:
            json.dump(obj, f, indent=1, ensure_ascii=False)
    print(f"  wrote {os.path.relpath(p, REPO)}  ({os.path.getsize(p)} B)")
    return p


# =====================================================================================
# The claim ledger. Each entry is ONE load-bearing sentence of the manuscript, the estimand it
# asserts written out in words and symbols, the artifact that would have to be wrong for it to be
# wrong, and -- separately and last -- the number the paper actually prints.
# The `independent_route` field is a SPECIFICATION. It never names one of our functions.
# =====================================================================================
NM_CLAIMS = [
 dict(id="NM-C1", label="claim:readout", axis="readout scale",
      section="Results / the conventional readout on this corpus",
      estimand="Two contrasts on the SAME molecules and the SAME strata, each standardised by its "
               "own within-stratum standard deviation: (a) the interventional contrast, the "
               "matched stratified difference in mean rescue factor between G-C-target and "
               "A-U-target interventions; (b) the conventional contrast, the same matched "
               "stratified difference computed on a structure-agreement column. Claim: (a) is "
               "larger than (b) after standardisation, and (b) additionally moves by a factor when "
               "the set of admitted pairs is changed in a defensible way.",
      falsifier="the two contrasts are not on the same weights/strata; or the agreement column is "
                "not what the paper says it is; or the standardisation is not by within-stratum SD",
      independent_route=[
        "From OK7a_M2R_data.v4.5.1.csv alone, rebuild the admitted set with the criterion in "
        "criterion.md, then compute the matched stratified difference in means with weight "
        "w_s = min(n_A(s), n_B(s)) over strata s = (design cell, min(depth,3), crossing).",
        "From OpenKnotBench_data.v4.5.1.csv, take the agreement column the paper names in "
        "claims.json:reference_column_disclosure and repeat on identical weights and strata.",
        "Standardise each by the pooled within-stratum SD of ITS OWN outcome column before "
        "comparing magnitudes. Comparing the raw contrasts is the error this claim exists to "
        "correct; a recomputation that compares them raw has not reproduced the claim.",
        "The claim is now made against FIVE agreement columns, not one. Four are the panel "
        "implementations' own structures; the fifth is folded from this deposit's own chemical "
        "probing and contains no sequence-to-structure model. Reproducing the claim on the "
        "shipped column alone reproduces the WEAKEST of the five gaps, which is the one the "
        "manuscript reports; check that the four others are larger, since that ordering is the "
        "reason the shipped column is not a flattering choice.",
        "One of the five does not survive a second interval method, and the manuscript says so. "
        "The percentile interval on the shipped column excludes zero; its bias-corrected and "
        "accelerated interval covers zero, and NOT because it is wider -- the width ratio is "
        "near one and the bias correction shifts it down. An audit that recomputes only the "
        "percentile interval has not tested this sentence."],
      artifacts=["m36_readout_scale.json", "m25_estimator_benchmark.json",
                 "m54_probing_reference.json", "m56_bca.json"],
      printed=["cvIntvSd", "cvConvSd", "cvPairedSd", "cvPairedSdCi", "cvSwing",
               "prNcol", "prReadout", "prReadoutCi", "prReadoutN", "prReadoutNCi",
               "bcTwo", "bcTwoWidth", "bcTwoBasic"]),
 dict(id="NM-C2", label="claim:identity", axis="pair identity",
      section="Results / what the instrument reads out",
      estimand="For each of three predictors, the matched stratified difference in the predictor's "
               "OWN output for the mutant target pair, between G-C-target and A-U-target "
               "interventions, on the same strata as the measured outcome; plus the same "
               "difference on the measured rescue factor. Claim: the neural predictor's output "
               "separates the two identities, the two nearest-neighbour predictors differ from "
               "each other, and the separation decomposes into an architecture axis and a "
               "training-data axis whose difference has an interval. NARROWED SINCE THE LAST "
               "PACKAGE: a second transformer sharing neither backbone nor training corpus does "
               "NOT reproduce the architecture axis, and a pseudoknot-capable thermodynamic "
               "implementation does not resolve the identity separation either, so the axis is "
               "stated as a property of one model and not of its architecture class.",
      falsifier="any pair of predictors is compared without an interval on the DIFFERENCE; or the "
                "three predictor columns are not read on the same records; or the 1-based vs "
                "0-based indexing of the two prediction sources is crossed; or the manuscript "
                "anywhere states the architecture axis as a fact about transformers rather than "
                "about this transformer, which the second-transformer result forbids",
      independent_route=[
        "Recompute each predictor's pair probability for the named (i,j) in the mutant sequence "
        "from a partition function of your own choosing for the thermodynamic arm; for the neural "
        "and learned-parameter arms use the released weights named in artifacts.json.",
        "Take differences of contrasts PAIRED ON THE SAME BOOTSTRAP RESAMPLE, clustering on the "
        "20 puzzles. A recomputation that reports two separate intervals and no interval on their "
        "difference has not reproduced the claim.",
        "Indexing is the trap: one prediction source is 0-based and the other is 1-based. A "
        "recomputation that gets the identity contrast right but the pairwise differences wrong "
        "should check this first."],
      artifacts=["m35_paired_predictor_differences.json", "m26_model_scored.json",
                 "m55_second_transformer.json", "m58_pk_member.json", "m52_sensitivity.json"],
      printed=["resNeural", "resNeuralCi", "resVienna", "resViennaCi", "efContrast",
               "efContrastCi", "pdNVCi", "pdNECi", "pdVECi", "axArch", "axArchCi", "axData",
               "axDataCi", "axGap", "axGapCi", "fmContrast", "fmContrastCi", "pkContrast",
               "pkContrastCi", "senK", "senKCi", "senMaxOne", "senMaxAll"]),
 dict(id="NM-C3", label="claim:depth", axis="free baseline",
      section="Results / which flips rescue, and the free baseline nobody reports",
      estimand="Within each design cell of at least 8 admitted interventions, the Spearman rank "
               "correlation between a score and the measured rescue factor, averaged across cells "
               "weighted by cell size, with a cluster bootstrap over the 20 puzzles. Scores: each "
               "predictor's pair probability, and HELIX DEPTH -- an integer read off the declared "
               "target structure, requiring no model. Claim: the paired difference (predictor "
               "minus depth) has an interval covering zero for every predictor, so no predictor "
               "beats the free integer; and three purpose-built mutation-effect scores lose to it "
               "with intervals that EXCLUDE zero on the losing side. NEW SINCE THE LAST PACKAGE: "
               "the claim now carries a measured DOMAIN. Split the interventions on whether a "
               "probing-derived reference confirms the intervened pair. Depth's rank agreement is "
               "much larger in the confirmed stratum than in the unconfirmed one, and in the "
               "unconfirmed stratum NOTHING is resolved -- every interval covers zero and the "
               "transformer's point estimate is the larger. Two further implementations, a second "
               "transformer and a pseudoknot-capable thermodynamic one, also lose to the integer, "
               "the latter by an interval excluding zero.",
      falsifier="the manuscript offers a MECHANISM for why helix depth works -- it may not, and "
                "says so: depth could be carrying real structural depth, the accuracy of the "
                "designer's declaration, or both, and the paper separates neither. What it "
                "establishes is that depth is not purely a declaration artefact, since depth read "
                "off the probing-derived reference is LARGER than depth read off the declaration; "
                "or the difference is read as two point estimates rather than as one interval on the "
                "difference -- which is the error the paper itself retracts here; or depth is "
                "computed from a predicted structure rather than from the declared target; or the "
                "bootstrap resamples interventions rather than puzzles; or the domain split is "
                "read as depth failing in the unconfirmed stratum rather than as nothing being "
                "resolved there, which is the paper's own bounded-negative rule",
      independent_route=[
        "Depth of a pair (a,b) in the declared target structure = the number of consecutive "
        "nested pairs stacked outside it, capped by the number stacked inside it; formally "
        "min(dep, up) where dep counts k with (a-k-1, b+k+1) paired and up counts k with "
        "(a+k+1, b-k-1) paired. No model, no folding, no training data.",
        "Compute the difference INSIDE each bootstrap resample and take the interval of the "
        "differences. Two separately-resampled intervals do not answer this claim.",
        "The riboSNitch-style scores are: Euclidean distance between the wild-type and mutant "
        "base-pair probability matrices; base-pair distance between the two MFE structures; and "
        "|dG_ensemble(mut) - dG_ensemble(wt)|. Their declared sign is NEGATIVE (a rescuing flip "
        "preserves structure), so negate before ranking."],
      artifacts=["m31_which_flips_rescue.json", "m31r_review.json",
                 "m38_ribosnitch_baseline.json", "m40v_verify_m38_m39.json",
                 "m57_criterion_validity.json", "m55_second_transformer.json",
                 "m58_pk_member.json"],
      printed=["rkRhoD", "rkRhoDCi", "rkRhoN", "rkDiffNCi", "rkDiffVCi", "efVsDepthCi",
               "rkPartN", "rkPartNCi", "rsBppm", "rsBppmCi", "rsBppmVs", "rsBppmVsCi",
               "rsMfeVs", "rsMfeVsCi", "rsEnsVs", "rsEnsVsCi",
               "vaDepConf", "vaDepUnc", "vaDepGap", "vaNconf", "vaNunc", "vaDNC", "vaDNCCi",
               "vaNresUnc", "vaConfLvl", "vaDepVsConf", "vaDepVsConfCi",
               "fmLevel", "fmVsDepth", "fmVsDepthCi", "pkLevel", "pkVsDepth", "pkVsDepthCi"]),
 dict(id="NM-C4", label="sec:ledger", axis="the ledger",
      section="Results / the ledger: who chose the covariate that carries each number",
      estimand="Eight covariate-choice channels, each priced by a direct contrast in its own "
               "units. Claim: five carry the number (one of them against the reported effect), "
               "three are empty, and the empty three are leakage, probe chemistry and the folding "
               "model's own free energy. Separately: the prices are NOT shares and may not be "
               "summed, because a share is a mediated fraction and the estimator for it is "
               "withdrawn by this paper.",
      falsifier="any row is priced by a ratio of two contrasts (that is the withdrawn estimator); "
                "or the eight rows are presented as a decomposition of one total; or a channel is "
                "priced on a scale it does not share with the row above it",
      independent_route=[
        "This claim is structural, not numerical: check that no row's price is a ratio of two "
        "contrasts, that the table is never summed, and that each row's number appears in the "
        "section it points at with the same value.",
        "The one row that IS a decomposition (C2, placement) must close as an identity: "
        "marginal - matched = observed attenuation, and predicted/observed = the mediated "
        "fraction. Check both to 1e-4 and report the residual."],
      artifacts=["m37_channel_ledger.json"],
      printed=["chPriced", "chCarry", "chEmpty", "chUnpriced", "chEps", "chSplit", "chNoSplit",
               "chSplitCarry"]),
 dict(id="NM-C5", label="sec:leakage", axis="leakage / exposure",
      section="Results / the leakage audit, in the main text",
      estimand="Fraction of design cells whose wild-type sequence, inside the design window, "
               "reaches >=0.8 identity to an equal-length substring of some sequence in the "
               "neural model's training file; and the within-stratum skill slope on the exposed "
               "subset against the unexposed one, for the neural model AND for a thermodynamic "
               "model that cannot have memorised anything. Claim: exposure is higher than an "
               "earlier deposit-based proxy implied, the channel is still empty because the "
               "thermodynamic control falls the same way, and EVERY COUNT IS A LOWER BOUND "
               "because the training file obtained is a subset of the full corpus.",
      falsifier="any sentence claims a design is absent from the training set rather than not "
                "found in the portion obtained; or the exposure scan uses a candidate index that "
                "can record 'no candidate' as 'identity zero'; or the thermodynamic control is "
                "omitted, in which case a drop on the exposed subset cannot be told from memory",
      independent_route=[
        "Identity of a query window against a training sequence = max over offsets of "
        "(matching positions)/(query length), over every training sequence at least as long as "
        "the query. Do it exhaustively; do not build a k-mer candidate index, because an index "
        "that finds no candidate must not be allowed to return zero.",
        "The control is the whole argument. Compute the same exposed-versus-unexposed slope for "
        "a purely thermodynamic predictor. If it falls too, exposure marks difficulty, not memory."],
      artifacts=["m39_exposure_tiers.json", "m26d_clean_subset.json", "m26c_leakage.json"],
      printed=["lkExpDesigns", "lkExpFrac", "lkProxyDesigns", "lkVerbatimT3", "lkNeuralClean",
               "lkNeuralExp", "lkNeuralDiffCi", "lkViennaClean", "lkViennaExp", "lkViennaDiffCi",
               "vIdxGap"]),
 dict(id="NM-C6", label="sec:withdrawal", axis="retraction",
      section="Results / a metric we withdraw",
      estimand="A calibration that injects a KNOWN mediated fraction into the real mediator and "
               "asks how much of the injected range the estimator's reading covers. Claim: the "
               "reading covers only a small fraction of the range the truth covers, so the "
               "estimator is uninformative here and the quantity is withdrawn.",
      falsifier="the calibration reports a correlation at a single injection level rather than a "
                "span across the sweep -- a correlation at one point is an endpoint artefact and "
                "the paper says so",
      independent_route=[
        "Sweep the injected fraction f over [0, 0.75]. Report how far the READING moves against "
        "how far the TRUTH moves. A correlation coefficient at one f is not the quantity; "
        "leave-one-out the endpoint and see whether the sign survives."],
      artifacts=["m26r3_realistic_calibration.json", "m26r4_calibration_symmetry.json"],
      printed=["mcNeuralSpanFrac", "mcViennaSpanFrac"]),
 # -------------------------------------------------------------------------------------------
 # C7 was missing from every package built before 2026-09-01. It is the manuscript's Reading-2
 # headline -- the claim the whole second half of the paper exists to support -- and the auditor
 # was never given it. Recorded here rather than quietly added, because a claim ledger that
 # omitted the paper's most consequential claim is itself a process finding.
 # -------------------------------------------------------------------------------------------
 dict(id="NM-C7", label="claim:budget", axis="the budgeted decision",
      section="Results / what the benchmark's margin buys, in tubes",
      estimand="Convert a structure-agreement margin into an experimental decision. Within each "
               "design cell, rank the admitted interventions by a score, take the top k = "
               "ceil(frac * n) with k >= 2, and record the fraction of that cell's RESCUING "
               "interventions recovered, pooled over cells weighted by the number of rescuing "
               "interventions. Scores compared: each implementation's pair probability, and free "
               "helix depth. Claim: at every budget no implementation's recovery advantage over "
               "free depth excludes zero; against a margin fixed as the gain one extra construct "
               "buys, most comparisons are formally EQUIVALENT to free depth and the rest fail on "
               "the side of being worse; the panel's largest agreement margin converts to a small "
               "percentage of rescuing interventions; and the gap changes sign exactly once as "
               "the budget is swept.",
      falsifier="the manuscript asserts delta as a universal scientific equivalence margin rather "
                "than as a declared operational tolerance of one construct in the median design; "
                "or it asserts budgeted recovery as THE definition of experimental utility rather "
                "than as one operational decision; or the equivalence test is run without a "
                "pre-fixed margin delta, or with a delta "
                "chosen after seeing the recoveries; or 'no interval excludes zero' is reported "
                "as 'the scores are equal', which is the bounded-negative rule this paper states "
                "for itself; or the sign change is asserted from three budget points rather than "
                "measured over a sweep; or ties inside a cell are broken in an order correlated "
                "with the outcome",
      independent_route=[
        "Rebuild the admitted set from criterion.md, then implement recovery-at-budget exactly as "
        "the estimand states. The budget must be a FRACTION of each cell's size, floored at two "
        "constructs, not a fixed count -- a fixed count changes which cells can contribute.",
        "Tie handling is the trap. Most within-design pairs are tied on depth, so a stable sort "
        "makes the result depend on CSV row order. Average over random within-cell orders and "
        "report both; if the two disagree, that disagreement is the finding.",
        "Equivalence is a TOST-shaped statement, not a failure to reject. Fix delta first, from "
        "the gain one extra construct buys, and grade each comparison against it.",
        "Sweep the budget rather than testing three points, and count sign changes over the "
        "sweep. A crossing asserted from three points is not measured.",
        "Substitute your own delta. The manuscript reports delta-star -- the smallest margin at "
        "which two one-sided tests would declare equivalence -- for exactly this reason, so a "
        "reader whose construct costs or cell sizes differ can regrade every comparison without "
        "rerunning anything. If you regrade and the verdict flips at a defensible delta, that is "
        "a finding about the tolerance and not about the estimator."],
      artifacts=["m44_decision.json", "m44v_verify.json", "m47_equivalence.json",
                 "m58_pk_member.json"],
      printed=["eqNeq", "eqNcomp", "eqNfail", "eqDelta", "dcTransF1", "dcTransPct",
               "tvGapTen", "tvGapCiTen", "ivNcross", "ivCrossA", "ivCrossB",
               "pkRecPk", "pkRecDepth"]),
 dict(id="NM-C8", label="sec:clauseiii", axis="the criterion's own validity",
      section="Method / the eligibility criterion",
      estimand="Clause (iii) of the admission criterion requires the intervened pair to be a pair "
               "of the DECLARED target structure. The validity of that clause is the fraction of "
               "declared-target pairs, inside the design windows of admitted design cells, that a "
               "probing-derived reference also calls paired. Claim: that fraction is well above a "
               "half but well below one, so roughly a fifth of what the design programs declared "
               "is not what the molecules did, and every agreement number scored against a "
               "declared target inherits that ceiling.",
      falsifier="the fraction is computed over intervened pairs only and quoted as if it were "
                "over all declared pairs -- those are different populations and the manuscript "
                "reports both; or the reference used is the structure column the deposit ships, "
                "which is malformed on a large minority of rows and must not be used for this",
      independent_route=[
        "Fold each wild type with a pseudoknot-capable program restrained by that construct's own "
        "reactivities, restrict both the declared target and the folded structure to the design "
        "window, and take the fraction of declared pairs present in the fold. Pool over pairs AND "
        "report the per-design distribution; a pooled fraction alone hides that some designs "
        "confirm nothing.",
        "Do NOT use the deposit's own M2_structure column for this. A large minority of its "
        "strings are not balanced dot-bracket, and a fraction computed against it is a fraction "
        "against a defect. The manuscript reports the figure obtainable that way and does not "
        "use it; check that no claim rests on it."],
      artifacts=["m57_criterion_validity.json", "m53_probing_reference.json",
                 "m54_probing_reference.json"],
      printed=["vaPooled", "vaHit", "vaAll", "vaMean", "vaMeanCi", "vaMed", "vaNone", "vaNdes",
               "vaIntv", "vaReleased"]),
 dict(id="NM-C9", label="app:probing", axis="a reference nobody in the panel wrote",
      section="Appendix / a reference nobody in the panel wrote",
      estimand="Replace the agreement axis's reference -- the designer's declared target -- with a "
               "structure folded from this deposit's own chemical probing by a program that is "
               "not one of the scored implementations. Claim: the incumbent still ranks first and "
               "the order of the panel is unchanged, but its largest margin loses a large "
               "fraction of its size, with an interval on the CHANGE that excludes zero.",
      falsifier="the change is read off two separately-resampled margins rather than from an "
                "interval on the difference; or the two references are scored on different cell "
                "sets -- the agreement axis and the readout comparison are defined on cell sets "
                "that differ by one design, and conflating them is a real error this run made "
                "once and caught with a reverse control",
      independent_route=[
        "Build the reference yourself from the deposit's reactivities. Take only the "
        "lowest-free-energy structure; these programs emit a set and a later member is not it.",
        "DO NOT treat this reference as model-free, and check that the manuscript does not "
        "either. It is a nearest-neighbour fold restrained by pseudoenergies derived from the "
        "measured reactivities, and it therefore SHARES AN ENERGY MODEL with one implementation "
        "in the scored panel. The property the comparison needs is narrower: it contains no "
        "prediction of these molecules made from sequence. An earlier version of the abstract "
        "said 'no model at all'; that was wrong and is corrected, and an auditor should verify "
        "the correction reached every place the column is described.",
        "Resample the two margins on the SAME draw and take the interval of the difference.",
        "Check which cell set each quantity is defined on before comparing anything."],
      artifacts=["m54_probing_reference.json", "m53_probing_reference.json"],
      printed=["prMarginBuilt", "prMarginBuiltCi", "prMarginPct", "prOrderSame",
               "prDepthWorstCi", "prRestraintN"]),
 dict(id="NM-C10", label="sec:models", axis="what the panel can represent",
      section="Experimental setup / the predictors",
      estimand="Two controls on panel composition, both added after review. (a) A second "
               "transformer sharing neither backbone nor training corpus with the first, scored "
               "on the same interventions and strata. (b) A pseudoknot-capable nearest-neighbour "
               "implementation -- the panel's own energy model with the pseudoknot search on, "
               "given no probing data -- added because the corpus is a pseudoknot round while "
               "every previously scored implementation is pseudoknot-blind. Claim: neither "
               "resolves the identity separation, neither beats free helix depth, and the "
               "pseudoknot-capable one reaches second place on agreement while placing last on "
               "response.",
      falsifier="the pseudoknot-capable member is given the construct's own probing data on the "
                "input side, which would put the outcome's own experiment on both sides of the "
                "comparison; or the second transformer's index convention is not checked, since "
                "an off-by-one in the pair map would produce a null for free; or the "
                "agreement-versus-response inversion is reported with an interval, which it must "
                "not be, the pair having been chosen after the point estimates were seen",
      independent_route=[
        "Score both from sequence alone. For the pseudoknot-capable member, verify the "
        "sequence-only structures match the unrestrained control of the reference build exactly; "
        "if they do not, the two runs may not be placed in one panel.",
        "For any neural member, verify the index convention before reading any column: decode the "
        "top L/2 pairs and check the fraction that are canonical. A correctly loaded head is far "
        "above chance; a randomly initialised head is far below. The manuscript reports both."],
      artifacts=["m55_second_transformer.json", "m58_pk_member.json"],
      printed=["fmContrast", "fmContrastCi", "fmLevel", "fmVsDepth", "fmVsDepthCi",
               "pkBuilt", "pkDecl", "pkLeadBuilt", "pkRank", "pkLevel", "pkContrast",
               "pkContrastCi", "pkVsEtfAgree", "pkVsEtfDecl"]),
 dict(id="NM-C11", label="sec:inference", axis="interval method dependence",
      section="Method / inference",
      estimand="Recompute the three intervals that carry claims with a second method -- "
               "bias-corrected and accelerated, with the jackknife taken over the same clusters "
               "the bootstrap resamples. Claim: two of the three still exclude zero, and the "
               "third does not; the third is the readout comparison against the shipped agreement "
               "column, and the shift is caused by the bias correction rather than by extra "
               "width.",
      falsifier="the BCa endpoints are computed with the jackknife over records rather than over "
                "clusters; or the degenerate case is not checked -- with bias correction and "
                "acceleration set to zero, BCa must equal the percentile interval EXACTLY; or the "
                "manuscript reports the second method only where it agrees",
      independent_route=[
        "Implement BCa from the definition: z0 = Phi^-1(fraction of bootstrap draws below the "
        "observed), acceleration from the jackknife third moment, endpoints at "
        "Phi(z0 + (z0 + z_q)/(1 - a(z0 + z_q))).",
        "Check the degenerate identity first. If z0 = a = 0 does not return the percentile "
        "interval to the last bit, the implementation is wrong and nothing after it counts.",
        "Reproduce each published percentile interval before reading its BCa counterpart. Two "
        "traps are documented: the published estimator draws its permutations and its bootstrap "
        "from ONE generator, so the bootstrap stream starts after the permutations; and the three "
        "runs do not share a quantile convention."],
      artifacts=["m56_bca.json"],
      printed=["bcOne", "bcTwo", "bcThree", "bcTwoBasic", "bcTwoWidth"]),
 dict(id="NM-C12", label="ass:placement", axis="how strong a violation would have to be",
      section="Method / estimation, and the one assumption it needs",
      estimand="A closed-form sensitivity bound. The matched estimator is linear in the outcome, "
               "so an omitted covariate U entering with within-stratum response beta_u shifts the "
               "estimate by exactly beta_u * g_u, where g_u is the same weighted arm gap applied "
               "to U -- a product invariant to rescaling U. Claim: strata are keyed on the design "
               "cell, so any covariate constant within a design has g_u = 0 exactly and no "
               "design-level confounder can bias the estimate; the confounder that would explain "
               "the contrast away is a stated multiple of the placement bias the estimator "
               "already conditions on; and no measured covariate, nor the worst linear "
               "combination of all of them, comes near it.",
      falsifier="the multiplier's interval is reported as excluding one -- IT DOES NOT, and the "
                "manuscript grades that stake as failed; or the maximum over covariate indices is "
                "obtained by search rather than in closed form; or a covariate with no "
                "within-stratum variance is included, which makes the covariance matrix singular",
      independent_route=[
        "Verify the identity by injection rather than by algebra alone: add a synthetic covariate "
        "at a known coefficient and check the estimate moves by exactly beta_u * g_u, that it is "
        "odd under U -> -U, and that a design-level covariate moves it by zero.",
        "The maximum of (w'c)(w'd) over unit-within-variance indices has a closed form: with "
        "v = Sigma^{1/2} w, a = Sigma^{-1/2} c and b = Sigma^{-1/2} d, the maximum is "
        "(||a|| ||b|| + a'b)/2. Random directions must approach and never exceed it."],
      artifacts=["m52_sensitivity.json"],
      printed=["senK", "senKCi", "senBench", "senDiag", "senDiagSd", "senMaxOne", "senArgOne",
               "senMarginCi", "senMaxAll", "senMaxAllPct", "senNcov", "senNdeg"]),
]


R1_CLAIMS = [
 dict(id="R1-C1", label="prop:routes", axis="estimator theory",
      section="Method / two routes to a mediated fraction, and when they disagree",
      estimand="Let T be a binary exposure and M a bounded observed mediator, with "
               "E[Y|T,M] = alpha_T + h(M) (no T-by-M interaction). Let beta be the least-squares "
               "slope of Y on M under a fixed reference law, g = E[M|A]-E[M|B], and "
               "r(m) = h(m) - beta*m. Claim: (total - controlled-direct) - beta*g equals exactly "
               "E[r(M)|A]-E[r(M)|B]; it is bounded by osc(r) * d_TV(P_{M|A},P_{M|B}); and the two "
               "routes agree for EVERY pair of exposure laws IF AND ONLY IF h is affine on the "
               "support of M.",
      falsifier="a counterexample with h affine and the routes disagreeing; or a non-affine h for "
                "which no pair of exposure laws separates the routes; or a case violating the "
                "stated bound",
      independent_route=[
        "This is a proposition, not a measurement. Verify it symbolically or by construction. "
        "The 'only if' direction is the load-bearing half: given non-affine h, exhibit exposure "
        "laws that separate the routes (a point mass at a point where r differs from its mean "
        "against the reference law suffices).",
        "PRIORITY NOTE the auditor should check the paper makes: the NON-EQUIVALENCE of the two "
        "routes under a nonlinear outcome model is standard and the paper must disclaim priority "
        "for it. What is claimed as new is the closed-form gap, the bound by an observable, and "
        "the iff. If the manuscript claims the non-equivalence itself as novel, that is a finding."],
      artifacts=["m21_thermo_mediator.json"],
      printed=["eLinearGapPts"]),
 dict(id="R1-C2", label="sec:cal", axis="calibration / retraction",
      section="Results / which route is readable is a question for calibration",
      estimand="Inject a KNOWN mediated fraction f into the real mediator and record what each "
               "estimator reads back, sweeping f. Claim: the reading covers only a fraction of "
               "the range the truth covers, and returns a specific value at total mediation; "
               "therefore every mediated fraction this work previously reported is withdrawn.",
      falsifier="the sweep's zero rung does not return zero; or the injection changes the "
                "mediator's marginal law in a way that alone explains the shortfall",
      independent_route=[
        "Reimplement the injection from the specification in calibration.md and sweep f. Compare "
        "the SPAN of the reading against the SPAN of the truth, not the correlation between them.",
        "Check the f=0 rung returns zero. A ladder whose zero rung does not fall back to nominal "
        "is void by the paper's own definition."],
      artifacts=["m26r3_realistic_calibration.json", "m26r4_calibration_symmetry.json"],
      printed=["calNSpanFrac", "calVSpanFrac", "calNFull", "calVFull"]),
 dict(id="R1-C3", label="sec:mediation", axis="placement channel",
      section="Results / the placement gap and the depth slope",
      estimand="The matched stratified difference in mean rescue between G-C and A-U targets, "
               "computed twice: MARGINALLY (design cell only) and MATCHED (design cell x helix "
               "depth x crossing). Claim: the two differ, the difference is quantitatively "
               "predicted by the product of a depth slope and a depth gap estimated on disjoint "
               "arms, and the residual of that prediction is reported.",
      falsifier="the identity marginal - matched = observed attenuation does not close; or the "
                "depth slope and the depth gap are estimated on the same records, which would "
                "make the prediction circular",
      independent_route=[
        "Compute both contrasts from the public CSV with weight w_s = min(n_A(s), n_B(s)); strata "
        "missing an arm are DROPPED, never extrapolated into.",
        "Check the closure as an identity to 1e-4 and report the residual epsilon. The paper "
        "reports it; a recomputation that does not is not comparable."],
      artifacts=["m20_designer_confounding.json"],
      companion_artifacts=["m37_channel_ledger.json"],
      printed=["medMarginal", "medMatched", "medRatio", "medFrac"]),
 dict(id="R1-C4", label="sec:energy", axis="energy channel (a null)",
      section="Results / the energy the designs were optimised for",
      estimand="The same identity contrast, with the nearest-neighbour free energy the model "
               "assigns to the pair held fixed by stratification. Claim: the contrast does not "
               "move -- retention near 100% -- so the channel every design program was optimising "
               "carries none of the effect; and a placebo mediator, built to have no path to the "
               "outcome, does not fire.",
      falsifier="the placebo mediator also absorbs a large share, which would void the arm; or the "
                "denominator of the retention is taken on a different population from the "
                "estimator's own, which inflates it",
      independent_route=[
        "The retention's denominator MUST be computed on the estimator's own retained population. "
        "Using the full corpus as denominator is a known way to manufacture a large number here; "
        "the paper reports what that error would have produced.",
        "dG_loc is undefined at non-canonical partners; check the guard exists and report how "
        "many records it touches."],
      artifacts=["m21_thermo_mediator.json", "m21v_verify.json", "m22_review_arms.json"],
      printed=["eRet", "eMed", "eMedCi", "eZ"]),
 dict(id="R1-C5", label="sec:crossdomain", axis="external validity of the bound",
      section="Results / the proposition, tested where none of this paper's machinery applies",
      estimand="On a second, unrelated corpus of designed protein binders, the same route bound "
               "is put at risk. Claim: the bound does not merely hold, it TRACKS -- the overlap "
               "and the route gap move together across the two corpora by comparable ratios.",
      falsifier="the bound holds trivially because the gap is near zero on both corpora; or the "
                "two ratios are computed on quantities that are not comparable across corpora",
      independent_route=[
        "The bound has two sides. Check the paper reports the SAME side on both corpora -- "
        "reporting one side on one corpus and the other side on the other would make the "
        "comparison meaningless."],
      artifacts=["m29_binder_crossdomain.json", "m29r_review.json"],
      printed=["xdN", "xdDtvRatio", "xdGapRatio"]),
 dict(id="R1-C6", label="sec:frame", axis="measurement ceiling",
      section="Results / the measurement frame: what a point is worth",
      estimand="A reproducibility ceiling for the wet readout, estimated from repeated measurement "
               "groups by a variance-of-moments argument. Claim: the ceiling is a specific value "
               "with an interval, and it is the one interval in the paper that resamples repeated "
               "measurement groups rather than targets.",
      falsifier="the repeat groups are misdefined; or the interval is described as resampling "
                "targets, which would be the wrong cluster",
      independent_route=[
        "Identify the repeat groups independently from the corpus and re-estimate. The paper "
        "flags this interval as the exception to its own clustering convention; check the "
        "manuscript actually says so where the number appears."],
      artifacts=["r1b_ceiling_interval.json"],
      printed=["cZLo", "cZHi"]),
]


# =====================================================================================
# Process-track material. These are claims the manuscript makes ABOUT ITSELF, each with the file
# that would falsify it. This is what the auditor's process track can grade without any answer key.
# =====================================================================================
PROCESS_CLAIMS = [
 dict(id="P1", claim="Every measured number printed in the manuscript reaches LaTeX through a "
      "generated macro, so a printed figure cannot drift from the run that produced it.",
      falsify_by="find a decimal literal in the .tex body that is a measurement. make_check.py "
                 "is the gate; run it and see whether it actually refuses one.",
      check_command="python make_check.py   # in the manuscript directory"),
 dict(id="P2", claim="Every experiment classed 'preregistered' has a declaration that was frozen, "
      "committed and PUSHED before the code that produced it existed.",
      falsify_by="for each declaration in provenance.json, compare `git log --diff-filter=A` for "
                 "the declaration and for its script. If the script was added first, the claim is "
                 "false for that experiment.",
      check_command="git log --diff-filter=A --format='%aI %H' -- <declaration> <script>"),
 dict(id="P3", claim="Staked predictions are graded in the declaration's own wording, including "
      "where the prediction failed, and failures are printed in the manuscript.",
      falsify_by="read each declaration's `staked` block and find the corresponding row in the "
                 "manuscript's preregistration appendix. A stake with no row, or a row whose "
                 "wording differs from the declaration, is a finding.",
      check_command=None),
 dict(id="P4", claim="No claim in the manuscript rests on the withdrawn estimator (a mediated "
      "fraction / a ratio of two contrasts).",
      falsify_by="search the results and conclusion for any quantity that is a ratio of two "
                 "contrasts or is described as a share, fraction or percentage OF an effect.",
      check_command=None),
 dict(id="P5", claim="Where two magnitudes are compared, an interval on their DIFFERENCE is "
      "reported -- not two separate intervals.",
      falsify_by="find a comparison of two point estimates with no interval on the difference. "
                 "The paper retracts one such comparison itself; a second one that is not "
                 "flagged is a finding.",
      check_command=None),
 dict(id="P6", claim="Every artifact a generator reads exists and is tracked by git, and every "
      "artifact tracked is classified in provenance.json.",
      falsify_by="run the repository's own audit tool and read checks [D] and [H].",
      check_command="python paper/audit_manuscript.py <manuscript-dir>"),
 dict(id="P7", claim="Every key cited in the body exists in refs.bib, so no citation typesets "
      "as a question mark.",
      falsify_by="check [J] of the audit tool, or diff the cited keys against the bib keys. "
                 "This check exists because two such keys once reached a committed PDF.",
      check_command="python paper/audit_manuscript.py <manuscript-dir>   # check [J]"),
 dict(id="P8", claim="Bibliographic entries are real: every author list, volume and DOI was "
      "fetched rather than recalled.",
      falsify_by="resolve every DOI and arXiv id in refs.bib and compare the author lists. A "
                 "placeholder author field, or a DOI that does not resolve, is a finding. Two "
                 "placeholder author fields were present in an earlier commit.",
      check_command=None),
 dict(id="P9", claim="Boundary and unexpected results were put through a four-step verification "
      "before any conclusion was changed: (1) an independent reimplementation recomputing the "
      "same number, (2) a mathematically equivalent alternative form, (3) a reverse control "
      "reproducing the OLD number with the OLD code, (4) a zero/identity self-check.",
      falsify_by="for each result the manuscript calls a boundary or a failed stake, locate all "
                 "four steps in the committed artifacts. A missing step is a finding even when "
                 "the conclusion is right.",
      check_command=None),
 dict(id="P10", claim="Negative results are reported bounded: an interval containing zero is "
      "reported as unresolved, never as a refutation.",
      falsify_by="find a sentence that reads an interval containing zero as evidence of absence.",
      check_command=None),
 dict(id="P11", claim="Each declaration names the consequence a failed stake will have on the "
      "manuscript BEFORE the number exists, and where a stake failed that consequence was "
      "actually applied to the text.",
      falsify_by="for each declaration with a failed stake, read its declared_consequences block "
                 "and then read the manuscript at the place it names. A declared narrowing that "
                 "was not applied is the most serious finding available here, because the "
                 "declaration is what makes the stake meaningful. Eight stakes across five "
                 "experiments failed; each should be traceable to a specific change.",
      check_command=None),
 dict(id="P12", claim="Where a declaration itself was wrong, the error is RECORDED as an erratum "
      "and the declaration is not edited after the fact.",
      falsify_by="compare each declaration's git blob at its first commit against its current "
                 "blob. Any difference is a finding. Then check that every erratum described in "
                 "the manuscript's preregistration appendix corresponds to something real -- "
                 "three are recorded: a stake written on a point estimate where an interval was "
                 "needed, a verification identity that does not exist, and a verification wording "
                 "that named the wrong count.",
      check_command="git log --format='%H' -- <declaration> | tail -1  # then diff that blob"),
 dict(id="P14", claim="The manuscript's central quantity is defined once, canonically, and every "
      "later use is an application of that definition rather than a re-definition.",
      falsify_by="read submission.json:canonical_estimand_as_printed, then find a place in the "
                 "manuscript that scores something else and calls it the same thing. The obvious "
                 "candidates are the agreement sections, where a structure-level quantity sits "
                 "beside a pair-level one; check the paper never differences them without a "
                 "common scale.",
      check_command=None),
 dict(id="P15", claim="Where the manuscript makes a scoping choice -- an equivalence margin, a "
      "decision utility, a reference column -- it declares the choice as a choice and supplies "
      "what a reader needs to substitute their own.",
      falsify_by="for the equivalence margin, check delta-star is reported per comparison. For "
                 "the decision utility, check the manuscript says budgeted recovery is one "
                 "operational decision and not the definition of experimental utility. For the "
                 "reference column, check all five columns are reported and not only the "
                 "flattering one. A scoping choice presented as a fact is a finding.",
      check_command=None),
 dict(id="P13", claim="A quantity computed and not reported is not hidden: the repository's own "
      "audit tool lists every generated macro the manuscript never prints, and the count is "
      "disclosed rather than minimised.",
      falsify_by="run the audit tool and read check [B]. Then ask, for each unprinted quantity, "
                 "whether it would have weakened a claim. A quantity computed, unprinted and "
                 "unfavourable is the finding this check exists for.",
      check_command="python paper/audit_manuscript.py <manuscript-dir>   # check [B]"),
]


SANITY = [
 "The interventional outcome must contain no folding model. Read the definition: if any term in "
 "it is a model output, the paper's central separation from profile-based scoring is void.",
 "A matched stratified difference must drop strata missing an arm, never extrapolate into them. "
 "Check the weight is min(n_A, n_B) and that the dropped count is reported.",
 "A permutation test after caliper matching must permute WITHIN strata. Permuting the treatment "
 "label across the matched set rebuilds a different matched set and narrows the null; the "
 "manuscript states the size of that error, so check the direction of its own correction.",
 "A cluster bootstrap on 20 clusters is optimistic. Any interval whose width matters at the "
 "second decimal should be read with that in mind, and the paper should say so.",
 "Skill on an exposed subset falling BELOW skill on an unexposed subset is not evidence of "
 "memorisation; it is evidence the exposed subset is harder. The control that separates them is "
 "a model that cannot have memorised anything.",
 "Every exposure count computed against a subset of a training corpus is a LOWER bound. A "
 "sentence of the form 'not in the training set' is unsupportable from a subset.",
 "A ratio of two contrasts is the estimator this work withdraws. Any number of that form anywhere "
 "in either manuscript is a finding, including in a table caption.",
 "A predictor scored as a panel member must see SEQUENCE ONLY. One program appears in this work "
 "twice, once restrained by the construct's own probing data to build a reference and once "
 "unrestrained as a scored member. If the restrained form were scored, the outcome's own "
 "experiment would sit on both sides of the comparison.",
 "The deposit's own probing-derived structure column is malformed on a large minority of rows -- "
 "the strings are not balanced dot-bracket. Any quantity computed against that column is computed "
 "against a defect. The well-formed subset is not a fallback either: it is a structure-class "
 "selection, the malformed rows being the more pseudoknotted ones.",
 "A dot-bracket parser for this corpus must keep ONE STACK PER BRACKET TYPE. It is a pseudoknot "
 "benchmark and the strings use several bracket families; a single-stack parser silently "
 "mis-pairs and every downstream number inherits it.",
 "The agreement axis and the readout comparison are defined on cell sets that DIFFER. Scoring one "
 "on the other's set reproduces published numbers to about 1e-3 and no better, which looks like a "
 "tolerance problem and is not.",
 "A number printed to four decimals carries +/-5e-5 of rounding. A reproduction tolerance of 5e-5 "
 "against a printed value is arithmetically unreachable; targets must be read from artifacts.",
 "A permutation-plus-bootstrap routine that draws both from one generator does not give the "
 "bootstrap a fresh stream. Reproducing such an interval requires advancing the generator by the "
 "permutation count first.",
 "An interval that covers zero after a second method, where the first excluded it, has not "
 "refuted anything. Check whether the shift came from extra width or from a bias correction; "
 "those are different findings and the manuscript distinguishes them.",
 "NOTHING in this work is model-free. A structure folded from probing data is a thermodynamic "
 "model restrained by an experiment, not an observation; the outcome the predictors are scored "
 "against is arithmetic on reactivities and contains no folding model, which is a different and "
 "much narrower statement. Any sentence conflating the two is a finding -- one such sentence "
 "reached a committed draft of the abstract and was corrected only after review.",
 "The declared target enters this work in four places that must be named rather than waved at: "
 "it says which pair is the target, it supplies helix depth, it supplies the crossing indicator, "
 "and it gates eligibility. What it never does is serve as the reference the measured outcome is "
 "scored against. A claim of the form 'no reference structure enters' is too strong; the "
 "supportable form is 'no inferred or model-generated structure of these molecules enters the "
 "measured outcome'.",
 "An equivalence margin is a choice. Check whether the manuscript defends its delta as universal "
 "or declares it as an operational tolerance and reports the statistic that lets you substitute "
 "your own. The second is defensible; the first is not.",
 "A decision utility is also a choice. Recovery under a fixed synthesis budget is not the only "
 "experimental utility, and a paper that converts a benchmark margin into ONE decision has said "
 "nothing about information gain, cost per synthesis or active-learning efficiency.",
]


def build(tag, folder, paper_dir, texname, numname, claims, title, one_line):
    print(f"\n=== {tag}")
    out = os.path.join(HERE, folder)
    os.makedirs(out, exist_ok=True)
    tex = open(os.path.join(paper_dir, texname), encoding="utf-8").read()
    mac = macros(paper_dir, numname)
    prov = json.load(open(os.path.join(paper_dir, "provenance.json")))

    # ---- submission.json : what is being audited, in the paper's own words
    abstract = expand(slice_between(tex, "\\begin{abstract}", "\\end{abstract}"), mac)
    quotes = []
    for c in claims:
        m = re.search(r"\\claimhead\{" + re.escape(c["label"]) + r"\}(.*?)\\end\{claimbox\}",
                      tex, re.S)
        src = None
        if m:
            src = m.group(1)
        else:
            # the companion manuscript states its claims as propositions and section leads rather
            # than in claim boxes, so fall back to the prose that follows the claim's own label
            pm = re.search(r"\\begin\{proposition\}[^\n]*\\label\{" + re.escape(c["label"])
                           + r"\}(.*?)\\end\{proposition\}", tex, re.S)
            if pm:
                src = pm.group(1)
            else:
                lm = re.search(r"\\label\{" + re.escape(c["label"]) + r"\}", tex)
                if lm:
                    src = tex[lm.end():lm.end() + 1100]
                    src = src.split("\\begin{")[0]
        if src:
            quotes.append({"claim_id": c["id"], "verbatim_as_printed": expand(src, mac)[:1400],
                           "where": c["section"]})
    sub = {
      "request_id": f"{tag}-manuscript-audit",
      "audited_object": "a finished manuscript plus the committed run artifacts behind it",
      "recommended_schema": "Sample A (manuscript / proof audit): decompose into atomic claims, "
                            "per claim a logic pass (T) with quoted evidence and a deterministic "
                            "recomputation pass (D); result track = verify_gate + reproducibility "
                            "by an independent route.",
      "title": title,
      "one_line": one_line,
      "abstract_as_printed": abstract,
      # The estimand is the first thing an auditor has to agree with us about, and leaving it to
      # be assembled from context is how a result track ends up auditing a different quantity.
      # Sliced from the manuscript so it cannot drift from what the paper says.
      "canonical_estimand_as_printed": (
        expand(slice_between(tex, "\\paragraph{What this instrument measures, stated once.}",
                             "\\paragraph{The two readings.}"), mac)
        if "What this instrument measures, stated once." in tex else None),
      "claim_quotes_as_printed": quotes,
      "independence_conditions": [
        "Do not import, call or execute any module under rna_harness/. The result track must not "
        "run the audited party's implementation.",
        "Do not take any number in claims.json:printed_value as an input. Those are the claims "
        "under audit, supplied so you know what to disagree with.",
        "The two public CSVs in artifacts.json:input_data are the substrate. Everything else in "
        "this repository is OUR output and is evidence about our process, not evidence for a "
        "number.",
        "Derive your own supervision plan. sanity_checks.md lists first-principles invariants "
        "this manuscript asserts; they are claims to be checked, not a plan to be adopted."],
      "_schema": "manuscript-audit-input/v1"}
    write(out, "submission.json", sub)

    # ---- claims.json : the ledger, with our numbers quarantined
    rows = []
    for c in claims:
        printed = {k: mac.get(k, None) for k in c["printed"]}
        rows.append({
          "id": c["id"], "axis": c["axis"], "section": c["section"],
          "estimand": c["estimand"],
          "what_would_falsify_it": c["falsifier"],
          "independent_route_SPECIFICATION_NOT_IMPLEMENTATION": c["independent_route"],
          "backing_artifacts": c["artifacts"],
          "companion_paper_artifacts": c.get("companion_artifacts", []),
          "printed_value": {"trust": "DO_NOT_TRUST_RECOMPUTE_INDEPENDENTLY", "macros": printed}})
    write(out, "claims.json", {
      "_note": "printed_value is the claim under audit, not an input to the audit. It is here so "
               "a disagreement can be located, and it is the only place our numbers appear.",
      "reference_column_disclosure":
        "The conventional structure-agreement readout on this corpus is scored against the column "
        "RNet_structure of OpenKnotBench_data.v4.5.1.csv. That column is the NEURAL PREDICTOR'S "
        "OWN PREDICTED STRUCTURE. The file contains no probing-derived structure column. This is "
        "disclosed because it makes one claim's comparator another claim's subject, and an "
        "auditor who does not know it will mis-scope both.",
      "claims": rows})

    # ---- artifacts.json : the inventory, with trust classes and checksums
    # C1: a package that only works on the machine that built it is not a package. The substrate
    # has to be obtainable, and the checksum is what makes "the same file" checkable.
    OBTAIN = {
      "release": "OpenKnotBench / OpenKnot AI design data, Eterna and the Das laboratory. Release "
                 "channel: https://github.com/eternagame/OpenKnotAIDesignData . A neighbouring "
                 "version of the M2 single-mutant SHAPE data is deposited at "
                 "https://doi.org/10.5281/zenodo.20101887 (v4.5.2).",
      "corpus_paper": "Townley J, Kladwang W, Baker D, et al. De novo design of RNA pseudoknots "
                      "with deep learning. Science 2026;393(6814):931-937. "
                      "doi:10.1126/science.aeg6829",
      "version_warning": "Both manuscripts use v4.5.1. If you obtain a different version the "
                         "sha256 below will not match. That is not a failure of the audit -- it "
                         "is a fact worth reporting, because a claim reproduced on a different "
                         "release is a weaker reproduction than one on the same bytes.",
      "local_path_note": "The absolute path in `path` is where the file sits on the machine that "
                         "produced the manuscripts. It is recorded so a reader of our run logs "
                         "can follow them, not because you are expected to have it."}
    inputs = [dict(role="INPUT DATA -- the substrate for the result track. Mutate-map-rescue "
                        "library; carries rescue_factor, target_structure, mutA/mutB, sub_start/"
                        "sub_end and the per-nucleotide reactivities.",
                   how_to_obtain=OBTAIN, **stat(M2R)),
              dict(role="INPUT DATA -- the benchmark panel. Source of the structure-agreement "
                        "column RNet_structure and of target_openknot_score.",
                   how_to_obtain=OBTAIN, **stat(BENCH))]
    ours = []
    for cls, entries in prov.items():
        if cls.startswith("_"):
            continue
        for k in entries:
            if k.startswith("_"):
                continue
            s = stat(os.path.join(CEIL, k))
            s.update(provenance_class=cls, tracked_by_git=tracked(os.path.join(CEIL, k)),
                     trust="OUR OUTPUT -- evidence about our process, never evidence for a number")
            ours.append(s)
    man = [dict(role="the manuscript", **stat(os.path.join(paper_dir, texname))),
           dict(role="generated numbers; every printed measurement lives here",
                **stat(os.path.join(paper_dir, "figs", numname))),
           dict(role="the number generator", **stat(os.path.join(paper_dir, "make_numbers.py"))),
           dict(role="the build gate that refuses a hand-typed measurement",
                **stat(os.path.join(paper_dir, "make_check.py"))),
           dict(role="process/provenance declaration",
                **stat(os.path.join(paper_dir, "provenance.json"))),
           dict(role="the repository's own audit tool",
                **stat(os.path.join(HERE, "audit_manuscript.py")))]
    comp = []
    for c in claims:
        for k in c.get("companion_artifacts", []):
            s2 = stat(os.path.join(CEIL, k))
            s2.update(trust="COMPANION PAPER OUTPUT -- inventoried because a claim points at it",
                      tracked_by_git=tracked(os.path.join(CEIL, k)))
            comp.append(s2)
    write(out, "artifacts.json", {
      "companion_paper_artifacts": comp,
      "input_data": inputs, "manuscript": man,
      "our_run_artifacts": sorted(ours, key=lambda r: r["path"]),
      "counts": {"input_data": len(inputs), "manuscript": len(man), "our_outputs": len(ours),
                 "companion": len(comp)}})

    # ---- process_claims.json
    write(out, "process_claims.json", {
      "_note": "claims the manuscript makes ABOUT ITS OWN PROCESS. Each is falsifiable from the "
               "repository without any answer key, which is what makes them the process track's "
               "natural material.",
      "claims": PROCESS_CLAIMS})

    # ---- sanity_checks.md
    write(out, "sanity_checks.md",
          "# First-principles invariants this manuscript asserts\n\n"
          "These are **not target values** and **not a supervision plan**. They are properties "
          "that must hold whatever the numbers turn out to be. Derive your own plan; use this "
          "list only to check that the manuscript did not violate something it itself relies on.\n\n"
          + "\n".join(f"{i+1}. {s}\n" for i, s in enumerate(SANITY)), raw=True)
    return out


README = """# Audit input package -- {title}

Generated by `paper/build_audit_packages.py`. Do not hand-edit: rerun the generator instead, so
every checksum here matches the repository state it describes.

## What is being audited

{one_line}

The audited object is a **finished manuscript plus the committed run artifacts behind it**. By the
audit system's own routing note, that is a **Sample A** job: decompose the submission into atomic
claims, run a logic pass (`T`) with quoted evidence per claim, run a deterministic recomputation
pass (`D`), and settle the result track with `verify_gate` + `reproducibility` **by an independent
route**.

## Files

| file | track | what it is |
|---|---|---|
| `submission.json` | both | the abstract and every claim box **as printed**, with macros expanded, plus the independence conditions |
| `claims.json` | result | one row per load-bearing claim: the estimand written out, what would falsify it, an **independent route stated as a specification**, and -- quarantined -- the number we print |
| `artifacts.json` | both | the full inventory with sha256: two public input CSVs, the manuscript files, and every one of our run outputs, each tagged with its trust class |
| `process_claims.json` | process | thirteen claims the manuscript makes **about its own process**, each with the command or comparison that would falsify it |
| `sanity_checks.md` | process | first-principles invariants the manuscript relies on. Not a plan, not target values |
| `criterion.md` | result | the eligibility criterion in prose, sufficient to rebuild the admitted set from the public CSV without our code |
| `estimator.md` | result | the estimators in prose and symbols: the matched stratified difference, the clustering, the permutation, the depth integer |

## The independence rules, restated because they are the point

1. **Do not run our code.** Nothing under `rna_harness/` may be imported or executed for the result
   track. If you run our estimator and get our number, you have measured nothing.
2. **Do not accept our numbers as input.** `claims.json:printed_value` is the claim under audit.
   It is supplied so a disagreement can be located, and it is the only place our numbers appear.
3. **The substrate is two public CSVs.** Everything else in the repository is our output: it is
   evidence about our process, never evidence for a number.
4. **Derive your own supervision plan.** `sanity_checks.md` is a list of things we assert, not a
   list of things you should assume.

## One disclosure the auditor needs before starting

The conventional structure-agreement readout on this corpus is scored against the column
`RNet_structure`, **which is the neural predictor's own predicted structure**. The benchmark file
contains no probing-derived structure column. This makes one claim's comparator another claim's
subject. It is stated in the manuscript; it is repeated here because an auditor who does not know
it will mis-scope both claims.

## Where the manuscript expects to be found wanting

Stated up front so that finding them is not mistaken for the whole audit:

- The cluster bootstrap has **20 clusters**. Intervals are optimistic and the manuscript says so.
- There is **no multiplicity correction** across arms.
- Two covariate-choice channels are **named and not priced**: who chose which pairs received a
  compensatory mutant, and the released checkpoint's pseudo-label stage.
- Exposure counts are **lower bounds**, because the training file we obtained is a subset.
- Two load-bearing review analyses have **no frozen declaration** (disclosed in the manuscript).
- **Eight preregistered stakes failed** across five experiments. Each failure has a declared
  consequence that was written before the number existed, and each consequence should be traceable
  to a specific change in the text. Checking that trace is process claim `P11`, and it is the most
  informative single thing an auditor can do with this package.
- **The free integer cannot be separated from "the declared pair is really there."** Its rank
  agreement exceeds that of the bare confirmation indicator, but the interval on the difference
  covers zero. The manuscript states this in the limitations; a reader who prefers the mechanical
  reading of the baseline is not contradicted by anything measured here.
- **One claim's interval is method-dependent.** The readout comparison against the shipped
  agreement column excludes zero under the percentile method and covers it under BCa. That column
  is also the smallest of the five gaps; the four others are not method-dependent.
- **The pseudoknot-capable member is not an independent opinion.** It is the panel's own energy
  model with the pseudoknot search on. It isolates a capability; it does not widen the panel.

A finding that lands on one of these is still a finding; it is not a surprise, and the manuscript
should be checked for whether it discloses the limitation where the number appears, not merely in
a limitations section.
"""

CRITERION = """# The eligibility criterion, in prose

Enough to rebuild the admitted set from `OK7a_M2R_data.v4.5.1.csv` **without our code**. Written
as a rule, not as an implementation. No threshold below can be tuned to raise the yield.

## Unit

One row of the CSV is one construct. A construct is admitted as an **intervention** when it is a
compensatory double mutant of a declared target base pair of its own wild type.

## The rule, in order, with every refusal counted

1. The row must carry an outcome and pass the corpus' own signal filter: `rescue_factor` present
   and `SN_filter == 1`. Otherwise **refused: no outcome**.
2. The row's wild type is the row of the same `(puzzle, method)` whose `mutA` and `mutB` are both
   empty. If there is none, **refused: no wild type**.
3. `mutA`, `mutB` are 1-based positions inside the design window. Convert to absolute 0-based
   indices with `a = mutA + sub_start - 2`, `b = mutB + sub_start - 2`. Unparsable positions are
   **refused**.
4. Parse `target_structure` into a pair map. `(a,b)` must be a declared pair of it. Otherwise
   **refused: not a declared pair**.
5. The unordered wild-type pair type at `(a,b)` must be one of `CG`, `AU`, `GU`. Otherwise
   **refused: non-canonical**.
6. **The intervention must be exactly two positions.** Compare the wild-type and mutant sequences
   over the design window `[sub_start-1, sub_end)`; the set of differing positions must equal
   `{a,b}` exactly. Anything else is **refused: not exactly two**. This refusal is the one that
   makes the experiment controlled, and its count is reported in the manuscript.

Ambiguity is refused rather than resolved at every step.

## Covariates carried, never used as selection

- **helix depth** `min(dep, up)`, where `dep` counts consecutive `k >= 0` with `(a-k-1, b+k+1)`
  paired and `up` counts consecutive `k >= 0` with `(a+k+1, b-k-1)` paired. Read off the
  **declared target structure**. No model, no folding, no training data.
- **crossing**: whether any declared pair `(x,y)` crosses `(a,b)`, i.e. `x < a < y < b` or
  `a < x < b < y`. The corpus is a pseudoknot round, so crossing is carried as a covariate and
  never as a selection rule.
- **design cell** `(puzzle, method)`.

## The outcome

`rescue_factor` is used **unaltered and untruncated, negative values included**. It is a released
column of the corpus; the manuscript reproduces it from the per-nucleotide reactivities rather
than quoting a formula from a paper, and states that it does not assert the normalisation is
identical. **The outcome contains no folding model.** If your reimplementation introduces one,
you have changed the claim.
"""

ESTIMATOR = """# The estimators, in prose and symbols

Enough to reimplement from the public CSV. No function of ours is named.

## Matched stratified difference in means

For a binary exposure `T in {A,B}` (here: target pair type G-C against A-U) and strata `s`:

    Delta = sum_s w_s * ( mean(Y | T=A, s) - mean(Y | T=B, s) )  /  sum_s w_s
    w_s   = min( n_A(s), n_B(s) )

**Strata missing an arm are dropped, never extrapolated into.** Report how many were dropped and
what weight they carried; the count is part of the estimand.

Two stratification sets are used and they answer different questions:

- **causal set** `s = (design cell, min(depth,3), crossing)` -- holds fixed what a design program
  chose, because placement confounds pair type;
- **prediction set** `s = (design cell)` alone -- for the question an experimenter faces given a
  design and a budget. Depth is information the experimenter already holds, so removing it would
  remove part of the answer.

Using one set to answer the other's question is the single easiest way to mis-audit this paper.

## Inference

- **Permutation**: labels are permuted **within stratum**. Permuting across the matched set
  rebuilds a different matched configuration and narrows the null; the manuscript quantifies that
  error rather than asserting it.
- **Bootstrap**: resample the **20 puzzles** (clusters), not the interventions. One interval in
  the companion manuscript is the exception -- it resamples repeated measurement groups -- and it
  is flagged where it appears.
- **Differences of two quantities** are computed **inside each resample** and the interval is
  taken on the differences. Two separately-resampled intervals do not answer a question about a
  difference. The manuscript retracts one comparison for exactly this reason.

## Within-design rank agreement

Within each design cell holding at least 8 admitted interventions, the Spearman rank correlation
between a score and the measured rescue factor; averaged across cells weighted by cell size;
cluster bootstrap over puzzles; permutation shuffling the outcome **inside the cell**.

## Standardisation before comparing scales

Two contrasts on different outcome columns are **not** comparable raw. Standardise each by the
pooled within-stratum standard deviation of its own outcome before comparing magnitudes. A
recomputation that compares them raw has reproduced a different claim from the one made.

## The quantity this work withdraws

A **mediated fraction** -- a ratio of two contrasts, presented as the share of an effect running
through a mediator. Calibration against an injected known truth shows the estimator's reading
covers only a small part of the range the truth covers. **Any number of that form, anywhere in
either manuscript including a table caption, is a finding.**
"""


NM_CHANGELOG = """# What changed since the last package

Previous package **2026-08-31 02:30**. Rebuilt twice on **2026-09-01**: once after a round of new
experiments, and again after a review round that found a factual error in our own wording.
Regenerate with `python paper/build_audit_packages.py`, validate with
`python paper/check_audit_packages.py`.

The independence rules in `README.md` are unchanged: do not run our code, do not accept our
numbers as input, the substrate is the two public CSVs.

---

## 0. Read this first: a sentence in our abstract was false, and this package repeated it

The previous package's specification for `NM-C9` described the probing-derived reference the way
the manuscript did --- as a structure built from probing data with **no model at all**.

**That was wrong.** The reference is a nearest-neighbour thermodynamic fold *restrained* by
pseudoenergies derived from the measured reactivities. It has an energy model, and that energy
model is **shared with one implementation in the scored panel**. Our own run log for that
experiment recorded the sharing; the abstract contradicted it.

The manuscript is corrected and so is `claims.json`. The narrower property, which is all the
comparison ever needed, is that the column contains **no prediction of these molecules made from
sequence**. `sanity_checks.md` now carries the general form of the trap: *nothing in this work is
model-free*, and the outcome containing no folding model is a different and much narrower
statement than a reference being model-free.

An auditor should treat this as a live check rather than a closed one: verify the correction
reached **every** place the column is described, not only the abstract.

## 1. A second wording claim was also too strong

The manuscript said the identity comparison is made "with no reference structure in the
comparison." The declared target does enter --- it says which pair is the target, and it keys the
strata through helix depth and crossing. The supportable form, now used, is that no **predicted
structure of these molecules** enters the measured outcome. `sanity_checks.md` names all four
places the declared target does enter, so an auditor can check the manuscript against the list
rather than against a vague claim.

## 2. The estimand is now stated canonically, and shipped in this package

`submission.json` gained `canonical_estimand_as_printed`, sliced from the manuscript:

> This instrument does not measure whether a predicted structure is globally correct. It measures
> whether a predictor's probability for a **named** base pair anticipates the measured consequence
> of a **specific** compensatory intervention on that pair, in a molecule whose own wild type is
> the internal control.

This is the first thing the result track has to agree with us about. A recomputation that scores a
structure-level quantity has audited something else. Process claim `P14` is the check.

## 3. Three scoping choices are now declared as choices

| choice | what the manuscript now says |
|---|---|
| equivalence margin | delta is a **declared operational tolerance** --- one construct in the median design --- not a universal scientific equivalence margin. Delta-star is reported per comparison so you can substitute your own and regrade without rerunning anything |
| decision utility | budgeted recovery is **one operational decision**, not the definition of experimental utility. Information gain, cost per synthesis and active-learning efficiency are different decisions and the paper says nothing about them |
| the free baseline | Claim 3 is about a **baseline**, not a mechanism. Helix depth may be carrying real structural depth, the accuracy of the designer's declaration, or both; the paper separates neither and says so |

Process claim `P15` asks whether each is presented as a choice or as a fact.

## 4. Ledger and package size

|  | 2026-08-31 | now |
|---|---|---|
| claims | 6 | **12** |
| process claims | 10 | **15** |
| sanity checks | 7 | **20** |
| printed values quarantined | 57 | **150** |
| our run artifacts inventoried | 31 | **53** |

## 5. The six claims added earlier today, still the substance of this update

| id | axis | why it exists |
|---|---|---|
| `NM-C7` | the budgeted decision | **it was missing from every previous package.** It is the manuscript's Reading-2 headline --- the claim the whole second half exists to support --- and the auditor was never handed it. Recorded rather than added quietly |
| `NM-C8` | the criterion's own validity | clause (iii) admits an intervention because a design program *declared* the pair; we now measure how often the molecule agrees, and it is not always |
| `NM-C9` | a reference nobody in the panel wrote | see item 0 |
| `NM-C10` | what the panel can represent | a second transformer, and a pseudoknot-capable implementation on a corpus that is a pseudoknot round |
| `NM-C11` | interval method dependence | the three claim-carrying intervals under a second method |
| `NM-C12` | how strong a violation would have to be | the closed-form sensitivity bound on the one identification assumption |

## 6. Three existing claims changed meaning, not just numbers

- **`NM-C2` is narrower.** A second transformer does not reproduce the architecture axis, and a
  pseudoknot-capable thermodynamic implementation does not resolve the identity separation. The
  axis is a property of **one model**. Any sentence still stating it as a class fact is a finding.
- **`NM-C3` gained a measured domain.** The free integer's advantage lives almost entirely where a
  probing-derived reference confirms the intervened pair; in the unconfirmed stratum **nothing is
  resolved**. Check the manuscript reports that as a domain and never as a defeat --- and equally
  that it does not report the unconfirmed stratum as a win.
- **`NM-C1` is made against five agreement columns**, and the shipped one --- the smallest of the
  five gaps --- is the one whose interval is method-dependent.

## 7. Where we already know we are weak

- **Eight preregistered stakes failed**, graded as failed in the declaration's own wording.
  Checking that each declared consequence was actually applied to the text is `P11`, and it is the
  highest-value single check in this package.
- **The free integer cannot be separated from "the declared pair is really there."** The interval
  on the difference covers zero. A reader who prefers the mechanical reading of the baseline is
  not contradicted by anything we measured.
- **One claim's interval is method-dependent** (see `NM-C11`).
- **The pseudoknot-capable member is not an independent opinion** --- it is the panel's own energy
  model with the pseudoknot search on. It isolates a capability; it does not widen the panel.
- **The predictor panel spans four families and is not a field**, which the manuscript states.
  Adding a genuinely orthogonal contemporary predictor is open work, not a completed control.

## 8. State of the manuscript

```
72 pages   build exit 0   0 overfull boxes   make_check clean
audit: 1 blocking finding
```

The single blocking finding is two placeholders in the code-and-data-availability paragraph: the
public repository URL and the Zenodo DOI, deferred until acceptance by decision. An auditor should
note that a software-methods venue will treat an unreleased tool as a submission-readiness
failure independently of anything scientific in this package.
"""


def main():
    specs = [
      ("NM", "intervention_benchmark_audit", os.path.join(HERE, "intervention_benchmark"),
       "NM_intervention.tex", "nm_numbers.tex", NM_CLAIMS,
       "Scoring RNA Structure Prediction on Interventions That Have Already Been Performed",
       "An instrument paper. It recovers controlled two-nucleotide interventions already executed "
       "in a public chemical-probing corpus, scores three structure predictors by their response "
       "rather than by agreement with a reference, and prices eight channels through which a "
       "benchmark number can be carried by a covariate somebody chose."),
      ("R1", "rna_placement_audit", os.path.join(HERE, "rna_placement"),
       "R1_placement.tex", "rna_numbers.tex", R1_CLAIMS,
       "The estimator companion: when a share of a benchmark number is estimable, and why here it "
       "is not",
       "An estimator paper. It gives the exact condition under which two standard routes to a "
       "mediated fraction agree, shows by calibration against an injected known truth that here "
       "neither is readable, withdraws every mediated fraction it previously reported, and prices "
       "two covariate channels directly instead."),
    ]
    made = []
    for tag, folder, pdir, tname, nname, claims, title, one in specs:
        out = build(tag, folder, pdir, tname, nname, claims, title, one)
        write(out, "README.md", README.format(title=title, one_line=one), raw=True)
        write(out, "criterion.md", CRITERION, raw=True)
        write(out, "estimator.md", ESTIMATOR, raw=True)
        if tag == "NM":
            # what changed since the previous package, so the recipient does not
            # have to diff two folders to find out what is new to audit
            write(out, "CHANGELOG.md", NM_CHANGELOG, raw=True)
        # the package checksums itself last, so the recipient can verify what they received
        files = sorted(f for f in os.listdir(out) if f != "MANIFEST.json")
        write(out, "MANIFEST.json", {
          "package": folder, "schema": "manuscript-audit-input/v1",
          "built_by": "paper/build_audit_packages.py",
          "regenerate_with": "python paper/build_audit_packages.py",
          "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                     capture_output=True, text=True).stdout.strip(),
          "files": [{"name": f, "bytes": os.path.getsize(os.path.join(out, f)),
                     "sha256": sha(os.path.join(out, f))} for f in files]})
        made.append(out)
    print("\nPackages built:")
    for m in made:
        n = sorted(os.listdir(m))
        print(f"  {os.path.relpath(m, REPO)}  ({len(n)} files: {', '.join(n)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
