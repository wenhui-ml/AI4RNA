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
        "correct; a recomputation that compares them raw has not reproduced the claim."],
      artifacts=["m36_readout_scale.json", "m25_estimator_benchmark.json"],
      printed=["cvIntvSd", "cvConvSd", "cvPairedSd", "cvPairedSdCi", "cvSwing"]),
 dict(id="NM-C2", label="claim:identity", axis="pair identity",
      section="Results / what the instrument reads out",
      estimand="For each of three predictors, the matched stratified difference in the predictor's "
               "OWN output for the mutant target pair, between G-C-target and A-U-target "
               "interventions, on the same strata as the measured outcome; plus the same "
               "difference on the measured rescue factor. Claim: the neural predictor's output "
               "separates the two identities, the two nearest-neighbour predictors differ from "
               "each other, and the separation decomposes into an architecture axis and a "
               "training-data axis whose difference has an interval.",
      falsifier="any pair of predictors is compared without an interval on the DIFFERENCE; or the "
                "three predictor columns are not read on the same records; or the 1-based vs "
                "0-based indexing of the two prediction sources is crossed",
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
      artifacts=["m35_paired_predictor_differences.json", "m26_model_scored.json"],
      printed=["resNeural", "resNeuralCi", "resVienna", "resViennaCi", "efContrast",
               "efContrastCi", "pdNVCi", "pdNECi", "pdVECi", "axArch", "axArchCi", "axData",
               "axDataCi", "axGap", "axGapCi"]),
 dict(id="NM-C3", label="claim:depth", axis="free baseline",
      section="Results / which flips rescue, and the free baseline nobody reports",
      estimand="Within each design cell of at least 8 admitted interventions, the Spearman rank "
               "correlation between a score and the measured rescue factor, averaged across cells "
               "weighted by cell size, with a cluster bootstrap over the 20 puzzles. Scores: each "
               "predictor's pair probability, and HELIX DEPTH -- an integer read off the declared "
               "target structure, requiring no model. Claim: the paired difference (predictor "
               "minus depth) has an interval covering zero for every predictor, so no predictor "
               "beats the free integer; and three purpose-built mutation-effect scores lose to it "
               "with intervals that EXCLUDE zero on the losing side.",
      falsifier="the difference is read as two point estimates rather than as one interval on the "
                "difference -- which is the error the paper itself retracts here; or depth is "
                "computed from a predicted structure rather than from the declared target; or the "
                "bootstrap resamples interventions rather than puzzles",
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
                 "m38_ribosnitch_baseline.json", "m40v_verify_m38_m39.json"],
      printed=["rkRhoD", "rkRhoDCi", "rkRhoN", "rkDiffNCi", "rkDiffVCi", "efVsDepthCi",
               "rkPartN", "rkPartNCi", "rsBppm", "rsBppmCi", "rsBppmVs", "rsBppmVsCi",
               "rsMfeVs", "rsMfeVsCi", "rsEnsVs", "rsEnsVsCi"]),
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
      artifacts=["m26r3_calibration.json", "m26r4_calibration_symmetry.json"],
      printed=["mcNeuralSpanFrac", "mcViennaSpanFrac"]),
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
      artifacts=[],
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
      artifacts=["m26r3_calibration.json", "m26r4_calibration_symmetry.json"],
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
      artifacts=["m20_designer_confounding.json", "m37_channel_ledger.json"],
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
      artifacts=[],
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
      artifacts=[],
      printed=[]),
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
    inputs = [dict(role="INPUT DATA -- public, independently re-derivable, the substrate for the "
                        "result track", **stat(M2R)),
              dict(role="INPUT DATA -- public, the benchmark panel; source of the agreement column",
                   **stat(BENCH))]
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
    write(out, "artifacts.json", {
      "input_data": inputs, "manuscript": man,
      "our_run_artifacts": sorted(ours, key=lambda r: r["path"]),
      "counts": {"input_data": len(inputs), "manuscript": len(man), "our_outputs": len(ours)}})

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
| `process_claims.json` | process | ten claims the manuscript makes **about its own process**, each with the command or comparison that would falsify it |
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
