# What changed since the last package

Rebuilt **2026-09-01**, third build of the day. The first two added experiments and corrected a
factual error; this one **realigns the package to the audit system's actual schema**, which we had
only a verbal description of before.

Regenerate with `python paper/build_audit_packages.py`, validate with
`python paper/check_audit_packages.py`.

---

## 0. Schema alignment (this build)

The package now speaks the schema's vocabulary instead of an invented one.

| added | what it does |
|---|---|
| `submission.json:route` | declares **Sample A** and says why: a finished document making discrete claims, not a live agent trajectory |
| `submission.json:original_submission[]` | `who` / `quote` / `maps_to`, so our printed sentences can be attached to whatever atomic-step decomposition the auditor builds |
| `submission.json` U-calibration | `claimed_level` + `matched_strong` / `matched_hedge` per claim, **extracted by token match on the printed sentence**, not declared by us. An auditor can re-run the same match |
| `verify_gate.json` | new file. Per claim: the artifacts holding the number with sha256, the verbatim printed sentence, and the items a gate must check. **Every `ok` and `evidence_sufficient` is null** |
| `process_claims.json` axes | each process claim tagged `integrity` or `validity`, matching the `deep_audit` split, so a finding routes without the auditor re-deriving that |
| `process_claims.json` forbidden shortcuts | six shortcuts we declare forbidden **against ourselves**, in the schema's own `forbidden_shortcuts` sense |
| checker `F3`/`F4` | the package's own validator now refuses a `verify_gate.json` with a pre-filled verdict. A gate file with `ok` already set is an answer sheet wearing a checklist's clothes |

**Two differences from the mathematical Sample A** that change how this should be graded:

1. The `D` pass here is **not exact arithmetic**. It runs against two public CSVs, so a
   deterministic check returns numeric agreement within a stated tolerance rather than an
   identity. Where an exact identity *is* available --- a closed-form maximum, a bias-correction
   degenerate case, a partition count --- `claims.json` says so and it should be checked as one.
2. There is **no single answer to reproduce**. Twelve claims are supplied and several are negative
   or unresolved by construction; reproducing one of those means reproducing an interval that
   covers zero, not a point.

Still deliberately absent, because they are the auditor's: **the supervision plan**, **the atomic
step decomposition**, and **every verdict**.

## 1. Read this next: a sentence in our abstract was false, and this package repeated it

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

## 2. A second wording claim was also too strong

The manuscript said the identity comparison is made "with no reference structure in the
comparison." The declared target does enter --- it says which pair is the target, and it keys the
strata through helix depth and crossing. The supportable form, now used, is that no **predicted
structure of these molecules** enters the measured outcome. `sanity_checks.md` names all four
places the declared target does enter, so an auditor can check the manuscript against the list
rather than against a vague claim.

## 3. The estimand is now stated canonically, and shipped in this package

`submission.json` gained `canonical_estimand_as_printed`, sliced from the manuscript:

> This instrument does not measure whether a predicted structure is globally correct. It measures
> whether a predictor's probability for a **named** base pair anticipates the measured consequence
> of a **specific** compensatory intervention on that pair, in a molecule whose own wild type is
> the internal control.

This is the first thing the result track has to agree with us about. A recomputation that scores a
structure-level quantity has audited something else. Process claim `P14` is the check.

## 4. Three scoping choices are now declared as choices

| choice | what the manuscript now says |
|---|---|
| equivalence margin | delta is a **declared operational tolerance** --- one construct in the median design --- not a universal scientific equivalence margin. Delta-star is reported per comparison so you can substitute your own and regrade without rerunning anything |
| decision utility | budgeted recovery is **one operational decision**, not the definition of experimental utility. Information gain, cost per synthesis and active-learning efficiency are different decisions and the paper says nothing about them |
| the free baseline | Claim 3 is about a **baseline**, not a mechanism. Helix depth may be carrying real structural depth, the accuracy of the designer's declaration, or both; the paper separates neither and says so |

Process claim `P15` asks whether each is presented as a choice or as a fact.

## 5. Ledger and package size

|  | 2026-08-31 | now |
|---|---|---|
| claims | 6 | **12** |
| process claims | 10 | **15** |
| sanity checks | 7 | **20** |
| printed values quarantined | 57 | **150** |
| our run artifacts inventoried | 31 | **53** |

## 6. The six claims added earlier today, still the substance of this update

| id | axis | why it exists |
|---|---|---|
| `NM-C7` | the budgeted decision | **it was missing from every previous package.** It is the manuscript's Reading-2 headline --- the claim the whole second half exists to support --- and the auditor was never handed it. Recorded rather than added quietly |
| `NM-C8` | the criterion's own validity | clause (iii) admits an intervention because a design program *declared* the pair; we now measure how often the molecule agrees, and it is not always |
| `NM-C9` | a reference nobody in the panel wrote | see item 0 |
| `NM-C10` | what the panel can represent | a second transformer, and a pseudoknot-capable implementation on a corpus that is a pseudoknot round |
| `NM-C11` | interval method dependence | the three claim-carrying intervals under a second method |
| `NM-C12` | how strong a violation would have to be | the closed-form sensitivity bound on the one identification assumption |

## 7. Three existing claims changed meaning, not just numbers

- **`NM-C2` is narrower.** A second transformer does not reproduce the architecture axis, and a
  pseudoknot-capable thermodynamic implementation does not resolve the identity separation. The
  axis is a property of **one model**. Any sentence still stating it as a class fact is a finding.
- **`NM-C3` gained a measured domain.** The free integer's advantage lives almost entirely where a
  probing-derived reference confirms the intervened pair; in the unconfirmed stratum **nothing is
  resolved**. Check the manuscript reports that as a domain and never as a defeat --- and equally
  that it does not report the unconfirmed stratum as a win.
- **`NM-C1` is made against five agreement columns**, and the shipped one --- the smallest of the
  five gaps --- is the one whose interval is method-dependent.

## 8. Where we already know we are weak

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

## 9. State of the manuscript

```
72 pages   build exit 0   0 overfull boxes   make_check clean
audit: 1 blocking finding
```

The single blocking finding is two placeholders in the code-and-data-availability paragraph: the
public repository URL and the Zenodo DOI, deferred until acceptance by decision. An auditor should
note that a software-methods venue will treat an unreleased tool as a submission-readiness
failure independently of anything scientific in this package.
