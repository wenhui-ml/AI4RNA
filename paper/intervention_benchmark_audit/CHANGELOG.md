# What changed since the last package

Previous package built **2026-08-31 02:30**. This one **2026-09-01**. Regenerate with
`python paper/build_audit_packages.py`, validate with `python paper/check_audit_packages.py`.

Everything below is a change to **what is being audited**, not to how. The independence rules in
`README.md` are unchanged: do not run our code, do not accept our numbers as input, the substrate
is the two public CSVs.

---

## 1. A gap in the previous package, stated first because it is ours

**The manuscript's Reading-2 headline claim was not in the claim ledger.** `claim:budget` --- that
no implementation outperforms a free integer once an agreement margin is spent as experiments ---
had no row in `claims.json` in any package built before today. The auditor was given five claims
and a retraction, and was never handed the claim the entire second half of the paper exists to
support.

It is now **`NM-C7`**, with its estimand, its falsifiers and an independent route. We record the
omission here rather than adding the row quietly, because a claim ledger that dropped the paper's
most consequential claim is itself a process finding, and `process_claims.json:P11` invites you to
look for others of the same kind.

## 2. Ledger size

|  | before | now |
|---|---|---|
| claims | 6 | **12** |
| printed values quarantined | 57 | **150** |
| our run artifacts inventoried | 31 | **53** |

## 3. The six new claims

| id | axis | why it exists |
|---|---|---|
| `NM-C7` | the budgeted decision | the omission above |
| `NM-C8` | the criterion's own validity | clause (iii) admits an intervention because a design program *declared* the pair. We now measure how often the molecule agrees, and it is not always |
| `NM-C9` | a reference nobody in the panel wrote | the agreement axis re-scored against a structure folded from this deposit's own probing rather than against a designer's intention |
| `NM-C10` | what the panel can represent | two panel-composition controls added after review: a second transformer, and a pseudoknot-capable implementation on a corpus that is a pseudoknot round |
| `NM-C11` | interval method dependence | the three claim-carrying intervals recomputed by a second method |
| `NM-C12` | how strong a violation would have to be | the closed-form sensitivity bound on the paper's one identification assumption |

## 4. Three existing claims changed meaning, not just numbers

- **`NM-C2` (pair identity) is narrower than it was.** A second transformer sharing neither
  backbone nor training corpus does **not** reproduce the architecture axis, and a
  pseudoknot-capable thermodynamic implementation does not resolve the identity separation either.
  The axis is now stated as a property of **one model**, not of an architecture class. If you find
  any sentence in the manuscript still stating it as a class fact, that is a finding.
- **`NM-C3` (the free baseline) gained a measured domain.** Split on whether a probing-derived
  reference confirms the intervened pair, the free integer's advantage lives almost entirely in
  the confirmed stratum; in the unconfirmed stratum **nothing is resolved** and the transformer's
  point estimate is the larger. Check that the manuscript reports that as a domain and never as a
  defeat, and equally that it does not report the unconfirmed stratum as a win.
- **`NM-C1` (readout scale) is now made against five agreement columns, not one**, and one of the
  five is method-dependent. See §6.

## 5. Three new process claims

- **`P11` --- declared consequences were actually applied.** Eight preregistered stakes failed
  across five experiments. Each declaration names, *before the number existed*, what a failure
  would do to the manuscript. **This is the highest-value single check in the package**: a declared
  narrowing that was not applied to the text is the most serious finding available here, because
  the declaration is the only thing that makes a stake mean anything.
- **`P12` --- declarations were not edited after the fact.** Where a declaration was itself wrong,
  the error is recorded as an erratum and the file is left alone. Three errata are claimed. Diff
  each declaration's first git blob against its current one; any difference is a finding.
- **`P13` --- quantities computed and not printed are disclosed.** Run our audit tool and read
  check `[B]`. Then ask, for each unprinted quantity, whether printing it would have weakened a
  claim.

## 6. Where we already know we are weak (added to `README.md`)

- **Eight stakes failed.** They are graded as failed in the manuscript's preregistration appendix,
  in the declaration's own wording.
- **The free integer cannot be separated from "the declared pair is really there."** Its rank
  agreement exceeds the bare confirmation indicator's, but the interval on the difference covers
  zero. A reader who prefers the mechanical reading of the baseline is not contradicted by anything
  we measured, and the limitations say so.
- **One claim's interval is method-dependent.** The readout comparison against the shipped
  agreement column excludes zero under the percentile method and covers zero under BCa --- and not
  because BCa is wider here, but because the bias correction shifts it. That same column is the
  *smallest* of the five gaps; the four others are not method-dependent.
- **The pseudoknot-capable member is not an independent fifth opinion.** It is the panel's own
  energy model with the pseudoknot search switched on. It isolates a capability. It does not widen
  the panel, and the paper does not claim it does.

## 7. Eight new first-principles traps in `sanity_checks.md`

Two are worth reading before you write any code:

- **A dot-bracket parser for this corpus must keep one stack per bracket type.** It is a pseudoknot
  benchmark; the strings use several bracket families. A single-stack parser silently mis-pairs and
  every downstream number inherits it.
- **A number printed to four decimals carries ±5e-5 of rounding.** A reproduction tolerance of 5e-5
  against a printed value is arithmetically unreachable. Read targets from the artifacts.

The rest cover the malformed structure column shipped with the deposit, the two cell sets that are
not the same set, the one-generator permutation-then-bootstrap stream, and the rule that a scored
panel member must see sequence only.

## 8. State of the manuscript itself

```
71 pages   build exit 0   0 overfull boxes   make_check clean
audit: 1 blocking finding
```

The single blocking finding is two placeholders in the code-and-data-availability paragraph: the
public repository URL and the Zenodo DOI. Both are deliberately deferred until the manuscript is
accepted, by decision. Nothing else in the audit is blocking.
