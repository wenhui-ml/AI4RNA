# M26c audit — the neural model, and what it did not absorb

Declaration `c22b1ddd`, frozen before the code existed. Offline, $0.

## What ran

RibonanzaNet-SS, the secondary-structure head of a transformer trained on ~2M chemical-mapping
profiles, scored on the **same 4317 interventions** as M26 (P1 asserted byte-identical admission,
and it passed: 4317 = 4317, 0 dropped). The head is a (1,64) linear on symmetrised pairwise
features; the checkpoint loaded with **0 missing and 0 unexpected keys**, so the architecture is
not a reconstruction of ours -- it is theirs.

## P6, which had to pass before any retention number could be read

    max asymmetry        0.00e+00
    range                [0.0000, 1.0000]
    on-target mean       0.7812
    random position pair 0.0026          -> 303x

The index convention is right and the model is not emitting noise.

## The eight staked predictions

| # | staked | outcome |
|---|--------|---------|
| 1 | L1 exact overlap = 0 | **deferred** — needs the training sequences (Kaggle, auth) |
| 2 | N1 positive skill, p<=0.005 | **correct** (+0.20692, p=0.0020) |
| 3 | neural slope beats ViennaRNA's +0.12059 | **correct** (+0.207 vs +0.121) |
| 4 | N3 calibration monotone | **WRONG** — bin4 (+0.0294) < bin3 (+0.0355) |
| 5 | N2 retention >= 50% | **correct** (100.7%) |
| 6 | mediated CI contains zero | **correct** ([-13.5%, +12.1%]) |
| 7 | N5 correlation < 0.95 | **correct** (Pearson +0.488, Spearman +0.575) |
| 8 | N4 joint retention >= 40% | **correct** (97.1%) |

## The result

    N2   CDE (fixing P_neural)        +0.0574   p 0.00050   cluster [0.0248, 0.0902]
         same-population baseline     +0.0570
         retention                    100.7%    mediated -0.7% [-13.5%, +12.1%]
         placebo                      103.9% +/- 6.5%   z = -0.48

    N4   CDE (fixing BOTH models)     +0.0752   p 0.00050
         retention                    97.1%     mediated +2.9%

Fixing a thermodynamic model's ensemble prediction left 96.3%. Fixing a neural model's prediction
leaves 100.7%. Fixing **both at once** leaves 97.1%. The two model classes together absorb
essentially none of the pair-identity effect.

## Three things that must travel with the number

**(a) The neural model is nearly binary here.** Bins 1-4 all sit at P_neural >= 0.83, and bin 4 is
0.9994. Its skill slope is real and it is larger than ViennaRNA's, but most of it comes from
separating bin 0 from everything else. That is also why calibration is not monotone at the top --
prediction 4 lost, and it lost for a reason worth reporting rather than explaining away.

**(b) L4 is the only leakage evidence we have so far.** Skill on wild types (+0.24929) and on
mutants (+0.20692) is comparable. Had the model memorised these molecules, wild-type skill should
tower over mutant skill, because the M2R mutant library postdates the weights. It does not. But
this is a backstop, not the audit: **L1/L2/L3 remain unrun**, and until they run the phrase
"across model classes" carries an asterisk in the manuscript, not a footnote in the appendix.

**(c) Two classes is two classes.** The declaration forbids "all existing models" in advance and
that stands. Both classes evaluated here take sequence in and emit a per-pair number; neither was
given the wet data for the molecule it is scoring.

## Retention above 100%

Recorded as `shape_1` per the declaration: not read as stronger evidence. The same-population
denominator was used throughout (+0.0570 on the retained subset vs +0.0539 on the full set), so
the excess is not the selection artefact that produced M21's first 153%.
