# M26d + M26r audit — what the review found, including in itself

## M26d: the clean-subset rerun

    leaked design cells        14 / 158 = 8.9%      staked <=10%   correct
    clean interventions        3993     leaked 324

    RibonanzaNet   all    slope +0.20692   retention 100.7%
                   clean  slope +0.21549   retention 100.2%   staked >=50%   correct
                   leaked slope +0.10022   K3 fired, weight 22, retention not reported
    ViennaRNA      all    slope +0.12059   retention  96.3%
                   clean  slope +0.12881   retention  95.1%   staked >=80%   correct
                   leaked slope -0.01761 (p 0.63)   K3 fired, weight 32

Staked prediction 3 -- that the clean/leaked skill gap would be below 0.05 -- **lost**. The gap is
0.115 (neural) and 0.146 (thermodynamic).

### The automatic verdict contradicts the declaration, and that is my bug

The script printed `memorisation_explains_skill`. The declaration's K2 is directional: *"F3 泄漏子集
技能**显著高于**干净子集"* -- leaked HIGHER than clean. The code tested `abs(diff) >= 0.05`. The
observed direction is the opposite: leaked is lower, for both models.

The artefact is committed with its wrong verdict rather than re-run with a corrected criterion,
because re-running a verdict function after seeing the result is how a criterion stops being a
criterion. The correction lives here and travels with the number.

### But the direction is not readable either (M26r, R3)

A random subsample of the same size (n=324) has a 95% slope interval of **[+0.0374, +0.4417]**.
The leaked subset's +0.10022 sits inside it. So "leaked is lower" cannot be read off this data at
all, and the argument I first drew from it does not stand.

What does still stand is qualitative and comes from the negative control: ViennaRNA's skill drops
on the same 324 records, and ViennaRNA has never been trained on anything. Whatever makes those
records hard is not memorisation. That is an argument about the control, not about the gap.

## M26r: four checks aimed at invalidating committed numbers

**R2 conditioning strength -- passes, and it answers a real worry.** RibonanzaNet is nearly binary
here, so "five quantile bins may fix nothing" was a live threat. It does not hold:

    P_vienna   total var 0.19547 -> within stratum 0.00443   97.7% removed
    P_neural   total var 0.12721 -> within stratum 0.00418   96.7% removed

**R4 window consistency -- passes.** 160 design cells, 0 with a non-unique (sub_start, sub_end).
M26d's leakage query used the right window.

**R3 leaked-subset interval -- fails, see above.**

**R1 positive control -- the control itself was malformed.** It built a mediator
`M = frac*1[CG] + noise` and conditioned on it. That mediator depends on the treatment but does
not predict the outcome, so it is not on a causal path; conditioning on it destroys CG/AU overlap
without shrinking the within-stratum contrast. At frac=1.0 the weight went to zero and the script
called that "mediation detected". It is not. Redone properly in M26r2 with a synthetic outcome
that actually flows through the mediator.

## Two reporting defects found in my own scripts

`m26d` printed "no-candidate identity mean 0.0000" from an empty array -- a fill value presented as
a measurement, which is the same class of error this project has already recorded once. The real
content of that line is that **zero queries lacked a k-mer candidate**, which is what should have
been printed.

`m27`'s TTYH2 entry (22 flips) came from the sequence fallback route rather than the construct-name
route, because that deposit has no per-row names. It is therefore verified to a lower standard than
the other ten deposits and is labelled that way.

## One foundational check that was never run before today, and passes exactly

`rescue_factor` is the outcome for every number in this line of work, and it is supplied by the
corpus. If it were computed with a folding model, everything here would be circular. It is not:

    R = 1 - |d_ij| / hypot(|d_i|, |d_j|)      on the design window, from reactivity profiles

Recomputed from the raw `reactivity_*` columns for all 4688 double mutants that have both single
mutants and a wild type:

    corpus mean +0.4124  SD 0.2534        reproduced mean +0.4124  SD 0.2534
    Pearson +1.0000   Spearman +1.0000   max |diff| 0.0014   median |diff| 0.0001

The outcome is measurement-derived end to end. No model appears in it.
