# The retention metric does not survive its own calibration

## What was asked

Every number of the form "retention 96.3%" in M26, M26b, M26c and M26d says a mediator explains
almost none of the pair-identity contrast. That reading is only available if the estimator can
detect mediation when mediation is present. Nobody had checked.

## Two attempts, the first one wrong

**m26r2** built a mediator `M = frac*1[CG] + noise` and conditioned on it. That mediator depends on
the treatment but does not predict the outcome, so it lies on no causal path. Conditioning on it
destroys CG/AU overlap without shrinking the within-stratum contrast; at frac=1.0 the weight went
to zero and the script reported "mediation detected". It had detected overlap collapse.

It was also calibrating in the wrong regime, by a measurable amount. Its synthetic mediator
separated the arms by 1.67 pooled SD. The real predictor separates them by **0.053** pooled SD --
31x weaker -- and the real contrast is +0.0539 against an outcome SD of 0.2501, two orders of
magnitude below the synthetic effect sizes.

Kept anyway, because what it shows in the easy regime is already bad: with a mediator 31x stronger
than the real one, a truth of 100% mediated comes back as 74.9% retention, and the mean absolute
error over the sweep is 0.5026. The metric is directionally right there (corr +0.906) and
compressed hard toward 100%.

**m26r3** changes one thing. The mediator IS the real predictor -- real near-binary distribution,
real correlation with design cells, real binning behaviour -- and only the outcome is synthetic:

    Ysyn = gamma * M_real + delta * 1[CG] + N(0, 0.2501)

with gamma, delta set so the total stratified contrast equals the corpus's own +0.05393 and a known
fraction f of it runs through the mediator.

## Result

    RibonanzaNet     true f      0%    10%    25%    50%    75%   100%
                     measured 103.2%  96.3%  93.5% 100.4%  99.9% 131.4%
                     corr -0.7019   MAE 0.4743

    ViennaRNA        measured  95.7% 109.8% 101.6% 106.0% 117.3% -842.5%
                     corr +0.7026   MAE 1.7358

At this corpus's actual signal-to-noise the retention ratio is two small noisy numbers divided by
one another, and it carries no information about the mediated fraction. For the neural arm the
correlation with truth is **negative**.

**The committed retention numbers -- 96.3%, 100.7%, 100.2%, 95.1%, and the four-parameter panel's
96.3-100.9% -- are withdrawn.** They may not be read as "the model explains ~0%". The withdrawal
belongs in the main text, not an appendix: our own calibration invalidated our own metric.

## What the calibration produced instead

Two quantities that are direct measurements on matched strata, with no ratio and no mediation
estimator anywhere in them:

    stratified CG-AU contrast in the OUTCOME            total = +0.05393   (weight 902)
    stratified CG-AU contrast in the MEDIATOR (alpha)
        RibonanzaNet                                          +0.06055
        ViennaRNA                                             -0.03039
    within-stratum slope of outcome on mediator (gamma)
        RibonanzaNet                                          +0.20692
        ViennaRNA                                             +0.12059

**ViennaRNA's alpha is negative.** In matched strata the thermodynamic model assigns *lower*
pairing probability to C-G targets than to A-U targets, while the measurement puts C-G *higher*,
and the model's own slope is positive -- higher predicted probability does go with higher rescue.
The model has the identity axis pointing the wrong way.

That is a sign, not a ratio. It needs none of the machinery that just failed.

The product route gives an upper bound on what could run through each predictor -- 23.2% for
RibonanzaNet, 6.8% for ViennaRNA -- and rule R4 confines it to that: an upper bound, never a point
estimate, because the bin-wise conditional mean is not affine.

## Verified separately

The sign is now the paper's headline, so it is recomputed in m26v against
`designaudit.core._weighted`, a bincount implementation written months apart from
`M26.matched`, and broken out per depth, per crossing and per design cell.
