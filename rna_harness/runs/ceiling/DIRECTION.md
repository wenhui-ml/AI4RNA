# Direction, revised: the bottleneck is a missing objective, not a scheduling decision

Supersedes the premise of `METHOD.md` and `PREREG.md` on this branch. Those documents asked how to
allocate a compute budget using a search-free coordinate. The measurements below say that question was
aimed at the wrong thing.

## What the allocation programme actually delivered

| claim | status |
|---|---|
| Δ predicts failure on Eterna100, AUC 0.870, robust to `N`, not reducible to length (LR p = 9.2e-8) | **holds**, n = 100, CI [0.801, 0.939] |
| Δ costs 8 draws, not 60 (AUC 0.862 at N = 5) | **holds** |
| Δ = 0 is a useful screen on the crossing set (19 of 21 held-out stored wins inside 13 of 30 targets) | **holds** |
| allocating call DEPTH by Δ buys efficiency | **refuted**: 0.20x at 10% budget against a pre-registered 3.04x |
| extreme-value extrapolation to Δ* | **refuted**: AUC 0.818 vs 0.875 |
| a two-sided Δ band | **refuted** on its own fit split; I built the mechanism on Poisson noise |
| p0, the measure of the satisfying set | **refuted**: Spearman +0.004, p = 0.989 |

Corrected headroom for trajectory reallocation, after fixing an arithmetic error of mine (`k·p` where the
within-replicate cap makes it `1-(1-p)^k`): **1.36x blind, 2.26x with oracle target knowledge**. The
5.53x/6.36x/7.06x figures I quoted earlier were oracle skylines presented as achievable values.

For comparison, this project already has a **learned** abstention result cutting SOTA cost 8-13x at
identical held-out quality across 3 splits. Trading that for a search-free 1.36x was only worth it for a
paper's sake. With the paper dropped, the search-free coordinate programme is closed.

## The measurement that changes the direction

Of the 530 in-loop solves the panel operator produced across the committed 15-cell sweep
(`bottleneck.py`):

```
  pKiss      agrees  530/530 = 100.0%     <- in loop
  ProbKnot   agrees  530/530 = 100.0%     <- in loop
  ShapeKnots agrees   48/530 =   9.1%     <- OUT OF LOOP

  marginal rejection, among solutions passing the other two:
    pKiss          0/48  =  0.0%
    ProbKnot       0/48  =  0.0%
    ShapeKnots   482/530 = 90.9%
```

**Every one of the 482 failures is ShapeKnots alone.** The operator drives both of its in-loop oracles to
100% and a single oracle it never consults rejects 90.9%. At the target level ShapeKnots halves the result:
42 of 60 targets reach an in-loop solve and reach pKiss+ProbKnot agreement, 21 reach unanimity.

Two supporting facts, both well powered:

* Attempts that pass the panel and attempts that fail it are **statistically identical in every resource**
  (Mann-Whitney p = 0.88 on ProbKnot calls, p = 0.83 on pKiss calls, n = 530). The verdict is decided at
  the moment the operator stops, and it stops because it hit *its own* target: a successful attempt reaches
  no limit at all (pKiss 37 of 150, ProbKnot 33 of 120, iterations 36 of 800).
* Across 5190 design cells the operator used 43-50% of any pKiss allowance and only 16-33% of attempts
  ever reached the cap. Depth is not purchasable, which is exactly why the allocation experiment could not
  win.

And there is an induction over the operator family: `autonomous` optimises pKiss and gets pKiss 100%,
ProbKnot 5%, ShapeKnots 3.4%. `panel_auto` optimises pKiss+ProbKnot and gets both at 100%, ShapeKnots
9.1%. **Adding an oracle to the loop has twice produced 100% agreement on it.** The third has never been
tried.

## Why it was never tried is not cost

`sk_cost.py`, timing all three oracles on 12 operator-produced sequences at this benchmark's lengths
(21-42 nt), 3 repetitions:

```
  median   pKiss 210.1 ms    ProbKnot 120.1 ms    ShapeKnots 64.8 ms
  ShapeKnots / pKiss = 0.31   (range 0.26-0.49)
```

ShapeKnots is the **cheapest** of the three. Gating adds about 0.5% to the cost of a panel-passing solve,
because the ~11 candidates a success needs are already being generated and thrown away -- today they are
spread over ~11 separate executions, which is why the per-execution rate (3.2/60) is 6.5x below the
ever-solved rate (21/60).

## The change

`DecomposePKPanelTool(sk_gate=True)`, off by default so every previously reported number is unchanged.
Reaching `(pk_bp, pb_bp) = (0,0)` stops being terminal: the candidate goes to ShapeKnots, and on rejection
it is remembered, the walk is kicked out of that basin, and the search continues for a different
pKiss+ProbKnot solution. `solved` then means all three agree, and the returned sequence is the one that
passed -- `best_seq` can be a (0,0) sequence ShapeKnots rejected, so it must not be returned.

The gate needs a larger pKiss allowance because it now needs ~11 candidates rather than 1. At
`max_pkiss=150` a smoke test screened only 2-5 candidates and correctly refused to certify any of them;
at 400 it can afford ~11.

## The prediction, before the run

From 9.1% per candidate and ~37 pKiss calls per candidate: 400 calls buys ~11 candidates, so
`1 - (1-0.091)^11 = 65%` for the 42 targets that reach an in-loop solve. Discounted by the fact that 21 of
those 42 never passed ShapeKnots in 15 stored executions -- for them it may be a target property -- the
honest expectation is:

| | now | with the gate |
|---|---|---|
| panel-unanimous per execution | 3.2/60 | **17-20/60** |
| per-target compute | 34.5 s | ~130 s (3.8x) |
| cost per panel-passing solve | — | ~1.5x better |
| capability | — | **5-6x** |

`off_400` (gate off, same 400-call allowance) is the falsifier: if extra pKiss budget alone lifts the panel
rate, the gate is not what did the work.

## The statistics problem, and why this fixes it too

Power, from the measured 5.33% per-attempt event rate:

```
  to detect 1.20x at 80% power:  ~135 replicates
            1.36x             :   ~45          <- I ran 15
            1.50x             :   ~24
            2.00x             :    ~7
```

My step-2 design was underpowered by 3x for the effect I predicted. The fix is cheap (5.7 core-h per arm
for 45 replicates), but the better fix is a bigger effect: if the gate raises the event rate from 5.3% to
~30%, detecting a 1.5x difference needs ~4 replicates instead of 24. **The change that raises capability
also repairs the power problem.**

## What is still owed

* The gate's per-target compute multiplier is a prediction (3.8x), not yet measured.
* For the 21 targets that never passed ShapeKnots in 15 executions, whether that is a target property or
  bad luck is unknown. If it is a property, the target-level ceiling is well below 42/60.
* The gate helps only where a three-oracle-satisfying sequence exists. Nothing here measures how often
  that is true, which is the designability question one level up.
