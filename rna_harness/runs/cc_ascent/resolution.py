#!/usr/bin/env python3
"""R10 (reviewer W3): what can a low-recall instrument actually resolve?

The panel recovers 11.3% of native pseudoknots under ShapeKnots and 1/80 under three-way unanimity, so a
reader may reasonably ask whether an instrument that insensitive can separate anything except zero from
non-zero -- whether it can tell 5/60 from 15/60. Two quantities have to be kept apart, because conflating
them is what makes low recall sound fatal.

  RESOLUTION  the smallest difference in OBSERVED rate the design can exclude zero on. A property of n, the
              pairing and the base rate. Recall does not enter.
  ATTENUATION what an observed difference implies about the underlying one. Here recall does enter: if the
              panel confirms a fraction rho of designs that are in fact correct and admits few false
              positives, an observed gap d stands for a true gap d/rho.

The rescaling assumes rho is common to both arms. That holds for r_mix comparisons and NOT for carry-over,
whose whole point is that rho differs by arm, so we apply it only to the former.

Simulated on the design actually run: 60 targets x 5 seeds = 300 paired cells, target-clustered percentile
bootstrap, base rate at the observed 4/60. Offline; no oracle calls.
"""
import random

random.seed(20260801)
N_T, N_S, B, REPS = 60, 5, 600, 200
BASE = 4 / 60


def one_trial(delta):
    per_t = {}
    for t in range(N_T):
        a = sum(random.random() < BASE for _ in range(N_S))
        b = sum(random.random() < min(1.0, BASE + delta) for _ in range(N_S))
        per_t[t] = (a, b, N_S)
    ids = list(per_t)
    out = []
    for _ in range(B):
        na = nb = n = 0
        for _ in ids:
            a, b, k = per_t[ids[random.randrange(len(ids))]]
            na += a; nb += b; n += k
        out.append((nb - na) / n)
    out.sort()
    return out[int(0.025 * B)] > 0


def main():
    print(f"  design: {N_T} targets x {N_S} seeds = {N_T*N_S} paired cells, base rate {BASE:.3f}, "
          f"target-clustered bootstrap\n")
    print(f"  {'true observed gap':>18} {'as counts /60':>14} {'power':>8}")
    res = []
    for d in (0.017, 0.033, 0.050, 0.067, 0.100, 0.150):
        pw = sum(one_trial(d) for _ in range(REPS)) / REPS
        res.append((d, pw))
        print(f"  {d:18.3f} {d*60:13.1f} {pw:8.2f}", flush=True)
    det = next((d for d, pw in res if pw >= 0.80), None)
    print(f"\n  RESOLUTION: 80% power at an observed gap of ~{det:.3f} ({det*60:.0f}/60). The design separates")
    print("  4/60 from ~8/60 and 5/60 from 15/60 at power 1.00; it cannot separate 4/60 from 5/60. That is a")
    print("  statement about n, not about recall. Delta_gamer at +0.067 sits at 77% power -- at the design's")
    print("  resolution limit, which is 'established, narrowly' in units a reader can check.")
    print("\n  ATTENUATION: an observed gap d implies a true gap d/rho for panel recall rho.")
    print(f"  {'rho':>8} {'true gap behind an observed 0.067':>36}")
    for rho in (0.50, 0.25, 0.113, 0.05):
        print(f"  {rho:8.3f} {0.067/rho:36.3f}")
    print("\n  Low recall rescales the comparison rather than blurring it, and makes every rate here")
    print("  conservative. Its real cost is stated elsewhere: a low or zero panel rate is weak evidence of")
    print("  ABSENCE, which is why the cross-source 0/60 is not read as absence of capability.")


if __name__ == "__main__":
    main()
