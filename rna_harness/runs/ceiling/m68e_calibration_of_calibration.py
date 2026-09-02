#!/usr/bin/env python3
"""M68 arm D: an interval for the corrected number, and a second perturbation shape.

Declaration: m68_review_gaps_declaration.json, arm D; predictions S8, S9; kill condition K4.

Section 5.9 prints an inverted value -- the true linear share that would produce the reading the
pipeline printed -- as a point read off a synthetic curve, with no interval. This paper argues
elsewhere that a six-point relation driven by one point is not evidence, so the same ruler is
applied here: bootstrap the WHOLE injection over targets (the clustering unit the manuscript uses
everywhere else) and report the inverted value's interval. Then repeat with a second perturbation
shape -- concave, steep-then-flat, the opposite curvature to the quadratic M62 swept -- and report
how far the inverted value moves.

K4 is live: if the interval covers the naive reading, the corrected number is withdrawn and only
the direction is reported.

COST: $0, offline, CPU only.
"""
from __future__ import annotations
import collections, json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, "/data/wenhui/openclaude_rna/rna_harness/tools")
import m16_rescue as M16
import m62_product_route_calibration as M62

SEED = 20260902
NBOOT = 400
NDRAW = 12                       # draws per lambda inside a bootstrap replicate
LAMBDAS = M62.LAMBDAS
PUBLISHED = M62.PUBLISHED
WORKERS = 24


def orthog(q, mc):
    """Remove the linear-in-depth component, then scale. Orthogonality to m is what matters."""
    q = q - q.mean()
    q = q - (float((mc * q).sum() / (mc * mc).sum())) * mc
    s = q.std()
    return q / s if s > 0 else q


def shapes(m):
    """Two perturbation shapes with opposite curvature, both orthogonal to depth.

    'quadratic' is M62's own: convex, accelerating. 'concave' is the steep-then-flat form the
    real conditional mean resembles. Neither is chosen after seeing an answer; both are reported.
    """
    mc = m - m.mean()
    return {"quadratic": orthog(m ** 2, mc),
            "concave": orthog(np.sqrt(m - m.min() + 1.0), mc)}


def invert_once(recs, r0, ndraw, seed):
    """One full injection sweep over lambda on THIS record set; return the inverted linear share."""
    rng0 = np.random.default_rng(seed)
    V = M62.M20.mediation(recs)
    total = V["matched"]; g = V["gap_g"]
    if not np.isfinite(total) or not np.isfinite(g) or g == 0 or total == 0:
        return None
    m = np.array([float(r["depth"]) for r in recs])
    sigma = M62.sigma_within(recs)
    for r, v in zip(recs, r0):
        r["_r0"] = float(v)
    dr0 = M62.marginal_gap(recs, lambda r: r["_r0"])
    if not np.isfinite(dr0) or dr0 == 0:
        return None
    f0 = abs(PUBLISHED * V["observed_attenuation"] / total)
    reads = []
    for lam in LAMBDAS:
        a = f0 * total / g
        c = a * g * (1.0 - lam) / (lam * dr0)
        delta = total - (a * g + c * dr0)
        ra = [M62.run(recs, m, r0, a, c, delta, sigma,
                      np.random.default_rng(int(rng0.integers(1, 2 ** 62))))["mediated_fraction"]
              for _ in range(ndraw)]
        ra = [x for x in ra if np.isfinite(x)]
        if not ra:
            return None
        reads.append(float(np.median(ra)))
    lt = np.array(LAMBDAS, float); lr = np.array(reads, float)
    o = np.argsort(lr)
    inv = float(np.interp(PUBLISHED, lr[o], lt[o]))
    clamped = bool(PUBLISHED < lr[o][0] or PUBLISHED > lr[o][-1])
    return inv, clamped, reads


_G = {}


def _init(shape_name):
    recs = [r for r in M16.build() if r["pt"] in ("CG", "AU")]
    m = np.array([float(r["depth"]) for r in recs])
    _G["recs"] = recs
    _G["r0_full"] = shapes(m)[shape_name]
    _G["puz"] = np.array([r["puzzle"] for r in recs])
    _G["targets"] = sorted({r["puzzle"] for r in recs})


def _boot(b):
    rng = np.random.default_rng(SEED + 104729 * b)
    T = _G["targets"]
    pick = rng.choice(len(T), size=len(T), replace=True)
    idx = []
    for k in pick:
        idx.extend(np.flatnonzero(_G["puz"] == T[k]).tolist())
    if len(idx) < 500:
        return None
    sub = [dict(_G["recs"][i]) for i in idx]
    r0 = _G["r0_full"][np.array(idx)]
    r0 = r0 - r0.mean()
    s = r0.std()
    if s > 0:
        r0 = r0 / s
    out = invert_once(sub, r0, NDRAW, SEED + 7 * b + 1)
    return None if out is None else (out[0], out[1])


def main():
    OUT = {"seed": SEED, "nboot": NBOOT, "ndraw_per_lambda": NDRAW,
           "published_reading": PUBLISHED, "lambdas": list(LAMBDAS)}
    recs = [r for r in M16.build() if r["pt"] in ("CG", "AU")]
    m = np.array([float(r["depth"]) for r in recs])
    SH = shapes(m)
    print(f"records {len(recs)}  targets {len({r['puzzle'] for r in recs})}", flush=True)
    # the two shapes are genuinely different curves, and we say how different
    corr = float(np.corrcoef(SH["quadratic"], SH["concave"])[0, 1])
    OUT["shape_correlation"] = round(corr, 4)
    print(f"shape correlation (quadratic vs concave, both orthogonal to depth): {corr:+.4f}",
          flush=True)

    for name in ("quadratic", "concave"):
        pt = invert_once([dict(r) for r in recs], SH[name], 40, SEED)
        inv, clamped, reads = pt
        print(f"\n[{name}] point estimate: inverted lambda {inv:.4f}"
              f"   readings {['%.4f' % x for x in reads]}"
              f"{'   CLAMPED' if clamped else ''}", flush=True)
        with ProcessPoolExecutor(WORKERS, initializer=_init, initargs=(name,)) as ex:
            res = [x for x in ex.map(_boot, range(NBOOT), chunksize=4) if x is not None]
        v = np.array([x[0] for x in res], float)
        ncl = sum(1 for x in res if x[1])
        lo, hi = float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))
        OUT[name] = {"point": round(inv, 4), "clamped_point": clamped,
                     "readings": [round(x, 4) for x in reads],
                     "n_boot_used": len(v), "n_clamped": ncl,
                     "boot_median": round(float(np.median(v)), 4),
                     "ci": [round(lo, 4), round(hi, 4)],
                     "covers_published": bool(lo <= PUBLISHED <= hi),
                     "bias_at_published": round(PUBLISHED - inv, 4)}
        print(f"[{name}] bootstrap over targets: median {np.median(v):.4f}"
              f"   95% [{lo:.4f}, {hi:.4f}]   ({len(v)} replicates, {ncl} clamped)"
              f"   covers the naive {PUBLISHED}: {OUT[name]['covers_published']}", flush=True)

    q, c = OUT["quadratic"], OUT["concave"]
    move = abs(c["point"] - q["point"])
    dist = abs(PUBLISHED - q["point"])
    OUT["shape_move"] = round(move, 4)
    OUT["distance_from_naive"] = round(dist, 4)
    OUT["S8"] = bool(not q["covers_published"])
    OUT["S9"] = bool(move < 0.25 * dist) if dist > 0 else None
    OUT["K4_fires"] = bool(q["covers_published"])
    print(f"\nS8 (the interval excludes the naive reading {PUBLISHED}): {OUT['S8']}", flush=True)
    print(f"S9 (shape moves the inverted value less than a quarter of its distance from naive): "
          f"{OUT['S9']}   moved {move:.4f} against a quarter of {dist:.4f} = {0.25*dist:.4f}",
          flush=True)
    if OUT["K4_fires"]:
        print("K4 FIRES: the corrected number is withdrawn; only the direction is reported.",
              flush=True)
    json.dump(OUT, open(os.path.join(HERE, "m68e_calibration_of_calibration.json"), "w"), indent=1)
    print("\nwrote m68e_calibration_of_calibration.json", flush=True)


if __name__ == "__main__":
    main()
