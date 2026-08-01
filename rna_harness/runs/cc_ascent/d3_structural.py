#!/usr/bin/env python3
"""R9, D3 hypothesis (b), STAGE 1: is there a STRUCTURAL property of accepted designs that explains the gap?

The state of the question. The frozen agent-written operators carry over to held-out ShapeKnots at 0.293
against the hand-built operator's 0.095 on paired, difficulty-matched units. Five explanations are dead:
oracle-call volume in both directions (3x allowance moves nothing; the agents in fact spend 4.6-10x FEWER
calls, and an arm held to their volume gives 0.107), target selectivity (matching costs 0.009), overall GC
composition (stratified, the advantage holds in both bins), denominator collapse (r_in is comparable), and the
four mechanisms legible in the source. What is left is narrow: at comparable in-loop success and lower search
volume, the accepted candidates differ in some property the held-out predictor rewards.

Why STRUCTURAL rather than compositional. Overall GC is excluded. But the panel's disagreements are about
PAIRING, and ShapeKnots is the member that rejects almost everything (11.3% recall on natives). The
mechanistically motivated version of "the agents' designs are different" is therefore: their crossing stem is
more dominant relative to the alternatives, so a folding model that was never optimized still finds it. That
is a property of where the nucleotides sit relative to the target's pair set, not of how many G and C there
are.

THE TWO GATES, FIXED BEFORE ANY NUMBER IS COMPUTED, AND THEN REPAIRED WHEN THE CONTROL CAUGHT THEM. This is where a study like this goes wrong, so the rule is
stated first and applied to all six features, with every result reported:

  (a) BETWEEN-ARM: the feature differs between the agent designs and ours.
  (b) WITHIN-ARM:  inside the HAND-BUILT arm alone, the feature predicts whether ShapeKnots confirms.

Gate (b) is the one that matters and the one a fishing expedition fails. A feature can differ between two
programs for a hundred incidental reasons; only if it also tracks carry-over WITHIN a single arm -- where the
program is held fixed -- is it a candidate mechanism rather than a fingerprint. A feature passing (a) alone is
reported as a fingerprint and used for nothing.

Passing both gates still is NOT D3. It licenses Stage 2: bias the hand-built operator toward the feature and
re-measure carry-over under the same rule every transplant faced (it must RISE while r_in does NOT collapse).
Only a reproduced gain closes D3.

The six features, pre-stated, all computed for every accepted design, all reported:
  gc_cross  GC fraction among the pairs of the target that CROSS another pair
  gc_nest   GC fraction among the target's non-crossing pairs
  gc_diff   gc_cross - gc_nest: is the crossing stem preferentially stiffened relative to the rest?
  a_loop    A fraction at unpaired positions -- the classic anti-spurious-pairing heuristic
  decoy     off-target complementary windows that could compete with the crossing stem (oracle-free)
  gc_all    overall GC -- the NEGATIVE CONTROL: already known to differ and already stratified out, so it
            should pass (a) and fail (b). If it passes (b) too, gate (b) is not discriminating.

WHAT THE CONTROL CAUGHT, AND THE REPAIR. On the first pass gc_all passed gate (b) outright (AUC 0.749,
p=1.9e-10): inside a single arm, GC-richer designs are likelier to be confirmed by ShapeKnots. So an unadjusted
gate (b) admits any feature merely correlated with GC, which is most of them, and it would have handed a
Stage-2 transplant to gc_cross on the strength of its GC content alone. We therefore run BOTH gates adjusted
for gc_all by logistic regression, and report the raw and adjusted columns side by side so the size of the
confound is visible rather than quietly removed. The adjusted negative control is a permuted feature, which
must come back null; gc_all cannot serve as its own control once it is the covariate.

Six features on one dataset is multiple comparison; we report all six and read nothing from a bare p<0.05.
Offline over stored sequences; no oracle calls, no episodes.
"""
import glob, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, HERE)

from rna_harness.benchmarks.rnainvbench import load_pseudobase, pk_pairs   # noqa: E402

OPS = ["gemini31pro", "kimik3"]
SEEDS, EXECS = [0, 1, 2, 3, 4], [0, 1, 2]
FEATURES = ["gc_cross", "gc_nest", "gc_diff", "a_loop", "decoy", "gc_all"]
WC = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}
STRONG = {("G", "C"), ("C", "G")}


def crossing_split(pairs):
    """Partition a target's pairs into those that cross some other pair and those that do not."""
    ps = sorted(pairs)
    cross = set()
    for i, (a, b) in enumerate(ps):
        for (c, d) in ps[i + 1:]:
            if a < c < b < d or c < a < d < b:
                cross.add((a, b)); cross.add((c, d))
    return cross, set(ps) - cross


def decoy_count(seq, cross_pairs, w=4):
    """Oracle-free competition proxy: how many OTHER windows could pair with the crossing stem's 5' arm.

    Take the 5' arm of the crossing stem, and count windows of length w elsewhere in the sequence that are
    fully complementary to it. Every such window is an alternative partner a different folding model might
    prefer, so a design with fewer of them has a crossing stem that is easier for an un-optimized predictor to
    find. This is a property of the sequence relative to the target, computable without folding anything.
    """
    if not cross_pairs or len(seq) < 2 * w:
        return 0
    arm = "".join(seq[i] for i, _ in sorted(cross_pairs)[:w])
    if len(arm) < w:
        return 0
    partner = {i for _, i in cross_pairs} | {i for i, _ in cross_pairs}
    n = 0
    for s in range(len(seq) - w + 1):
        if any(s + k in partner for k in range(w)):
            continue                                     # skip the real stem and its own arm
        win = seq[s:s + w][::-1]
        if all((a, b) in WC for a, b in zip(arm, win)):
            n += 1
    return n


def featurize(seq, target):
    pairs = set(pk_pairs(target))
    cross, nest = crossing_split(pairs)

    def gc(ps):
        if not ps:
            return float("nan")
        return sum((seq[i], seq[j]) in STRONG for i, j in ps) / len(ps)

    paired = {i for p in pairs for i in p}
    loops = [c for k, c in enumerate(seq) if k not in paired]
    gcx, gcn = gc(cross), gc(nest)
    return {"gc_cross": gcx, "gc_nest": gcn,
            "gc_diff": (gcx - gcn) if not (math.isnan(gcx) or math.isnan(gcn)) else float("nan"),
            "a_loop": (loops.count("A") / len(loops)) if loops else float("nan"),
            "decoy": float(decoy_count(seq, cross)),
            "gc_all": sum(c in "GC" for c in seq) / len(seq)}


def collect_agent(targets):
    out = []
    for op in OPS:
        for f in sorted(glob.glob(os.path.join(HERE, f"arep_{op}_dev", "s*_e*.json"))):
            for r in json.load(open(f))["rows"]:
                if r.get("seq") and r.get("pkiss") and r.get("probknot"):
                    out.append((featurize(r["seq"], targets[r["id"]]), bool(r.get("shapeknots"))))
    return out


def collect_hand(targets):
    out = []
    for s in SEEDS:
        for e in EXECS:
            p = os.path.join(HERE, f"gvp_e{e}_s{s}", "sweep.json")
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            if len(d.get("rows") or []) < 60:
                continue
            for r in d["rows"]:
                v = r["arms"].get("panel_auto") or {}
                if v.get("error") or not v.get("solved") or not v.get("seq"):
                    continue
                pn = v.get("panel") or {}
                out.append((featurize(v["seq"], targets[r["id"]]), bool(pn.get("shapeknots"))))
    return out


def mean_sd(xs):
    xs = [x for x in xs if not math.isnan(x)]
    if not xs:
        return float("nan"), float("nan"), 0
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1))
    return m, sd, len(xs)


def welch(a, b):
    ma, sa, na = mean_sd(a); mb, sb, nb = mean_sd(b)
    if na < 2 or nb < 2 or (sa == 0 and sb == 0):
        return float("nan"), float("nan")
    se = math.sqrt(sa * sa / na + sb * sb / nb)
    if se == 0:
        return float("nan"), float("nan")
    z = (ma - mb) / se
    return z, math.erfc(abs(z) / math.sqrt(2))          # normal approx; n is in the hundreds


def logistic(X, y, iters=200, ridge=1e-3):
    """IRLS logistic regression -> (coefs, standard errors). Hand-rolled: no statsmodels here, and the design
    is three columns on a few hundred rows, where IRLS is exact enough and auditable.

    A small ridge penalty on the non-intercept columns keeps the fit finite when a feature separates the
    outcome (48 positives in 530 makes that easy). Without it IRLS diverges and the routine returns nan, which
    is worse than a slightly shrunk coefficient: a nan reads as "no signal" when it often means the opposite.
    The penalty biases coefficients toward zero, so it can only make a gate HARDER to pass."""
    n, k = len(X), len(X[0])
    b = [0.0] * k
    for _ in range(iters):
        eta = [sum(b[j] * row[j] for j in range(k)) for row in X]
        mu = [1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, e)))) for e in eta]
        w = [max(m * (1 - m), 1e-9) for m in mu]
        # Gauss-Newton step: solve (X' W X) d = X' (y - mu)
        A = [[sum(w[i] * X[i][r] * X[i][c] for i in range(n)) + (ridge if r == c and r > 0 else 0.0)
              for c in range(k)] for r in range(k)]
        g = [sum(X[i][r] * (y[i] - mu[i]) for i in range(n)) - (ridge * b[r] if r > 0 else 0.0)
             for r in range(k)]
        d = solve(A, g)
        if d is None:
            return None, None
        b = [b[j] + d[j] for j in range(k)]
        if max(abs(x) for x in d) < 1e-10:
            break
    inv = inverse(A)
    if inv is None:
        return None, None
    return b, [math.sqrt(max(inv[j][j], 0.0)) for j in range(k)]


def solve(A, g):
    k = len(A)
    M = [row[:] + [g[i]] for i, row in enumerate(A)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-12:
            return None
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        M[c] = [v / pv for v in M[c]]
        for r in range(k):
            if r != c and M[r][c]:
                f = M[r][c]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[c])]
    return [M[i][k] for i in range(k)]


def inverse(A):
    k = len(A)
    M = [row[:] + [1.0 if i == j else 0.0 for j in range(k)] for i, row in enumerate(A)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-12:
            return None
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        M[c] = [v / pv for v in M[c]]
        for r in range(k):
            if r != c and M[r][c]:
                f = M[r][c]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[c])]
    return [row[k:] for row in M]


def zstd(xs):
    m, s, _ = mean_sd(xs)
    return [(x - m) / s if s else 0.0 for x in xs], m, s


def adjusted_p(rows, ys, key, cov="gc_all"):
    """p-value for `key` in logit(P(confirmed)) ~ 1 + gc_all + key, both standardized."""
    good = [(r, y) for r, y in zip(rows, ys)
            if not math.isnan(r[key]) and not math.isnan(r[cov])]
    globals()["_LAST_N"] = len(good)
    if len(good) < 30:
        return float("nan"), float("nan")
    f, _, _ = zstd([r[key] for r, _ in good])
    c, _, _ = zstd([r[cov] for r, _ in good])
    X = [[1.0, c[i], f[i]] for i in range(len(good))]
    y = [1.0 if yy else 0.0 for _, yy in good]
    b, se = logistic(X, y)
    if b is None or se is None or se[2] == 0:
        return float("nan"), float("nan")
    z = b[2] / se[2]
    return z, math.erfc(abs(z) / math.sqrt(2))


def auc(pos, neg):
    """Probability a randomly drawn confirmed design scores above a randomly drawn unconfirmed one."""
    pos = [x for x in pos if not math.isnan(x)]; neg = [x for x in neg if not math.isnan(x)]
    if not pos or not neg:
        return float("nan")
    wins = ties = 0
    for p in pos:
        for q in neg:
            wins += p > q; ties += p == q
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def main():
    tasks = load_pseudobase(limit=60)
    targets = {f"pseudobase:{i}": t.target for i, t in enumerate(tasks)}
    ag, hb = collect_agent(targets), collect_hand(targets)
    print("=" * 100)
    print("D3 hypothesis (b), STAGE 1: structural properties of accepted designs")
    print("=" * 100)
    print(f"  agent designs: {len(ag)}   hand-built designs: {len(hb)}")
    print("  Gate (a) between-arm difference; gate (b) predicts carry-over WITHIN the hand-built arm.")
    print("  A feature must pass BOTH to earn a Stage-2 transplant. gc_all is the negative control:")
    print("  it should pass (a) and FAIL (b); if it passes (b), gate (b) is not discriminating.\n")

    hb_pos = [f for f, y in hb if y]
    hb_neg = [f for f, y in hb if not y]
    ag_pos = [f for f, y in ag if y]
    ag_neg = [f for f, y in ag if not y]
    print(f"  within hand-built: {len(hb_pos)} ShapeKnots-confirmed, {len(hb_neg)} not")
    print(f"  within agent:      {len(ag_pos)} confirmed, {len(ag_neg)} not\n")
    hb_rows = [f for f, _ in hb]; hb_y = [y for _, y in hb]
    ag_rows = [f for f, _ in ag]
    arm_rows = hb_rows + ag_rows
    arm_y = [False] * len(hb_rows) + [True] * len(ag_rows)   # "is this an agent design?"

    print(f"  {'feature':10} {'agent':>15} {'hand-built':>15} | {'(a) raw p':>10} {'(a) adj p':>10} "
          f"| {'(b) AUC':>8} {'(b) raw p':>10} {'(b) adj p':>10}  verdict")
    verdicts = {}
    for k in FEATURES:
        A = [f[k] for f, _ in ag]; H = [f[k] for f, _ in hb]
        ma, sa, _ = mean_sd(A); mh, sh, _ = mean_sd(H)
        _, pa_raw = welch(A, H)
        P = [f[k] for f in hb_pos]; N = [f[k] for f in hb_neg]
        a_hb = auc(P, N)
        _, pb_raw = welch(P, N)
        n_def = sum(1 for f in hb_rows if not math.isnan(f[k]))
        if k == "gc_all":
            pa_adj = pb_adj = float("nan")               # cannot adjust the covariate for itself
        else:
            _, pa_adj = adjusted_p(arm_rows, arm_y, k)
            _, pb_adj = adjusted_p(hb_rows, hb_y, k)
        ga = (not math.isnan(pa_adj)) and pa_adj < 0.05
        gb = (not math.isnan(pb_adj)) and pb_adj < 0.05
        v = "BOTH -> stage 2" if (ga and gb) else ("(a) only: fingerprint" if ga else
                                                   ("(b) only: lever, not mechanism" if gb else "neither"))
        if k == "gc_all":
            v = "covariate (adjusted for)"
        elif math.isnan(pb_adj):
            # Not a numerical failure: gc_nest and gc_diff are UNDEFINED on a bare H-type pseudoknot,
            # where every pair crosses some other pair and the nested set is empty. That is 97% of this
            # pool, so the feature simply does not exist for the topology the benchmark is made of, and
            # its raw within-arm AUC was computed on the handful of targets carrying an extra hairpin.
            v = f"undefined on {len(hb_rows) - n_def}/{len(hb_rows)} (no non-crossing pairs)"
        verdicts[k] = (ga, gb)
        print(f"  {k:10} {ma:7.3f}+-{sa:<5.3f} {mh:7.3f}+-{sh:<5.3f} | {pa_raw:10.3g} {pa_adj:10.3g} "
              f"| {a_hb:8.3f} {pb_raw:10.3g} {pb_adj:10.3g}  {v}")

    # Adjusted negative control: a permuted copy of the strongest feature must come back null.
    import random as _r
    _r.seed(20260801)
    perm_rows = [dict(f) for f in hb_rows]
    vals = [f["gc_cross"] for f in perm_rows]; _r.shuffle(vals)
    for f, v_ in zip(perm_rows, vals):
        f["perm"] = v_
    _, p_perm = adjusted_p(perm_rows, hb_y, "perm")
    print(f"  {'perm':10} {'--':>15} {'--':>15} | {'--':>10} {'--':>10} "
          f"| {'--':>8} {'--':>10} {p_perm:10.3g}  adjusted negative control (must be null)")

    print(f"\n  {'feature':10} {'(b) AUC within AGENT arm (replication of gate b)':>52}")
    for k in FEATURES:
        print(f"  {k:10} {auc([f[k] for f in ag_pos], [f[k] for f in ag_neg]):52.3f}")

    ok = [k for k in FEATURES if k != "gc_all" and all(verdicts[k])]
    levers = [k for k in FEATURES if k != "gc_all" and verdicts[k][1] and not verdicts[k][0]]
    print("\n  " + "-" * 96)
    if not math.isnan(p_perm) and p_perm < 0.05:
        print("  ADJUSTED NEGATIVE CONTROL FAILED: a permuted feature is significant, so the adjusted gate is")
        print("  not calibrated and nothing below can be read.")
    elif ok:
        print(f"  Features earning a Stage-2 transplant: {', '.join(ok)}")
        print("  Stage 2 is the test that counts: bias the operator toward the feature, re-measure carry-over")
        print("  deterministically at 5 seeds, and require it to RISE while r_in does NOT collapse.")
    else:
        print("  No feature passes both GC-adjusted gates. D3 hypothesis (b) is refuted at this resolution:")
        print("  once overall composition is controlled for, no pre-stated structural statistic both")
        print("  distinguishes the arms AND predicts carry-over inside one. Recorded as the sixth null.")
        if levers:
            print(f"\n  One asymmetry worth stating rather than burying: {', '.join(levers)} predicts")
            print("  carry-over WITHIN an arm but does NOT differ between arms. That cannot be the mechanism")
            print("  behind the gap -- the agents do not have more of it -- but it is a lever on our own")
            print("  carry-over, and a different experiment from the one D3 asks for. We report it as such.")
    print("  Six features on one dataset is multiple comparison. Nothing here is read from a bare p<0.05;")
    print("  the gate is the conjunction, and the decision it licenses is a transplant, not a conclusion.")


if __name__ == "__main__":
    main()
