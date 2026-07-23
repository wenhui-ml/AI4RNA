#!/usr/bin/env python3
"""Second domain for chomsky_climb: a DECIDABLE two-sided class-ascent in Boolean constraint definability.

The RNA thesis (a two-sided certificate of a formal-class ascent) is instantiated on a second, unrelated
formal hierarchy to show it is not RNA-specific. The class axis here is the minimum CNF clause-WIDTH needed
to define a Boolean relation R subseteq {0,1}^n:
  * width <= 2  <=>  R is closed under the ternary MAJORITY operation  (Schaefer / Baker-Pixley);
    2-CNF plays the role of the context-free (nested) class.
  * width  = k  is the analog of a higher Chomsky tier.

Two-sided certificate (both sides DECIDABLE here -- unlike RNA, whose positive side is model-relative because
its ground-truth folder is heuristic; SAT has an exact verifier, which localizes where model-relativity comes
from -- the substrate, not the certificate):
  - NEGATIVE (theorem, O(|R|^3)): a 2-CNF-confined designer's solution set is ALWAYS majority-closed, so a
    target R that is NOT majority-closed is provably unrealizable by it  (floor = 0 on width>=3 targets).
  - POSITIVE (exact, O(2^n)): a general-CNF designer returns a CNF phi whose model set == R, verified by
    enumerating {0,1}^n.
  - ASCENT: the realizable class moves from {width<=2 = majority-closed relations} up to the target's width.

No LLM, no external solver; pure decidable combinatorics on small n. Run with `--smoke` for the self-tests.
"""
import itertools, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sat")


def assignments(n):
    return list(itertools.product((0, 1), repeat=n))


def majority(a, b, c):
    return tuple(1 if (x + y + z) >= 2 else 0 for x, y, z in zip(a, b, c))


def is_majority_closed(R, n):
    """Decidable test == 2-CNF-definability (Baker-Pixley: majority is a near-unanimity term for 2-CNF)."""
    Rs = set(R)
    for a in Rs:
        for b in Rs:
            for c in Rs:
                if majority(a, b, c) not in Rs:
                    return False
    return True


def clauses_upto_width(w, n):
    """All non-trivial CNF clauses over n vars with <= w literals, as frozensets of (var, sign);
    sign=1 means the literal is x_var, sign=0 means ~x_var. A clause is a disjunction of its literals."""
    out = []
    for width in range(1, w + 1):
        for vars_ in itertools.combinations(range(n), width):
            for signs in itertools.product((0, 1), repeat=width):
                out.append(frozenset(zip(vars_, signs)))
    return out


def clause_satisfied(clause, x):
    # disjunction: satisfied if ANY literal is true. literal (v,s): true iff x[v]==s
    return any(x[v] == s for (v, s) in clause)


def tightest_cnf(R, w, n):
    """The tightest width-<=w CNF ENTAILED by R = every clause (width<=w) satisfied by ALL of R.
    Its model set is the smallest width-<=w-definable superset of R."""
    keep = [cl for cl in clauses_upto_width(w, n) if all(clause_satisfied(cl, x) for x in R)]
    return keep


def models_of(clauses, n):
    return set(x for x in assignments(n) if all(clause_satisfied(cl, x) for cl in clauses))


def width_definable(R, w, n):
    """Is R exactly definable by a width-<=w CNF?  <=>  models(tightest width-<=w CNF of R) == R."""
    return models_of(tightest_cnf(R, w, n), n) == set(R)


def cnf_width(R, n):
    """Minimum clause width needed to define R exactly (the decidable class grade). Empty relation and full
    relation are trivial; a general R is always definable at width n (forbid each non-member)."""
    R = set(R)
    for w in range(0, n + 1):
        if width_definable(R, w, n):
            return w
    return n  # unreachable: width n always works


def full_cnf(R, n):
    """General-CNF designer: forbid every non-member with a full clause -> models == R exactly."""
    non = [m for m in assignments(n) if m not in set(R)]
    # clause forbidding m: OR of (x_i != m_i) = literal (i, 1-m_i)
    return [frozenset((i, 1 - m[i]) for i in range(n)) for m in non]


def two_cnf_designer(R, n):
    """2-CNF-confined designer: outputs the tightest 2-CNF; realizes R exactly iff R is majority-closed."""
    cls = tightest_cnf(R, 2, n)
    return cls, (models_of(cls, n) == set(R))


def general_designer(R, n):
    """General-CNF designer: realizes any R exactly; also minimizes to the true width."""
    w = cnf_width(R, n)
    cls = tightest_cnf(R, w, n) if w >= 1 else []
    ok = (models_of(cls, n) == set(R))
    if not ok:  # width w should define R by construction; fall back to the forbidding CNF
        cls = full_cnf(R, n); ok = (models_of(cls, n) == set(R))
    return cls, w, ok


def gen_targets(n, k, seed):
    """A graded pool of relations: random subsets of {0,1}^n of varied size + a few structured ones."""
    g = random.Random(seed)
    allx = assignments(n)
    pool = []
    # structured anchors (n>=3): implication (2), NAE3 (3), parity/XOR (n)
    if n >= 3:
        pool.append(("implication", set(x for x in allx if not (x[0] == 1 and x[1] == 0))))       # x0 -> x1
        pool.append(("NAE3", set(x for x in allx if not (x[0] == x[1] == x[2]))))                  # not-all-equal
        pool.append(("parity", set(x for x in allx if sum(x[:min(n, 3)]) % 2 == 1)))               # odd parity on first 3
    seen = set(frozenset(r) for _, r in pool)
    tries = 0
    while len(pool) < k and tries < 4000:
        tries += 1
        size = g.randint(2, len(allx) - 1)
        R = frozenset(g.sample(allx, size))
        if R in seen:
            continue
        seen.add(R); pool.append((f"rand{len(pool)}", set(R)))
    return pool[:k]


def run(n=4, k=48, seed=0):
    os.makedirs(OUT, exist_ok=True)
    targets = gen_targets(n, k, seed)
    rows, floor_solves, asc_solves = [], 0, 0
    from collections import Counter
    grade_dist = Counter()
    for name, R in targets:
        w = cnf_width(R, n)
        grade_dist[w] += 1
        _, floor_ok = two_cnf_designer(R, n)      # floor: 2-CNF only
        _, aw, asc_ok = general_designer(R, n)     # ascent: general CNF
        maj = is_majority_closed(R, n)
        # decidable-negative check: majority-closed <=> width<=2 <=> 2-CNF realizable
        neg_theorem_ok = (maj == (w <= 2) == floor_ok)
        floor_solves += int(floor_ok); asc_solves += int(asc_ok)
        rows.append({"name": name, "|R|": len(R), "cnf_width": w, "majority_closed": maj,
                     "floor_2cnf_realizes": floor_ok, "ascent_realizes": asc_ok,
                     "neg_theorem_consistent": neg_theorem_ok})
    higher = [r for r in rows if r["cnf_width"] >= 3]
    summary = {
        "n": n, "n_targets": len(targets), "grade_dist(cnf_width)": dict(sorted(grade_dist.items())),
        "floor_2cnf_realizes_total": floor_solves, "ascent_realizes_total": asc_solves,
        "higher_class_targets(width>=3)": len(higher),
        "floor_on_higher_class": sum(r["floor_2cnf_realizes"] for r in higher),   # must be 0 (the theorem)
        "ascent_on_higher_class": sum(r["ascent_realizes"] for r in higher),       # should be all
        "negative_theorem_holds_all": all(r["neg_theorem_consistent"] for r in rows),
        "rows": rows}
    json.dump(summary, open(os.path.join(OUT, "sat_ascent.json"), "w"), indent=1)
    print(f"SAT class-ascent (n={n}, {len(targets)} targets): grade(cnf_width) dist "
          f"{summary['grade_dist(cnf_width)']}", flush=True)
    print(f"  floor (2-CNF) realizes {floor_solves}/{len(targets)}; ascent (general CNF) realizes {asc_solves}/{len(targets)}", flush=True)
    print(f"  higher-class targets (width>=3): {len(higher)}; floor on them = "
          f"{summary['floor_on_higher_class']}/{len(higher)} (theorem: must be 0); "
          f"ascent on them = {summary['ascent_on_higher_class']}/{len(higher)}", flush=True)
    print(f"  decidable NEGATIVE theorem (majority-closed <=> width<=2 <=> 2-CNF-realizable) holds for all: "
          f"{summary['negative_theorem_holds_all']}", flush=True)
    print("-> sat/sat_ascent.json", flush=True)
    return summary


def smoke():
    """Self-tests of the decidable machinery on known relations."""
    n = 3
    allx = assignments(n)
    impl = set(x for x in allx if not (x[0] == 1 and x[1] == 0))     # x0->x1 : 2-CNF
    nae = set(x for x in allx if not (x[0] == x[1] == x[2]))          # NAE3   : width 3, not maj-closed
    par = set(x for x in allx if sum(x) % 2 == 1)                     # parity : width 3, not maj-closed
    full = set(allx)                                                  # tautology: width 0
    checks = [
        ("implication majority-closed", is_majority_closed(impl, n), True),
        ("implication cnf_width==2", cnf_width(impl, n), 2),
        ("implication 2-CNF realizes", two_cnf_designer(impl, n)[1], True),
        ("NAE3 not majority-closed", is_majority_closed(nae, n), False),
        ("NAE3 cnf_width==3", cnf_width(nae, n), 3),
        ("NAE3 2-CNF FAILS", two_cnf_designer(nae, n)[1], False),
        ("NAE3 general realizes", general_designer(nae, n)[2], True),
        ("parity not majority-closed", is_majority_closed(par, n), False),
        ("parity cnf_width==3", cnf_width(par, n), 3),
        ("parity 2-CNF FAILS", two_cnf_designer(par, n)[1], False),
        ("tautology cnf_width==0", cnf_width(full, n), 0),
        ("tautology 2-CNF realizes", two_cnf_designer(full, n)[1], True),
    ]
    ok = True
    for label, got, want in checks:
        good = (got == want)
        ok = ok and good
        print(f"  [{'OK ' if good else 'FAIL'}] {label}: got {got}, want {want}")
    print("SMOKE", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        smoke()
    else:
        run(n=int(sys.argv[1]) if len(sys.argv) > 1 else 4,
            k=int(sys.argv[2]) if len(sys.argv) > 2 else 48)
