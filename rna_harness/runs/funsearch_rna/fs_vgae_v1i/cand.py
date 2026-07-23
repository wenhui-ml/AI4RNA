
import random, time, RNA
def design(target, budget_s, seed=0):
    rng = random.Random(seed); n = len(target); B = "ACGU"
    def fold(s): return RNA.fold_compound(s).mfe()[0]
    best, bestd, t0 = "A"*n, 10**9, time.time()
    while time.time() - t0 < budget_s:
        s = [rng.choice(B) for _ in range(n)]
        for _ in range(400):
            st = fold("".join(s)); d = RNA.bp_distance(st, target)
            if d < bestd: bestd, best = d, "".join(s)
            if d == 0: return "".join(s)
            bad = [i for i in range(n) if st[i] != target[i]] or [rng.randrange(n)]
            i = rng.choice(bad); s[i] = rng.choice(B)
            if time.time() - t0 >= budget_s: break
    return best


if __name__ == "__main__":
    import json, sys, time, RNA
    targets = json.load(open(sys.argv[1])); budget = float(sys.argv[2])
    solved = 0; tot = 0.0; per = []
    for t in targets:
        t0 = time.time()
        try:
            s = design(t, budget)
        except Exception:
            s = ""
        dt = time.time() - t0; tot += dt
        ok = isinstance(s, str) and len(s) == len(t) and RNA.fold_compound(s.upper().replace("T","U")).mfe()[0] == t
        solved += int(ok); per.append(round(dt, 1))
    print("RESULT " + json.dumps({"solved": solved, "n": len(targets), "cpu": round(tot, 1), "per": per}))
