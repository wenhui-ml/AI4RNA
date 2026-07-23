def design(target,budget_s,seed=0):
    x=[0]*(10**9)
    return 'A'*len(target)


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
