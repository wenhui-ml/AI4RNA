def design_pk(target, prim):
    '''Return a candidate RNA sequence (str of A/C/G/U, same length as target) intended to fold, UNDER THE
    PSEUDOKNOT ORACLE pKiss, into `target` (extended dot-bracket with () [] {} <> pairing layers).
    You may ONLY use these primitives from the dict `prim` (do not import anything):
      prim['inverse_fold'](start_seq, nested_dotbracket) -> (seq, dist)  # ViennaRNA MFE inverse fold; NESTED
                                                                         # '()' only -- it ignores [] {}.
      prim['pkiss_fold'](seq) -> extended_dotbracket_str                 # the pseudoknot MFE oracle. EXPENSIVE
                                                                         # (~0.5-2s). Budget-limited: after
                                                                         # prim['budget_calls'] calls it stops you.
      prim['pk_pairs'](dotbracket) -> set of (i,j) base pairs (all bracket families)
      prim['pk_bp_distance'](a, b) -> int   # size of the symmetric difference of the two pair SETS
      prim['round_only'](target) -> nested '()' reduction of target (drops [] {} <>)
      prim['rng']  -> a seeded random.Random you should use for any randomness
      prim['budget_calls'] -> int  # how many pkiss_fold calls you may make total for this target
    Facts you can exploit: DesiRNA and REMC (the existing tools) optimise the ViennaRNA MFE energy and NEVER
    call pKiss during search, so the crossing ('[]{}') stem is only satisfied incidentally and the last
    base-pair is never repaired against the true oracle. In extended dot-bracket every position is in at most
    one pair, so crossing-stem positions are disjoint from nested ones.'''
    rng = prim['rng']
    n = len(target)
    target_pairs = prim['pk_pairs'](target)
    nested_target = prim['round_only'](target)
    nested_pairs = prim['pk_pairs'](nested_target)
    crossing_pairs = target_pairs - nested_pairs

    def random_seq():
        return ''.join(rng.choice('ACGU') for _ in range(n))

    def repair_crossing(seq):
        s = list(seq)
        for i, j in crossing_pairs:
            if rng.random() < 0.7:
                s[i] = 'G'
                s[j] = 'C'
            else:
                s[i] = 'A'
                s[j] = 'U'
        return ''.join(s)

    budget = prim['budget_calls']
    calls = 0
    best_seq = None
    best_dist = None
    best_fold = None
    best_pairs = None

    def evaluate(seq):
        nonlocal calls, best_seq, best_dist, best_fold, best_pairs
        fold = prim['pkiss_fold'](seq)
        calls += 1
        pairs = prim['pk_pairs'](fold)
        dist = prim['pk_bp_distance'](pairs, target_pairs)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_seq = seq
            best_fold = fold
            best_pairs = pairs
        return fold, pairs, dist

    # Phase 1: initial candidates via inverse_fold + crossing repair
    num_initial = min(15, budget)
    for _ in range(num_initial):
        if calls >= budget:
            break
        start = random_seq()
        seq, _ = prim['inverse_fold'](start, nested_target)
        seq = repair_crossing(seq)
        _, _, dist = evaluate(seq)
        if dist == 0:
            return seq

    # Phase 2: greedy local search
    while calls < budget and best_dist > 0:
        r = rng.random()
        if r < 0.2:
            # inverse_fold restart from best sequence
            if calls >= budget:
                break
            start = best_seq if best_seq is not None else random_seq()
            seq, _ = prim['inverse_fold'](start, nested_target)
            seq = repair_crossing(seq)
            _, _, dist = evaluate(seq)
            if dist == 0:
                return seq
        elif r < 0.6:
            # fix a random missing pair
            missing = target_pairs - best_pairs
            if missing:
                i, j = rng.choice(list(missing))
                seq = list(best_seq)
                seq[i] = 'G'
                seq[j] = 'C'
                if calls >= budget:
                    break
                _, _, dist = evaluate(''.join(seq))
                if dist == 0:
                    return ''.join(seq)
                if dist >= best_dist and calls < budget:
                    seq2 = list(best_seq)
                    seq2[i] = 'A'
                    seq2[j] = 'U'
                    _, _, dist2 = evaluate(''.join(seq2))
                    if dist2 == 0:
                        return ''.join(seq2)
            else:
                if calls >= budget:
                    break
                seq = list(best_seq)
                pos = rng.randrange(n)
                seq[pos] = rng.choice('ACGU')
                _, _, dist = evaluate(''.join(seq))
                if dist == 0:
                    return ''.join(seq)
        elif r < 0.8:
            # break a random extra pair
            extra = best_pairs - target_pairs
            if extra:
                i, j = rng.choice(list(extra))
                seq = list(best_seq)
                seq[i] = 'A'
                if calls >= budget:
                    break
                _, _, dist = evaluate(''.join(seq))
                if dist == 0:
                    return ''.join(seq)
                if dist >= best_dist and calls < budget:
                    seq2 = list(best_seq)
                    seq2[j] = 'A'
                    _, _, dist2 = evaluate(''.join(seq2))
                    if dist2 == 0:
                        return ''.join(seq2)
            else:
                if calls >= budget:
                    break
                seq = list(best_seq)
                pos = rng.randrange(n)
                seq[pos] = rng.choice('ACGU')
                _, _, dist = evaluate(''.join(seq))
                if dist == 0:
                    return ''.join(seq)
        else:
            # random mutation of 1-2 positions
            if calls >= budget:
                break
            seq = list(best_seq)
            num_mut = rng.randint(1, 2)
            for _ in range(num_mut):
                pos = rng.randrange(n)
                seq[pos] = rng.choice('ACGU')
            _, _, dist = evaluate(''.join(seq))
            if dist == 0:
                return ''.join(seq)

    return best_seq if best_seq is not None else random_seq()


def should_call_sota(features, skill_bp):
    '''Return True to FALL BACK to the expensive SOTA solver (DesiRNA, ~$0.043/target) when your cheap
    design_pk did not solve the target, or False to ABSTAIN (give up cheaply). `skill_bp` is the pKiss
    base-pair distance your design_pk achieved (0 = solved; larger = worse; None = your code errored).
    `features` is a dict of mechanical structural features with keys incl.: 'len', 'n_pairs', 'frac_paired',
    'n_pk_brackets', 'n_units', 'n_junctions', 'max_junction_degree', 'n_hairpins', 'max_stem', 'min_stem',
    'n_lonely_pairs', 'nesting_depth'. GOAL: minimise total cost at EQUAL solve -- abstain on targets where
    SOTA will ALSO fail (so you lose no real solve) but keep firing SOTA where it would succeed.'''
    if skill_bp == 0:
        return False
    n_hairpins = features.get('n_hairpins', 0)
    min_stem = features.get('min_stem', 0)
    n_lonely_pairs = features.get('n_lonely_pairs', 0)
    length = features.get('len', 0)
    # SOTA succeeds on simple H-type pseudoknots: single hairpin, no bulges/lonely pairs, min_stem >= 3, len <= 65
    if n_hairpins == 1 and min_stem >= 3 and n_lonely_pairs == 0 and length <= 65:
        return True
    return False