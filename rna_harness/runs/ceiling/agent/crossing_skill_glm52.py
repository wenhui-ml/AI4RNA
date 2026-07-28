def design_pk(target, prim):
    '''Return a candidate RNA sequence (str of A/C/G/U, same length as target) intended to fold, UNDER THE
    PSEUDOKNOT ORACLE pKiss, into `target` (extended dot-bracket with () [] {} <> pairing layers).'''
    rng = prim['rng']
    n = len(target)
    if n == 0:
        return ""
    nested = prim['round_only'](target)
    target_pairs = prim['pk_pairs'](target)
    nested_pairs = prim['pk_pairs'](nested)
    crossing_pairs = target_pairs - nested_pairs

    start = ''.join(rng.choice('ACGU') for _ in range(n))
    seq, _ = prim['inverse_fold'](start, nested)

    if not target_pairs:
        return seq

    bracket_map = {'(': ')', '[': ']', '{': '}', '<': '>'}
    open_set = set(bracket_map.keys())
    close_set = set(bracket_map.values())
    comp = {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C'}

    def fix_crossing(s):
        s_list = list(s)
        for i, j in crossing_pairs:
            if 0 <= i < n and 0 <= j < n:
                s_list[i] = 'G'
                s_list[j] = 'C'
        return ''.join(s_list)

    def mutate_seq(s, pairs):
        s_list = list(s)
        for i, j in pairs:
            if 0 <= i < n and 0 <= j < n:
                s_list[i] = rng.choice('GCA')
                s_list[j] = comp.get(s_list[i], 'C')
        return ''.join(s_list)

    best_seq = seq
    best_dist = prim['pk_bp_distance'](prim['pkiss_fold'](seq), target)

    if best_dist == 0:
        return best_seq

    for _ in range(prim['budget_calls'] - 1):
        if not crossing_pairs:
            cand = mutate_seq(best_seq, nested_pairs)
        else:
            if rng.random() < 0.5:
                cand = fix_crossing(best_seq)
            else:
                cand = mutate_seq(best_seq, crossing_pairs)

        cand, _ = prim['inverse_fold'](cand, nested)

        if rng.random() < 0.3:
            cand = fix_crossing(cand)

        dist = prim['pk_bp_distance'](prim['pkiss_fold'](cand), target)

        if dist < best_dist:
            best_dist = dist
            best_seq = cand
            if best_dist == 0:
                break

    return best_seq


def should_call_sota(features, skill_bp):
    '''Return True to FALL BACK to the expensive SOTA solver (DesiRNA, ~$0.043/target) when your cheap
    design_pk did not solve the target, or False to ABSTAIN (give up cheaply).'''
    if skill_bp is not None and skill_bp == 0:
        return False
    if skill_bp is None:
        return False

    L = features.get('len', 0)
    n_pk = features.get('n_pk_brackets', 0)
    n_pairs = features.get('n_pairs', 0)
    max_stem = features.get('max_stem', 0)
    min_stem = features.get('min_stem', 0)
    n_lonely = features.get('n_lonely_pairs', 0)
    max_deg = features.get('max_junction_degree', 0)

    if L <= 0:
        return False
    if n_pk <= 0:
        return True
    if L < 30:
        return True
    if n_pk >= 14:
        return False
    if L >= 30 and n_pk <= 12:
        return False
    if L >= 45 and n_pk >= 10:
        return False
    if max_stem <= 3:
        return False
    if min_stem <= 2:
        return False
    if n_lonely > 0:
        return False
    if max_deg >= 4:
        return False
    if n_pairs > 0 and (n_pk / n_pairs) > 0.6:
        return False
    return True