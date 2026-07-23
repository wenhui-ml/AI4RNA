def design_pk(target, prim):
    nested_structure = prim['round_only'](target)
    start_seq = ''.join(prim['rng'].choice('ACGU') for _ in range(len(target)))
    candidate_seq, _ = prim['inverse_fold'](start_seq, nested_structure)

    best_seq = candidate_seq
    best_dist = prim['pk_bp_distance'](prim['pk_pairs'](prim['pkiss_fold'](best_seq)), prim['pk_pairs'](target))

    budget_remaining = prim['budget_calls'] - 1

    while budget_remaining > 0 and best_dist > 0:
        candidate_seq = ''.join(prim['rng'].choice('ACGU') for _ in range(len(target)))
        candidate_seq, _ = prim['inverse_fold'](candidate_seq, nested_structure)
        folded_structure = prim['pkiss_fold'](candidate_seq)
        current_dist = prim['pk_bp_distance'](prim['pk_pairs'](folded_structure), prim['pk_pairs'](target))

        if current_dist < best_dist:
            best_seq = candidate_seq
            best_dist = current_dist

        # Early exit if perfectly solved
        if best_dist == 0:
            break

        budget_remaining -= 1

    return best_seq

def should_call_sota(features, skill_bp):
    if skill_bp == 0:
        return False

    # Refined decision based on more features
    if features['n_pk_brackets'] > 10 and features['len'] <= 40:
        return True

    if features['n_pk_brackets'] >= 12:
        return True

    if features['len'] > 50:
        return False

    # Additional conditions based on junctions and stems
    if features['n_junctions'] > 3 or features['max_stem'] < 4:
        return True

    # Consider abstaining if the structure is complex but not too large
    if features['len'] <= 35 and features['n_pk_brackets'] <= 10:
        return False

    return skill_bp > 4