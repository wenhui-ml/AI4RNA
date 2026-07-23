def design_pk(target, prim):
    nested_target = prim['round_only'](target)
    start_seq = ''.join(prim['rng'].choice('ACGU') for _ in range(len(target)))
    
    seq, dist = prim['inverse_fold'](start_seq, nested_target)
    best_seq = seq
    best_dist = dist
    
    if best_dist == 0:
        return seq
    
    for _ in range(prim['budget_calls']):
        current_structure = prim['pkiss_fold'](best_seq)
        current_dist = prim['pk_bp_distance'](prim['pk_pairs'](current_structure), prim['pk_pairs'](target))
        
        if current_dist < best_dist:
            best_dist = current_dist
            best_seq = best_seq
        
        if best_dist == 0:
            break
        
        mutated_seq = list(best_seq)
        mutation_index = prim['rng'].randint(0, len(best_seq) - 1)
        mutated_seq[mutation_index] = prim['rng'].choice('ACGU')
        mutated_seq = ''.join(mutated_seq)
        
        seq, dist = prim['inverse_fold'](mutated_seq, nested_target)
        
        if dist < best_dist:
            best_seq = seq
            best_dist = dist

    return best_seq

def should_call_sota(features, skill_bp):
    if skill_bp is None:
        return True
    
    if skill_bp == 0:
        return False
    
    if features['n_pk_brackets'] > 12 and features['len'] > 50:
        return False
    
    if features['n_pk_brackets'] <= 10 and features['len'] <= 40:
        return True
    
    if features['n_units'] > 5 and features['n_junctions'] > 3:
        return False
    
    return True