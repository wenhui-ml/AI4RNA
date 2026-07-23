```python
def design_pk(target, prim):
    '''Return a candidate RNA sequence (str of A/C/G/U, same length as target) intended to fold, UNDER THE
    PSEUDOKNOT ORACLE pKiss, into `target` (extended dot-bracket with () [] {} <> pairing layers).'''
    round_target = prim['round_only'](target)
    seq, _ = prim['inverse_fold'](''.join(prim['rng'].choice('ACGU') for _ in range(len(target))), round_target)
    pkiss_calls = 0
    best_seq = seq
    best_fold = prim['pkiss_fold'](seq)
    best_distance = prim['pk_bp_distance'](prim['pk_pairs'](best_fold), prim['pk_pairs'](target))
    pkiss_calls +=