def design_pk(target, prim):
    '''Return an RNA sequence (str of A/C/G/U, len == len(target)) intended to fold, under the ViennaRNA MFE
    model, EXACTLY into the NESTED dot-bracket `target` (only '(' ')' '.'; no pseudoknots here). Primitives:
      prim['inverse_fold'](start_seq, target) -> (seq, dist)  # ViennaRNA inverse fold; dist = #positions off
      prim['fold'](seq) -> mfe_structure_str                  # ViennaRNA MFE fold (CHEAP, microseconds)
      prim['bp_distance'](structA, structB) -> int            # base-pair distance between two structures
      prim['rng']          -> seeded random.Random
      prim['budget_calls'] -> int  # max fold()+inverse_fold() calls (MFE is cheap, so this is generous)
    The function name is design_pk for interface compatibility; the task here is NESTED design.'''
    rng = prim['rng']
    n = len(target)
    budget = prim['budget_calls']
    calls = 0
    
    # Parse target structure
    target_pairs = [0] * n
    stack = []
    for i, c in enumerate(target):
        if c == '(':
            stack.append(i)
        elif c == ')':
            j = stack.pop()
            target_pairs[j] = i
            target_pairs[i] = j
    
    # Generate a structured seed: complementary pairs, GC-rich stems, AU for loops
    def make_seed():
        seq = list('A' * n)
        used = [False] * n
        for i in range(n):
            if used[i]:
                continue
            if target_pairs[i] == 0 and target[i] == '.':
                seq[i] = rng.choice(['A', 'C', 'G', 'U'])
                used[i] = True
            elif target_pairs[i] > 0:
                j = target_pairs[i]
                pair = rng.choice(['GC', 'CG', 'AU', 'UA'])
                seq[i] = pair[0]
                seq[j] = pair[1]
                used[i] = True
                used[j] = True
        return ''.join(seq)
    
    # Initialize with a structured seed
    best_seq = make_seed()
    calls += 1
    best_struct = prim['fold'](best_seq)
    best_dist = prim['bp_distance'](best_struct, target)
    
    if best_dist == 0:
        return best_seq
    
    # Main loop: alternate between inverse_fold and mutation-based local search
    while calls < budget:
        # Try inverse_fold from best sequence
        if calls + 1 <= budget:
            seq, dist = prim['inverse_fold'](best_seq, target)
            calls += 1
            if dist == 0:
                # Verify with fold
                if calls + 1 <= budget:
                    calls += 1
                    struct = prim['fold'](seq)
                    if prim['bp_distance'](struct, target) == 0:
                        return seq
                    else:
                        # inverse_fold dist was 0 but MFE doesn't match; keep if better
                        d = prim['bp_distance'](struct, target)
                        if d < best_dist:
                            best_seq, best_dist = seq, d
                else:
                    return seq  # trust inverse_fold result
        
        # Local search: mutate and refold
        if calls + 1 > budget:
            break
        
        # Mutate: either random point mutation or structured re-seed of a region
        candidate = list(best_seq)
        if rng.random() < 0.5 and n > 10:
            # Mutate a contiguous region
            start = rng.randint(0, n - 5)
            length = rng.randint(3, min(10, n - start))
            for k in range(start, start + length):
                candidate[k] = rng.choice('ACGU')
        else:
            # Point mutations
            num_muts = rng.randint(1, max(1, n // 20))
            for _ in range(num_muts):
                pos = rng.randint(0, n - 1)
                candidate[pos] = rng.choice('ACGU')
        
        candidate_seq = ''.join(candidate)
        calls += 1
        candidate_struct = prim['fold'](candidate_seq)
        candidate_dist = prim['bp_distance'](candidate_struct, target)
        
        if candidate_dist == 0:
            return candidate_seq
        
        # Accept if better, or with some probability if not much worse (simulated annealing)
        if candidate_dist < best_dist:
            best_seq = candidate_seq
            best_dist = candidate_dist
        elif candidate_dist <= best_dist + 2 and rng.random() < 0.3:
            best_seq = candidate_seq
            best_dist = candidate_dist
    
    return best_seq


def should_call_sota(features, skill_bp):
    '''Return True to fall back to the expensive SOTA solver (DesiRNA) when design_pk did not solve, else
    False to ABSTAIN. skill_bp = the MFE bp-distance your design_pk reached (0 = solved; None = errored).
    features keys incl: 'len','n_pairs','frac_paired','n_units','n_junctions','max_junction_degree',
    'n_hairpins','max_stem','min_stem','n_lonely_pairs','n_triloops','nesting_depth'. Minimise total cost at
    equal solve: abstain where SOTA will ALSO fail; keep firing SOTA where it would succeed.'''
    if skill_bp is None:
        return False
    if skill_bp == 0:
        return False  # Already solved
    
    L = features.get('len', 0)
    n_pairs = features.get('n_pairs', 0)
    n_junctions = features.get('n_junctions', 0)
    max_jdeg = features.get('max_junction_degree', 0)
    n_units = features.get('n_units', 0)
    n_hairpins = features.get('n_hairpins', 0)
    nesting_depth = features.get('nesting_depth', 0)
    n_lonely = features.get('n_lonely_pairs', 0)
    frac_paired = features.get('frac_paired', 0)
    
    # SOTA fails on long structures with many independent units (multi-way junctions spread out)
    # The two SOTA-fail cases in training: len 81 with 4 units, len 110 with 3 units
    # Both have long unpaired stretches between units and moderate-to-high unit counts
    
    # Abstain (SOTA will also fail) when:
    # 1. Very long with many independent units and low nesting (spread-out architecture)
    # 2. Long with high unit count and long internal loops
    
    if L >= 75 and n_units >= 3 and nesting_depth <= 3:
        # Spread-out multi-unit: SOTA struggles
        if n_junctions >= 2 and max_jdeg <= 2:
            return False  # Abstain
        if L >= 100 and n_units >= 3:
            return False  # Abstain
    
    # SOTA also likely fails on very long sequences with low pairing density
    if L >= 75 and frac_paired < 0.25:
        return False  # Abstain
    
    # SOTA succeeds on most other structures: call it
    return True