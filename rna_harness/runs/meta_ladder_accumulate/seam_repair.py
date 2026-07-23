import RNA, random, sys

target = open('/data/wenhui/openclaude_rna/rna_harness/runs/meta_ladder_accumulate/target370.txt').read().strip()
N = len(target)

units = [
 'GGUCCGCUCCGGCGGCUCAUGUCCUACAUAAGAGAAAGUUUCGAGCCCGAUUCCGGGACC',
 'CGGGUCCUCAAGGGUCAUCGUAUAUUACGGUCUUCAUGGAAGGUGACGGGUAACCGCCUG',
 'UAUGCCGAAAUCGGGCGCCCAUUGGAUGGUAGUUAACAUGACGCGCCGCUGCUGCGCAUA',
 'CGUUUGGACCCCCGACCAACGAUCUUCGUCUAGUGUAGCACUUGGUCCCUUCUGGAAACG',
 'UCGUACCAACCGGGAUGGAGUCUAGGACUGUUACAGUGUGUACCAUCGGAAGCCCUACGA',
]
starts = [31,93,155,217,279]

# scaffold (non-unit) positions
unit_pos = set()
for s in starts:
    for i in range(s, s+60):
        unit_pos.add(i)
scaffold_pos = [i for i in range(N) if i not in unit_pos]
print('scaffold positions:', len(scaffold_pos), file=sys.stderr)

# init candidate: units placed; scaffold init by complementary
def init_scaffold():
    seq = list('A'*N)
    # place units
    for s,u in zip(starts, units):
        for k in range(60):
            seq[s+k] = u[k]
    # init scaffold positions: pair map
    # build target pair map
    stack=[]
    pair=[-1]*N
    for i,c in enumerate(target):
        if c=='(':
            stack.append(i)
        elif c==')':
            j=stack.pop()
            pair[i]=j; pair[j]=i
    # assign complementary pairs at scaffold positions; unpaired -> A
    for i in scaffold_pos:
        if pair[i]>=0 and pair[i] in scaffold_pos:
            # scaffold-scaffold pair
            pass
        elif target[i]=='.':
            seq[i]='A'
    # for scaffold-scaffold pairs, set G-C
    done=set()
    for i in scaffold_pos:
        j=pair[i]
        if j>=0 and j in set(scaffold_pos) and i not in done:
            seq[i]='G'; seq[j]='C'; done.add(i); done.add(j)
    return ''.join(seq)

cand = init_scaffold()
print('cand len', len(cand), file=sys.stderr)
print(cand)

# fold
fc = RNA.fold_compound(cand)
mfe, _ = fc.mfe()
# bp distance
def bpdist(s, t):
    # count pairs in s not matching t
    import re
    def pairs(ss):
        st=[]; p=set()
        for i,c in enumerate(ss):
            if c=='(': st.append(i)
            elif c==')':
                if st: p.add((st.pop(),i))
        return p
    ps=pairs(s); pt=pairs(t)
    return len(ps ^ pt), len(ps-pt), len(pt-ps)

d,sp,mi = bpdist(mfe, target)
print('initial bpdist', d, 'spurious', sp, 'missing', mi, file=sys.stderr)
