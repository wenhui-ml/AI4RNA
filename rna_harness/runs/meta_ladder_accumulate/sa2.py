import RNA, random, sys, time
from multiprocessing import Pool

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

unit_pos = set()
for s in starts:
    for i in range(s, s+60):
        unit_pos.add(i)
scaffold_pos = [i for i in range(N) if i not in unit_pos]

stack=[]; pair=[-1]*N
for i,c in enumerate(target):
    if c=='(':
        stack.append(i)
    elif c==')':
        j=stack.pop(); pair[i]=j; pair[j]=i

def pairs_of(ss):
    st=[]; p=set()
    for i,c in enumerate(ss):
        if c=='(': st.append(i)
        elif c==')':
            if st: p.add(tuple(sorted((st.pop(),i))))
    return p
pt = pairs_of(target)

def init_mix(gc_frac=0.5):
    seq = list('A'*N)
    for s,u in zip(starts, units):
        for k in range(60):
            seq[s+k] = u[k]
    scset=set(scaffold_pos)
    done=set()
    # scaffold-scaffold pairs: mix GC/AU
    for i in scaffold_pos:
        j=pair[i]
        if j>=0 and j in scset and i not in done:
            if random.random()<gc_frac:
                seq[i]='G'; seq[j]='C'
            else:
                seq[i]='A'; seq[j]='U'
            done.add(i); done.add(j)
        elif target[i]=='.':
            seq[i]=random.choice('ACGU')
    return ''.join(seq)

def fold(seq):
    fc=RNA.fold_compound(seq)
    mfe,_=fc.mfe()
    return mfe

def score(seq):
    mfe=fold(seq)
    ps=pairs_of(mfe)
    return len(ps ^ pt)

def run(seed):
    random.seed(seed)
    cand = init_mix(0.5)
    cs = score(cand)
    best=cand; bs=cs
    t0=time.time()
    T=6.0
    while time.time()-t0 < 140 and bs>0:
        if random.random()<0.7:
            i=random.choice(scaffold_pos)
        else:
            i=random.randrange(N)
        old=cand[i]
        new=random.choice([c for c in 'ACGU' if c!=old])
        trial=list(cand); trial[i]=new; trial=''.join(trial)
        ts=score(trial)
        if ts<=cs or random.random()<pow(2.718,(cs-ts)/max(T,0.01)):
            cand=trial; cs=ts
            if ts<bs:
                best=trial; bs=ts
        T*=0.9997
    return bs, best

if __name__=='__main__':
    seeds=range(40)
    with Pool(40) as p:
        results=p.map(run, seeds)
    results.sort()
    print('best bpdist', results[0][0], file=sys.stderr)
    print(results[0][1])
