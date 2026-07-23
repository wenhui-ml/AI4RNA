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

unit_pos = set()
for s in starts:
    for i in range(s, s+60):
        unit_pos.add(i)
scaffold_pos = [i for i in range(N) if i not in unit_pos]

# pair map
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

def init_scaffold():
    seq = list('A'*N)
    for s,u in zip(starts, units):
        for k in range(60):
            seq[s+k] = u[k]
    done=set()
    scset=set(scaffold_pos)
    for i in scaffold_pos:
        j=pair[i]
        if j>=0 and j in scset and i not in done:
            seq[i]='G'; seq[j]='C'; done.add(i); done.add(j)
        elif target[i]=='.':
            seq[i]='A'
    return ''.join(seq)

def fold(seq):
    fc=RNA.fold_compound(seq)
    mfe,_=fc.mfe()
    return mfe

def score(seq):
    mfe=fold(seq)
    ps=pairs_of(mfe)
    return len(ps ^ pt), mfe

cand = init_scaffold()
best = cand
bs, _ = score(cand)
print('init bpdist', bs, file=sys.stderr)

# SA over scaffold positions + unit flanks (allow unit mutation too but mostly scaffold)
alphabet='ACGU'
import time
t0=time.time()
cur = cand; cs = bs
T = 8.0
random.seed(1)
iters=0
while time.time()-t0 < 150 and cs>0:
    iters+=1
    # pick a position: 70% scaffold, 30% unit
    if random.random()<0.7:
        i=random.choice(scaffold_pos)
    else:
        i=random.randrange(N)
    old=cur[i]
    new=random.choice([c for c in alphabet if c!=old])
    trial=list(cur); trial[i]=new; trial=''.join(trial)
    ts,_=score(trial)
    if ts<=cs or random.random()<pow(2.718,(cs-ts)/max(T,0.01)):
        cur=trial; cs=ts
        if ts<bs:
            best=trial; bs=ts
            print('best',bs,'at iter',iters, file=sys.stderr)
    T*=0.9995
print('final best bpdist', bs, 'iters', iters, file=sys.stderr)
print(best)
