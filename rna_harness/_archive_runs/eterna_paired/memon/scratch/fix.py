import RNA, random, time, multiprocessing as mp

target = "((((....).))).((((....).))).((((....).))).((((....).))).((((....).))).((((....).))).((((((((((((((((((((((...)))).))))))).(((((..((((((((((..(((((((...))))))).))))))))))((((.((((((...)))))))))))))))...))))))))))).((((....).))).((((....).))).((((....).))).((((....).))).((((....).)))"
n=len(target)
stack=[];pairs={}
for i,c in enumerate(target):
    if c=='(':stack.append(i)
    elif c==')':
        j=stack.pop();pairs[j]=i;pairs[i]=j
paired=set(pairs.keys())

base="UUACGAAGGAUAAACCGCAGAUGACGGAAGGCGAAAGACCUACCACGAUAGAUGGAGCACGCAGGUUGCAGGGAAGCGUACCCGCUUCCUGCCAAUGUCCGCGUCUCCCGGGCUGUGGACAGGGUACGUUGAGCGUGUUCAUGCACGAUGGUCGUGCGCAACACGCUCGGGUGAUACACGGUGCGUGUACGCUGUGCCAACUUGGCAGGAAGAGUCCGUGAGAGACAGGCCGCAAGAGCCAAACCGAAGGAGUUAGCACGAAGGAUGCAGUAUUGACAAUAC"

# Unit0 pairs: (0,12)(1,11)(2,10)(3,8); unpaired 4,5,6,7,9
# Unit10 (last) pairs: (269,281)(270,280)(271,279)(272,277); unpaired 273-276,278
# Cross pairs formed: (1,271)(2,270)(3,269). So pos1~pos271, pos2~pos270, pos3~pos269 are complementary.
# To break: make pos1 NOT complementary to pos271, etc.
# Currently base[1]='U', base[271]='A' -> complementary. Change base[1] to G (pairs with C/U).
# But base[1] is paired with base[11] in target. So change (1,11) pair.
# Let's set unit0: (0,12)=GC,(1,11)=CG,(2,10)=GC,(3,8)=CG, loop GAAA, pos9=A
# unit10: (269,281)=GC,(270,280)=CG,(271,279)=GC,(272,277)=CG, loop GAAA, pos278=A
# Check cross: pos1=C, pos271=G -> still complementary! Bad.
# Need pos1 and pos271 non-complementary. pos1 paired with pos11.
# Strategy: unit0 use A/U for (1,11), unit10 use A/U for (271,279) but different.
# pos1=U pos271=A -> comp. pos1=A pos271=A -> NOT comp (A-A). 
# So unit0 (1,11)=AU (pos1=A,pos11=U); unit10 (271,279)=UA (pos271=U,pos279=A). pos1=A,pos271=U -> comp! 
# Try: unit0(1,11)=AU, unit10(271,279)=GC. pos1=A,pos271=G -> not comp (A-G). 
# And (2,270),(3,269): pos2 in unit0, pos270 in unit10.
# Let me just brute force small: try many unit0/unit10 combos and pick best full bpd.

small_target="((((....).)))"
def fold(s): return RNA.fold_compound(s).mfe()[0]
def ok_small(s): return fold(s)==small_target

loops=['GAAA','GCAA','CUUG','UUCG','GAGA']
combs=[('G','C'),('C','G'),('G','U'),('A','U'),('U','A'),('U','G')]
pp=[(0,12),(1,11),(2,10),(3,8)]

# Generate unit candidates with constraint that pos1,pos2,pos3 avoid complementarity with unit10
import itertools
def gen_units():
    units=[]
    for lp in loops:
        for c0 in combs:
            for c1 in combs:
                for c2 in combs:
                    for c3 in combs:
                        s=['A']*13
                        s[9]='A'
                        for k in range(4): s[4+k]=lp[k]
                        s[pp[0][0]]=c0[0];s[pp[0][1]]=c0[1]
                        s[pp[1][0]]=c1[0];s[pp[1][1]]=c1[1]
                        s[pp[2][0]]=c2[0];s[pp[2][1]]=c2[1]
                        s[pp[3][0]]=c3[0];s[pp[3][1]]=c3[1]
                        seq=''.join(s)
                        if ok_small(seq):
                            units.append(seq)
    return units

units=gen_units()
print("num units",len(units))

# unit10 occupies 269-281. The cross pairs are (1,271)(2,270)(3,269) 
# i.e. unit0 pos1 <-> unit10 pos2 (271-269=2), unit0 pos2 <-> unit10 pos1, unit0 pos3 <-> unit10 pos0
# So unit0[1..3] pairs with unit10[2,1,0] (reversed). 
# Break: for k in 1,2,3: unit0[k] not complementary to unit10[3-k]
COMP={'A':'U','U':'A','G':'C','C':'G','G':'U','U':'G'}
def cross_ok(u0,u10):
    for k in [1,2,3]:
        if u10[3-k] == COMP.get(u0[k],''):
            return False
    return True

def dist(s):
    return RNA.bp_distance(RNA.fold_compound(s).mfe()[0], target)

# Also fix (73,78) - unit5 closing pair. Let me find unit5: pos 70-82, pairs (70,82)(71,81)(72,80)(73,78)
# base[73],base[78] currently. Let me just allow the hill climb to fix that.

best_d=999;best=None
# sample random pairs of units
random.seed(0)
# precompute full sequence builder
def build(u0,u10):
    s=list(base)
    for k in range(13):
        s[k]=u0[k]
        s[269+k]=u10[k]
    return ''.join(s)

# Try combos with cross_ok
import random as rng
tried=0
for _ in range(3000):
    u0=rng.choice(units); u10=rng.choice(units)
    if not cross_ok(u0,u10): continue
    s=build(u0,u10)
    d=dist(s)
    if d<best_d:
        best_d=d;best=s
        print("best",best_d)
    tried+=1
print("FINAL",best_d)
print(best)
