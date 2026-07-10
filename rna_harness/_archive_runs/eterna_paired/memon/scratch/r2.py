import RNA, random, time, multiprocessing as mp

target = "((((....).))).((((....).))).((((....).))).((((....).))).((((....).))).((((....).))).((((((((((((((((((((((...)))).))))))).(((((..((((((((((..(((((((...))))))).))))))))))((((.((((((...)))))))))))))))...))))))))))).((((....).))).((((....).))).((((....).))).((((....).))).((((....).)))"
n=len(target)
stack=[];pairs={}
for i,c in enumerate(target):
    if c=='(':stack.append(i)
    elif c==')':
        j=stack.pop();pairs[j]=i;pairs[i]=j
paired=set(pairs.keys())

start_seq="UUACGAAGGAUAAACCGCAGAUGACGGAAGGCGAAAGACCUACCACGAUAGAUGGAGCACGCAGGUUGCAGGGAAGCGUACCCGCUUCCUGCCAAUGUCCGCGUCUCCCGGGCUGUGGACAGGGUACGUUGAGCGUGUUCAUGCACGAUGGUCGUGCGCAACACGCUCGGGUGAUACACGGUGCGUGUACGCUGUGCCAACUUGGCAGGAAGAGUCCGUGAGAGACAGGCCGCAAGAGCCAAACCGAAGGAGUUAGCACGAAGGAUGCAGUAUUGACAAUAC"

def dist(s):
    return RNA.bp_distance(RNA.fold_compound(s).mfe()[0], target)

def worker(seed):
    random.seed(seed*1000003+7)
    cur=start_seq; cur_d=dist(cur)
    best,best_d=cur,cur_d
    deadline=time.time()+200
    while time.time()<deadline and best_d>0:
        c=list(cur)
        m=random.choice([1,1,1,1,2,2,3])
        for _ in range(m):
            i=random.randrange(n)
            if i in paired:
                j=pairs[i]
                r=random.choice(['GC','CG','AU','UA','GU','UG','GC','CG','AU'])
                c[i]=r[0]; c[j]=r[1]
            else:
                c[i]=random.choice('ACGU')
        c=''.join(c); d=dist(c)
        if d<cur_d or (d==cur_d and random.random()<0.45):
            cur,cur_d=c,d
            if d<best_d: best_d,best=d,c
    return best_d,best

if __name__=='__main__':
    args=list(range(96))
    with mp.Pool(48) as p:
        results=p.map(worker, args)
    results.sort()
    print("best", results[0][0])
    print(results[0][1])
