import RNA, random, time, multiprocessing as mp, os, sys
target='....((....((....((....((..((..((...))..))....))......)).......)).........)).............'
base='UAUAGCCAAACCGAGAGCGACGCCAACUUACGAAACGGAAGACAGGGAUAUGAGCGAAAAUAGGGAGACAUAGGCAAUUGUAAUUGUG'
def dist(s): return RNA.bp_distance(RNA.fold_compound(s).mfe()[0], target)
def worker(seed):
    random.seed(seed ^ os.getpid())
    cur=base; cur_d=dist(cur); best,best_d=cur,cur_d
    deadline=time.time()+150
    while time.time()<deadline and best_d>0:
        c=list(cur)
        for _ in range(random.choice([1,1,2])):
            i=random.randrange(len(c)); c[i]=random.choice('ACGU')
        c=''.join(c); d=dist(c)
        if d<cur_d or (d==cur_d and random.random()<0.5):
            cur,cur_d=c,d
            if d<best_d: best,best_d=c,d
    return best,best_d
if __name__=='__main__':
    with mp.Pool(64) as p:
        results=p.map(worker,range(128))
    results.sort(key=lambda x:x[1])
    out=sys.argv[1]
    with open(out,'w') as f:
        for b,d in results[:5]: f.write(f'{d} {b}\n')
    if results[0][1]==0:
        print('SOLVED',results[0][1],results[0][0],flush=True)
    print('DONE',flush=True)
