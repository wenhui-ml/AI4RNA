import RNA, random, time, multiprocessing as mp, os, sys
target='....((....((....((....((..((..((...))..))....))......)).......)).........)).............'
def worker(args):
    sid, init_seed = args
    random.seed(init_seed ^ os.getpid())
    deadline=time.time()+140
    while time.time()<deadline:
        start=''.join(random.choice('ACGU') for _ in range(len(target)))
        seq,d=RNA.inverse_fold(start,target)
        if d==0 and RNA.fold_compound(seq).mfe()[0]==target:
            return seq
    return None
if __name__=='__main__':
    found=None
    with mp.Pool(64) as p:
        for r in p.imap_unordered(worker, list(enumerate(range(128)))):
            if r:
                found=r; p.terminate(); break
    if found:
        with open(sys.argv[1],'w') as f: f.write(found)
        print('SOLVED',found,flush=True)
    print('DONE',flush=True)
