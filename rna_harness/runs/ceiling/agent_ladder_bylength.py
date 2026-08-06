"""HYPOTHESIS ONLY, generated after the result -- not a decision."""
import json,glob,os
from collections import defaultdict
H="/data/wenhui/openclaude_rna/rna_harness/runs/ceiling/"
DROP="desirna@180s"
for name,sub in (("eterna100","verify_ladder"),("bprna_nested","verify_ladder_bprna")):
    by=defaultdict(list)
    for f in sorted(glob.glob(os.path.join(H,sub,"p*.json"))):
        try: r=json.load(open(f))
        except Exception: continue
        if r.get('trace'): by[r['id']].append(r)
    print(f"=== {name} ===")
    print(f"{'长度区间':>12s}{'靶标':>6s}{'@180 花费':>11s}{'@180 解出':>10s}{'单价':>12s}")
    for lo,hi,lab in ((0,120,"<120 nt"),(120,200,"120-200 nt"),(200,100000,">=200 nt")):
        cost=0.0; solv=0; nt=0
        for tid,runs in by.items():
            if not (lo<=runs[0]['n']<hi): continue
            nt+=1
            for r in runs:
                for t in r['trace']:
                    if t['tool'].startswith(DROP):
                        cost+=t.get('cost_usd') or 0.0
                        solv+= 1 if t.get('solved') else 0
        unit=("$%.4f"%(cost/solv)) if solv else "从未解出"
        print(f"{lab:>12s}{nt:6d}{cost:11.4f}{solv:10d}{unit:>12s}")
    tot=sum(sum(t.get('cost_usd') or 0.0 for r in runs for t in r['trace'] if t['tool'].startswith(DROP)) for runs in by.values())
    short=sum(sum(t.get('cost_usd') or 0.0 for r in runs for t in r['trace'] if t['tool'].startswith(DROP)) for runs in by.values() if runs[0]['n']<200)
    ssol=sum(sum(1 for r in runs for t in r['trace'] if t['tool'].startswith(DROP) and t.get('solved')) for runs in by.values() if runs[0]['n']<200)
    print(f"  -> <200 nt 的靶标付了 @180 总花费的 {short/tot:.1%},换回 {ssol} 次成功\n")
