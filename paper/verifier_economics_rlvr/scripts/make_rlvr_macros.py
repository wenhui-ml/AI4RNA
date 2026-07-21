"""Emit LaTeX macros from RLVR run data -> paper/verifier_economics/figs/rlvr_numbers.tex.

Every headline RLVR number in the paper is a \newcommand defined here, computed from
runs/*/curve.csv (training-distribution: strong_acc, gaming, J, cum strong-calls) and
runs/*/heldout.jsonl (held-out strict acc). Numbers not yet available render as a visible
red \textbf{[PENDING]} so a draft can never silently pass a prediction off as a measurement.

Run: python scripts/make_rlvr_macros.py
"""
import csv
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE.parent / 'verifier_economics' / 'figs' / 'rlvr_numbers.tex'
PENDING = r'{\color{red}\textbf{[PENDING]}}'


def load_curve(run):
    p = BASE / 'runs' / run / 'curve.csv'
    return list(csv.DictReader(open(p))) if p.exists() else None


def load_heldout(run):
    p = BASE / 'runs' / run / 'heldout.jsonl'
    if not p.exists():
        return None
    rs = sorted((json.loads(l) for l in open(p)), key=lambda x: x['step'])
    return rs or None


def window(rows, key, lo, hi):
    xs = [float(r[key]) for r in rows if lo <= int(r['step']) < hi]
    return sum(xs) / len(xs) if xs else None


def emit(macros, name, val, fmt='{:.1f}', scale=1.0, suffix=''):
    if val is None:
        macros[name] = PENDING
    else:
        macros[name] = fmt.format(val * scale) + suffix


def main():
    m = {}
    complete = {}  # run -> bool(final curve present)

    def do_arm(prefix, run, full_steps=800):
        cur = load_curve(run)
        ho = load_heldout(run)
        thr = int(full_steps * 0.97)  # "final" checkpoint threshold (tolerate save cadence)
        done = cur is not None and len(cur) >= thr
        complete[run] = done
        # on-policy strong accuracy: first-100 vs last-100 window
        if cur:
            a0 = window(cur, 'strong_acc', 0, 100)
            steps = [int(r['step']) for r in cur]
            last = max(steps)
            aL = window(cur, 'strong_acc', last - 99, last + 1)
            g0 = window(cur, 'gaming_rate', 0, 100)
            gL = window(cur, 'gaming_rate', last - 99, last + 1)
            emit(m, prefix + 'AccStart', a0, '{:.1f}', 100)
            emit(m, prefix + 'AccEnd', aL, '{:.1f}', 100)
            emit(m, prefix + 'GameStart', g0, '{:.1f}', 100)
            emit(m, prefix + 'GameEnd', gL, '{:.1f}', 100)
            m[prefix + 'StrongCalls'] = cur[-1]['cum_strong_calls']
        else:
            for s in ['AccStart', 'AccEnd', 'GameStart', 'GameEnd', 'StrongCalls']:
                m[prefix + s] = PENDING
        # held-out: base (step 0) and final. "final" is the last available checkpoint's held-out once
        # training is done — normally the step-`full_steps` checkpoint, but if that checkpoint was lost
        # (e.g. a disk-full save failure) we honestly report the last VALID checkpoint (max step present).
        if ho:
            emit(m, prefix + 'HoBase', ho[0]['heldout_strong_acc'], '{:.1f}', 100)
            emit(m, prefix + 'HoFinal', (ho[-1]['heldout_strong_acc'] if done else None), '{:.1f}', 100)
        else:
            m[prefix + 'HoBase'] = PENDING
            m[prefix + 'HoFinal'] = PENDING

    def savings(prefix, cascade_run, strong_run):
        cs, ss = load_curve(cascade_run), load_curve(strong_run)
        if cs and ss and complete.get(cascade_run) and complete.get(strong_run):
            emit(m, prefix, 1 - int(cs[-1]['cum_strong_calls']) / max(1, int(ss[-1]['cum_strong_calls'])),
                 '{:.0f}', 100)
        else:
            m[prefix] = PENDING

    # 7B main (800 steps, lr 2e-6): on-policy separation + cost; held-out flat at this budget.
    do_arm('mSevenBCheap', 'main7b_cheap_s0')
    do_arm('mSevenBStrong', 'main7b_strong_s0')
    do_arm('mSevenBCascade', 'main7b_cascade_s0')
    savings('mSevenBSavings', 'main7b_cascade_s0', 'main7b_strong_s0')

    # 7B long-horizon probe (2000 steps, lr 5e-6): held-out separation emerges.
    do_arm('longCheap', 'long7b_cheap_s0', 2000)
    do_arm('longStrong', 'long7b_strong_s0', 2000)
    do_arm('longStrongB', 'long7b_strong_s1', 2000)
    do_arm('longCascade', 'long7b_cascade_s0', 2000)
    savings('longSavings', 'long7b_cascade_s0', 'long7b_strong_s0')

    # Cross-family Llama-3.1-8B (800 steps, lr 2e-6).
    do_arm('llamaCheap', 'llama8b_cheap_s0')
    do_arm('llamaStrong', 'llama8b_strong_s0')
    do_arm('llamaCascade', 'llama8b_cascade_s0')
    savings('llamaSavings', 'llama8b_cascade_s0', 'llama8b_strong_s0')

    # Dial frontier points (held-out final + cumulative strong-calls) at p in {0.25,0.5,0.75}.
    for pp, pfx in [('025', 'dialTwentyFive'), ('050', 'dialFifty'), ('075', 'dialSeventyFive')]:
        do_arm(pfx, f'main7bd{pp}_dial_s0')
        cur = load_curve(f'main7bd{pp}_dial_s0')
        m[pfx + 'Calls'] = cur[-1]['cum_strong_calls'] if cur and complete.get(f'main7bd{pp}_dial_s0') else PENDING

    # pilot 3B savings (already complete) — register runs so completeness is known
    do_arm('pilotThreeBCascade', 'pilot3b_cascade_s0', 250)
    do_arm('pilotThreeBStrong', 'lr3e6_strong_s0', 250)
    do_arm('pilotThreeBStrongBase', 'pilot3b_strong_s0', 250)
    savings('threeBSavings', 'pilot3b_cascade_s0', 'lr3e6_strong_s0')
    if isinstance(m.get('threeBSavings'), str) and 'PENDING' in m['threeBSavings']:
        savings('threeBSavings', 'pilot3b_cascade_s0', 'pilot3b_strong_s0')

    m['mSevenBArmsComplete'] = str(sum(1 for r in ('main7b_cheap_s0', 'main7b_strong_s0',
                                                   'main7b_cascade_s0') if complete.get(r)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        f.write('% AUTO-GENERATED by scripts/make_rlvr_macros.py — do not edit by hand.\n')
        f.write('% [PENDING] = experiment not yet complete; rerun the generator to fill.\n')
        for k in sorted(m):
            f.write(f'\\newcommand{{\\{k}}}{{{m[k]}}}\n')
    print(f'wrote {len(m)} macros -> {OUT}  ({m["mSevenBArmsComplete"]}/3 main7b arms complete)')
    for k in sorted(m):
        print(f'  \\{k} = {m[k]}')


if __name__ == '__main__':
    main()
