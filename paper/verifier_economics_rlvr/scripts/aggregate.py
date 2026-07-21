"""Aggregate per-sample verifier logs (samples.rank*.jsonl) into per-step training curves.

For each run dir under runs/, computes per global step:
  - train reward mean (the arm's own reward signal)
  - strong-gold accuracy on rollouts (mean strong verdict = true training quality proxy)
  - gaming rate  = mean(cheap-accept AND strong-reject)
  - Youden's J of the ARM's accept decision vs strong-gold: J = TPR - FPR
    (TPR = P(arm accepts | strong correct), FPR = P(arm accepts | strong wrong))
  - verifier cost: per-step and cumulative wall seconds; cheap/strong call counts
    (cheap arm: 1 cheap call; strong arm: 1 strong call; cascade: 1 cheap + [cheap-accept] strong)

Usage: python aggregate.py RUN_DIR [RUN_DIR ...]   -> writes RUN_DIR/curve.csv + prints summary
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def aggregate(run_dir: Path):
    rows = []
    for p in sorted(run_dir.glob('samples.rank*.jsonl')):
        with open(p) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # tolerate a torn tail line from a live run
    if not rows:
        print(f'{run_dir}: no sample logs')
        return None
    arm = rows[0]['arm']
    by_step = defaultdict(list)
    for r in rows:
        by_step[r['step']].append(r)

    out, cum_cost, cum_strong_calls = [], 0.0, 0
    for step in sorted(by_step):
        rs = by_step[step]
        n = len(rs)
        accepts = [r['reward'] > 0.5 for r in rs]
        strong = [bool(r['strong']) for r in rs]
        cheap = [bool(r['cheap']) for r in rs]
        pos = [a for a, s in zip(accepts, strong) if s]
        neg = [a for a, s in zip(accepts, strong) if not s]
        tpr = sum(pos) / len(pos) if pos else float('nan')
        fpr = sum(neg) / len(neg) if neg else float('nan')
        step_cost = sum(r['cost_s'] for r in rs)
        cum_cost += step_cost
        # strong-verifier calls: strong pays on all; cheap pays none; cascade/dial pay on escalations.
        # Prefer the logged 'escalated' flag (cascade/dial); fall back to cheap-accepts for older
        # cascade logs written before the 'escalated' field existed (backward compatible).
        if arm == 'strong':
            strong_calls = n
        elif arm == 'cheap':
            strong_calls = 0
        elif any('escalated' in r for r in rs):
            strong_calls = sum(bool(r.get('escalated')) for r in rs)
        else:
            strong_calls = sum(cheap)  # legacy cascade logs
        cum_strong_calls += strong_calls
        out.append({
            'step': step, 'n': n, 'arm': arm,
            'reward_mean': round(sum(r['reward'] for r in rs) / n, 4),
            'strong_acc': round(sum(strong) / n, 4),
            'gaming_rate': round(sum(1 for c, s in zip(cheap, strong) if c and not s) / n, 4),
            'tpr': round(tpr, 4) if pos else '', 'fpr': round(fpr, 4) if neg else '',
            'J': round(tpr - fpr, 4) if pos and neg else '',
            'cost_s': round(step_cost, 5), 'cum_cost_s': round(cum_cost, 4),
            'strong_calls': strong_calls, 'cum_strong_calls': cum_strong_calls,
            'len_mean': round(sum(r['len'] for r in rs) / n, 1),
        })
    csv_path = run_dir / 'curve.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    last = out[-1]
    print(f"{run_dir.name}: arm={arm} steps={len(out)} samples={len(rows)} | "
          f"last: reward={last['reward_mean']} strong_acc={last['strong_acc']} "
          f"gaming={last['gaming_rate']} J={last['J']} | cum_cost={last['cum_cost_s']}s "
          f"cum_strong_calls={last['cum_strong_calls']} -> {csv_path}")
    return out


if __name__ == '__main__':
    targets = [Path(a) for a in sys.argv[1:]]
    if not targets:
        base = Path(__file__).resolve().parent.parent / 'runs'
        targets = sorted(d for d in base.iterdir() if d.is_dir() and list(d.glob('samples.rank*.jsonl')))
    for t in targets:
        aggregate(t)
