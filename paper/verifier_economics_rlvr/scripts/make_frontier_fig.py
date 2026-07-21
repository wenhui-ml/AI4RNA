"""Draw the dial cost x robustness frontier: held-out accuracy vs cumulative strong-verifier calls,
for p in {0 (cheap), 0.25, 0.5, 0.75, 1 (cascade)} on the 7B policy. -> figs/f1_frontier.pdf
Run: python scripts/make_frontier_fig.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
OUT = BASE.parent / 'verifier_economics' / 'figs' / 'f1_frontier.pdf'

# (label, run dir) in dial order; endpoints reuse the main7b cheap/cascade runs.
POINTS = [
    ('cheap ($p{=}0$)', 'main7b_cheap_s0'),
    ('$p{=}0.25$', 'main7bd025_dial_s0'),
    ('$p{=}0.5$', 'main7bd050_dial_s0'),
    ('$p{=}0.75$', 'main7bd075_dial_s0'),
    ('cascade ($p{=}1$)', 'main7b_cascade_s0'),
]


def final_heldout(run):
    p = BASE / 'runs' / run / 'heldout.jsonl'
    rs = sorted((json.loads(l) for l in open(p)), key=lambda x: x['step'])
    return rs[-1]['heldout_strong_acc'] * 100


def strong_calls(run):
    rows = list(csv.DictReader(open(BASE / 'runs' / run / 'curve.csv')))
    return int(rows[-1]['cum_strong_calls'])


def main():
    xs, ys, labels = [], [], []
    for lab, run in POINTS:
        xs.append(strong_calls(run) / 1000.0)
        ys.append(final_heldout(run))
        labels.append(lab)

    plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(xs, ys, '-', color='#888', lw=1.2, zorder=1)
    for x, y, lab in zip(xs, ys, labels):
        ax.scatter([x], [y], s=45, zorder=2)
        ax.annotate(lab, (x, y), textcoords='offset points', xytext=(6, 5), fontsize=9)
    ax.set_xlabel('cumulative strong-verifier calls (thousands)')
    ax.set_ylabel('held-out strict accuracy (\\%)')
    ax.set_title('Verifier dial: quality vs.\\ verifier cost (7B, 800 steps)', fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches='tight')
    print(f'wrote {OUT}')
    for lab, x, y in zip(labels, xs, ys):
        print(f'  {lab}: calls={x*1000:.0f} heldout={y:.1f}%')


if __name__ == '__main__':
    main()
