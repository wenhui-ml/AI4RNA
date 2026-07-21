"""Draw the dial cost x robustness frontier: held-out accuracy vs cumulative strong-verifier calls,
for p in {0 (cheap), 0.25, 0.5, 0.75, 1 (cascade)} on the 7B policy.
Output: figs/f_dial_frontier.pdf  (distinct name -- must NOT collide with the RNA Axis-2 fig f1_frontier.pdf)
Run: python scripts/make_frontier_fig.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
OUT = BASE.parent / 'verifier_economics' / 'figs' / 'f_dial_frontier.pdf'

POINTS = [
    ('cheap ($p{=}0$)', 'main7b_cheap_s0', 'left'),
    ('$p{=}0.25$', 'main7bd025_dial_s0', 'up'),
    ('$p{=}0.5$', 'main7bd050_dial_s0', 'up'),
    ('$p{=}0.75$', 'main7bd075_dial_s0', 'up'),
    ('cascade ($p{=}1$)', 'main7b_cascade_s0', 'right'),
]


def final_heldout(run):
    rs = sorted((json.loads(l) for l in open(BASE / 'runs' / run / 'heldout.jsonl')), key=lambda x: x['step'])
    return rs[-1]['heldout_strong_acc'] * 100


def strong_calls(run):
    return int(list(csv.DictReader(open(BASE / 'runs' / run / 'curve.csv')))[-1]['cum_strong_calls'])


def main():
    xs = [strong_calls(r) / 1000.0 for _, r, _ in POINTS]
    ys = [final_heldout(r) for _, r, _ in POINTS]

    plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'mathtext.fontset': 'dejavuserif'})
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    ax.plot(xs, ys, '-', color='#999', lw=1.3, zorder=1)
    ax.scatter(xs, ys, s=48, color='#1f4e79', zorder=3)

    # labels placed to stay strictly inside the axes (no overflow, no floating in blank space)
    for (lab, _, pos), x, y in zip(POINTS, xs, ys):
        if pos == 'left':
            ax.annotate(lab, (x, y), textcoords='offset points', xytext=(8, -3), ha='left', va='center', fontsize=9)
        elif pos == 'right':
            ax.annotate(lab, (x, y), textcoords='offset points', xytext=(-8, -12), ha='right', va='top', fontsize=9)
        else:
            ax.annotate(lab, (x, y), textcoords='offset points', xytext=(0, 7), ha='center', va='bottom', fontsize=9)

    # x-range with margin so the leftmost (cheap) and rightmost (cascade) labels fit inside
    xmax = max(xs)
    ax.set_xlim(-0.10 * xmax, 1.18 * xmax)
    # y-range: place the data band in the LOWER-CENTER of the frame with headroom above (uniform framing)
    lo, hi = min(ys), max(ys)
    span = max(hi - lo, 1.0)
    ax.set_ylim(lo - 1.2 * span, hi + 2.2 * span)

    ax.set_xlabel('cumulative strong-verifier calls (thousands)')
    ax.set_ylabel('held-out strict accuracy (%)')
    ax.set_title('Verifier dial: held-out quality vs. verifier cost (7B, 800 steps)', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches='tight')
    print(f'wrote {OUT}')
    for (lab, _, _), x, y in zip(POINTS, xs, ys):
        print(f'  {lab}: calls={x*1000:.0f} heldout={y:.1f}%')


if __name__ == '__main__':
    main()
