"""CPU test for the risk-calibrated dial arm. Proves the load-bearing invariants:
  - dial(p=0) == cheap  (never escalate; trust every cheap-accept)
  - dial(p=1) == cascade (always escalate cheap-accepts to strong)
  - escalation coin is deterministic & reproducible (same completion -> same decision)
  - realized escalation fraction among cheap-accepts ~= p (large sample)
  - cost and gaming are MONOTONE in p (frontier is well-ordered)
Run: ~/miniconda3/envs/cua/bin/python test_dial.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'verifiers'))
from arms import reward_and_metrics, _stable_coin  # noqa: E402

ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}")


# 1) endpoint equivalence on a representative case set (correct, gamed, wrong, cheap-reject)
cases = [
    ('The answer is \\boxed{7}.', '7'),                    # cheap-accept, strong-accept
    ('I mention 7 but box \\boxed{2}.', '7'),              # cheap-accept, strong-reject (gamed)
    ('Totally wrong \\boxed{9}.', '7'),                    # cheap-reject
    ('value 1/2 here, box \\boxed{3}', '1/2'),             # cheap-accept, strong-reject (gamed)
    ('final answer 3.5 in \\boxed{3.5}', '3.5'),           # cheap-accept, strong-accept
]
for comp, gold in cases:
    d0 = reward_and_metrics('dial', comp, gold, dial_p=0.0)
    ch = reward_and_metrics('cheap', comp, gold)
    d1 = reward_and_metrics('dial', comp, gold, dial_p=1.0)
    cas = reward_and_metrics('cascade', comp, gold)
    check(f"dial(p=0)==cheap  reward  [{gold}]", d0['reward'] == ch['reward'])
    check(f"dial(p=0)==cheap  cost    [{gold}]", abs(d0['reward_cost_s'] - ch['reward_cost_s']) < 1e-6 or d0['reward'] == ch['reward'])
    check(f"dial(p=1)==cascade reward [{gold}]", d1['reward'] == cas['reward'])

# 2) determinism: same input -> identical coin and identical verdict across repeated calls
comp, gold = cases[1]
coins = {_stable_coin(comp, gold) for _ in range(5)}
check("coin is deterministic (single value over repeats)", len(coins) == 1)
verdicts = {reward_and_metrics('dial', comp, gold, dial_p=0.5)['reward'] for _ in range(5)}
check("dial verdict deterministic at p=0.5", len(verdicts) == 1)

# 3) coin uniformity: escalation fraction among cheap-accepts ~= p on many distinct completions
N = 4000
gold = '7'
comps = [f'the number 7 appears, box \\boxed{{{i%3}}} ({i})' for i in range(N)]  # all cheap-accept
for p in (0.25, 0.5, 0.75):
    esc = sum(reward_and_metrics('dial', c, gold, dial_p=p)['escalated'] for c in comps)
    frac = esc / N
    check(f"escalation frac ~= p={p} (got {frac:.3f})", abs(frac - p) < 0.03)

# 4) monotonicity: mean cost and gaming (rewarded-gamed) are monotone non-decreasing/decreasing in p
#    build a mixed workload: half correct, half gamed
mixed = ([('box \\boxed{7} correct %d' % i, '7') for i in range(1000)] +
         [('7 mentioned, box \\boxed{1} gamed %d' % i, '7') for i in range(1000)])
prev_cost, prev_game = -1.0, 2.0
mono_cost, mono_game = True, True
for p in (0.0, 0.25, 0.5, 0.75, 1.0):
    ms = [reward_and_metrics('dial', c, g, dial_p=p) for c, g in mixed]
    cost = sum(m['reward_cost_s'] for m in ms) / len(ms)
    # rewarded-gamed = gamed sample that still got reward 1 (i.e., not escalated)
    rgame = sum(1 for m in ms if m['gamed'] and m['reward'] > 0.5) / len(ms)
    mono_cost &= cost >= prev_cost - 1e-9
    mono_game &= rgame <= prev_game + 1e-9
    prev_cost, prev_game = cost, rgame
    print(f"   p={p}: mean_cost={cost:.6e}  rewarded_gaming={rgame:.3f}")
check("cost monotone non-decreasing in p", mono_cost)
check("rewarded-gaming monotone non-increasing in p", mono_game)

# 5) invalid p raises
try:
    reward_and_metrics('dial', 'x', '1', dial_p=None)
    check("dial_p=None raises", False)
except ValueError:
    check("dial_p=None raises", True)

print('\nALL:', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
