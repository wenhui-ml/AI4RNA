"""ms-swift GRPO external reward plugin: the three verifier arms of the cost-repair experiment.

Registers three ORMs — ve_cheap / ve_strong / ve_cascade — one per arm (HANDOFF §5/§7). Each call
computes the arm's reward via verifiers/arms.py AND appends per-sample instrumentation (cheap/strong
verdicts, gamed flag, verifier cost) to $VE_RUN_DIR/samples.rank{RANK}.jsonl, from which J = TPR-FPR,
gaming rate and cumulative verifier cost over training are aggregated (scripts/aggregate.py).

Usage:
    VE_RUN_DIR=/path/to/run swift rlhf --rlhf_type grpo \
        --external_plugins .../verifiers/plugin.py --reward_funcs ve_cascade ...
Dataset must carry a `solution` column holding the canonical gold answer (data/prepare_data.py).
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import List

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))
from arms import reward_and_metrics  # noqa: E402

from swift.rewards import ORM, orms  # noqa: E402


def _rank() -> int:
    for var in ('RANK', 'LOCAL_RANK'):
        v = os.environ.get(var)
        if v is not None and v.isdigit():
            return int(v)
    return 0


class _VerifierArmORM(ORM):
    arm: str = None
    dial_p = None  # only used by the 'dial' arm; read from VE_DIAL_P at construction

    def __init__(self, args=None, **kwargs):
        super().__init__(args)
        if self.arm == 'dial':
            self.dial_p = float(os.environ['VE_DIAL_P'])  # required; fail loudly if unset
        run_dir = Path(os.environ.get('VE_RUN_DIR', str(_DIR.parent / 'runs' / 'debug')))
        run_dir.mkdir(parents=True, exist_ok=True)
        self._fh = open(run_dir / f'samples.rank{_rank()}.jsonl', 'a', buffering=1)

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        ts = kwargs.get('trainer_state')
        step = int(getattr(ts, 'global_step', -1)) if ts is not None else -1
        now = round(time.time(), 3)
        rewards = []
        for comp, gold in zip(completions, solution):
            m = reward_and_metrics(self.arm, comp, gold, dial_p=self.dial_p)
            rewards.append(m['reward'])
            rec = {
                't': now, 'step': step, 'arm': self.arm, 'reward': m['reward'],
                'cheap': int(m['cheap_accept']), 'strong': int(m['strong_accept']),
                'gamed': int(m['gamed']), 'escalated': int(m['escalated']),
                'cost_s': round(m['reward_cost_s'], 7),
                'audit_s': round(m['audit_cost_s'], 7), 'len': len(comp),
            }
            if self.dial_p is not None:
                rec['dial_p'] = self.dial_p
            self._fh.write(json.dumps(rec) + '\n')
        return rewards


class VECheap(_VerifierArmORM):
    arm = 'cheap'


class VEStrong(_VerifierArmORM):
    arm = 'strong'


class VECascade(_VerifierArmORM):
    arm = 'cascade'


class VEDial(_VerifierArmORM):
    arm = 'dial'  # risk-calibrated knob; escalation probability from env VE_DIAL_P in [0,1]


orms['ve_cheap'] = VECheap
orms['ve_strong'] = VEStrong
orms['ve_cascade'] = VECascade
orms['ve_dial'] = VEDial
