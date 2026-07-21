"""CPU test: the ms-swift plugin registers, rewards match arms.py, and JSONL instrumentation lands.

Run: VE_RUN_DIR=/tmp/ve_plugin_test python test_plugin.py  (in an env with ms-swift installed)
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

run_dir = Path(os.environ.get('VE_RUN_DIR') or tempfile.mkdtemp(prefix='ve_plugin_test_'))
os.environ['VE_RUN_DIR'] = str(run_dir)
if run_dir.exists():
    shutil.rmtree(run_dir)

sys.path.insert(0, str(Path(__file__).resolve().parent / 'verifiers'))
import plugin  # noqa: E402  (registers orms, must import after VE_RUN_DIR is set)
from swift.rewards import orms  # noqa: E402

GOLD = ['7', '1/2', '3.5']
COMPLETIONS = [
    'Compute 3+4. The answer is \\boxed{7}.',            # correct boxed -> all arms accept
    'The value 1/2 appears here but I box \\boxed{2}.',  # gamed: cheap accepts, strong rejects
    'Something entirely wrong: \\boxed{9}.',              # wrong -> all arms reject
]
EXPECT = {'ve_cheap': [1.0, 1.0, 0.0], 've_strong': [1.0, 0.0, 0.0], 've_cascade': [1.0, 0.0, 0.0]}


class _TS:  # stand-in for trainer_state
    global_step = 3


ok = True
for name, want in EXPECT.items():
    fn = orms[name](args=None)
    got = fn(COMPLETIONS, GOLD, trainer_state=_TS())
    status = 'PASS' if got == want else 'FAIL'
    ok &= got == want
    print(f'{name:12s} rewards={got} expected={want} {status}')

logs = sorted(run_dir.glob('samples.rank*.jsonl'))
rows = [json.loads(line) for p in logs for line in open(p)]
print(f'\nlog rows: {len(rows)} (expect {len(EXPECT) * len(GOLD)}) in {run_dir}')
ok &= len(rows) == len(EXPECT) * len(GOLD)

gamed = [r for r in rows if r['arm'] == 'cheap' and r['gamed']]
ok &= len(gamed) == 1 and gamed[0]['reward'] == 1.0 and gamed[0]['strong'] == 0
print('cheap arm logs the gamed sample with reward 1.0:', 'PASS' if gamed else 'FAIL')
casc = [r for r in rows if r['arm'] == 'cascade']
ok &= all(r['step'] == 3 for r in rows)
esc = [r for r in casc if r['cheap']]
ok &= all(r['cost_s'] > 0 for r in esc)
print('cascade escalation costs > cheap-only:', 'PASS' if esc else 'FAIL')

print('\nALL:', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
