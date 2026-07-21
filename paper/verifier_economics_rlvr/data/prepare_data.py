"""Build RLVR train/eval sets from cached AI-MO/NuminaMath-CoT.

Protocol (cost-repair RLVR, see ../HANDOFF.md §5):
  - keep only problems whose reference solution ends in a \boxed{} answer that canonicalizes to a
    plain numeric value (int / decimal / fraction) -> verifier semantics are airtight for all arms.
  - eval  = 500 problems from source=='math' (Hendrycks MATH -> MATH500-style held-out).
  - train = TRAIN_N problems sampled from all OTHER sources (source=='math' excluded entirely),
    deduped by problem text. Written in ms-swift GRPO format with a `solution` column carrying the
    canonical gold answer string consumed by verifiers/arms.py.
Run: HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python prepare_data.py
"""
import json
import random
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / 'verifiers'))
from arms import extract_boxed, strong_verify, cheap_verify  # noqa: E402

SEED = 0
TRAIN_N = 50_000
EVAL_N = 500
SMOKE_N = 256
MAX_PROBLEM_CHARS = 1500

_FRAC = re.compile(r'(-?)\\[dt]?frac\{(-?\d+)\}\{(-?\d+)\}')


def canon_gold(boxed):
    """Canonicalize a boxed answer to 'int', 'a/b' or decimal string; None if not plainly numeric."""
    if not boxed:
        return None
    s = boxed.strip()
    for junk in ['\\!', '\\,', '\\;', '\\ ', '$', '{,}', '\\left', '\\right']:
        s = s.replace(junk, '')
    s = s.replace('^\\circ', '').replace('\\circ', '').replace('\N{DEGREE SIGN}', '')
    s = s.replace('\\%', '').replace('%', '')
    s = s.replace(' ', '').replace(',', '')
    m = _FRAC.fullmatch(s)
    if m:
        sign = -1 if m.group(1) == '-' else 1
        a, b = sign * int(m.group(2)), int(m.group(3))
        if b == 0:
            return None
        if b < 0:
            a, b = -a, -b
        return f'{a}/{b}'
    if re.fullmatch(r'-?\d+', s):
        return str(int(s))
    if re.fullmatch(r'-?\d*\.\d+', s):
        return s
    if re.fullmatch(r'-?\d+/\d+', s):
        a, b = s.split('/')
        return None if int(b) == 0 else f'{int(a)}/{int(b)}'
    return None


def _extract(row):
    gold = canon_gold(extract_boxed(row['solution']))
    return {'gold': gold if gold is not None else ''}


def _keep(row):
    if not row['gold'] or len(row['problem']) > MAX_PROBLEM_CHARS:
        return False
    p = row['problem']
    low = p.lower()
    if 'prove' in low or '证明' in p:  # exclude proof-style problems
        return False
    return True


def main():
    from datasets import load_dataset
    rng = random.Random(SEED)
    ds = load_dataset('AI-MO/NuminaMath-CoT')['train']
    ds = ds.map(_extract, num_proc=32)
    ds = ds.filter(_keep, num_proc=32)
    print(f'usable rows with numeric gold: {len(ds)}')

    # round-trip sanity on a sample: arms must accept the gold under both verifiers
    for i in rng.sample(range(len(ds)), 200):
        g = ds[i]['gold']
        assert strong_verify(f'The answer is \\boxed{{{g}}}.', g)[0], (i, g)
        assert cheap_verify(f'The answer is \\boxed{{{g}}}.', g)[0], (i, g)

    math_rows = [i for i, s in enumerate(ds['source']) if s == 'math']
    other_rows = [i for i, s in enumerate(ds['source']) if s != 'math']
    print(f'source==math rows: {len(math_rows)}, other: {len(other_rows)}')

    eval_idx = rng.sample(math_rows, EVAL_N)
    eval_set = ds.select(eval_idx)
    eval_problems = set(eval_set['problem'])

    rng.shuffle(other_rows)
    seen, train_idx = set(), []
    for i in other_rows:
        p = ds[i]['problem']
        if p in seen or p in eval_problems:
            continue
        seen.add(p)
        train_idx.append(i)
        if len(train_idx) >= TRAIN_N:
            break
    train_set = ds.select(train_idx)

    with open(BASE / 'eval.jsonl', 'w') as f:
        for r in eval_set:
            f.write(json.dumps({'problem': r['problem'], 'gold': r['gold'], 'source': r['source']},
                               ensure_ascii=False) + '\n')

    def train_line(r):
        return json.dumps({'messages': [{'role': 'user', 'content': r['problem']}],
                           'solution': r['gold']}, ensure_ascii=False) + '\n'

    with open(BASE / 'train.jsonl', 'w') as f:
        for r in train_set:
            f.write(train_line(r))
    with open(BASE / 'train_smoke.jsonl', 'w') as f:
        for r in train_set.select(range(SMOKE_N)):
            f.write(train_line(r))

    from collections import Counter
    meta = {'seed': SEED, 'train_n': len(train_set), 'eval_n': len(eval_set), 'smoke_n': SMOKE_N,
            'max_problem_chars': MAX_PROBLEM_CHARS,
            'train_sources': dict(Counter(train_set['source'])),
            'filter': 'numeric-canonical boxed gold; no proof problems; math source held out entirely'}
    with open(BASE / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
