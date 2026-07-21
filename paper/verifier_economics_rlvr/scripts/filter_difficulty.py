"""Difficulty-filter train prompts by base-model pass rate (DAPO-style dynamic-sampling prep).

Rationale: pilot showed 60–90% of GRPO groups have zero reward variance (all-correct or all-wrong)
-> zero gradient. Keep only prompts where the base policy passes 1..7 of 8 samples, so every group
carries signal.

Shard mode (one process per GPU):
  CUDA_VISIBLE_DEVICES=i python filter_difficulty.py --shard i --num-shards 8 \
      --model /path/7B --in data/train.jsonl --out-dir data/filter7b [--max-prompts 24000]
Merge mode (after all shards):
  python filter_difficulty.py --merge --out-dir data/filter7b --train-out data/train_7b_filtered.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'verifiers'))
from arms import strong_verify  # noqa: E402

SYSTEM = (BASE / 'configs' / 'system_prompt.txt').read_text().strip()
N_SAMPLES = 8


def run_shard(args):
    rows = [json.loads(l) for l in open(args.inp)]
    rows = rows[:args.max_prompts][args.shard::args.num_shards]
    print(f'shard {args.shard}/{args.num_shards}: {len(rows)} prompts x {N_SAMPLES} samples')
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, gpu_memory_utilization=0.55, max_model_len=3072, dtype='bfloat16')
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': r['messages'][0]['content']}],
        tokenize=False, add_generation_prompt=True) for r in rows]
    sp = SamplingParams(n=N_SAMPLES, temperature=1.0, max_tokens=1536, seed=args.shard)
    outs = llm.generate(prompts, sp)
    out_path = Path(args.out_dir) / f'shard{args.shard}.jsonl'
    with open(out_path, 'w') as f:
        for r, o in zip(rows, outs):
            k = sum(strong_verify(c.text, r['solution'])[0] for c in o.outputs)
            f.write(json.dumps({'messages': r['messages'], 'solution': r['solution'], 'pass8': k},
                               ensure_ascii=False) + '\n')
    print(f'shard {args.shard} done -> {out_path}')


def run_merge(args):
    rows = []
    for p in sorted(Path(args.out_dir).glob('shard*.jsonl')):
        rows += [json.loads(l) for l in open(p)]
    hist = {}
    for r in rows:
        hist[r['pass8']] = hist.get(r['pass8'], 0) + 1
    keep = [r for r in rows if 1 <= r['pass8'] <= 7]
    import random
    random.Random(0).shuffle(keep)
    with open(args.train_out, 'w') as f:
        for r in keep:
            f.write(json.dumps({'messages': r['messages'], 'solution': r['solution']},
                               ensure_ascii=False) + '\n')
    print(f'pass8 histogram: {dict(sorted(hist.items()))}')
    print(f'kept {len(keep)}/{len(rows)} informative prompts -> {args.train_out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=None)
    ap.add_argument('--num-shards', type=int, default=8)
    ap.add_argument('--model')
    ap.add_argument('--in', dest='inp', default=str(BASE / 'data' / 'train.jsonl'))
    ap.add_argument('--out-dir', default=str(BASE / 'data' / 'filter7b'))
    ap.add_argument('--max-prompts', type=int, default=24000)
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--train-out', default=str(BASE / 'data' / 'train_7b_filtered.jsonl'))
    a = ap.parse_args()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    run_merge(a) if a.merge else run_shard(a)
