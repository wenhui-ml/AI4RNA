"""Held-out eval: strict(strong)-verified accuracy of checkpoints on data/eval.jsonl via vLLM.

Sweeps RUN_DIR/ckpt/**/checkpoint-*/ (plus optionally the base model as step 0), greedy decoding,
appends one record per checkpoint to RUN_DIR/heldout.jsonl and skips checkpoints already evaluated.
This produces the y-axis of the headline figure (held-out quality vs cumulative verifier cost).

Usage:
  CUDA_VISIBLE_DEVICES=6 python eval_checkpoints.py RUN_DIR [--base Qwen/Qwen2.5-3B-Instruct]
                         [--util 0.4] [--max-tokens 2048]
"""
import argparse
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'verifiers'))
from arms import strong_verify  # noqa: E402

SYSTEM = (BASE / 'configs' / 'system_prompt.txt').read_text().strip()


def eval_model(model_path: str, evalset, util: float, max_tokens: int):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_path, gpu_memory_utilization=util, max_model_len=4096,
              dtype='bfloat16', enforce_eager=False)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': r['problem']}],
        tokenize=False, add_generation_prompt=True) for r in evalset]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens))
    n_ok = 0
    for r, o in zip(evalset, outs):
        ok, _ = strong_verify(o.outputs[0].text, r['gold'])
        n_ok += ok
    del llm
    import gc, torch  # noqa: E401
    gc.collect()
    torch.cuda.empty_cache()
    return n_ok / len(evalset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('--base', default=None, help='also eval this base model as step 0')
    ap.add_argument('--util', type=float, default=0.4)
    ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--min-age-s', type=float, default=0.0,
                    help='skip checkpoints modified more recently than this (avoid half-written saves)')
    args = ap.parse_args()

    evalset = [json.loads(l) for l in open(BASE / 'data' / 'eval.jsonl')]
    out_path = args.run_dir / 'heldout.jsonl'
    done = set()
    if out_path.exists():
        done = {json.loads(l)['ckpt'] for l in open(out_path)}

    todo = []
    if args.base and args.base not in done:
        todo.append((0, args.base))
    import time
    for d in sorted(args.run_dir.glob('ckpt/**/checkpoint-*')):
        m = re.fullmatch(r'checkpoint-(\d+)', d.name)
        if m and d.is_dir() and str(d) not in done:
            if args.min_age_s and time.time() - d.stat().st_mtime < args.min_age_s:
                continue
            todo.append((int(m.group(1)), str(d)))
    todo.sort()
    print(f'{len(todo)} checkpoints to eval on {len(evalset)} held-out problems')

    for step, path in todo:
        acc = eval_model(path, evalset, args.util, args.max_tokens)
        rec = {'step': step, 'ckpt': path, 'heldout_strong_acc': round(acc, 4), 'n': len(evalset)}
        with open(out_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        print(rec)


if __name__ == '__main__':
    main()
