"""Minimal LLM cross-domain validation of the paper's central mechanism (Axis-2 + Goodhart), in the
domain RLVR actually uses: code with unit-test verification. Mirrors the RNA construction exactly.

  cheap rung  (\match analogue): passes the small set of PROVIDED unit tests  (gameable)
  strong rung (\pfe analogue):   passes a large HELD-OUT / mutated test population (robustness)

Fixed proposer (one LLM at fixed temperature), vary only the reward used to SELECT among N sampled
candidates (best-of-N = the optimization budget). If the cheap proxy is optimized while robustness lags
--- cheap-arm PROVIDED-pass rises with N but its HELD-OUT-pass stalls, while strong-arm HELD-OUT-pass
rises --- that is the same proxy/objective divergence we measured in RNA, now on a real LLM verifier.

Auth: source the 0600 .orkey into OPENAI_API_KEY (see runs/launch_p1.sh); never echoed. Code execution is
sandboxed in a subprocess with a wall timeout and address-space cap."""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, tempfile

from ..llm.openrouter import OpenRouterClient

OUT_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "runs", "llm_verifier_goodhart")

# Each problem: a prompt, a WEAK provided test set (cheap rung), a COMPREHENSIVE held-out set (strong rung).
# A mix of edge-case-heavy problems (where a WEAK provided test does not pin down the spec, so a
# provided-passing solution can still fail the true objective) and easy controls (which stay robust).
PROBLEMS = [
    {"name": "floor_div", "sig": "def floor_div(a, b):",
     "desc": "Return the mathematical floor of a divided by b (round toward negative infinity), like Python's // operator, for integers.",
     "provided": ["floor_div(7, 2) == 3"],
     "held": ["floor_div(-7, 2) == -4", "floor_div(7, -2) == -4", "floor_div(-8, 2) == -4",
              "floor_div(0, 3) == 0", "floor_div(9, 3) == 3", "floor_div(-1, 3) == -1"]},
    {"name": "is_leap_year", "sig": "def is_leap_year(y):",
     "desc": "Return True iff year y is a leap year in the proleptic Gregorian calendar.",
     "provided": ["is_leap_year(2020) == True"],
     "held": ["is_leap_year(1900) == False", "is_leap_year(2000) == True", "is_leap_year(2100) == False",
              "is_leap_year(2023) == False", "is_leap_year(2024) == True", "is_leap_year(1600) == True"]},
    {"name": "days_in_month", "sig": "def days_in_month(m, y):",
     "desc": "Return the number of days in month m (1-12) of year y, accounting for leap years.",
     "provided": ["days_in_month(1, 2021) == 31"],
     "held": ["days_in_month(2, 2020) == 29", "days_in_month(2, 2021) == 28", "days_in_month(2, 1900) == 28",
              "days_in_month(2, 2000) == 29", "days_in_month(4, 2021) == 30", "days_in_month(12, 2021) == 31"]},
    {"name": "median", "sig": "def median(nums):",
     "desc": "Return the median of a non-empty list of numbers (for even length, the average of the two middle values).",
     "provided": ["median([1, 2, 3]) == 2"],
     "held": ["median([1,2,3,4]) == 2.5", "median([1]) == 1", "median([4,1,3,2]) == 2.5",
              "median([7,7,7]) == 7", "median([1,2]) == 1.5", "median([5,3,1,2,4]) == 3"]},
    {"name": "mode", "sig": "def mode(nums):",
     "desc": "Return the most frequent value in a non-empty list; if several values tie, return the smallest of them.",
     "provided": ["mode([1, 1, 2]) == 1"],
     "held": ["mode([2,2,3,3]) == 2", "mode([5]) == 5", "mode([4,4,1,1,1]) == 1",
              "mode([9,8,8]) == 8", "mode([3,3,3,2,2,2]) == 2"]},
    {"name": "count_overlap", "sig": "def count_overlap(s, sub):",
     "desc": "Return the number of occurrences of sub in s, counting OVERLAPPING occurrences (sub is non-empty).",
     "provided": ["count_overlap('abcabc', 'abc') == 2"],
     "held": ["count_overlap('aaa', 'aa') == 2", "count_overlap('aaaa', 'aa') == 3", "count_overlap('abc', 'x') == 0",
              "count_overlap('aaaaa', 'aa') == 4", "count_overlap('xyxyx', 'xyx') == 2"]},
    {"name": "primes_up_to", "sig": "def primes_up_to(n):",
     "desc": "Return the sorted list of all primes strictly less than n (n may be 0, 1, or negative).",
     "provided": ["primes_up_to(10) == [2, 3, 5, 7]"],
     "held": ["primes_up_to(0) == []", "primes_up_to(1) == []", "primes_up_to(2) == []",
              "primes_up_to(3) == [2]", "primes_up_to(-5) == []", "len(primes_up_to(100)) == 25"]},
    {"name": "fib", "sig": "def fib(n):",
     "desc": "Return the n-th Fibonacci number with fib(0)=0, fib(1)=1.",
     "provided": ["fib(5) == 5"],
     "held": ["fib(0) == 0", "fib(1) == 1", "fib(2) == 1", "fib(10) == 55", "fib(20) == 6765"]},
    {"name": "is_palindrome", "sig": "def is_palindrome(s):",
     "desc": "Return True iff s reads the same forwards and backwards, ignoring case and non-alphanumeric characters.",
     "provided": ["is_palindrome('racecar') == True"],
     "held": ["is_palindrome('') == True", "is_palindrome('Ab') == False",
              "is_palindrome('A man a plan a canal Panama') == True", "is_palindrome('No lemon, no melon') == True",
              "is_palindrome('abc') == False"]},
    {"name": "gcd", "sig": "def gcd(a, b):",
     "desc": "Return the greatest common divisor of non-negative integers a and b, with gcd(0,0)=0.",
     "provided": ["gcd(12, 8) == 4"],
     "held": ["gcd(0, 0) == 0", "gcd(0, 5) == 5", "gcd(5, 0) == 5", "gcd(17, 5) == 1", "gcd(100, 60) == 20"]},
    {"name": "first_index", "sig": "def first_index(lst, x):",
     "desc": "Return the index of the first occurrence of x in lst, or -1 if x is not present.",
     "provided": ["first_index([1, 2, 3], 2) == 1"],
     "held": ["first_index([1, 2], 5) == -1", "first_index([], 1) == -1", "first_index([2, 2], 2) == 0",
              "first_index([1,2,3,2], 2) == 1", "first_index([9], 9) == 0"]},
    {"name": "round_half_even", "sig": "def round_half_even(x):",
     "desc": "Round x to the nearest integer, rounding exact halves to the nearest EVEN integer (banker's rounding).",
     "provided": ["round_half_even(2.4) == 2"],
     "held": ["round_half_even(0.5) == 0", "round_half_even(1.5) == 2", "round_half_even(2.5) == 2",
              "round_half_even(3.5) == 4", "round_half_even(-0.5) == 0"]},
    {"name": "chunk", "sig": "def chunk(lst, k):",
     "desc": "Split lst into consecutive chunks of length k; the final chunk may be shorter if lst is not divisible by k.",
     "provided": ["chunk([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]"],
     "held": ["chunk([1,2,3], 2) == [[1,2],[3]]", "chunk([], 2) == []", "chunk([1], 2) == [[1]]",
              "chunk([1,2,3,4,5], 2) == [[1,2],[3,4],[5]]", "chunk([1,2,3], 1) == [[1],[2],[3]]"]},
    {"name": "binary_search", "sig": "def binary_search(lst, x):",
     "desc": "Return the index of x in the sorted list lst using binary search, or -1 if x is absent.",
     "provided": ["binary_search([1, 3, 5, 7], 5) == 2"],
     "held": ["binary_search([], 1) == -1", "binary_search([1], 1) == 0", "binary_search([1,2,3], 4) == -1",
              "binary_search([1,3,5], 1) == 0", "binary_search([1,3,5,7,9], 9) == 4", "binary_search([2,4], 3) == -1"]},
    {"name": "str_to_int", "sig": "def str_to_int(s):",
     "desc": "Parse s as an integer (optional leading/trailing spaces and a leading +/- sign); return 0 if s is not a valid integer.",
     "provided": ["str_to_int('42') == 42"],
     "held": ["str_to_int('-5') == -5", "str_to_int(' 10 ') == 10", "str_to_int('') == 0",
              "str_to_int('abc') == 0", "str_to_int('+7') == 7", "str_to_int('3.5') == 0"]},
    {"name": "avg", "sig": "def avg(nums):",
     "desc": "Return the arithmetic mean of the list nums, or 0.0 if the list is empty.",
     "provided": ["avg([2, 4]) == 3"],
     "held": ["avg([]) == 0.0", "avg([5]) == 5", "avg([1, 2]) == 1.5", "avg([1,2,3,4]) == 2.5"]},
    {"name": "is_anagram", "sig": "def is_anagram(a, b):",
     "desc": "Return True iff strings a and b are anagrams of each other (same multiset of characters, case-sensitive).",
     "provided": ["is_anagram('listen', 'silent') == True"],
     "held": ["is_anagram('', '') == True", "is_anagram('a', 'ab') == False", "is_anagram('rat', 'car') == False",
              "is_anagram('aab', 'aba') == True", "is_anagram('Ab', 'ab') == False"]},
    {"name": "nth_largest", "sig": "def nth_largest(nums, k):",
     "desc": "Return the k-th largest value in nums (k is 1-indexed; duplicates count as distinct positions).",
     "provided": ["nth_largest([3, 1, 2], 1) == 3"],
     "held": ["nth_largest([3,1,2], 3) == 1", "nth_largest([5], 1) == 5", "nth_largest([2,2,1], 1) == 2",
              "nth_largest([2,2,1], 2) == 2", "nth_largest([9,7,8], 2) == 8"]},
    {"name": "count_words", "sig": "def count_words(s):",
     "desc": "Return the number of whitespace-separated words in s (collapsing runs of whitespace; empty string has 0 words).",
     "provided": ["count_words('hello world') == 2"],
     "held": ["count_words('') == 0", "count_words('  a  b  ') == 2", "count_words('one') == 1",
              "count_words('   ') == 0", "count_words('a b c d') == 4"]},
]

_CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def extract_code(text):
    m = _CODE_RE.search(text or "")
    return (m.group(1) if m else (text or "")).strip()


def run_tests(code, tests, timeout=6):
    """Execute `code` then evaluate each test expression in a sandboxed subprocess. Returns list of bools."""
    harness = code + "\n\n_T = " + json.dumps(tests) + "\n" + (
        "import json as _j\n_r=[]\n"
        "for _t in _T:\n"
        "    try:\n        _r.append(bool(eval(_t)))\n    except Exception:\n        _r.append(False)\n"
        "print('RESULT'+_j.dumps(_r))\n")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write("import resource; resource.setrlimit(resource.RLIMIT_AS,(2**31,2**31))\n" + harness)
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        for line in p.stdout.splitlines():
            if line.startswith("RESULT"):
                return json.loads(line[6:])
    except Exception:
        pass
    finally:
        os.unlink(path)
    return [False] * len(tests)


def frac(bools):
    return sum(bools) / len(bools) if bools else 0.0


def run(model, n_max, budgets, temperature, outdir, provided_ks=(1, 3)):
    os.makedirs(outdir, exist_ok=True)
    client = OpenRouterClient(model=model)
    if not client.available():
        print("FATAL: no OPENROUTER/OPENAI API key in env (source the .orkey)", flush=True); sys.exit(1)
    print(f"LLM verifier Goodhart: model={model}, {len(PROBLEMS)} problems, N_max={n_max}, T={temperature}, "
          f"provided-k={list(provided_ks)}", flush=True)
    # For each candidate we run the FULL ordered test pool (provided first, then held-out edge cases) ONCE.
    # cheap rung(k) = pass the first k tests; strong rung = pass all. The ablation dials k (weaker->stronger
    # cheap verifier), all from one execution and nested by construction (first-k subset of all).
    per_problem = []
    for prob in PROBLEMS:
        pool = prob["provided"] + prob["held"]
        sysmsg = ("You are a careful Python programmer. Respond with ONLY one Python function inside a "
                  "```python code block, no prose, no tests.")
        user = f"{prob['desc']}\n\nWrite the function with this exact signature:\n{prob['sig']}"
        cands = []
        for _ in range(n_max):
            r = client.chat([{"role": "user", "content": user}], system=sysmsg,
                            temperature=temperature, max_tokens=512)
            vec = run_tests(extract_code(r.text), pool)
            cands.append(vec)
        per_problem.append({"name": prob["name"], "n_tests": len(pool), "vecs": cands})
        strong_ok = sum(all(v) for v in cands)
        print(f"  {prob['name']:14} sampled {n_max} | strong-correct: {strong_ok}/{n_max}", flush=True)
        json.dump({"per_problem": per_problem}, open(os.path.join(outdir, "llm_goodhart.json"), "w"), indent=1)

    def cheap_ok(vec, k): return all(vec[:k])
    def strong_ok(vec): return all(vec)

    agg = {}
    for k in provided_ks:
        # Axis-2 survival: among candidates passing the cheap rung(k), fraction also passing strong.
        allc = [v for pp in per_problem for v in pp["vecs"]]
        cp = [v for v in allc if cheap_ok(v, k)]
        surv = 100 * sum(strong_ok(v) for v in cp) / len(cp) if cp else 0.0
        # best-of-N: cheap arm selects blind (max cheap-score = sum(vec[:k]), first on ties); strong by all.
        byN = {}
        for N in budgets:
            cproxy = crob = srob = 0
            for pp in per_problem:
                c = pp["vecs"][:N]
                cheap = max(c, key=lambda v: sum(v[:k]))
                strong = max(c, key=lambda v: (sum(v), sum(v[:k])))
                cproxy += int(cheap_ok(cheap, k)); crob += int(strong_ok(cheap)); srob += int(strong_ok(strong))
            m = len(per_problem)
            byN[N] = {"cheap_proxy": round(100*cproxy/m, 1), "cheap_robust": round(100*crob/m, 1),
                      "strong_robust": round(100*srob/m, 1)}
        agg[f"k{k}"] = {"survival": round(surv, 1), "cheap_accept": len(cp), "total": len(allc), "byN": byN}
        hi = budgets[-1]
        print(f"\n=== provided-k={k} === Axis-2 survival: {len(cp)}/{len(allc)} pass cheap rung; "
              f"{surv:.1f}% also pass strong (FP-mass {100-surv:.1f}%). "
              f"best-of-{hi}: cheap-robust={byN[hi]['cheap_robust']}% strong-robust={byN[hi]['strong_robust']}%",
              flush=True)
    print(f"\n  tokens: prompt={client.total_prompt_tokens} completion={client.total_completion_tokens}", flush=True)
    json.dump({"agg": agg, "per_problem": per_problem, "model": model, "n_max": n_max, "budgets": budgets,
               "provided_ks": list(provided_ks)}, open(os.path.join(outdir, "llm_goodhart.json"), "w"), indent=1)
    print("LLM_GOODHART_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--n-max", type=int, default=12)
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8, 12])
    ap.add_argument("--provided-ks", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--outdir", default=OUT_DEFAULT)
    a = ap.parse_args()
    run(a.model, a.n_max, [b for b in a.budgets if b <= a.n_max], a.temperature, a.outdir, a.provided_ks)
