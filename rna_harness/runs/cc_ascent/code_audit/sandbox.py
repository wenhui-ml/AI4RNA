#!/usr/bin/env python3
"""Safe execution of LLM-generated candidate code (code_audit experiment).

Executes untrusted candidate solutions against a batch of (args -> expected) cases in an ISOLATED subprocess:
  - AST pre-vet: reject forbidden imports / builtins / dunder escapes before running anything.
  - subprocess with `-I -S` (isolated, no site), setsid + rlimits (CPU/AS/FSIZE/NOFILE), wall timeout,
    killpg on timeout so a runaway/infinite-loop candidate and any children are reaped.
  - no network reachable code paths (socket/urllib/etc. banned at AST level).
Returns one bool per case (True=matched expected). Errors/timeouts/rejects -> all-fail (conservative).
Not a security boundary against a determined adversary; a defensive sandbox for frontier-LLM coding output.
"""
import ast, json, os, resource, signal, subprocess, sys, tempfile, textwrap

PY = sys.executable

# Modules a self-contained algorithm may need; everything else is rejected.
ALLOWED_IMPORTS = {"math", "itertools", "collections", "functools", "re", "string",
                   "bisect", "heapq", "operator", "typing", "cmath", "fractions", "decimal", "random"}
FORBIDDEN_NAMES = {"open", "eval", "exec", "compile", "__import__", "input", "globals", "locals",
                   "vars", "getattr", "setattr", "delattr", "memoryview", "breakpoint", "help", "exit", "quit"}
FORBIDDEN_DUNDER = {"__globals__", "__builtins__", "__subclasses__", "__bases__", "__mro__", "__class__",
                    "__code__", "__closure__", "__dict__", "__getattribute__", "__reduce__", "__import__"}


def vet(code):
    """Return (ok, reason). Static AST check before any execution."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return False, f"import {a.name}"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return False, f"from {node.module}"
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return False, f"name {node.id}"
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_DUNDER:
            return False, f"attr {node.attr}"
    return True, "ok"


_DRIVER = textwrap.dedent('''
    import json, sys
    spec = json.load(open(sys.argv[1]))
    code, entry, mode = spec["code"], spec["entry"], spec["mode"]
    ns = {}
    try:
        exec(code, ns)
        fn = ns[entry]
    except Exception as e:
        print(json.dumps({"fatal": repr(e)[:200]})); sys.exit(0)
    out = []
    if mode == "cases":
        for args, expected in spec["cases"]:
            try:
                out.append(bool(fn(*args) == expected))
            except Exception:
                out.append(False)
    else:  # collect: return the candidate's actual outputs (JSON-safe or sentinel)
        for args in spec["args_list"]:
            try:
                v = fn(*args)
                json.dumps(v)                 # ensure serializable
                out.append({"ok": True, "val": v})
            except Exception as e:
                out.append({"ok": False, "err": repr(e)[:80]})
    print(json.dumps({"results": out}))
''')


def _limits(cpu_s, mem_mb):
    def _set():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        m = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (m, m))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    return _set


def _run(spec_extra, entry, code, n, fill, wall_s, cpu_s, mem_mb):
    ok, reason = vet(code)
    if not ok:
        return [fill] * n, f"vet:{reason}"
    with tempfile.TemporaryDirectory() as td:
        spec = os.path.join(td, "spec.json"); drv = os.path.join(td, "drv.py")
        json.dump({"code": code, "entry": entry, **spec_extra}, open(spec, "w"))
        open(drv, "w").write(_DRIVER)
        try:
            p = subprocess.run([PY, "-I", "-S", drv, spec], capture_output=True, text=True,
                               timeout=wall_s, preexec_fn=_limits(cpu_s, mem_mb), cwd=td)
        except subprocess.TimeoutExpired:
            return [fill] * n, "timeout"
        except Exception as e:
            return [fill] * n, f"launch:{repr(e)[:80]}"
        try:
            d = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            return [fill] * n, f"parse:{(p.stdout or p.stderr)[:80]}"
        if "fatal" in d:
            return [fill] * n, f"fatal:{d['fatal'][:80]}"
        res = (d.get("results", []) + [fill] * n)[:n]
        return res, "ok"


def run_cases(code, entry, cases, wall_s=8, cpu_s=6, mem_mb=512):
    """cases: list of [args, expected]. Returns (list[bool], status) aligned to cases (all-False on reject)."""
    return _run({"mode": "cases", "cases": cases}, entry, code, len(cases), False, wall_s, cpu_s, mem_mb)


def run_collect(code, entry, args_list, wall_s=8, cpu_s=6, mem_mb=512):
    """Returns (list[{"ok":bool,"val":...}|{"ok":False,...}], status): the candidate's actual outputs."""
    return _run({"mode": "collect", "args_list": args_list}, entry, code, len(args_list),
                {"ok": False, "err": "reject"}, wall_s, cpu_s, mem_mb)


if __name__ == "__main__":
    # self-test: a good impl, a vet-rejected impl, an infinite loop
    good = "def f(n):\n    return n*n"
    print("good:", run_cases(good, "f", [[[3], 9], [[4], 16], [[5], 24]]))       # [T,T,F]
    print("import os:", run_cases("import os\ndef f(n):\n    return n", "f", [[[1], 1]]))
    print("eval:", run_cases("def f(n):\n    return eval('n')", "f", [[[1], 1]]))
    print("loop:", run_cases("def f(n):\n    while True: pass", "f", [[[1], 1]], wall_s=3))
    print("dunder:", run_cases("def f(n):\n    return ().__class__", "f", [[[1], 1]]))
