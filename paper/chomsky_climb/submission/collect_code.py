#!/usr/bin/env python3
"""Resolve the transitive closure of first-party imports reachable from the supplement's entry points.

Why this exists rather than a hand-written file list in build.sh. The previous supplement shipped
runs/cc_ascent plus five source files chosen by reading the paper, and it did not run: the analysis scripts
import rna_harness.benchmarks.rnainvbench, which imports ..constraints, which was never listed, and the
package had no __init__.py at all so nothing was importable in the first place. A hand list encodes what we
believed the dependencies were on the day we wrote it; this walks the import graph, so a new import in any
shipped file pulls its target in automatically.

Emits one repo-relative path per line on stdout. Entry points are every .py under runs/cc_ascent (the
analysis scripts a reviewer re-executes), the operator sources the paper names, and the figure builder.
"""
import ast
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
PKG = "rna_harness"
PKG_DIR = os.path.join(ROOT, PKG)


def module_to_path(mod):
    """rna_harness.a.b -> the file that defines it, or None if it is not first-party."""
    rel = mod.replace(".", os.sep)
    for cand in (os.path.join(ROOT, rel + ".py"), os.path.join(ROOT, rel, "__init__.py")):
        if os.path.isfile(cand):
            return cand
    return None


def imports_of(path):
    """Every module name this file imports, with relative imports resolved against its own package."""
    try:
        tree = ast.parse(open(path, encoding="utf8", errors="ignore").read(), filename=path)
    except SyntaxError:
        return set()
    # package this file lives in, as a dotted name
    rel = os.path.relpath(path, ROOT)
    parts = rel.split(os.sep)
    pkg_parts = parts[:-1] if parts[-1] != "__init__.py" else parts[:-1]
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # `from ..x import y` inside a/b/c.py: drop (level-1) trailing package components
                base = pkg_parts[: len(pkg_parts) - (node.level - 1)] if node.level > 1 else pkg_parts
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            if not mod:
                continue
            out.add(mod)
            # `from pkg.mod import name` where name is itself a submodule
            for a in node.names:
                out.add(mod + "." + a.name)
    return out


def main():
    entries = []
    runs = os.path.join(PKG_DIR, "runs", "cc_ascent")
    for dp, dn, fn in os.walk(runs):
        dn[:] = [d for d in dn if d not in ("__pycache__", "_archive_runs")]
        entries += [os.path.join(dp, f) for f in fn if f.endswith(".py")]
    for extra in ("cognitive/skills/pk_decompose_panel.py", "cognitive/skills/pk_decompose.py",
                  "experiments/track_b_pk.py", "oracles_pk.py", "benchmarks/rnainvbench.py"):
        p = os.path.join(PKG_DIR, extra)
        if os.path.isfile(p):
            entries.append(p)
    figs = os.path.join(ROOT, "paper", "chomsky_climb", "make_figs.py")
    if os.path.isfile(figs):
        entries.append(figs)

    def package_inits(path):
        """Every __init__.py on the path from the package root down to this file.

        These belong INSIDE the fixpoint, not appended after it: an __init__.py re-exports, and
        benchmarks/__init__.py does `from .hybrid import synthesize_hybrid_tasks`. Collecting them last
        ships the file but never follows its imports, so importing the package still fails.
        """
        out, d = [], os.path.dirname(path)
        while d.startswith(PKG_DIR):
            ini = os.path.join(d, "__init__.py")
            if os.path.isfile(ini):
                out.append(ini)
            d = os.path.dirname(d)
        return out

    seen, stack = set(), list(entries)
    while stack:
        path = stack.pop()
        if path in seen:
            continue
        seen.add(path)
        stack += [p for p in package_inits(path) if p not in seen]
        for mod in imports_of(path):
            if not (mod == PKG or mod.startswith(PKG + ".")):
                continue
            tgt = module_to_path(mod)
            if tgt and tgt not in seen:
                stack.append(tgt)

    for p in sorted(os.path.relpath(x, ROOT) for x in seen):
        print(p)
    print("  %d files (%d entry points, %d pulled in transitively)"
          % (len(seen), len(entries), len(seen) - len(entries)), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
