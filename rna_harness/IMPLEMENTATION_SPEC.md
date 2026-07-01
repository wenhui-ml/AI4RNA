# RNA Inverse-Design / Hybrid-Constraint Harness — Implementation Spec

Target environment: **conda env `rna`, Python 3.10.20, ViennaRNA 2.7.2** (both verified live on this box; the system `base` python is 3.13 and must NOT be used for LinearFold/LinearPartition, which ship cp310-only wheels). Driven by Claude Opus 4.8 via OpenRouter; verification is CPU-heavy and runs against local folding oracles.

Convention throughout: sequences over `{A,C,G,U}` uppercase; structures are dot-bracket over `( ) .`; `n = len(seq)`; energies in kcal/mol; `RT = RNA.exp_param().kT/1000.0 ≈ 0.6163` at 37 °C.

---

## 1. Oracles: install + API

### 1.1 ViennaRNA — PRIMARY oracle (REQUIRED)

Already installed in `rna` (2.7.2). If ever reinstalling: `conda install -c bioconda viennarna` or `pip install ViennaRNA`. Free, headless, no registration.

**Canonical call pattern** — build a `fold_compound`, call `pf()` BEFORE any of `bpp()`/`ensemble_defect()`/`positional_entropy()`/`pr_structure()`:

```python
import RNA

def make_fc(seq, md=None):
    return RNA.fold_compound(seq) if md is None else RNA.fold_compound(seq, md)

fc = RNA.fold_compound(seq)
mfe_struct, mfe_energy = fc.mfe()      # (str, float kcal/mol)  -- ONE ground state (may be degenerate)
ens_struct, dG_ens     = fc.pf()       # (str, float); dG_ens = -RT*ln(Z). MUST precede bpp/defect/entropy.
ed  = fc.ensemble_defect(target)       # float in [0,1], normalized (needs pf() first)
bpp = fc.bpp()                         # tuple-of-tuples, 1-indexed upper triangle p_ij (needs pf())
```

Module shortcuts also exist and return **lists** `[str, float]` (not tuples): `RNA.fold(seq)`, `RNA.pf_fold(seq)`.

**Model details / parameter sets** (all verified live). Loaders mutate GLOBAL state, are NOT thread-safe, and each RESETS temperature to 37 °C — build the `fold_compound` AFTER the load, and serialize oracle calls (one process per param-set, or a lock):

```python
RNA.params_load_RNA_Turner2004()       # default; returns 1 on success
RNA.params_load_RNA_Turner1999()       # switch to Turner 1999
RNA.read_parameter_file(path)          # load a .par file
md = RNA.md(); md.temperature = 25.0; md.dangles = 2   # per-fold model details (live-tested to change MFE)
fc = RNA.fold_compound(seq, md)
```
> NOTE (verified): `RNA.md()` has NO `param_file` attribute in 2.7.2 — select param sets via the global `params_load_*` / `read_parameter_file` route only. T1999 vs T2004 change the tables but produce near-identical MFE for canonical sequences; do not expect large divergence.

### 1.2 LinearFold / LinearPartition — OPTIONAL 2nd/3rd oracle (cheap, headless)

Best fully-headless secondary engine. **Requires Python 3.10 (cp310 wheels; the 3.13 build fails).** Install into the `rna` env:

```bash
conda run -n rna pip install linearfold-unofficial linearpartition-unofficial
```

```python
import linearfold, linearpartition
struct, energy = linearfold.fold(seq)      # (str, float). Signature is fold(seq) only — no options.
r = linearpartition.partition(seq)         # dict:
#   r['structure']   -> dot-bracket (MEA/MFE-proxy)
#   r['free_energy'] -> ensemble free energy kcal/mol
#   r['bpp']         -> numpy STRUCTURED array, dtype [('i','<i4'),('j','<i4'),('prob','<f8')]
#                       SPARSE (i,j,prob) triples, 0-INDEXED. Densify yourself for a matrix.
```
Caveats (verified): no native ensemble-defect function — compute it yourself from `r['bpp']` vs the target, or use ViennaRNA for defect. Uppercase, T→U conversion is your responsibility.

### 1.3 EternaFold via arnie — OPTIONAL learned-model oracle

`pip install arnie` (0.2.7, pure-python, headless). arnie is only a wrapper; it shells out to a compiled binary.

**Do NOT use the bioconda EternaFold binary** — it is MPI-linked (`eternafold-1.3.1-h9948957_2`) and the EternaFold README itself documents that it "may hang on some systems." Build from source instead (plain `make` = non-MPI):

```bash
git clone https://github.com/eternagame/EternaFold && cd EternaFold/src && make   # non-MPI contrafold binary
# make multi  == MPI build; DO NOT use.
```

arnie config — set `$ARNIEFILE` to a file with `key: /path` lines (format from repo `example_arnie_file.txt`):
```
eternafold: /abs/path/EternaFold/src
vienna_2:   /abs/path/vienna/bin
linearpartition: /abs/path/LinearPartition/bin
TMP: /tmp
```
Optional override: `export ETERNAFOLD_PARAMETERS=/abs/EternaFold/parameters/EternaFoldParams.v1`.

```python
from arnie.mfe import mfe
from arnie.bpps import bpps
from arnie.pfunc import pfunc
mfe(seq, package='eternafold')                 # -> dot-bracket str
bpps(seq, package='eternafold')                # -> NxN numpy matrix
pfunc(seq, package='eternafold', return_free_energy=True)
```
> UNVERIFIED — implement defensively: EternaFold end-to-end was never run in research (bioconda hung; source build not compiled). Wrap arnie calls in try/except + timeout; treat EternaFold as an optional cross-check, never a hard gate. arnie's default `package` is `vienna_2` for mfe/pfunc/free_energy but bare `vienna` for bpps; pass `package=` explicitly.

### 1.4 NUPACK 4 — OPTIONAL, license-gated (do NOT auto-install)

Not on PyPI. Requires a one-time free **registration** at https://www.nupack.org/downloads, license agreement, manual download, then:
```bash
python3 -m pip install -U nupack -f ~/Downloads/nupack-4.0.x/package
```
Non-commercial academic license only. Python-only (no CLI). Add ONLY if the license fits and someone does the registered download.

```python
from nupack import Strand, Model, mfe, pfunc, pairs, defect
a = Strand(seq, name='a'); m = Model(material='rna', celsius=37)
mfe([a], model=m)[0].structure, mfe([a], model=m)[0].energy
Z, dG = pfunc([a], model=m)
d = defect(strands=[a], structure=target, model=m)   # normalized; == ViennaRNA ensemble_defect. Prefer keyword args.
```

### Oracle recommendation

| Oracle | Status | Why |
|---|---|---|
| ViennaRNA 2.7.2 | REQUIRED | Ground truth; matches the Eterna100 benchmark semantics. |
| LinearFold/LinearPartition | OPTIONAL, recommended | Free headless cross-check; cheap. cp310 only. |
| EternaFold (arnie, source build) | OPTIONAL | Distinct learned model; needs source build; wrap defensively. |
| NUPACK 4 | OPTIONAL | License + manual registration gate; only if licensing fits. |

Minimum viable harness = **ViennaRNA alone**. Add LinearPartition for a second independent engine when you want cross-validation.

---

## 2. Benchmark data sources

### 2.1 Eterna100 (v1 + v2) — PRIMARY benchmark (uMFE-solved)

Repo: https://github.com/eternagame/eterna100-benchmarking — **default branch is `master`** (the `main` raw URL 404s).

Working raw URLs (all HTTP 200, verified):
```bash
# One TSV holds BOTH v1 and v2 (101 lines = header + 100 puzzles, 12 tab-separated columns):
curl -sL https://raw.githubusercontent.com/eternagame/eterna100-benchmarking/master/data/eterna100_puzzles.tsv -o eterna100_puzzles.tsv
# Reference eval script + fold util (for semantics only; see note on binaries):
#   https://raw.githubusercontent.com/eternagame/eterna100-benchmarking/master/scripts/benchmark.py
#   https://raw.githubusercontent.com/eternagame/eterna100-benchmarking/master/scripts/util/fold.py
```

**File format** — 12 tab-separated columns, header verbatim:
```
Puzzle #	Puzzle Name	Secondary Structure V1	Secondary Structure V2	Eterna ID (V1/Vienna1)	Eterna ID (V1/Vienna2)	Eterna ID (V2/Vienna1)	Eterna ID (V2/Vienna2)	Sample Solution (V1/Vienna1)	Sample Solution (V1/Vienna2)	Sample Solution (V2/Vienna1)	Sample Solution (V2/Vienna2)
```
0-based columns: `[0]` Puzzle # (1–100), `[1]` Name, `[2]` Structure V1, `[3]` Structure V2, `[4–7]` four Eterna IDs, `[8–11]` four sample solutions.

Verified facts to bake in:
- Structure columns contain ONLY `( ) .` — pure dot-bracket, no pseudoknots, no extended alphabet.
- Length range **12–400 nt** (min 12 is puzzle #8 "G-C Placement", NOT #1; #1 "Simple Hairpin" is 16 nt). Largest 400 nt: #90 Gladius, #97 Hoglafractal.
- **Exactly 19** puzzles have `V2 structure != V1 structure`: `50, 52, 57, 60, 61, 67, 78, 80, 81, 86, 87, 88, 90, 91, 92, 96, 97, 99, 100`. The other 81 are identical.
- Placeholder tokens: cells may be `(Unsolved)` or `(Undisclosed)`. #100 "Teslagon" col-8 is `(Undisclosed)` — the only row whose col-8 sample solution is not pure ACGU.
- **NO locked/constrained-base column.** Eterna100 targets are unconstrained pure inverse folding — treat all positions as free.

**Parser:**
```python
import csv
def load_eterna100(path, version='V2'):   # version in {'V1','V2'}
    col = 2 if version == 'V1' else 3
    out = []
    with open(path, newline='') as f:
        r = csv.reader(f, delimiter='\t'); next(r)  # header
        for row in r:
            struct = row[col].strip()
            if set(struct) <= set('().'):          # skip placeholder cells defensively
                out.append({'id': int(row[0]), 'name': row[1], 'target': struct})
    return out
```

**uMFE-solved success rule** (read from `benchmark.py`, verified verbatim): a puzzle is solved iff the designed sequence's MFE structure equals the target dot-bracket by **case-sensitive character-by-character string equality** under the matching Vienna params. Reference code: `folded = fold(sequence, '1.8.5' if folder=='vienna1' else '2.6.4')[0]; Success = '1' if structure == folded else '0'`. Pair the target column to the engine: V1↔Vienna 1.8.5, V2↔Vienna 2.6.4. Benchmark score = count of the 100 solved.

> IMPLEMENTATION NOTE (verified caveat): the repo's `benchmark.py` shells out to locally-**built** RNAfold binaries at a hardcoded path `external/ViennaRNA-{version}/build/bin/RNAfold`. A plain `pip/conda` ViennaRNA does NOT satisfy that path. For this harness, do NOT reproduce their subprocess path — implement the equality check directly against the installed `RNA` module (2.7.2 uses Turner-2004-era params, i.e. the "vienna2 / V2" convention). If you need exact bit-for-bit reproduction against 1.8.5/2.6.4, build those from source separately. For our purposes: evaluate the **V2** structures under installed ViennaRNA 2.7.2.

### 2.2 RNAInvBench — SECONDARY (dataset provenance + wrappers), clone-and-run only

Repo: https://github.com/COLA-Laboratory/RNAInvBench (branch `main`, MIT). NOT pip-installable, no top-level `setup.py`. Heavy multi-env (Docker or per-folder conda). We use it as a **data + algorithm reference**, not a runtime dependency.

- Secondary data format: CSV/txt with header `Id,str`, e.g. `data/Eterna100v2.txt` rows like `1,(((((......)))))`.
- Also gzipped pandas pickles: `data/inverse_rna_folding_{train,valid,benchmark}.plk.gz`, `..._dotbracket.pkl.gz`, and `constrained_design_{train,valid,benchmark}.plk.gz` (constrained design = IUPAC/`N`-mask + GC target within the secondary task, not a separate task).
- Per-run output convention: one pandas pickle per structure `<algo>_<dataset>_<index>.pkl` with columns `target_structure, seq_constraints, sequence, time, distance`. Worth mirroring for compatibility with their `evaluations/` scripts (`successRate.py`, `structureDistance.py`, `MFEgcContent.py`, `meanEntropy.py`), which group by `filename.split('_')[0]`.
- Reference wrapper signature: `call_rnainverse(target_structure, constraint=None, repeat=1, save_file=...)` — default `constraint = 'N'*len(target_structure)`.

> UNVERIFIED — implement defensively: RNAInvBench's environments were never built end-to-end; large data is Google-Drive-hosted via `data/download.py`. Do not depend on running their code. The README's `docker exec` example is malformed. Use their TSV/CSV data and their output-pickle schema as a compatibility target only.

---

## 3. Metrics & constraints — exact ViennaRNA calls

All verified live on 2.7.2 unless flagged. Always `fc.pf()` before defect/bpp/entropy/pr_structure.

### 3.1 uMFE-solved (unique MFE match)

`fc.mfe()` returns only ONE of possibly several ground states, so `mfe_struct == target` is necessary but **not sufficient** for uniqueness. Enumerate exact ground states with `subopt`. **`subopt`'s delta is in DEKA-CAL (10·cal/mol)** — `subopt(0)` = exactly the MFE, `subopt(100)` = within 1.0 kcal/mol. Getting units wrong silently breaks the check.

```python
def is_umfe(seq, target):
    fc = RNA.fold_compound(seq)
    fc.mfe()
    structs = {sol.structure for sol in fc.subopt(0)}   # sol.structure (str), sol.energy (float)
    return (target in structs) and (len(structs) == 1)

def is_mfe_solved(seq, target):        # weaker, matches Eterna100 benchmark equality
    return RNA.fold(seq)[0] == target
```
Use `is_mfe_solved` to reproduce the Eterna100 score; use `is_umfe` when you want the stricter uniqueness guarantee.

### 3.2 Structural distances

```python
d_bp = RNA.bp_distance(a, b)          # symmetric diff of pair sets; options default RNA.BRACKETS_RND. 0 iff pair-identical.
d_h  = RNA.hamming_distance(a, b)     # dot-bracket positionwise (equal length). Over-counts vs d_bp.
```

### 3.3 Ensemble defect (normalized, [0,1])

Formula (matches docstring, hand-verified): `ED(s) = 1 - (1/n)·Σ_{(i,j)∈s} p_ij - (1/n)·Σ_i (1-s_i)·q_i`, `q_i = 1 - Σ_j p_ij`.
```python
fc = RNA.fold_compound(seq); fc.pf()
ed = fc.ensemble_defect(target)       # float; VERIFIED 0.0911 on GGGGAAAACCCC vs ((((....))))
```
> CORRECTION: the pair-table variant is the MODULE function `RNA.ensemble_defect_pt(...)`, NOT `fc.ensemble_defect_pt` (that attribute does not exist on the fold_compound).

### 3.4 Boltzmann / ensemble probability of target

```python
fc = RNA.fold_compound(seq); _, dG = fc.pf()
p_target = fc.pr_structure(target)    # p(s)=exp(-E(s)/RT)/Z. VERIFIED 0.84005 on the toy.
E_s      = fc.eval_structure(target)  # E(s) kcal/mol float. (eval_structure_pt returns int deka-cal.)
```

### 3.5 Positional entropy & diffuseness (designability triage)

```python
S = fc.positional_entropy()           # DoubleVector length n+1; INDEX 0 IS A DUMMY sentinel — use S[1:] / 1..n.
mbd = fc.mean_bp_distance()           # <d>=Σ p_ij(1-p_ij); large => diffuse ensemble => harder target.
```

### 3.6 Sequence-level constraints (checkable predicates)

```python
def gc_ok(x, lo, hi):
    g = sum(c in 'GC' for c in x.upper()); return lo <= g/len(x) <= hi

def no_forbidden(x, forbidden):        # forbidden: iterable of substrings
    xu = x.upper(); return all(m not in xu for m in forbidden)

def fixed_motifs_ok(x, motifs):        # motifs: list of (pos, literal)
    return all(x[p:p+len(m)] == m for p, m in motifs)

def codes_for(x, aa, frame=0, table=CODON):   # CODON: dict RNA-codon->aa char (harness-supplied)
    cds = x[frame:frame+3*len(aa)]
    if len(cds) < 3*len(aa): return False
    return all(table[cds[3*k:3*k+3]] == aa[k] for k in range(len(aa)))
```
> The organism codon table and any forbidden-motif list are external inputs the harness must supply.

### 3.7 Negative design (competitor discrimination)

```python
gap = fc.eval_structure(competitor) - fc.eval_structure(target)   # >0 = target favored; larger = better
p_competitor = fc.pr_structure(competitor)                        # minimize this
# next-best distinct structure: scan fc.subopt(delta) for lowest-energy structure != target, maximize its E - E(target)
```
Composite design loss (recommended): `loss = ensemble_defect(target) + λ·max(0, E(target) - E(competitor))`, or `loss = -log pr_structure(target)`.

> UNVERIFIED — implement defensively: `fc.pbacktrack(n)` (Boltzmann sampling of competitors) returned EMPTY on this 2.7.2 build in research — likely compiled without stochastic backtracking. Do NOT rely on `pbacktrack`. Discover competitors via `fc.subopt(delta)` instead.

### 3.8 Designability triage (heuristic, NOT proof)

Flag a target as likely-hard/undesignable if, after a short local search, the best candidate still has `ensemble_defect > ~0.1–0.2`, low `pr_structure(target)`, or persistently high positional entropy at target-paired positions. Structural red flags: isolated (lonely) base pairs, very short stems flanked by strong alternative stems, small symmetric loops. Only the rival-motif method (arXiv:2402.17206, RECOMB 2024) *proves* undesignability — treat these as triage only.

---

## 4. Classical baseline generator

Implement **two generators and ensemble** (return first solved / best across both):

### 4.1 Generator A — RNAinverse wrapper (trivial, strong on easy targets)

```python
def rnainverse_design(target, tries=20, gc_only=False):
    RNA.cvar.symbolset = 'GC' if gc_only else 'AUGC'   # global GC lever
    best_seq, best_cost = None, float('inf')
    for _ in range(tries):
        start = RNA.random_string(len(target), RNA.cvar.symbolset)
        seq, cost = RNA.inverse_fold(start, target)     # returns [sequence, cost]; cost==0.0 iff MFE(seq)==target
        if cost < best_cost:
            best_seq, best_cost = seq, cost
        if cost == 0.0:
            break
    return best_seq, (best_cost == 0.0)
```
Related globals (verified present): `RNA.cvar.final_cost` (0.0, stop threshold for `inverse_pf_fold`), `RNA.cvar.give_up` (0 → set 1 to bail early on provably-impossible), `RNA.cvar.inv_verbose`. `RNA.inverse_pf_fold(start, target)` maximizes the target's ensemble probability (slower, better-defined designs).

### 4.2 Generator B — SAMFEO-style structure-aware local search (best non-learned)

This is the algorithmic core of SAMFEO (Zhou et al., Bioinformatics 2023; repo https://github.com/shanry/SAMFEO, deps `numpy>=1.23, pandas>=1.4, ViennaRNA>=2.6`, CPU-only). Implement it directly (~300–500 lines) for full control, or run their `main.py --t 1 --k 10 --object pd --path data/eterna/eterna100.txt --repeat 5`.

**Objective:** normalized ensemble defect (via `fc.ensemble_defect(target)`) as the smooth cost; `RNA.bp_distance(RNA.fold(seq)[0], target)` (==0 iff solved) as the hard solved-check.

**Initialization (constraint-guided):** parse `pt = RNA.ptable(target)` (1-indexed; `pt[i]` = partner or 0). Unpaired positions → `A`; each target pair `(i,j)` → randomly `(C,G)` or `(G,C)`. Generate `k` seeds.

**Structured mutation (the key idea — mutate correlated positions together):**
- Position `i` is a target pair `(i,j)`: jointly resample the *pair* from the 6 valid pairs `{CG, GC, AU, UA, GU, UG}` (weight GC/CG by `p_gc`). NEVER mutate one half of a target pair alone.
- Position `i` unpaired in target: resample `i` from `{A,U,C,G}` weighted by `p_gc`.
- ~20% of the time, co-mutate a stacked neighbor pair (`i±1` with its partner) together with `i`, to fix helices needing two-pair changes.

**Main multi-frontier loop (top-k frontier, defect-weighted sampling):**
```
frontier = k constraint-guided seeds, scored by objective
for step in range(max_steps):
    x  = sample from frontier with P(x_m) ∝ exp((1 - f(x_m)) / T)          # T=1 default
    per-position defect ε_i from fc.bpp() vs target
    i  = sample position with P(i) ∝ exp(ε_i / T'), T' = 2T                  # high-defect positions mutated more
    x' = structured_mutation(x, i, target, p_gc)
    if bp_distance(fold(x')[0], target) == 0: record solution
    update frontier to keep top-k by objective
```
GC control: track running GC; set per-mutation `p_gc' = clip(p_gc + κ·(gc_target - gc_current), 0, 1)`.

**Lighter fallback (stand up in a day, then upgrade to B):** single-best adaptive walk over mismatched positions (`pt_cur[i] != pt_target[i]`) with the same structured mutation, plus escape mechanisms — sideways moves (accept equal-distance with p≈0.5), simulated-annealing acceptance `exp(-(d_new-d_old)/T)` annealing T from ~1.0, random restarts after R stall steps, and K parallel multi-starts.

> SKIP as primary effort: antaRNA (Python 2, unmaintained — borrow only its GC-control idea; cite PMC4576691 for the ACO algorithm, PMC4652366 is the separate pseudoknot paper) and LEARNA/Meta-LEARNA (heavy DL env, marginal over SAMFEO).

---

## 5. Recommended module layout

```
rna_harness/
├── oracles/
│   ├── __init__.py
│   ├── vienna.py         # fold_compound wrappers: mfe, pf, ensemble_defect, bpp, pr_structure,
│   │                     #   is_umfe, is_mfe_solved, bp_distance; param-set context manager (global-state lock)
│   ├── linear.py         # OPTIONAL: linearfold/linearpartition wrappers; densify sparse bpp
│   ├── eternafold.py     # OPTIONAL: arnie wrappers, try/except + timeout, feature-flagged
│   └── nupack.py         # OPTIONAL: license-gated; import guarded, disabled by default
├── metrics.py            # gc_ok, no_forbidden, fixed_motifs_ok, codes_for, negative_design gap,
│                         #   positional-entropy triage, designability heuristics
├── constraints.py        # constraint spec dataclass (gc window, forbidden set, fixed motifs,
│                         #   coding frame+aa, competitor structs) + validate(seq, spec) -> report
├── generators/
│   ├── __init__.py
│   ├── rnainverse.py     # Generator A (§4.1)
│   ├── samfeo.py         # Generator B: structured mutation + multi-frontier search (§4.2)
│   └── local_search.py   # lighter adaptive-walk fallback (SA + restarts + multi-start)
├── data/
│   ├── eterna100.py      # download + parse eterna100_puzzles.tsv (§2.1), V1/V2 selectable
│   └── eterna100_puzzles.tsv
├── benchmark.py          # run generators over Eterna100, score uMFE-solved, emit per-puzzle results
├── results_io.py         # write <algo>_<dataset>_<index>.pkl (RNAInvBench-compatible schema)
├── llm/
│   └── openrouter.py     # Claude Opus 4.8 client (proposal/repair loop over designs)
└── config.py             # oracle feature flags, param set (T2004 default), temperature, GC targets, seeds
```

Key cross-cutting rules:
- **Thread-safety:** ViennaRNA `params_load_*` and `RNA.cvar` are global. Confine parameter switching to a single serialized code path (process-per-param-set for CPU parallelism, or a lock). Multi-start / batch parallelism should parallelize over *sequences within one fixed param set*, not switch params concurrently.
- **Every generated candidate is verified by the ViennaRNA oracle** (`is_mfe_solved` / `is_umfe`) — never trust a generator's self-reported success.
- Oracle wrappers behind feature flags in `config.py`; ViennaRNA hard-required, others degrade gracefully if unavailable.
- Fix and record `{param_set, temperature, dangles}` in every result for reproducibility — uMFE/designability are all defined relative to these.

---

## Appendix — verified vs flagged

CONFIRMED live/by-source: all §1.1 ViennaRNA calls; LinearFold/LinearPartition install + API (cp310); arnie install + config mechanism; Eterna100 TSV format, URLs, 19-diff list, success rule; all §3 metric calls except those below.

FLAGGED (implement defensively): EternaFold end-to-end (never run — MPI hang / source build uncompiled); NUPACK (registration/license gate, not installed); `fc.pbacktrack` (returned empty on this 2.7.2 build — use `subopt` instead); RNAInvBench end-to-end environments (never built; use only its data + output schema); exact SAMFEO structured-mutation combinatorics (from paper, not line-by-line source — cross-check Algorithm 1 / eqs 8–13 if reimplementing precisely).
