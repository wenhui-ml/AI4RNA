# Target lists

These are the pools of App. D, emitted as manifests so the supplement is self-contained. Each file is in
the format `sweep.py` reads from `TARGETS_JSON`, so any sweep re-executes without an RNAInvBench checkout:

    TARGETS_JSON=rna_harness/runs/cc_ascent/targets/pseudobase_primary.json \
      python3 rna_harness/runs/cc_ascent/sweep.py mytag 0 60 0 always_cheap,autonomous

`g` is the intrinsic grade of App. D: the crossing-pair count, computed from the dot-bracket in O(|P|^2)
and fixed before design. `seq_restr` is the IUPAC restraint string. Neither `sweep.py` code path passes
the restraint to the design operators, so these manifests reproduce the runs exactly; `seq_restr` is here
only because `split_audit.py` measures edit distance over it.

| file | pool | n | grade distribution | sha256 |
|---|---|---|---|---|
| `pseudobase_primary.json` | held-out (primary) | 60 | `3:4 4:6 5:22 6:18 7:7 8:1 9:2` | `f52127c17f586c04` |
| `pseudobase_pool2.json` | second pool | 50 | `3:2 4:12 5:11 6:16 7:6 9:2 12:1` | `ea2461170ec2367b` |
| `pseudobase_all251.json` | full Pseudobase++ table | 251 | `3:13 4:48 5:75 6:71 7:28 8:5 9:7 10:2 12:2` | `3f21361af92f0216` |
| `eterna100_v2.json` | EteRNA100-V2 (nested CF baseline) | 100 | `0:100` | `e375530b2490a732` |

Full digests:

    f52127c17f586c042fd823eb1eb12e460ecbc80d11918e782377bf865406e508  pseudobase_primary.json
    ea2461170ec2367b1a0a00e611981d6d3fd3adced7defe04aefbe5ef3d07606c  pseudobase_pool2.json
    3f21361af92f021631fbe8b5a1ed4bf3b663ee6b16dbb4e90f7e7519154a6f01  pseudobase_all251.json
    e375530b2490a73248c1c7afe9387e47a9cee9aee248e0f6887ff4f1e0a8d3e6  eterna100_v2.json

## Upstream sources and licences

- **Pseudobase++** via RNAInvBench (COLA-Laboratory), MIT licence. `data/Pseudobase.csv`, 251 tasks,
  21-137 nt. MIT permits redistribution; these manifests are derived from it. **Record the commit SHA
  of the checkout used.**
- **bpRNA / ArchiveII** via RNAInvBench `inverse_rna_folding_train_dotbracket.pkl.gz`, same licence.
  Reconstructed to crossing dot-bracket by `bprna_pk.py`; multiplets excluded.
- **EteRNA100-V2**, eternagame/eterna100-benchmarking. The loader reads the `master` branch at run time,
  which is not a pin, and the commit that was actually fetched during the paper's runs was never recorded
  -- so it cannot be reconstructed now, and claiming today's `master` SHA would assert something we did not
  verify. What *is* verifiable is the snapshot: `eterna100_v2.json` above is content-addressed, and every
  number in the paper that touches the nested baseline is reproducible against that file. Anyone re-running
  from upstream should compare their own digest to it before comparing results.

## Grade distributions to check against App. D

The manifests must reproduce Table `tab:appdata`. Primary: g:n = 3:4, 4:6, 5:22, 6:18, 7:7, 8:1, 9:2
(60 targets). Cross-source: 1:4, 4:13, 5:6, 6:16, 7:8, 8:5, 9:3, 10:3, 12:1, 13:1 (60 targets). If a
distribution disagrees, the index ranges or the upstream table have moved and the manuscript's pool
description is stale -- fix that before shipping.
