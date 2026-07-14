# OpenScience MCP hook (E5) — the P1 spectrum-experiment bridge

Our `rna_harness/mcp/motif_server.py` is already a FastMCP stdio server, and OpenScience ships an MCP
client that spawns local stdio servers (`backend/cli/src/mcp/index.ts`, `StdioClientTransport`). So our
whole RNA capability (motif analysis, decomposition, parallel search, finisher, **and `desirna_solve`**)
plugs into an OpenScience workspace with **one config entry and zero code changes**.

This is kept as a **hook, not activated** — its purpose is Path-1 (the science): run the SAME
memory-on/off + de-scaffold ablation inside OpenScience's real-database / expensive-oracle loop, to test
whether the green-zone null breaks (self-improvement ∝ oracle headroom). Do NOT wire this for a product
(Plan A is parked).

## Config

Add to the OpenScience config (`openscience.json` / project config; the `mcp` block, schema in
`backend/cli/src/config/config.ts:516-577`):

```json
{
  "mcp": {
    "rna_motifs": {
      "type": "local",
      "command": [
        "/home/ubuntu/miniconda3/envs/rna/bin/python", "-m", "rna_harness.mcp.motif_server"
      ],
      "environment": {
        "PYTHONPATH": "/data/wenhui/openclaude_rna",
        "DESIRNA_CMD": "/home/ubuntu/miniconda3/envs/rna/bin/python /data/wenhui/DesiRNA/DesiRNA.py"
      },
      "timeout": 30000
    }
  }
}
```

## What the OpenScience agent then sees (namespaced `{client}_{tool}`)

- `rna_motifs_analyze_target` — structural facts / decomposition of a dot-bracket.
- `rna_motifs_parallel_search`, `rna_motifs_finish` — bounded classical search / finisher.
- `rna_motifs_inverse_fold_search` — native ViennaRNA inverse fold (crutched mode only).
- `rna_motifs_desirna_solve` — **call DesiRNA (SOTA), oracle-verified** (crutched mode only).

## Notes / gotchas

- Tool registration is gated by `RNA_AUTONOMOUS` in `motif_server.py`: `desirna_solve` /
  `inverse_fold_search` / advice tools are in `crutch_only` (present when the flag is OFF; hidden when
  `RNA_AUTONOMOUS=1`). For the P1 de-scaffold arm, set `"RNA_AUTONOMOUS": "1"` in `environment` to hide
  the crutch tools; leave it unset for the crutched arm. This is the exact knob the ablation needs.
- MCP connect timeout defaults to 30 s in code (`mcp/index.ts:28`); a Python cold start is well within it.
- Every MCP tool call is permission-gated (`ask` by default) on the OpenScience side; auto-approve
  `rna_motifs_*` for an autonomous run.
- Reward stays exact and ours: these tools verify with our ViennaRNA oracle; OpenScience's provenance
  DAG can additionally trace each returned metric (see S3/E3 — deferred).
