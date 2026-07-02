"""MCP (Model Context Protocol) servers for the RNA specialization.

These expose curated, DETERMINISTIC RNA domain knowledge to the openclaude agent as real
MCP tools (`mcp__rna_motifs__*`). The design choice is deliberate: a local curated
motif knowledge base rather than a live web search, so the capability is offline, reproducible,
and consistent with the determinism thesis (the MCP supplies priors; RnaVerify remains the
ground-truth oracle -- the agent must still VERIFY any suggested motif against the folding
oracle). See `motif_kb.py` for the (testable) knowledge base and `motif_server.py` for the
stdio MCP server wired into the agent via `--mcp-config`.
"""
