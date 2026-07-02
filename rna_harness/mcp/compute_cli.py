"""Clean-process entrypoint for the CPU-parallel compute tools.

The MCP server runs an asyncio loop; a fork-based process pool cannot be created inside it
(fork + live event loop deadlocks). So the MCP tools shell out to THIS module as a fresh
process, where a normal (fast, fork-based) ProcessPoolExecutor is safe. Reads a JSON command
from argv[1] (or stdin) and prints the JSON result to stdout.

    python -m rna_harness.mcp.compute_cli '{"cmd":"parallel_search","target":"((((....))))"}'
"""
from __future__ import annotations

import json
import sys

from .compute import batch_fold_impl, parallel_search_impl


def main() -> int:
    raw = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
    try:
        payload = json.loads(raw)
    except Exception as e:
        sys.stdout.write(json.dumps({"error": f"bad payload: {e}"}))
        return 1
    cmd = payload.get("cmd")
    if cmd == "batch_fold":
        out = batch_fold_impl(payload.get("sequences", []), payload.get("target", ""))
    elif cmd == "parallel_search":
        out = parallel_search_impl(payload["target"], n_restarts=int(payload.get("n_restarts", 0)),
                                   steps=int(payload.get("steps", 3000)))
    else:
        out = {"error": f"unknown cmd {cmd!r}"}
    sys.stdout.write(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
