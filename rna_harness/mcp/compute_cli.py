"""Clean-process entrypoint for the CPU-parallel compute tools.

The MCP server runs an asyncio loop; a fork-based process pool cannot be created inside it
(fork + live event loop deadlocks). So the MCP tools shell out to THIS module as a fresh
process, where a normal (fast, fork-based) ProcessPoolExecutor is safe. Reads a JSON command
from argv[1] (or stdin) and prints the JSON result to stdout.

    python -m rna_harness.mcp.compute_cli '{"cmd":"parallel_search","target":"((((....))))"}'
"""
from __future__ import annotations

import json
import os
import signal
import sys

from .compute import batch_fold_impl, parallel_search_impl
from .finisher import parallel_finish, parallel_inverse_fold

# Hard self-limit: if this process (or its pool workers) ever hangs or is orphaned by a parent
# kill, SIGALRM fires and we SIGKILL our OWN process group -- reaping every pool worker. Prevents
# the orphaned-worker leak even when the parent times us out or dies mid-search. 240s >> the 180s
# parallel_search timeout, so a normal call finishes long before this triggers.
def _watchdog(_signum, _frame):
    try:
        os.killpg(os.getpgid(0), signal.SIGKILL)
    except Exception:
        os._exit(1)


try:
    signal.signal(signal.SIGALRM, _watchdog)
    signal.alarm(240)
except Exception:
    pass


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
    elif cmd == "inverse_fold_search":
        out = parallel_inverse_fold(payload["target"], warm=payload.get("warm") or None,
                                    n_restarts=int(payload.get("n_restarts", 0)),
                                    tries=int(payload.get("tries", 40)))
    elif cmd == "finish":
        out = parallel_finish(payload["candidate"], payload["target"],
                              n_restarts=int(payload.get("n_restarts", 0)),
                              iters=int(payload.get("iters", 20000)))
    else:
        out = {"error": f"unknown cmd {cmd!r}"}
    sys.stdout.write(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
