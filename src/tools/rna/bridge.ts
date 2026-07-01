/**
 * Bridge from the openclaude TS agent to the Python `rna_harness` science layer.
 *
 * The TS tools are deliberately THIN: all deterministic RNA ground truth (ViennaRNA
 * folding, the canonical reward, constraints, diagnosis, benchmarks) lives in Python and is
 * invoked here over a stateless JSON-on-stdin / JSON-on-stdout contract
 * (`python -m rna_harness.cli <command>`). This keeps the agentic intelligence in openclaude
 * and the exact verifier in Python, and never reimplements a weak parallel agent in TS.
 *
 * Config via env (with deployment defaults):
 *   RNA_HARNESS_PYTHON  path to the `rna` conda env python (has ViennaRNA)
 *   RNA_HARNESS_DIR     directory containing the `rna_harness` package (repo root)
 */
import { execa } from 'execa'

const DEFAULT_PYTHON = '/home/ubuntu/miniconda3/envs/rna/bin/python'
const DEFAULT_DIR = '/data/wenhui/openclaude_rna'

export function rnaPython(): string {
  return process.env.RNA_HARNESS_PYTHON || DEFAULT_PYTHON
}

export function rnaDir(): string {
  return process.env.RNA_HARNESS_DIR || DEFAULT_DIR
}

export function isRnaEnabled(): boolean {
  // Feature-flagged so the RNA specialization never disturbs general openclaude use.
  return process.env.OPENCLAUDE_RNA === '1' || process.env.OPENCLAUDE_RNA === 'true'
}

export type RnaResponse = { ok: boolean; error?: string; [k: string]: unknown }

export async function callRnaHarness(
  command: string,
  request: Record<string, unknown>,
  opts?: { timeoutMs?: number },
): Promise<RnaResponse> {
  const result = await execa(rnaPython(), ['-m', 'rna_harness.cli', command], {
    cwd: rnaDir(),
    input: JSON.stringify(request ?? {}),
    timeout: opts?.timeoutMs ?? 180_000,
    reject: false,
    env: { ...process.env },
  })
  const stdout = typeof result.stdout === 'string' ? result.stdout : ''
  if (!stdout) {
    const err = typeof result.stderr === 'string' ? result.stderr : String(result.exitCode)
    throw new Error(`rna_harness "${command}" produced no output. stderr: ${err.slice(0, 500)}`)
  }
  let parsed: RnaResponse
  try {
    parsed = JSON.parse(stdout)
  } catch {
    throw new Error(`rna_harness "${command}" returned non-JSON: ${stdout.slice(0, 400)}`)
  }
  if (!parsed.ok) {
    throw new Error(`rna_harness "${command}" error: ${parsed.error ?? 'unknown'}`)
  }
  return parsed
}

/** Shared tool-result block: surface the full structured JSON to the model. */
export function rnaResultBlock(content: unknown, toolUseID: string) {
  return {
    tool_use_id: toolUseID,
    type: 'tool_result' as const,
    content: typeof content === 'string' ? content : JSON.stringify(content, null, 2),
  }
}
