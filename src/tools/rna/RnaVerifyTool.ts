/**
 * RnaVerify — the canonical, harness-controlled verifier for an RNA design.
 *
 * This is the official scorer: given a target structure (and optional hybrid constraints),
 * it folds the submitted sequence with ViennaRNA and returns the deterministic verdict +
 * reward + an exact folding-mismatch diagnosis. The agent may fold sequences freely to
 * explore (RnaFold / its own code), but "solved" and "reward" are decided HERE so the agent
 * cannot grade its own homework. Deterministic: same input -> same result (R_det).
 */
import { z } from 'zod/v4'
import { buildTool, type ToolDef } from '../../Tool.js'
import { lazySchema } from '../../utils/lazySchema.js'
import { callRnaHarness, rnaResultBlock } from './bridge.js'

const constraintsSchema = () =>
  z
    .object({
      gc_min: z.number().min(0).max(1).optional().describe('minimum GC fraction'),
      gc_max: z.number().min(0).max(1).optional().describe('maximum GC fraction'),
      forbidden: z.array(z.string()).optional().describe('substrings that must NOT appear'),
      motifs: z
        .array(z.tuple([z.number(), z.string()]))
        .optional()
        .describe('fixed motifs as [position, literal] pairs (0-indexed, IUPAC ok)'),
      iupac_mask: z.string().optional().describe('per-position IUPAC allowed-set mask, length n'),
      coding_aa: z.string().optional().describe('required amino-acid translation'),
      coding_frame: z.number().optional().describe('reading frame for coding_aa (default 0)'),
      competitors: z
        .array(z.string())
        .optional()
        .describe('dot-bracket competitor structures to avoid (negative design)'),
    })
    .optional()

const oracleSchema = () =>
  z
    .object({
      param_set: z.enum(['turner2004', 'turner1999', 'andronescu2007', 'langdon2018']).optional(),
      temperature: z.number().optional(),
      dangles: z.number().optional(),
    })
    .optional()

const inputSchema = lazySchema(() =>
  z.strictObject({
    sequence: z.string().describe('the candidate RNA sequence (A/C/G/U)'),
    target: z.string().describe('target secondary structure in dot-bracket notation'),
    constraints: constraintsSchema().describe('optional hybrid constraints'),
    oracle: oracleSchema().describe('optional oracle config (defaults: turner2004, 37C, dangles=2)'),
    cross_oracle: z
      .boolean()
      .optional()
      .describe('also fold under the multi-oracle panel for a robustness score (default true)'),
  }),
)
type InputSchema = ReturnType<typeof inputSchema>

const PROMPT = `Verify an RNA design against a target secondary structure with the canonical ViennaRNA oracle.

This is the OFFICIAL scorer. Use it to check whether a candidate sequence solves a design task and to get the exact reward and a precise folding-mismatch diagnosis to guide repair.

## Returns
- **solved**: true iff the MFE structure equals the target (and all constraints pass)
- **mfe_match / is_umfe**: MFE string-match / unique-MFE flags
- **reward** + **reward_breakdown**: the deterministic shaped reward
- **ensemble**: MFE structure/energy, ensemble defect, p(target), bp-distance
- **constraints**: per-constraint pass/fail (GC, forbidden, motif, coding, ...)
- **cross_oracle**: fraction of the independent-energy-model panel that also solves it (robustness)
- **diagnosis**: exact missing/spurious base pairs and which positions to fix (present when unsolved)

## Notes
- The reward is deterministic and computed by the harness, not by you — you cannot game it.
- Only claim "design success under a computational folding oracle" — never in-vivo function.`

export const RnaVerifyTool = buildTool({
  name: 'RnaVerify',
  searchHint: 'verify RNA design fold reward diagnosis',
  maxResultSizeChars: 100_000,
  async description() {
    return 'Verify an RNA sequence against a target structure (canonical ViennaRNA reward + mismatch diagnosis).'
  },
  async prompt() {
    return PROMPT
  },
  get inputSchema(): InputSchema {
    return inputSchema()
  },
  userFacingName() {
    return 'RnaVerify'
  },
  isConcurrencySafe() {
    return true
  },
  isReadOnly() {
    return true
  },
  renderToolUseMessage() {
    return null
  },
  mapToolResultToToolResultBlockParam(content, toolUseID) {
    return rnaResultBlock(content, toolUseID)
  },
  async call(input: z.infer<InputSchema>) {
    const res = await callRnaHarness('verify', input as Record<string, unknown>)
    return { data: res }
  },
} satisfies ToolDef<InputSchema>)
