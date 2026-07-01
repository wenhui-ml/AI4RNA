/**
 * RnaConstraints — check hybrid sequence-level constraints (GC / forbidden / motif / IUPAC /
 * coding) on a sequence, independent of folding. Exact, deterministic predicates.
 */
import { z } from 'zod/v4'
import { buildTool, type ToolDef } from '../../Tool.js'
import { lazySchema } from '../../utils/lazySchema.js'
import { callRnaHarness, rnaResultBlock } from './bridge.js'

const inputSchema = lazySchema(() =>
  z.strictObject({
    sequence: z.string().describe('RNA sequence (A/C/G/U)'),
    constraints: z
      .object({
        gc_min: z.number().min(0).max(1).optional(),
        gc_max: z.number().min(0).max(1).optional(),
        forbidden: z.array(z.string()).optional(),
        motifs: z.array(z.tuple([z.number(), z.string()])).optional(),
        iupac_mask: z.string().optional(),
        coding_aa: z.string().optional(),
        coding_frame: z.number().optional(),
      })
      .describe('the hybrid constraints to check'),
  }),
)
type InputSchema = ReturnType<typeof inputSchema>

const PROMPT = `Check hybrid sequence-level constraints on an RNA sequence (GC window, forbidden substrings, fixed motifs, IUPAC mask, coding/translation). Returns per-constraint pass/fail and details. Structure is checked separately by RnaVerify/RnaFold.`

export const RnaConstraintsTool = buildTool({
  name: 'RnaConstraints',
  searchHint: 'check RNA sequence constraints GC motif coding',
  maxResultSizeChars: 100_000,
  async description() {
    return 'Check hybrid sequence-level constraints (GC / forbidden / motif / IUPAC / coding).'
  },
  async prompt() {
    return PROMPT
  },
  get inputSchema(): InputSchema {
    return inputSchema()
  },
  userFacingName() {
    return 'RnaConstraints'
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
    const res = await callRnaHarness('constraints', input as Record<string, unknown>)
    return { data: res }
  },
} satisfies ToolDef<InputSchema>)
