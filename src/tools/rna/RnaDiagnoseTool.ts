/**
 * RnaDiagnose — localize exactly how a sequence's fold differs from the target.
 *
 * This is the T_trace (trace-determinism) signal: because the oracle is exact, a failure
 * localizes precisely to specific base pairs / positions. The returned directive is the
 * repair instruction whose adoption the harness measures as actuation fidelity (A).
 */
import { z } from 'zod/v4'
import { buildTool, type ToolDef } from '../../Tool.js'
import { lazySchema } from '../../utils/lazySchema.js'
import { callRnaHarness, rnaResultBlock } from './bridge.js'

const inputSchema = lazySchema(() =>
  z.strictObject({
    target: z.string().describe('target structure (dot-bracket)'),
    sequence: z.string().optional().describe('a sequence to fold+diagnose (either this or `folded`)'),
    folded: z.string().optional().describe('an already-known folded structure (dot-bracket)'),
    oracle: z
      .object({
        param_set: z.enum(['turner2004', 'turner1999', 'andronescu2007', 'langdon2018']).optional(),
        temperature: z.number().optional(),
        dangles: z.number().optional(),
      })
      .optional(),
  }),
)
type InputSchema = ReturnType<typeof inputSchema>

const PROMPT = `Diagnose exactly how a candidate's MFE fold differs from the target: which target base pairs are MISSING, which SPURIOUS pairs formed, which positions are mismatched, plus a concrete repair directive. Use this to guide targeted fixes rather than blind mutation.`

export const RnaDiagnoseTool = buildTool({
  name: 'RnaDiagnose',
  searchHint: 'diagnose RNA fold mismatch repair directive',
  maxResultSizeChars: 100_000,
  async description() {
    return 'Localize how a sequence folds vs the target (missing/spurious pairs + repair directive).'
  },
  async prompt() {
    return PROMPT
  },
  get inputSchema(): InputSchema {
    return inputSchema()
  },
  userFacingName() {
    return 'RnaDiagnose'
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
    const res = await callRnaHarness('diagnose', input as Record<string, unknown>)
    return { data: res }
  },
} satisfies ToolDef<InputSchema>)
