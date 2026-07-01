/**
 * RnaFold — fold an RNA sequence and report MFE + ensemble properties.
 *
 * A convenience over the exact oracle for exploration. (You may also just write and run
 * ViennaRNA code with Bash — this tool standardizes the common query.) Folding here is for
 * exploration; the OFFICIAL verdict/reward comes from RnaVerify.
 */
import { z } from 'zod/v4'
import { buildTool, type ToolDef } from '../../Tool.js'
import { lazySchema } from '../../utils/lazySchema.js'
import { callRnaHarness, rnaResultBlock } from './bridge.js'

const inputSchema = lazySchema(() =>
  z.strictObject({
    sequence: z.string().describe('RNA sequence (A/C/G/U)'),
    target: z.string().optional().describe('optional target structure for ensemble-defect / p(target)'),
    oracle: z
      .object({
        param_set: z.enum(['turner2004', 'turner1999', 'andronescu2007', 'langdon2018']).optional(),
        temperature: z.number().optional(),
        dangles: z.number().optional(),
      })
      .optional(),
    cross_oracle: z.boolean().optional().describe('also fold under the multi-oracle panel (needs target)'),
  }),
)
type InputSchema = ReturnType<typeof inputSchema>

const PROMPT = `Fold an RNA sequence with ViennaRNA and report its MFE structure, energy, ensemble free energy, and (if a target is given) ensemble defect, p(target), and multi-oracle agreement.

Use for exploration while designing. The available param sets are genuinely different energy models: turner2004 (default), turner1999, andronescu2007, langdon2018 (EternaFold-derived). The OFFICIAL score for a task still comes from RnaVerify.`

export const RnaFoldTool = buildTool({
  name: 'RnaFold',
  searchHint: 'fold RNA sequence MFE ensemble defect',
  maxResultSizeChars: 100_000,
  async description() {
    return 'Fold an RNA sequence (MFE structure/energy, ensemble defect, p(target), multi-oracle agreement).'
  },
  async prompt() {
    return PROMPT
  },
  get inputSchema(): InputSchema {
    return inputSchema()
  },
  userFacingName() {
    return 'RnaFold'
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
    const res = await callRnaHarness('analyze', input as Record<string, unknown>)
    return { data: res }
  },
} satisfies ToolDef<InputSchema>)
