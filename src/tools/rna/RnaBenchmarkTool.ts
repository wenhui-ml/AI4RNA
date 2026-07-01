/**
 * RnaBenchmark — load benchmark design tasks (Eterna100) as structured task specs the agent
 * can then solve autonomously.
 */
import { z } from 'zod/v4'
import { buildTool, type ToolDef } from '../../Tool.js'
import { lazySchema } from '../../utils/lazySchema.js'
import { callRnaHarness, rnaResultBlock } from './bridge.js'

const inputSchema = lazySchema(() =>
  z.strictObject({
    source: z.enum(['eterna100']).optional().describe('benchmark source (default eterna100)'),
    version: z.enum(['V1', 'V2']).optional().describe('Eterna100 target version (default V2)'),
    ids: z.array(z.number()).optional().describe('specific puzzle ids to load'),
    max_len: z.number().optional().describe('only puzzles with target length <= this'),
  }),
)
type InputSchema = ReturnType<typeof inputSchema>

const PROMPT = `Load RNA inverse-design benchmark tasks (Eterna100). Returns each task's id, name, target dot-bracket structure, and length. Use max_len to select a difficulty range. Solve each task by designing a sequence and confirming with RnaVerify.`

export const RnaBenchmarkTool = buildTool({
  name: 'RnaBenchmark',
  searchHint: 'load Eterna100 RNA design benchmark tasks',
  maxResultSizeChars: 200_000,
  async description() {
    return 'Load RNA inverse-design benchmark tasks (Eterna100 V1/V2).'
  },
  async prompt() {
    return PROMPT
  },
  get inputSchema(): InputSchema {
    return inputSchema()
  },
  userFacingName() {
    return 'RnaBenchmark'
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
    const res = await callRnaHarness('benchmark_task', input as Record<string, unknown>)
    return { data: res }
  },
} satisfies ToolDef<InputSchema>)
