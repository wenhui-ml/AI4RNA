/**
 * RNA inverse-design specialization tools for openclaude.
 *
 * These are ADDITIVE and feature-flagged (OPENCLAUDE_RNA=1) so they never disturb general
 * openclaude use. Together with the RNA knowledge/persona (skills + system prompt) and the
 * Python `rna_harness` verifier they turn the general code agent into an RNA inverse-design
 * expert, while leaving all of openclaude's native capabilities (bash, code, web, memory,
 * MCP, sub-agents) fully available for autonomous problem solving.
 */
import type { Tool } from '../../Tool.js'
import { isRnaEnabled } from './bridge.js'
import { RnaVerifyTool } from './RnaVerifyTool.js'
import { RnaFoldTool } from './RnaFoldTool.js'
import { RnaDiagnoseTool } from './RnaDiagnoseTool.js'
import { RnaConstraintsTool } from './RnaConstraintsTool.js'
import { RnaBenchmarkTool } from './RnaBenchmarkTool.js'

export { isRnaEnabled } from './bridge.js'

export const RNA_TOOLS: Tool[] = [
  RnaVerifyTool,
  RnaFoldTool,
  RnaDiagnoseTool,
  RnaConstraintsTool,
  RnaBenchmarkTool,
]

export function getRnaTools(): Tool[] {
  return isRnaEnabled() ? RNA_TOOLS : []
}
