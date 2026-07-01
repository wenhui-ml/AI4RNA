import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import type { Command } from '../../commands.js'
import { isRnaEnabled, rnaDir } from '../../tools/rna/bridge.js'

const FALLBACK_KNOWLEDGE = `You are an expert RNA inverse-design scientist. Design an RNA sequence over {A,C,G,U} whose ViennaRNA MFE secondary structure equals the target (dot-bracket), satisfying any stated constraints. Use RnaVerify to check candidates and its diagnosis to guide targeted repairs (strengthen missing pairs with G-C, break spurious pairs); you may also use RnaFold/RnaDiagnose or write ViennaRNA code via Bash. Red line: only claim design success under the computational folding oracle, never in-vivo function.`

function loadKnowledge(): string {
  try {
    return readFileSync(join(rnaDir(), 'rna_harness', 'knowledge', 'rna_expert.md'), 'utf8')
  } catch {
    return FALLBACK_KNOWLEDGE
  }
}

const command = {
  type: 'prompt',
  name: 'rna-design',
  description: 'RNA inverse-design specialist: solve an RNA design task autonomously',
  argumentHint: '<target dot-bracket structure and any constraints>',
  contentLength: 0,
  progressMessage: 'designing RNA',
  source: 'builtin',
  isEnabled: () => isRnaEnabled(),
  async getPromptForCommand(args: string) {
    const task = (args || '').trim()
    const taskBlock = task
      ? `# Task\n${task}\n`
      : `# Task\nAsk the user for the target secondary structure (dot-bracket) and any constraints (GC window, motifs, forbidden subsequences, coding), then solve it.\n`
    return [
      {
        type: 'text',
        text:
          `${loadKnowledge()}\n\n---\n\n${taskBlock}\n` +
          `Work autonomously with the RNA tools. Confirm the final design with RnaVerify ` +
          `(it is the official, deterministic scorer). Report the solved sequence, its ` +
          `ensemble defect, and cross-oracle agreement.`,
      },
    ]
  },
} satisfies Command

export default command
