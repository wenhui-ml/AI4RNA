"""领域 B:密码子约束的 mRNA 设计 —— 一个标准工具在原理上表达不了的真实领域。

WHY THIS DOMAIN. Two conditions have to hold before a domain can be used to test this agent: experts must
actually struggle in it, and an EXACT verifier must exist. That pair rules out most of RNA -- aptamer
binding, ribozyme activity and translation efficiency have no verifier and are therefore out of scope here,
whatever their importance. Coding-sequence design passes both: the objectives genuinely conflict (codon
choice against structure against composition against manufacturability), and every one of them is exactly
checkable -- translation is a table lookup, folding is the same oracle used everywhere else, and composition
and forbidden motifs are string operations.

WHY THE STANDARD TOOL CANNOT EXPRESS IT, demonstrated rather than asserted. DesiRNA's sequence restraint is
PER POSITION (IUPAC). A codon constraint is JOINT over three positions, and the projection onto positions
over-generates for most amino acids:

    Leu  6 codons  ->  per-position YUN  ->  8 triplets allowed, 2 of them (UUU, UUC) encode Phe
    Ser  6 codons  ->  per-position WSN  -> 16 triplets allowed, 10 of them wrong
    Arg  6 codons  ->  per-position MGN  ->  8 triplets allowed, 2 of them encode Ser
    Val  4 codons  ->  per-position GUN  ->  4 triplets, exact (the codons happen to form a cube)

So this is not a soft-versus-hard distinction and not a missing flag: a joint constraint does not decompose
into per-position constraints. The tool is given its best available approximation anyway, because a benchmark
that withholds the closest thing a tool can do measures our framing rather than the tool.

FEASIBLE BY CONSTRUCTION, so a failure is never "no answer exists": the target structure for each task is the
MFE of one particular synonymous variant of that peptide. At least one sequence therefore satisfies both the
coding constraint and the structural one, and it is a sequence the search is not told.

JUDGED EXACTLY: the answer must translate to the requested peptide AND fold to the target structure.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

CODONS: Dict[str, List[str]] = {
    "A": ["GCU", "GCC", "GCA", "GCG"], "R": ["CGU", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "N": ["AAU", "AAC"], "D": ["GAU", "GAC"], "C": ["UGU", "UGC"],
    "Q": ["CAA", "CAG"], "E": ["GAA", "GAG"], "G": ["GGU", "GGC", "GGA", "GGG"],
    "H": ["CAU", "CAC"], "I": ["AUU", "AUC", "AUA"],
    "L": ["UUA", "UUG", "CUU", "CUC", "CUA", "CUG"], "K": ["AAA", "AAG"],
    "M": ["AUG"], "F": ["UUU", "UUC"], "P": ["CCU", "CCC", "CCA", "CCG"],
    "S": ["UCU", "UCC", "UCA", "UCG", "AGU", "AGC"], "T": ["ACU", "ACC", "ACA", "ACG"],
    "W": ["UGG"], "Y": ["UAU", "UAC"], "V": ["GUU", "GUC", "GUA", "GUG"],
}
AA_OF = {c: aa for aa, cs in CODONS.items() for c in cs}
IUPAC = {"A": "A", "C": "C", "G": "G", "U": "U", "R": "AG", "Y": "CU", "S": "GC", "W": "AU",
         "K": "GU", "M": "AC", "B": "CGU", "D": "AGU", "H": "ACU", "V": "ACG", "N": "ACGU"}

# Peptides chosen for the amino acids whose codons do NOT form an IUPAC cube -- L, S, R -- because those are
# exactly where a per-position restraint stops being equivalent to the real constraint.
PEPTIDES = {
    "LSR-rich-12": "MLSRLSRLSRLV",
    "LSR-rich-16": "MLSRVLSRVLSRVLSR",
    "mixed-14": "MGSLLRSVLGKSTA",
    "mixed-18": "MKLSRDGVLSRAFTELSR",
}


def translate(rna: str) -> Optional[str]:
    """Exact. Returns None if the sequence is not a whole number of known codons."""
    if len(rna) % 3:
        return None
    out = []
    for i in range(0, len(rna), 3):
        aa = AA_OF.get(rna[i:i + 3])
        if aa is None:
            return None
        out.append(aa)
    return "".join(out)


def iupac_restraint(peptide: str) -> str:
    """The closest thing a PER-POSITION restraint can say. Over-generates wherever the codons of an amino
    acid do not form a product of per-position sets -- which is most of them."""
    def code(bases):
        for k, v in IUPAC.items():
            if set(v) == set(bases):
                return k
        return "N"
    out = []
    for aa in peptide:
        cs = CODONS[aa]
        out.append("".join(code({c[i] for c in cs}) for i in range(3)))
    return "".join(out)


def restraint_leak(peptide: str) -> dict:
    """How much the per-position restraint over-generates, exactly: the fraction of triplets it admits that
    encode the WRONG amino acid, position by position."""
    from itertools import product
    bad = tot = 0
    for aa in peptide:
        cs = set(CODONS[aa])
        per = [set(c[i] for c in CODONS[aa]) for i in range(3)]
        allowed = set("".join(p) for p in product(*per))
        tot += len(allowed)
        bad += len(allowed - cs)
    return {"admitted": tot, "wrong": bad, "leak": round(bad / tot, 3)}


def synonymous(peptide: str, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice(CODONS[aa]) for aa in peptide)


def build() -> List[Dict]:
    """Deterministic. The target structure is the MFE of one synonymous variant, so a solution exists."""
    import RNA
    items = []
    for name, pep in PEPTIDES.items():
        ref = synonymous(pep, seed=7)
        target = RNA.fold(ref)[0]
        if target.count("(") < 3:          # a target with almost no structure would test nothing
            ref = synonymous(pep, seed=11)
            target = RNA.fold(ref)[0]
        items.append({"id": name, "peptide": pep, "n": len(ref), "target": target,
                      "reference": ref, "seq_restr": iupac_restraint(pep),
                      "leak": restraint_leak(pep),
                      "why": "the target is the MFE of a synonymous variant, so at least one answer exists"})
    return items


if __name__ == "__main__":
    for it in build():
        print(f"  {it['id']:<12} {it['n']:>3} nt  肽={it['peptide']}")
        print(f"      靶结构   {it['target']}")
        print(f"      逐位约束 {it['seq_restr']}")
        print(f"      约束泄漏 {it['leak']['wrong']}/{it['leak']['admitted']} = {it['leak']['leak']:.0%} "
              f"的被允许三联体编码错误氨基酸")
        print(f"      参考解翻译校验: {translate(it['reference']) == it['peptide']}")
