# M27 survey — is there a second corpus, and I got it wrong twice before I got it right

A survey, not an experiment. No effect is estimated here and none may be.

## The question

The instrument needs compensatory **double** mutants. Standard mutate-and-map libraries are
**single**-mutant scans. So "does another corpus contain minable interventions" is not a question
about how much data exists — it is a question about which libraries were built with rescue in mind.

## Two false negatives I produced before the answer

**First: the wrong annotation key.** RMDB writes per-row sequences under `ANNOTATION_DATA:`; I
keyed on `DATA_ANNOTATION`, which the older deposits use. All 57 candidates came back
"no per-row sequences", the survey printed *zero interventions anywhere*, and it printed a
conclusion to go with it: `mine` does not transfer. That conclusion was a parser result wearing a
scientific result's clothes.

**Second: whole-sequence Hamming distance.** With the key fixed, deposits parsed and still showed
zero. The reason is that every construct carries a barcode and padding outside the designed
region, so a wild type and its compensatory mutant differ at a dozen positions, not two. Masking
high-variability positions did not rescue it either — in a library of 75,250 *different* designs
the designed region is the variable region, so masking it leaves only the constant adapters.

Both times the output was `0` and both times `0` was wrong. The lesson is specific enough to be
worth stating: **on this kind of data, "we found nothing" and "our reader is broken" produce
byte-identical output**, so a null from a scan is not reportable until the scan has been shown to
find something somewhere.

## What actually decides it

RMDB deposits encode the design cell and the mutations in the construct **name**:

    SL5_SARS_CoV_2                              <- wild type
    SL5_SARS_CoV_2_27C-A_97G-A_5pad6_w53barcode <- two mutations, positions 27 and 97

Counting mutation tags is exact, needs no window inference, and takes milliseconds per deposit.

## Result

Scanning the 57 title-matched deposits:

| deposit | families | mutation histogram | doubles | orientation flips |
|---|---|---|---|---|
| `M2RFPK_DMS_0001` | 250 | {0: 250, 1: 75000} | 0 | 0 |
| `M2SL5_DMS_0000` | 6 | — | 1314 | **160** |
| `M2SL5_2A3_0000` | 6 | — | 1314 | **160** |

`M2RFPK` is mutate-and-map on Rfam pseudoknots: 250 wild types, 300 single mutants each, no
doubles at all. That is what a pure M2 library looks like, and it confirms the distinction the
survey was built to test.

`M2SL5` (R. Huang, R. Das; 206 nt; DMS and 2A3) is different. Its three families are

    SL5_SARS_CoV_2              797 constructs
    SL5_MERS_GCadded            887
    SL5_BtCoV_HKU5_GCadded      812

**three natural coronavirus 5'UTR stem-loops**, carrying a systematic double-mutant scan.

## What this is and is not

It **is** an independent, natural-RNA, intervention-bearing corpus, from a different release, with
two chemical probes. If the mining criterion admits its flips and a predictor can be scored on
them, the limitation "this only works on designed RNA" is gone.

It is **not** a matched replacement for OK7a. 160 flips per probe against 4,317 is roughly
thirty-fold less, spread over three molecules rather than hundreds of designs — so the design
stratification that carries every estimate in the paper has almost nothing to stratify over.
Power has to be measured before anything is claimed, and the readout is a 2D reactivity matrix
rather than a `rescue_factor` column, so the outcome has to be derived rather than read.

Also: the natural/designed labels in the survey come from a regex over deposit titles and are
unreliable — `M2SL5` was labelled `unclear` because "SL5" is not in the pattern. The families had
to be opened to find out what the corpus was. Titles do not decide; construct names do.

## Still open

A full-library scan over all 1024 deposits is running under the same name-based criterion, because
the title filter is exactly the kind of pre-filter that produced the first false negative.

---

# Full-library sweep — 1024 deposits, and the auto-verdict was wrong a third time

    scanned      1024 deposits (no title pre-filter, no disk cache)
    total flips  8658
    deposits with minable interventions   11
    runtime      399s

| deposit | families | doubles | orientation flips | what it is |
|---|---|---|---|---|
| `M3SARS_{2A3,DMS}_0000` | 3 | 15759 | **3954** | coronavirus frameshift elements |
| `M2SL5_{2A3,DMS}_0000` | 6 | 1314 | 160 | coronavirus 5'UTR stem-loops |
| `15KREP_{2A3,DMS}_0001` | 35070 | 3600 | 136 | DasLab 15K replicate library |
| `15KLIB_{2A3,DMS}_0000` | 8781 | 900 | 34 | DasLab 15k library |
| `15KREP_{2A3,DMS}_0000` | 8781 | 900 | 34 | replicates from previous |
| `TTYH2_DMS_0001` | 173 | 121 | 22 | human TTYH2 ECS dsRNA (Li JB lab) |

## M3SARS is the find, not M2SL5

    fse__NC_045512.2_13470-13569    5357 constructs   SARS-CoV-2
    fse__NC_004718.3_13400-13499    5357              SARS-CoV (2003)
    fse__NC_019843.3_13440-13539    5357              MERS-CoV

Three natural coronavirus **frameshift stimulation elements**, 207 nt, two chemical probes,
R. Huang and R. Das. **3954 orientation flips against our 4317** — the same order of magnitude,
on natural viral RNA, from an independent release.

And the position-separation histogram is 1, 2, 3, 4, 5, 6, ... in near-equal counts: this is an
**exhaustive** two-dimensional double-mutant scan, every position pair. That matters more than the
count does. In OK7a the experimenters chose which pairs to rescue, which is the confound the whole
placement-conditioning apparatus exists to handle. An exhaustive scan has no such choice in it.

The corollary is that most of those 3954 flips are **not at base pairs** — `_0G-C_1G-C` is
positions 0 and 1, adjacent. The usable count is the intersection with the declared structure and
it has not been computed. Nothing is claimed about it here.

## The third false negative, and why the verdict line is gone

The sweep printed **"no natural corpus contains a minable intervention"** — again. Both M2SL5 and
M3SARS were sitting in its own results table when it printed that. The `kind` label comes from a
regex over deposit titles, and these deposits are titled with library names
(`DasLabBigLib2_OneMil2_fse`), not RNA names, so both came back `unclear` and the `if not nat`
branch fired.

That line is now removed rather than repaired. A classifier that cannot classify should not print
a conclusion, and this one had produced a wrong one on every single run.

## Coverage limit, stated

Roughly 900 of the 1024 deposits skipped with "no construct naming and no per-row sequence
annotation" — older deposits use formats this reader does not parse. **The sweep is a lower bound
on what RMDB holds, not a census**, and the eleven hits should be read that way.
