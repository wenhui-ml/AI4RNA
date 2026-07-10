# RNA design playbook (learned strategies)

- Cap each hairpin with a stable 4-nt tetraloop, but VARY the loop across adjacent hairpins. Stable
  choices include GAAA, UUCG, GCAA, GAGA, CUUG, GGAA — rotate through them so that neighbouring
  hairpins do NOT share the same loop sequence.
- Close helices with strong pairs but ALTERNATE the orientation (G-C on one helix, C-G on the next;
  an occasional A-U/U-A is fine) to break sequence repetition between hairpins.
- Why: repeated identical loops + identical stems create spurious cross-hairpin base pairs that
  misfold the whole assembly. Every hairpin must fold locally AND the full multi-hairpin structure
  must have no spurious inter-hairpin pairing. Diversity across units is the key.
