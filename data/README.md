# Data

## `starts/`

The benchmark starting sequences, 30 per task, one per line, in the order the
paper numbers them (`--start 1` is the first line).

They were selected so that the comparison is informative: the target property
sits near the 25th percentile of a 6,000-sequence candidate pool, leaving room
to improve, while the constrained properties sit in intermediate percentile
bands, so the constraint actually binds without starting at an extreme. A
minimum Hamming distance of `max(5, L/3)` between starts keeps them in distinct
neighbourhoods.

Two tasks have no file here because their starts are genomic intervals rather
than free sequences:

- `dna_heatshock` — 30 promoters from the GSE301246 held-out test set, extracted
  against **hg19**.
- `dna_splicing` — 48 splice sites on Pangolin's held-out chromosomes 1, 3, 5,
  7 and 9, with 10,001 bp of context each.

`docs/TASKS.md` gives the extraction recipe for both.

## Training data

The datasets behind the oracles are not redistributed here; each one keeps the
licence of its original publication. `docs/TASKS.md` names the source, the
accession and the split for every task.
