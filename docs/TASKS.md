# The 14 tasks

Every task names one property to improve and one or more to hold inside a
tolerance band around their value in the starting sequence. All oracles are
frozen before optimisation and never updated during search.

## At a glance

| Task | Domain | L | Maximise | Constrained within | K | Steps | lr | rho |
|---|---|---|---|---|---|---|---|---|
| `mrna_mrl` | rna | 50 | mean ribosome load | folding MFE ±2.0 kcal/mol | 16 | 800 | 0.25 | 10.0 |
| `rna_mfe` | rna | 40 | minimum free energy | GC fraction ±0.05 fraction of G+C | 16 | 500 | 0.2 | 100.0 |
| `rna_toehold` | rna | 30 | ON-state output | OFF-state leakage ±0.02 normalised 0-1 | 8 | 500 | 0.2 | 10.0 |
| `dna_heatshock` | dna | 209-384 | heat-shock inducibility (heat - control) | basal (control) activity ±0.2 PARM activity-score units; substitution budget ±10 substitutions | 8 | 300 | 0.1 | 1.0 |
| `dna_hepg2` | dna | 200 | HepG2 activity | K562 activity ±0.1 log2 RNA/DNA; SK-N-SH activity ±0.1 log2 RNA/DNA | 8 | 500 | 0.01 | 10.0 |
| `dna_k562` | dna | 200 | K562 activity | HepG2 activity ±0.1 log2 RNA/DNA; SK-N-SH activity ±0.1 log2 RNA/DNA | 8 | 500 | 0.01 | 10.0 |
| `dna_sknsh` | dna | 200 | SK-N-SH activity | K562 activity ±0.1 log2 RNA/DNA; HepG2 activity ±0.1 log2 RNA/DNA | 8 | 500 | 0.01 | 10.0 |
| `dna_splicing` | dna | 10001 | liver splice-site usage | heart usage ±0.05 fraction 0-1; brain usage ±0.05 fraction 0-1; testis usage ±0.05 fraction 0-1; substitution budget ±30 substitutions | 2 | 300 | 0.05 | 10.0 |
| `protein_canya` | protein | 20 | CANYA nucleation score | net charge ±1.0 elementary charges | 8 | 400 | 0.1 | 0.5 |
| `protein_kcat` | protein | 40 | log10 k_cat | Kyte-Doolittle hydrophobicity (SUM, not GRAVY) ±2.0 summed scale units | 16 | 800 | 0.1 | 0.1 |
| `protein_solubility` | protein | 40 | eSOL solubility score | net charge ±0.5 elementary charges | 16 | 800 | 0.2 | 1.0 |
| `protein_stability_designed` | protein | 43 | proteolysis stability score | net charge ±1.0 elementary charges | 16 | 800 | 0.1 | 0.1 |
| `protein_tem_activity` | protein | 285 | ampicillin-resistance fitness | solubility ±0.1 log2 enrichment; folding stability ±0.1 kcal/mol; substitution budget ±171 substitutions | 8 | 500 | 0.03 | 10.0 |
| `protein_tem_solubility` | protein | 285 | yeast-display solubility | ampicillin-resistance fitness ±0.1 relative; folding stability ±0.1 kcal/mol; substitution budget ±171 substitutions | 8 | 500 | 0.03 | 10.0 |

The target is always stated so that **higher is better**; two tasks
(`rna_mfe`, `protein_canya`) minimise their native score and carry
`target_sign: -1` in their config, which the code applies for you.

`K`, `steps`, `lr` and `rho` are the published settings and are the defaults.
Shared across every task: active-set threshold 0.3, projection damping 1e-6,
multiplier clipping [0, 100], dual step 0.05 (0.02 on TEM-1 and on every
DNA task: the three enhancers, heat shock and splicing).

Hyperparameters were selected on the **first 5** of the 30 starting sequences.
The remaining 25 are held out and were never used for tuning.

## Starting sequences

30 per task, in `data/starts/<task>.txt`, numbered from 1 as `--start N`.

They were chosen so the comparison is informative: the target sits near the
25th percentile of a 6,000-sequence candidate pool, leaving headroom, while the
constrained properties sit in intermediate percentile bands so the constraint
binds without starting at an extreme. Consecutive starts are at least
`max(5, L/3)` substitutions apart.

Two tasks were selected from genomic loci rather than a library:

- **`dna_heatshock`** — 30 promoters from the GSE301246 held-out test set.
  Basal activity between the 60th and 95th percentile of eligible test
  promoters, predicted inducibility below its 70th percentile, one promoter per
  gene, separated loci. **Coordinates are hg19.** Extracting the same MH4
  fragments against hg38 produces an oracle that appears to train normally and
  reaches Pearson 0.07 against measured activity. The 30 extracted promoter
  sequences ship in `data/starts/dna_heatshock.txt`, so you do not need the
  genome to run this task.
- **`dna_splicing`** — 48 canonical splice sites on Pangolin's held-out human
  chromosomes 1, 3, 5, 7, 9, one per gene, measured liver usage in [0.2, 0.8].
  Each carries 10,001 bp of context; a ±500 bp window is editable, the splice
  core and its immediate flanks are frozen, leaving 993 mutable positions.
  Starting sequences are **not** shipped: each is a 10 kb genomic interval, and
  redistributing 48 of them means redistributing the reference genome. This
  task is the one benchmark entry you cannot reproduce from this artifact
  alone; see below.

## What runs out of the box

| | Tasks | Needs |
|---|---|---|
| Runs after `download_models.sh` | the 3 RNA, the 3 enhancers, the 4 CNN protein tasks, and `dna_heatshock` — **11 of 14** | the default model bundle; heat shock additionally needs the PARM model class (below) |
| Needs a larger download | `protein_tem_activity`, `protein_tem_solubility` | the `esm/` bundle (3.8 GB of fine-tuned ESM-2 weights) plus `pip install fair-esm` |
| Not reproducible from this artifact | `dna_splicing` | the Pangolin package and weights, a reference genome, and the 48 genomic intervals |

**PARM model class** (for `dna_heatshock`). The weights are ours and ship in the
bundle under `parm/`; the network class is PARM's own and we do not
redistribute it:

```bash
git clone https://github.com/vansteensellab/PARM third_party/PARM
export PARM_REPO=$PWD/third_party/PARM
```

**ESM-2 oracles** (for the two TEM-1 tasks). Fetch the `esm/` directory from the
Zenodo record into your `GRACE_MODELS` directory, then `pip install fair-esm`.
The two tasks share one 285-aa reference sequence rather than 30 starts, so
`--start all` gives a single run per seed.

**Pangolin** (for `dna_splicing`). Install from
<https://github.com/tkzeng/Pangolin>, fetch a reference genome, and extract the
48 intervals. We do not ship an extraction script.

## Oracles and data

| Task | Source dataset | n | Split | Held-out accuracy |
|---|---|---|---|---|
| `rna_toehold` | toehold ON/OFF library, Angenent-Mari et al. | 60,750 ON / 78,455 OFF | random 90/10 | rho 0.660 ON, 0.524 OFF |
| `mrna_mrl` | Optimus 5-Prime polysome MPRA, Sample et al. | 30,000 | random 90/10 | rho 0.838 MRL, 0.747 MFE |
| `rna_mfe` | ViennaRNA RNAfold labels (computational) | 45,000 | random 90/10 | rho 0.730; GC exact |
| `dna_k562` / `dna_hepg2` / `dna_sknsh` | three-cell lentiMPRA, Gosai et al. | 798,064 | 80/10/10 after dedup | Pearson 0.862 / 0.864 / 0.862 |
| `dna_heatshock` | PARM MH4 promoter MPRA, GSE301246 | 771,323 | official held-out fold | Pearson 0.838 control, 0.828 heat |
| `dna_splicing` | Pangolin, Zeng & Li (not retrained) | 48 sites | held-out chromosomes | published model |
| `protein_kcat` | DLKcat-derived turnover windows | 18,304 | random 90/10 | rho 0.270; hydrophobicity exact |
| `protein_solubility` | E. coli eSOL, Niwa et al. | 32,943 | random 90/10 | rho 0.457; charge exact |
| `protein_canya` | CANYA nucleation assay, Thompson et al. | 42,235 | random 90/10 | rho 0.354; charge exact |
| `protein_stability_designed` | Rocklin et al. via TAPE | 48,399 | random 90/10 | rho 0.755; charge exact |
| `protein_tem_activity` | TEM-1 DMS, Firnberg et al. | 5,198 | 5-fold, position-based | rho 0.494 +/- 0.077 |
| `protein_tem_solubility` | TEM-1 yeast display, Klesmith et al. | 4,130 | 5-fold, position-based | rho 0.519 +/- 0.059 |

## Two unit traps

- **Hydrophobicity is a SUM, not GRAVY.** The `protein_kcat` constraint is
  summed Kyte-Doolittle over the 40-aa window. Dividing by length moves the
  effective tolerance by a factor of 40.
- **GC fraction is exact and integer-quantised.** For a 40-nt RNA a +/-0.05
  band permits a change of at most two GC bases, so the constraint is satisfied
  on a lattice, not a continuum. The feasibility check recomputes it from
  integer counts to avoid float boundary cases.
