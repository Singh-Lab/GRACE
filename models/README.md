# Trained oracles

The oracles are **not** in this repository. They are on Zenodo
(DOI: *to be assigned*) and land here:

```
models/
├── MANIFEST.sha256                    checksums for every file below
├── rna_toehold/
│   ├── target_oracle.pt               ON-state output
│   └── constraint_oracle.pt           OFF-state leakage
├── mrna_mrl/
│   ├── target_oracle.pt               mean ribosome load
│   └── constraint_oracle.pt           folding MFE
├── rna_mfe/
│   └── target_oracle.pt               RNAfold MFE surrogate (GC is exact, no model)
├── protein_kcat/
│   └── target_oracle.pt               log10 k_cat   (hydrophobicity is exact)
├── protein_solubility/
│   └── target_oracle.pt               eSOL score    (net charge is exact)
├── protein_canya/
│   └── target_oracle.pt               amyloid nucleation (net charge is exact)
├── protein_stability_designed/
│   └── target_oracle.pt               proteolysis stability (net charge is exact)
├── dna_cnn/                           one directory per cell type
│   ├── k562/{best_model.pt,normalization.npy,train.csv,valid.csv,test.csv,...}
│   ├── hepG2/{...}
│   └── sknsh/{...}
└── parm/                              heat-shock promoter ensembles, 44 MB
    ├── induction.pt                   heat - control, the target
    └── control.pt                     basal activity, the constraint
```

`dna_cnn/` is most of the bundle's size: each cell type ships the lentiMPRA
train/valid/test splits and its training log next to the weights, so the CNNs
can be retrained rather than only reused.

Not in the default bundle, fetched separately:

```
esm/                                   TEM-1 heads, 3.8 GB (see below)
```

Get them:

```bash
bash scripts/download_models.sh                  # all of it, ~575 MB
bash scripts/download_models.sh protein_canya    # one task, ~1 MB
bash scripts/download_models.sh --from-local DIR # a bundle you already have
```

Keep them outside the repository if you prefer:

```bash
export GRACE_MODELS=/path/to/grace-models
```

`python scripts/optimize.py --list` reports, per task, whether its models are
present.

## What we cannot redistribute

- **Pangolin** (`dna_splicing`). Install the package and use its released
  weights; we add no training of our own.
- **ESM-2** (`protein_tem_*`). Our fine-tuned regression heads are on Zenodo,
  but the base model is `facebook/esm2_t33_650M_UR50D` from Hugging Face.
- **PARM's model class** (`dna_heatshock`). The CNN ensembles we trained ship
  in the bundle under `parm/`, but the network class is PARM's own:
  `git clone https://github.com/vansteensellab/PARM third_party/PARM` and set
  `PARM_REPO`. The MH4 library itself is GEO accession GSE301246, and its
  coordinates are **hg19** — extracting them against hg38 produces an oracle
  that trains normally and reaches Pearson 0.07 against measured activity.

## Provenance

Every oracle was frozen before any optimisation run and never updated during
search. `docs/TASKS.md` gives each one's dataset, split and held-out accuracy.
