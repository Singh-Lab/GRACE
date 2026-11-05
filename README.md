# GRACE

**Gradient-Resolved Adaptive Constrained Exploration for biological sequence redesign.**

GRACE improves one property of a sequence while holding others inside a
tolerance band around their starting values — stronger enhancer activity in one
cell type without touching the other two, higher catalytic turnover at constant
hydrophobicity, lower amyloid nucleation at constant net charge. It combines a
differentiable relaxation of the sequence, a null-space projection that removes
the part of the target gradient that would break an active constraint, and an
augmented Lagrangian that corrects whatever violation is left.

This repository contains the method, the 14-task benchmark, every baseline, and
the scripts that reproduce the paper's tables. Eleven of the fourteen tasks run
from the default model bundle; the two TEM-1 tasks need a 3.8 GB ESM-2 download
and splicing needs Pangolin plus a reference genome (see `docs/TASKS.md`).

---

## Install

```bash
git clone https://github.com/<org>/grace.git
cd grace
conda env create -f environment.yml && conda activate grace    # or: pip install -r requirements.txt
bash scripts/download_models.sh                                 # trained oracles, from Zenodo
```

Python 3.10+, PyTorch 2.x. CPU is enough for every task except TEM-1 (ESM-2)
and the heat-shock promoter (PARM), which want a GPU. Heat shock also needs
PARM's model class, which is a git clone rather than a pip package:

```bash
git clone https://github.com/vansteensellab/PARM third_party/PARM
export PARM_REPO=$PWD/third_party/PARM
```

Check what you have:

```bash
python scripts/optimize.py --list
```

---

## Redesign a sequence

The one command you need:

```bash
python scripts/optimize.py --task protein_canya --start 1 --out designs/
```

```
protein_canya / grace / seed 42
  start  -1.8041
  design -6.3672   gain +4.5631
  17 substitutions (15.0% identity), feasible
  6400 oracle calls over 400 steps
```

You get `designs/protein_canya_grace_start1_seed42.fasta` and a `.json` beside
it with the scores, the constraint values and the tolerances actually used.

### Choosing what to redesign

| | |
|---|---|
| `--task` | which problem; `--list` prints all 14 with their constraints |
| `--start N` | benchmark starting sequence *N*, numbered from 1 as in the paper |
| `--start all` | every benchmark start for that task |
| `--sequence ACGU...` | your own sequence instead |
| `--fasta file.fa` | a file of your own sequences |

### Choosing how hard to push

| | |
|---|---|
| `--steps` | optimisation steps (default: the task's published value) |
| `--K` | sequence samples drawn per step |
| `--lr`, `--rho` | learning rate on the logits, augmented-Lagrangian penalty |
| `--budget` | hard cap on oracle calls; defaults to `steps x K x (1 + #constraints)` |
| `--seeds 42,0,8` | run several seeds and keep the best |

### Choosing how tight the constraints are

| | |
|---|---|
| `--tolerance 0.5` | absolute tolerance, applied to every constraint |
| `--tolerance 0.5,0.1` | one value per constraint, in the order `--list` prints them |
| `--tolerance-scale 0.5` | half the published tolerance, i.e. twice as strict |

Tolerances are **reference-centred**: a constraint is satisfied when
`|c(x) - c(x₀)| <= eps`, measured against *your* starting sequence, not an
absolute target. Tightening them is the most informative knob in the whole
tool — the paper's lead over the baselines is largest when they are tight.

### A worked example

Push a toehold switch's ON state, keep OFF-state leakage within half the
benchmark tolerance, three seeds, write everything out:

```bash
python scripts/optimize.py --task rna_toehold --start all \
    --tolerance-scale 0.5 --seeds 42,0,8 --out designs/toehold_strict
```

### From Python

```python
from grace import optimize, list_tasks, load_task

load_task("rna_mfe").describe()        # what the task asks for
d = optimize("rna_mfe", start=1, tolerance_scale=0.5, steps=500)
print(d.sequence, d.gain, d.feasible)
d.to_fasta("design.fasta")
```

`feasible=False` means no feasible improvement was found and your starting
sequence is handed back unchanged. That is the paper's convention: a method
never returns something that breaks a constraint.

---

## The benchmark

14 tasks, each one target property and one to four constrained properties,
every oracle frozen before optimisation begins.

| Domain | Task | Maximise | Subject to |
|---|---|---|---|
| RNA | `rna_toehold` | ON-state output | OFF leakage ±0.02 |
| RNA | `mrna_mrl` | ribosome load | folding MFE ±2.0 kcal/mol |
| RNA | `rna_mfe` | folding stability (↓ MFE) | GC fraction ±0.05 |
| DNA | `dna_k562` | K562 enhancer activity | HepG2, SK-N-SH ±0.1 log₂ |
| DNA | `dna_hepg2` | HepG2 activity | K562, SK-N-SH ±0.1 log₂ |
| DNA | `dna_sknsh` | SK-N-SH activity | K562, HepG2 ±0.1 log₂ |
| DNA | `dna_heatshock` | heat-shock inducibility | basal activity ±0.2, ≤10 substitutions |
| DNA | `dna_splicing` | liver splice-site usage | heart/brain/testis ±0.05, ≤30 substitutions |
| Protein | `protein_kcat` | log₁₀ k_cat | hydrophobicity ±2.0 |
| Protein | `protein_solubility` | eSOL score | net charge ±0.5 |
| Protein | `protein_canya` | ↓ amyloid nucleation | net charge ±1.0 |
| Protein | `protein_stability_designed` | mini-protein stability | net charge ±1.0 |
| Protein | `protein_tem_activity` | TEM-1 activity | solubility, stability ±0.1 |
| Protein | `protein_tem_solubility` | TEM-1 solubility | activity, stability ±0.1 |

Each task ships 30 benchmark starting sequences in `data/starts/` — except the
two TEM-1 tasks, which share a single wild-type reference, and `dna_splicing`,
whose starts are 10 kb genomic intervals and are not redistributed. They are
chosen near
the 25th percentile of the target so there is room to improve, with the
constrained properties in intermediate percentile bands so the constraint is
informative. `docs/TASKS.md` gives the dataset, split, held-out accuracy and
selection rule for every one.

---

## Baselines

Every baseline runs through the same entry point, the same oracles, the same
starting sequences and the same selection rule, at a **matched budget of
predictor calls**:

```bash
python scripts/optimize.py --task protein_canya --start 1 --method adalead
```

`grace`, `projection_only`, `alm_only` (the two ablations), `soft_penalty`,
`fastseqprop`, `adalead`, `directed_evolution`, `simulated_annealing`,
`random`, `bayesopt`. Guided diffusion and Ctrl-DNA have their own drivers;
see `docs/BASELINES.md`.

Budget matching is on predictor calls, not steps: a GRACE run with `T` steps,
`K` samples and `m` constraint oracles spends `G = T·K·(1+m)` calls, and every
baseline is given the same `G`. Each run records what it actually spent in
`n_oracle_calls`.

---

## Reproducing the paper

```bash
python scripts/reproduce.py --table main          # main results
python scripts/reproduce.py --table tolerance     # the 0.5x - 2x tolerance sweep
```

`--table main` covers the tasks whose models are present: 11 of 14 with the
default bundle, 13 with the ESM-2 download. Runs happen locally in a thread
pool; `--slurm` submits them as one array job instead, and `--jobs` sets the
local pool size. Use `--dry-run` first to see the grid size — `main` is about
6,500 runs at 11 tasks.

---

## Layout

```
grace/              the package
  api.py            optimize() -> Design
  registry.py       task configs, starting sequences, model paths
  projection.py     null-space projection and active-set selection
  _core.py          the optimiser, vendored unchanged from the paper's code
  _core_dna.py      the DNA enhancer runner
configs/tasks/      one YAML per task: oracles, tolerances, published settings
data/starts/        30 benchmark starting sequences per task
models/             trained oracles (downloaded, not in git)
scripts/            command-line entry points
docs/               task details, baselines, what was vendored and why
tests/              smoke tests that run in a couple of minutes
```

`configs/tasks/*.yaml` is the single source of truth. Nothing in the code
hard-codes a tolerance, a learning rate or an oracle path — edit the YAML and
the change reaches the CLI, the Python API and the reproduction scripts at once.

---

## Models and data

Trained oracles are on Zenodo (DOI: https://doi.org/10.5281/zenodo.23975030) and are **not** in this
repository. `scripts/download_models.sh` fetches and verifies them against
`models/MANIFEST.sha256`. The default bundle is about 575 MB — most of it the
three DNA enhancer CNNs, which ship their train/valid/test splits alongside the
weights — plus 44 MB of PARM promoter ensembles for heat shock.
`scripts/download_models.sh protein_canya` fetches one task.

Larger and separate: the `esm/` directory, 3.8 GB of fine-tuned ESM-2 heads for
the two TEM-1 tasks.

Two things we do not redistribute at all: Pangolin (install the package and use
its released weights) and PARM's model class (clone it, as above). The ESM-2
base model comes from `facebook/esm2_t33_650M_UR50D`.

Three things silently change the results if you miss them, so they are worth
repeating here:

- **Heat-shock coordinates are hg19.** Extracting the same MH4 fragments against
  hg38 trains oracles that look healthy and reach Pearson 0.07 against measured
  activity.
- **Hydrophobicity is a sum, not a GRAVY average.** The `protein_kcat`
  constraint is summed Kyte–Doolittle over the window; dividing by length moves
  the tolerance by a factor of 40.
- **Compare at matched oracle calls, not matched steps.** Every result file
  records `n_oracle_calls` for exactly this reason.

---


## License

MIT for the code in this repository. The benchmark datasets keep the licences
of their original publications; `docs/TASKS.md` names the source of each one.
