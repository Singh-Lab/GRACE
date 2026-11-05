# GRACE

**Gradient-Resolved Adaptive Constrained Exploration**  
*Machine-learning-guided biological sequence design under competing property constraints*

**Improve the target. Respect the constraints.**

Increasing enzyme activity can compromise solubility. Strengthening an enhancer in one cell type can change its activity in others. Biological sequence design is rarely about optimizing a single property in isolation.

GRACE approaches sequence redesign as a **constrained optimization problem**: starting from a reference sequence, it searches for variants with better predicted target performance while keeping other predicted properties within specified tolerances of their reference values.

It combines three ideas:

1. **Straight-through gradient optimization** to search over discrete protein, DNA, and RNA sequences using differentiable property predictors.
2. **Null-space projection** to reduce target updates that conflict with currently active constraints.
3. **An augmented Lagrangian** to respond to constraint violations and guide the search back toward feasibility.

The repository includes the GRACE implementation, a **14-task benchmark** spanning proteins, DNA, and RNA, comparison methods, and scripts for reproducing the reported experiments.

**Explore:** [Quick start](#quick-start) · [Example](#redesign-a-sequence) · [Benchmark](#benchmark) · [Reproduce the experiments](#reproducing-the-experiments) · [Models and data](#models-and-data)

## Quick start

```bash
git clone https://github.com/Singh-Lab/GRACE.git
cd GRACE

conda env create -f environment.yml
conda activate grace

bash scripts/download_models.sh
python scripts/optimize.py --list
```

Python 3.10+ and PyTorch 2.x are required. You can also install Python dependencies with `pip install -r requirements.txt`. Most tasks run on CPU; the TEM-1 (ESM-2) and heat-shock promoter (PARM) tasks are better suited to a GPU. See [Models and data](#models-and-data) for additional dependencies and current model availability.

## Redesign a sequence

For example, search for a protein variant with reduced predicted amyloid nucleation while keeping its net charge close to the starting sequence:

```bash
python scripts/optimize.py --task protein_canya --start 1 --out designs/
```

GRACE writes a designed sequence to `designs/protein_canya_grace_start1_seed42.fasta` and a companion JSON file containing its predicted scores, constraint values, and tolerances. The benchmark default seed is 42.

Want to make the constraints stricter? This example optimizes the ON state of an RNA toehold switch while limiting OFF-state leakage to **half** the benchmark tolerance:

```bash
python scripts/optimize.py --task rna_toehold --start all \
    --tolerance-scale 0.5 --seeds 42,0,8 \
    --out designs/toehold_strict
```

### Choose the starting sequence and search settings

| Option | Meaning |
| --- | --- |
| `--task protein_canya` | Choose one of the 14 tasks; `--list` shows the full list |
| `--start 1` or `--start all` | Use one benchmark reference or all available references |
| `--sequence ACGU...` | Supply your own sequence |
| `--fasta sequences.fa` | Read input sequences from FASTA |
| `--tolerance 0.5` | Override the absolute tolerance for all constraints |
| `--tolerance 0.5,0.1` | Set tolerances separately, in the order shown by `--list` |
| `--tolerance-scale 0.5` | Multiply the task's default tolerances by 0.5 |
| `--steps`, `--K`, `--lr`, `--rho` | Control optimization steps, samples, learning rate, and penalty |
| `--budget` | Cap the total number of predictor calls |
| `--seeds 42,0,8` | Run multiple random seeds |

Constraints are **reference-centered**. For constraint predictor $c_j$, starting sequence $x_0$, and tolerance $\epsilon_j$, a returned design $x$ is feasible when

$$|c_j(x) - c_j(x_0)| \leq \epsilon_j$$

This means GRACE tries to preserve each auxiliary property's *starting value*, not force it toward a universal optimum. If no feasible improvement is found, the API returns the unchanged starting sequence rather than a constraint-violating design.

### Python API

```python
from grace import optimize, list_tasks, load_task

load_task("rna_mfe").describe()
design = optimize("rna_mfe", start=1, tolerance_scale=0.5, steps=500)

print(design.sequence, design.gain, design.feasible)
design.to_fasta("design.fasta")
```

## Benchmark

GRACE is evaluated on **14 redesign problems**: six protein tasks, five DNA tasks, and three RNA tasks. Each has a target to improve and one or more auxiliary properties to preserve.

| Domain | Task | Target to improve | Properties constrained |
| --- | --- | --- | --- |
| Protein | `protein_canya` | Lower amyloid nucleation | Net charge |
| Protein | `protein_kcat` | Catalytic turnover | Hydrophobicity |
| Protein | `protein_solubility` | Solubility | Net charge |
| Protein | `protein_stability_designed` | Mini-protein stability | Net charge |
| Protein | `protein_tem_activity` | TEM-1 activity | Solubility, stability |
| Protein | `protein_tem_solubility` | TEM-1 solubility | Activity, stability |
| DNA | `dna_k562` | K562 enhancer activity | HepG2, SK-N-SH activity |
| DNA | `dna_hepg2` | HepG2 enhancer activity | K562, SK-N-SH activity |
| DNA | `dna_sknsh` | SK-N-SH enhancer activity | K562, HepG2 activity |
| DNA | `dna_heatshock` | Heat-shock inducibility | Basal activity, mutation budget |
| DNA | `dna_splicing` | Liver splice-site usage | Heart/brain/testis usage, mutation budget |
| RNA | `rna_toehold` | Toehold ON-state output | OFF-state leakage |
| RNA | `mrna_mrl` | Translation-related ribosome load | Folding free energy |
| RNA | `rna_mfe` | Lower RNA folding free energy | GC fraction |

Predictors are held fixed during optimization. The benchmark compares methods under the same starting sequences, constraints, and **matched predictor-call budgets**, rather than simply matching iteration counts. Full tolerance values, data sources, and starting-sequence selection rules are in [`docs/TASKS.md`](docs/TASKS.md).

## Reproducing the experiments

```bash
# Run a baseline on the same task
python scripts/optimize.py --task protein_canya --start 1 --method adalead

# Reproduce the main results
python scripts/reproduce.py --table main

# Reproduce the tolerance sweep
python scripts/reproduce.py --table tolerance
```

The repository includes GRACE ablations (`projection_only`, `alm_only`) and comparison methods including soft penalties, Fast SeqProp, AdaLead, directed evolution, simulated annealing, random search, and Bayesian optimization. Guided diffusion and Ctrl-DNA have separate drivers; see [`docs/BASELINES.md`](docs/BASELINES.md).

**Plan compute resources before a full reproduction.** The main experiment grid is 5,940 runs for the 11 tasks supported by the default model bundle. Use `--dry-run` to inspect the grid, `--jobs` to control local parallelism, or `--slurm` for a Slurm array job. Every run records `n_oracle_calls` for budget-matched comparisons.

## Models and data

Pretrained property predictors are archived separately from the source code:

**[GRACE model archive on Zenodo](https://doi.org/10.5281/zenodo.23975030)** · Version 1.0.0

```bash
# Download the complete available bundle (~245 MB compressed)
bash scripts/download_models.sh

# Or download just the models for one task
bash scripts/download_models.sh protein_canya
```

The archive expands to approximately 575 MB and contains model assets for nine components, supporting 11 of the 14 benchmark tasks. The downloader verifies the complete bundle against its SHA-256 manifest.

**Additional requirements and limitations:**

- **TEM-1 activity and solubility:** require fine-tuned ESM-2 assets (~3.8 GB) that are **not included in the current Zenodo release**. The ESM-2 base model is `facebook/esm2_t33_650M_UR50D`.
- **Heat-shock promoters:** additionally require the PARM model class, installed from its source repository:

  ```bash
  git clone https://github.com/vansteensellab/PARM third_party/PARM
  export PARM_REPO=$PWD/third_party/PARM
  ```

- **Splicing:** requires Pangolin and a compatible reference genome. The 10 kb genomic starting intervals are not redistributed.

Most benchmark tasks include 30 starting sequences in `data/starts/`; the two TEM-1 tasks share a wild-type reference. See [`docs/TASKS.md`](docs/TASKS.md) for precise sources and setup.

### Details that matter for reproducibility

- **Heat-shock coordinates use hg19.** Using hg38 for the same fragments changes the predictor inputs and results.
- **Hydrophobicity is summed, not averaged.** The `protein_kcat` constraint uses summed Kyte–Doolittle scores, not GRAVY.
- **Compare matched oracle calls.** An optimizer step can invoke multiple target and constraint predictors; iteration counts alone are not a fair budget comparison.

## Repository structure

```text
grace/             Core optimization package and Python API
configs/tasks/     Task definitions, constraints, and default settings
data/starts/       Redistributable benchmark starting sequences
models/            Downloaded model assets and model documentation
scripts/           Optimization, download, and reproduction entry points
docs/              Task details, baseline implementations, and provenance
examples/          Usage examples
tests/             Smoke tests
```

Task YAML files in `configs/tasks/` define the predictors, tolerances, and default optimization settings shared by the CLI, Python API, and reproduction scripts.

## Citation and license

If GRACE supports your work, please cite the GRACE study (citation metadata are in [`CITATION.cff`](CITATION.cff)). For the pretrained model artifacts, cite the [Zenodo release](https://doi.org/10.5281/zenodo.23975030).

The repository's source code is released under the [MIT License](LICENSE). Underlying benchmark datasets retain the terms of their original sources; see [`docs/TASKS.md`](docs/TASKS.md).
