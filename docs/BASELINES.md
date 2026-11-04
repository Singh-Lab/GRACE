# Baselines

Every baseline runs through `scripts/optimize.py --method <name>`, against the
same frozen oracles, the same starting sequences and the same selection rule:
**return the best feasible discrete sequence encountered, or the starting
sequence if none was found.**

| `--method` | What it does |
|---|---|
| `grace` | the full method: null-space projection + augmented Lagrangian |
| `projection_only` | ablation: projection, no multipliers |
| `alm_only` | ablation: augmented Lagrangian, no projection |
| `soft_penalty` | fixed scalar penalty on the violation |
| `fastseqprop` | maximises the target only; constraints never steer the update |
| `adalead` | adaptive population-based greedy search |
| `directed_evolution` | iterative single-substitution scan and select |
| `simulated_annealing` | stochastic mutation with a cooling schedule |
| `random` | mutated variants of the start, no search model |
| `bayesopt` | GP surrogates, constrained `qLogExpectedImprovement` (BoTorch) |

Two more are driven separately because they need their own models:

- **Guided discrete diffusion** (`grace/_guided_diffusion.py`): a pretrained
  discrete diffusion prior steered by property gradients, following NOS. Run it
  with `python -m grace._guided_diffusion --help`.
- **Ctrl-DNA**: constrained RL on a HyenaDNA backbone, DNA tasks only. Use the
  authors' released implementation; we changed nothing but the feasibility
  criterion applied to its output.

## Budget matching

Budgets are matched on **predictor calls**, not steps. A GRACE run with `T`
steps, `K` samples per step and `m` constraint oracles spends

```
G = T · K · (1 + m)
```

calls, and every baseline is given the same `G` on the same task. Each
implementation takes its budget in its own units — candidates, draws, steps, an
evaluation cap — so the value passed to it is whatever makes its call count
equal `G`, and the achieved count is read back from the run's own counter
rather than assumed. Every result JSON carries `n_oracle_calls`.

**Constrained BO is the exception.** Its implementation caps acquisition
iterations at `min(budget − 20, 180)` because refitting the exact GP dominates
its cost, so it evaluates 150–200 candidates whatever budget it is given. It
runs at its published setting and reports the candidates it actually spent; it
is not described as budget-matched.

## A fair comparison in one command

```bash
for m in grace adalead directed_evolution simulated_annealing random fastseqprop; do
    python scripts/optimize.py --task protein_canya --start all --method $m \
        --out results/canya_$m --quiet
done
python scripts/reproduce.py --aggregate results
```
