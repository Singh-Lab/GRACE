"""One constrained redesign, start to finish.

    from grace import optimize
    d = optimize("protein_canya", start=1, steps=400)
    print(d.sequence, d.gain, d.feasible)

The optimiser itself is the code that produced the paper's numbers, vendored
unchanged in `grace/_core.py`. This module's job is only to turn a task name
and a handful of user choices into the arguments that core expects, so that a
user never has to name an oracle file or remember a sign convention.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

from .registry import MODEL_DIR, PREFIXED_RUNNERS, Task, TaskError, load_task

MODEL_DIR_DNA = MODEL_DIR / "dna_cnn"

CORE = Path(__file__).resolve().parent / "_core.py"
CORE_DNA = Path(__file__).resolve().parent / "_core_dna.py"

#: Method name -> the core's internal mode string.
METHODS = {
    "grace": "grace_lagrangian",          # the full method
    "projection_only": "grace_only",      # ablation: null-space projection alone
    "alm_only": "lagrangian_only",        # ablation: augmented Lagrangian alone
    "soft_penalty": "soft_penalty",
    "fastseqprop": "unconstrained",
    "adalead": "adalead",
    "directed_evolution": "directed_evolution",
    "simulated_annealing": "simulated_annealing",
    "random": "random",
    "bayesopt": "botorch",
}


def _method_budget(method: str, G: int, m: int) -> int:
    """Translate the shared call budget G into the units each method spends in.

    Every method is given G = steps x K x (1+m) predictor calls, GRACE's own
    spend, but they consume a budget argument in different units, so passing G
    verbatim does not match them. These are the conversions from the paper's
    budget-matching pass (reports/2026-11-03_matched_budget_n15.md), each
    derived from how that mode actually calls the oracles in _core.py:

      GRACE / ablations / soft penalty / FastSeqProp
          G, by construction -- same steps and K.
      Random search
          G/(1+m) draws: one consider() per draw.
      Simulated annealing
          G/(2+m) steps: one consider() plus one target-only acceptance call.
      AdaLead
          64G / (128(1+m) + 160): budget//64 rounds, each up to 128 children at
          (1+m) calls, then a pool of <=160 sorted on the target alone.
      Directed evolution
          has no candidate budget at all; it is capped with --max_evals G
          instead (see optimize(), which also passes --shuffle_scan).
      Bayesian optimisation
          not matched. Its implementation caps acquisition iterations at
          min(max(G,150) - 20, 180) because refitting the exact GP dominates
          the cost, so it evaluates 150-200 candidates whatever it is given.
          docs/BASELINES.md states this exception.

    These assume the published step counts. AdaLead in particular spends its
    budget in rounds of 64, and its population only reaches full width after
    the first round, so at a G small enough to buy one round (roughly
    --steps 25 on a one-constraint task) it underspends badly. That is a
    property of running a population method at a toy budget, not of the
    conversion; at the published settings it reads 54-66% on the consider()
    counter, which the target-only selection calls bring back to ~100%.
    """
    if method == "random":
        return max(1, round(G / (1 + m)))
    if method == "simulated_annealing":
        return max(1, round(G / (2 + m)))
    if method == "adalead":
        return max(64, round(64 * G / (128 * (1 + m) + 160)))
    return G


@dataclass
class Design:
    """The result of one run."""
    task: str
    method: str
    seed: int
    start_sequence: str
    sequence: str                  # the returned design
    start_target: float            # target score of the start, in task units
    target: float                  # target score of the design
    gain: float                    # target - start_target, oriented so higher is better
    edits: int                     # substitutions from the start
    identity: float                # % positions matching the start
    feasible: bool                 # a feasible improvement was found (else the
                                   # start is returned unchanged and gain is 0)
    constraint_start: list         # c_j(x_0)
    tolerances: list               # eps_j actually used
    n_oracle_calls: int
    steps: int
    tolerance_scale: float = 1.0   # the multiplier applied to the published eps

    def to_json(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    def to_fasta(self, path) -> None:
        head = (f">{self.task}|{self.method}|seed{self.seed}|"
                f"gain={self.gain:.4f}|edits={self.edits}|"
                f"{'feasible' if self.feasible else 'INFEASIBLE'}")
        Path(path).write_text(f"{head}\n{self.sequence}\n")

    def summary(self) -> str:
        flag = "feasible" if self.feasible else "NO FEASIBLE IMPROVEMENT FOUND"
        return (f"{self.task} / {self.method} / seed {self.seed}\n"
                f"  start  {self.start_target:+.4f}\n"
                f"  design {self.target:+.4f}   gain {self.gain:+.4f}\n"
                f"  {self.edits} substitutions ({self.identity:.1f}% identity), {flag}\n"
                f"  {self.n_oracle_calls} oracle calls over {self.steps} steps")


def _tolerances(task: Task, tolerance, tolerance_scale):
    """Resolve the per-constraint tolerances actually used by this run."""
    eps = [c.eps for c in task.constraints]
    if tolerance is not None:
        vals = [float(x) for x in str(tolerance).split(",")]
        if len(vals) == 1:
            eps = [vals[0]] * len(eps)
        elif len(vals) == len(eps):
            eps = vals
        else:
            raise TaskError(
                f"{task.key} has {len(eps)} constraints "
                f"({', '.join(c.name for c in task.constraints)}); "
                f"--tolerance got {len(vals)} values.")
    return [e * tolerance_scale for e in eps]


def optimize(task_key: str, *, start=1, sequence=None, method="grace", seed=42,
             steps=None, K=None, lr=None, rho=None, tolerance=None,
             tolerance_scale=1.0, budget=None, active_set_tol=None,
             keep_raw=None, quiet=False) -> Design:
    """Redesign one sequence under its task's constraints.

    Args:
        task_key: a key from `grace.list_tasks()`.
        start: which benchmark starting sequence to use, numbered from 1.
        sequence: your own starting sequence; overrides `start`.
        method: "grace" or any baseline in `grace.api.METHODS`.
        steps / K / lr / rho: override the task's published settings.
        tolerance: absolute tolerance, one value or one per constraint.
        tolerance_scale: multiply the published tolerances (0.5 = twice as tight).
        budget: cap on oracle calls; defaults to the task's matched budget.

    Returns:
        A `Design`. `feasible=False` means no feasible improvement was found and
        the starting sequence is returned unchanged, exactly as in the paper.
    """
    task = load_task(task_key)
    if method not in METHODS:
        raise TaskError(f"Unknown method {method!r}. Choose from: {', '.join(METHODS)}")
    missing = task.missing_models()
    if missing:
        raise TaskError(
            f"{task.key} needs model files that are not present:\n  "
            + "\n  ".join(missing)
            + f"\nDownload them with `bash scripts/download_models.sh {task.key}` "
              "or see models/README.md.")
    if task.runner not in ("generic", "dna") and task.runner not in PREFIXED_RUNNERS:
        raise TaskError(
            f"{task.key} runs through the '{task.runner}' backend, which needs weights "
            f"we cannot redistribute. See docs/TASKS.md for how to set it up.")

    wt = sequence or task.start(start)
    bad = set(wt) - set(task.alphabet)
    if bad:
        raise TaskError(f"Sequence contains {sorted(bad)}, outside the task alphabet "
                        f"{task.alphabet}.")

    steps = steps or task.steps
    K = K or task.K
    eps = _tolerances(task, tolerance, tolerance_scale)
    G = budget or steps * K * (1 + len(task.constraints))
    budget = _method_budget(method, G, len(task.constraints))

    if task.runner == "dna":
        if len(set(eps)) > 1:
            raise TaskError(
                f"{task.key} runs through the DNA backend, which applies one "
                f"tolerance to every constraint. You asked for "
                f"{', '.join(str(e) for e in eps)}. Pass a single --tolerance, or "
                f"use --tolerance-scale to move them together.")
        return _run_dna(task, wt, method, seed, steps, K, lr, rho, eps, G,
                        keep_raw, quiet, active_set_tol, tolerance_scale)

    argv = [
        "--target_oracle", task.resolve_oracle(task.target["oracle"]),
        "--wt", wt,
        "--alphabet", task.alphabet,
        "--target_sign", str(task.target_sign),
        "--mode", METHODS[method],
        "--K", str(K),
        "--steps", str(steps),
        "--lr", str(lr if lr is not None else task.lr),
        "--rho", str(rho if rho is not None else task.rho),
        "--dual_lr", str(task.dual_lr),
        "--active_set_tol", str(task.active_set_tol if active_set_tol is None
                                else active_set_tol),
        "--budget", str(budget),
        "--max_evals", str(G),
        "--seed", str(seed),
        "--eps", str(eps[0]),
        "--eps_list", ",".join(str(e) for e in eps),
    ]
    if method == "directed_evolution":
        # without this the --max_evals cap lands on the N-terminus every time
        argv.append("--shuffle_scan")
    if task.constraints:
        argv += ["--constraint_oracles"] + [
            task.resolve_oracle(c.oracle) for c in task.constraints]

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(keep_raw) if keep_raw else Path(tmp) / "run.json"
        raw.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(CORE), *argv, "--out", str(raw), "--log_history"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not raw.exists():
            raise TaskError(f"The optimiser failed.\n{proc.stderr[-2000:]}")
        if not quiet and proc.stdout.strip():
            print(proc.stdout.strip(), file=sys.stderr)
        r = json.loads(raw.read_text())

    design = r.get("best_feasible_seq") or wt
    edits = sum(a != b for a, b in zip(design, wt))
    # A run that never saw a feasible improvement writes null here; the contract
    # is that the start comes back unchanged with gain 0, not that we crash.
    start_target = float(r["wt_target"])
    raw_best = r.get("best_feasible_target")
    found = raw_best is not None
    target = float(raw_best) if found else start_target
    return Design(
        task=task.key, method=method, seed=seed, start_sequence=wt,
        sequence=design,
        start_target=start_target, target=target,
        gain=target - start_target,
        edits=edits, identity=100.0 * (1 - edits / len(wt)),
        feasible=found and (design != wt or target > start_target),
        constraint_start=r.get("c0", []), tolerances=eps,
        n_oracle_calls=int(r.get("n_evals", 0)), steps=steps,
        tolerance_scale=tolerance_scale)


#: DNA enhancer modes. The DNA runner names two of them differently.
_DNA_MODES = dict(METHODS, adalead="adalead", bayesopt="botorch_baseline")


def _run_dna(task, wt, method, seed, steps, K, lr, rho, eps, G,
             keep_raw, quiet, active_set_tol, tolerance_scale=1.0):
    """The three enhancer tasks run through the DNA CNN backend.

    Its result schema differs from the generic runner's, so the fields are read
    back with the paper's own harvest rule: prefer the best DECODED feasible
    sequence, then the hard-feasible one, then the best sampled one.
    """
    cells = [task.target["oracle"].split(":")[1]] + \
            [c.oracle.split(":")[1] for c in task.constraints]
    argv = [
        "--sequence", wt,
        "--target", cells[0],
        "--constraints", *cells[1:],
        "--model_dir", str(MODEL_DIR_DNA),
        "--mode", _DNA_MODES[method],
        "--steps", str(steps), "--K", str(K),
        "--lr", str(lr if lr is not None else task.lr),
        "--constraint_eps", str(eps[0]),
        "--constraint_type", "absolute",
        "--alm_rho", str(rho if rho is not None else task.rho),
        "--alm_dual_lr", str(task.dual_lr),
        "--active_set_tol", str(task.active_set_tol if active_set_tol is None
                                else active_set_tol),
        "--max_queries", str(G),
        "--seed", str(seed),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(keep_raw) if keep_raw else Path(tmp) / "run.json"
        raw.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run([sys.executable, str(CORE_DNA), *argv, "--out", str(raw)],
                              capture_output=True, text=True)
        if proc.returncode != 0 or not raw.exists():
            raise TaskError(f"The DNA optimiser failed.\n{proc.stderr[-2000:]}")
        r = json.loads(raw.read_text())

    c0 = r.get("real_c_initial") or r.get("c_initial") or {}
    start_target = float(c0[r["target_oracle"]])
    best = next((r[k] for k in ("best_real_feasible_target_score",
                                "best_hard_feasible_target_score",
                                "best_sampled_feasible_target_score")
                 if r.get(k) is not None), None)
    design = r.get("best_real_feasible_sequence") or wt
    found = best is not None
    if not found:
        best = start_target
    edits = sum(a != b for a, b in zip(design, wt))
    return Design(
        task=task.key, method=method, seed=seed, start_sequence=wt, sequence=design,
        start_target=start_target, target=float(best), gain=float(best) - start_target,
        edits=edits, identity=100.0 * (1 - edits / len(wt)), feasible=found,
        constraint_start=[c0.get(c, None) for c in cells[1:]], tolerances=eps,
        n_oracle_calls=int(r.get("n_queries") or r.get("n_evals") or 0), steps=steps,
        tolerance_scale=tolerance_scale)
