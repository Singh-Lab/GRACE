#!/usr/bin/env python3
"""Rebuild the paper's tables from scratch.

    python scripts/reproduce.py --table main --dry-run     # see the grid first
    python scripts/reproduce.py --table main --slurm       # one job per run
    python scripts/reproduce.py --table main --jobs 8      # locally, 8 at a time
    python scripts/reproduce.py --aggregate results/main   # tables from finished runs

The full main-results grid is about 10,000 runs, so look at --dry-run before
launching anything. Every run writes one JSON; --aggregate turns a directory of
them into the table.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from grace.api import METHODS, optimize                     # noqa: E402
from grace.registry import list_tasks, load_task            # noqa: E402

TABLES = {
    # (methods, seeds, tolerance scales)
    "main": (["grace", "projection_only", "alm_only", "adalead",
              "directed_evolution", "simulated_annealing", "random",
              "fastseqprop", "bayesopt"], [42, 0], [1.0]),
    "trajectory": (["grace", "projection_only", "alm_only", "adalead",
                    "directed_evolution", "simulated_annealing", "random",
                    "fastseqprop"], [42, 0], [1.0]),
    "tolerance": (["grace", "projection_only", "alm_only", "adalead",
                   "directed_evolution", "simulated_annealing", "fastseqprop"],
                  [42, 0, 8], [0.5, 0.75, 1.0, 1.5, 2.0]),
}


def grid(table, tasks):
    methods, seeds, scales = TABLES[table]
    for task in tasks:
        t = load_task(task)
        if t.missing_models():
            print(f"skipping {task}: {t.missing_models()[0]}", file=sys.stderr)
            continue
        n = len(t.starts())
        for m, s, sc, i in itertools.product(methods, seeds, scales, range(1, n + 1)):
            yield dict(task=task, method=m, seed=s, scale=sc, start=i)


def one(job, outdir):
    out = (Path(outdir) / job["task"] / job["method"] /
           f"start{job['start']:02d}_seed{job['seed']}_tol{job['scale']}.json")
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    d = optimize(job["task"], start=job["start"], method=job["method"],
                 seed=job["seed"], tolerance_scale=job["scale"], quiet=True)
    d.to_json(out)
    return out


def aggregate(outdir):
    import numpy as np
    import pandas as pd
    rows = [json.loads(p.read_text()) for p in Path(outdir).rglob("*.json")]
    if not rows:
        print(f"no results under {outdir}", file=sys.stderr)
        return 1
    df = pd.DataFrame(rows)
    if "tolerance_scale" not in df:
        df["tolerance_scale"] = 1.0
    # Collapse seeds to their per-start median first: the unit of inference is
    # the starting sequence, not the run. The tolerance scale has to stay in the
    # key, or a sweep collapses into a single number.
    keys = ["task", "method", "start_sequence", "tolerance_scale"]
    per_start = df.groupby(keys)["gain"].median().reset_index()
    scales = sorted(per_start["tolerance_scale"].unique())
    if len(scales) > 1:
        for sc in scales:
            sub = per_start[per_start.tolerance_scale == sc]
            print(f"\n--- tolerance x{sc}")
            print(sub.pivot_table(index="task", columns="method", values="gain",
                                  aggfunc="median").round(3).to_string(na_rep="  --"))
    else:
        table = per_start.pivot_table(index="task", columns="method", values="gain",
                                      aggfunc="median").round(3)
        print(table.to_string(na_rep="  --"))
    print(f"\n{len(df)} runs, {per_start.shape[0]} "
          f"(task, method, start, tolerance) cells")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", choices=sorted(TABLES))
    p.add_argument("--tasks", help="comma-separated subset (default: all)")
    p.add_argument("--out", default="results", help="output directory")
    p.add_argument("--dry-run", action="store_true", help="count the runs and stop")
    p.add_argument("--jobs", type=int, default=1, help="local parallelism")
    p.add_argument("--slurm", action="store_true", help="submit one array job")
    p.add_argument("--aggregate", help="build the table from a finished directory")
    p.add_argument("--task-id", type=int, dest="task_id",
                   help=argparse.SUPPRESS)   # one element of the grid, for --slurm
    a = p.parse_args()

    if a.aggregate:
        return aggregate(a.aggregate)
    if not a.table:
        p.error("one of --table or --aggregate is required")

    tasks = a.tasks.split(",") if a.tasks else list_tasks()
    jobs = list(grid(a.table, tasks))
    outdir = Path(a.out) / a.table

    if a.task_id:                              # one array element
        print(one(jobs[a.task_id - 1], outdir))
        return 0

    print(f"{a.table}: {len(jobs)} runs over {len({j['task'] for j in jobs})} tasks "
          f"-> {outdir}")
    if a.dry_run:
        for j in jobs[:5]:
            print("  ", j)
        print(f"   ... and {len(jobs) - 5} more")
        return 0

    if a.slurm:
        spec = outdir / "_jobs.jsonl"
        spec.parent.mkdir(parents=True, exist_ok=True)
        spec.write_text("\n".join(json.dumps(j) for j in jobs))
        script = outdir / "_array.slurm"
        script.write_text(f"""#!/bin/bash
#SBATCH --job-name=grace_{a.table}
#SBATCH --array=1-{len(jobs)}%200
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output={outdir}/logs/%A_%a.out
cd {ROOT}
python scripts/reproduce.py --table {a.table} --task-id $SLURM_ARRAY_TASK_ID --out {a.out}
""")
        (outdir / "logs").mkdir(exist_ok=True)
        print(subprocess.run(["sbatch", str(script)], capture_output=True,
                             text=True).stdout.strip())
        return 0

    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for i, _ in enumerate(ex.map(lambda j: one(j, outdir), jobs), 1):
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    return aggregate(outdir)


if __name__ == "__main__":
    sys.exit(main())
