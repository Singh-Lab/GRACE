#!/usr/bin/env python3
"""Redesign a sequence under constraints.

Examples
--------
List what is available:
    python scripts/optimize.py --list

Redesign benchmark start #3 of the amyloid task with the published settings:
    python scripts/optimize.py --task protein_canya --start 3

Your own sequence, a tighter charge tolerance, a shorter run:
    python scripts/optimize.py --task protein_canya \
        --sequence GSKTAWFVGQMNALYRPHDE \
        --tolerance 0.5 --steps 200 --out designs/mypeptide

Every start, three seeds, written to one folder:
    python scripts/optimize.py --task rna_toehold --start all --seeds 42,0,8 \
        --out designs/toehold
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grace.api import METHODS, optimize                     # noqa: E402
from grace.registry import TaskError, list_tasks, load_task  # noqa: E402


def print_catalogue() -> None:
    print("GRACE benchmark tasks\n" + "=" * 70)
    for key in list_tasks():
        t = load_task(key)
        missing = t.missing_models()
        state = "ready" if not missing else f"needs: {missing[0]}"
        print(f"\n{t.describe()}\n  status: {state}")
    print("\n" + "=" * 70)
    print("methods: " + ", ".join(METHODS))


def main() -> int:
    p = argparse.ArgumentParser(
        description="Constrained biological sequence redesign with GRACE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples\n--------\n")[1])
    p.add_argument("--list", action="store_true",
                   help="show every task, its constraints and whether its models are present")

    g = p.add_argument_group("what to redesign")
    g.add_argument("--task", help="task key; see --list")
    g.add_argument("--start", default="1",
                   help="benchmark starting sequence, numbered from 1, or 'all' (default: 1)")
    g.add_argument("--sequence", help="your own starting sequence, instead of --start")
    g.add_argument("--fasta", help="a FASTA file of starting sequences")

    g = p.add_argument_group("how to redesign")
    g.add_argument("--method", default="grace", choices=sorted(METHODS),
                   help="default: grace")
    g.add_argument("--seeds", default="42", help="comma-separated (default: 42)")
    g.add_argument("--steps", type=int, help="optimisation steps (default: the task's)")
    g.add_argument("--K", type=int, help="samples per step (default: the task's)")
    g.add_argument("--lr", type=float, help="learning rate on the logits")
    g.add_argument("--rho", type=float, help="augmented-Lagrangian penalty")
    g.add_argument("--budget", type=int,
                   help="cap on oracle calls (default: steps x K x (1 + #constraints))")

    g = p.add_argument_group("how tight the constraints are")
    g.add_argument("--tolerance",
                   help="absolute tolerance: one value for all constraints, or one "
                        "per constraint, comma-separated")
    g.add_argument("--tolerance-scale", type=float, default=1.0, dest="tolerance_scale",
                   help="multiply the published tolerances (0.5 = twice as tight)")

    g = p.add_argument_group("output")
    g.add_argument("--out", help="directory for the FASTA and JSON (default: print only)")
    g.add_argument("--quiet", action="store_true")

    a = p.parse_args()
    if a.list or not a.task:
        print_catalogue()
        return 0 if a.list else 2

    task = load_task(a.task)
    if a.fasta:
        # Records wrap at 60 or 80 columns in most FASTA files, so the lines of
        # one record have to be joined before the next ">" rather than taken as
        # separate sequences.
        starts, name, buf = [], None, []
        def _flush():
            if buf:
                starts.append((name or f"seq{len(starts)+1}", "".join(buf)))
        for line in Path(a.fasta).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                _flush()
                name = line[1:].split()[0] if len(line) > 1 else None
                buf = []
            else:
                buf.append(line)
        _flush()
        if not starts:
            raise SystemExit(f"No sequences found in {a.fasta}")
    elif a.sequence:
        starts = [("custom", a.sequence)]
    elif a.start == "all":
        starts = [(f"start{i+1}", s) for i, s in enumerate(task.starts())]
    else:
        i = int(a.start)
        starts = [(f"start{i}", task.start(i))]

    seeds = [int(s) for s in a.seeds.split(",")]
    out = Path(a.out) if a.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, seq in starts:
        for seed in seeds:
            d = optimize(task.key, sequence=seq, method=a.method, seed=seed,
                         steps=a.steps, K=a.K, lr=a.lr, rho=a.rho,
                         tolerance=a.tolerance, tolerance_scale=a.tolerance_scale,
                         budget=a.budget, quiet=True)
            rows.append((label, seed, d))
            if not a.quiet:
                print(d.summary() + "\n")
            if out:
                stem = out / f"{task.key}_{a.method}_{label}_seed{seed}"
                d.to_fasta(stem.with_suffix(".fasta"))
                d.to_json(stem.with_suffix(".json"))

    if out:
        best = max(rows, key=lambda r: r[2].gain)[2]
        best.to_fasta(out / f"{task.key}_{a.method}_best.fasta")
        print(f"{len(rows)} design(s) -> {out}/   best gain {best.gain:+.4f}")
    elif len(rows) > 1:
        best = max(rows, key=lambda r: r[2].gain)[2]
        print(f"best gain {best.gain:+.4f} ({best.edits} substitutions)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TaskError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
