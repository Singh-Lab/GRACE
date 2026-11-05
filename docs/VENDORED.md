# What is vendored, and the changes made to it

`grace/_core.py`, `grace/_core_dna.py`, `grace/projection.py` and
`grace/_guided_diffusion.py` are the files that produced the paper's numbers.
They are copied in rather than rewritten, because a cleaner reimplementation
would be a different program and the published results would no longer be
evidence about this code.

Four edits were made, all mechanical:

1. **`grace/_core.py`** — the argument parser that lived inside
   `if __name__ == '__main__':` is now returned by `build_parser()`, so it can
   be inspected without running the module. No default value and no line of the
   algorithm changed. The published settings themselves live in
   `configs/tasks/*.yaml`, which is the single source of truth the API reads.

2. **`grace/_core_dna.py`** — the oracle-call counter `_Q`, which the runner
   already maintained and logged, is now also written into the result JSON as
   `n_queries`. Budget matching is the paper's central fairness claim and it
   should be checkable from an output file without reading a log.

3. **import paths** — `from grace_core import ...` became a relative import of
   `grace/projection.py`, which is the same implementation under a clearer
   name. The old repository reached it through a shim and a `sys.path` insert.
   `grace/_guided_diffusion.py` likewise now imports the shared oracle helpers
   from `_core` rather than from `optimize_generic`, which is the same file
   under its new name.

4. **`grace/parm_oracle.py`** — added, not vendored unchanged: the heat-shock
   oracle loader from the original tree, with its hard-coded path to a PARM
   checkout replaced by the `PARM_REPO` environment variable. It keeps its
   un-prefixed name because `_core.py` does `from parm_oracle import
   PARMOracle` and the core is vendored unchanged.

You can verify each file:

```bash
diff <(git show HEAD:grace/_core.py)      /path/to/original/optimize_generic.py
diff <(git show HEAD:grace/_core_dna.py)  /path/to/original/optimize_dna.py
diff <(git show HEAD:grace/projection.py) /path/to/original/grace_core.py
```

## Why the CLI is a wrapper, not a rewrite

`scripts/optimize.py` builds an argument list and calls the vendored core in a
subprocess. That is deliberate. The core resolves oracle paths, sign
conventions and budget accounting in a particular order, and reproducing that
order in a second implementation is exactly the kind of near-copy that drifts.
The wrapper's whole job is to turn `--task protein_canya --tolerance 0.5` into
the right flags.

The cost is one process launch per run, about 0.2 s. For the matched-budget
grids in `scripts/reproduce.py` that is noise against a run's own runtime.

## Third-party code

None. The baselines are our own implementations of published methods, and the
files above are our own prior code. Where an implementation follows a paper
closely, `grace/projection.py` and `docs/BASELINES.md` name it. No source file
in this repository carries another project's licence header.

PARM's model class and the Pangolin package are *dependencies*, fetched by the
user from their own repositories; neither is redistributed here.
