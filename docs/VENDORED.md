# What is vendored, and the one change made to it

`grace/_core.py`, `grace/_core_dna.py`, `grace/projection.py` and
`grace/_guided_diffusion.py` are the files that produced the paper's numbers.
They are copied in rather than rewritten, because a cleaner reimplementation
would be a different program and the published results would no longer be
evidence about this code.

Exactly three edits were made, all mechanical:

1. **`grace/_core.py`** — the argument parser that lived inside
   `if __name__ == '__main__':` is now returned by `build_parser()`. This lets
   `scripts/optimize.py` reuse the same defaults instead of restating them, so
   there is one place where `steps`, `K`, `rho` and `active_set_tol` are
   defined. No default value and no line of the algorithm changed.

2. **`grace/_core_dna.py`** — the oracle-call counter `_Q`, which the runner
   already maintained and logged, is now also written into the result JSON as
   `n_queries`. Budget matching is the paper's central fairness claim and it
   should be checkable from an output file without reading a log.

3. **import paths** — `from grace_core import ...` became a relative import of
   `grace/projection.py`, which is the same implementation under a clearer
   name. The old repository reached it through a shim and a `sys.path` insert.

You can verify both:

```bash
diff <(git show HEAD:grace/_core.py) /path/to/original/optimize_generic.py
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
