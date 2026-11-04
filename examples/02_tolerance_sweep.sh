#!/usr/bin/env bash
# How much improvement survives as the constraint tightens. The paper's lead
# over the baselines is largest at the tight end.
set -e
for s in 0.5 0.75 1.0 1.5 2.0; do
    echo "=== tolerance x$s ==="
    python scripts/optimize.py --task protein_canya --start 1 --tolerance-scale "$s"
done
