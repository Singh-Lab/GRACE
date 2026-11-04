#!/usr/bin/env bash
# Same start, same oracles, same budget of predictor calls.
set -e
for m in grace adalead directed_evolution simulated_annealing random; do
    printf '%-22s' "$m"
    python scripts/optimize.py --task protein_canya --start 1 --method "$m" \
        | grep gain | tr -s ' '
done
