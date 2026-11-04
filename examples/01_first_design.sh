#!/usr/bin/env bash
# Reduce the amyloid nucleation of one peptide while holding its net charge
# within +/-1.0 of where it started.
set -e
python scripts/optimize.py --task protein_canya --start 1 --out designs/first
cat designs/first/protein_canya_grace_start1_seed42.fasta
