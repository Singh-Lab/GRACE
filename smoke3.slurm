#!/bin/bash
#SBATCH --job-name=grace_smoke3
#SBATCH --output=logs/smoke3.out
#SBATCH --error=logs/smoke3.err
#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
cd /scratch/gpfs/MONA/Toki/GRACE/recomb_grace
echo "### pytest"; python -m pytest -q tests/ 2>&1 | tail -6
echo; echo "### DNA enhancer backend (the one that failed)"
python scripts/optimize.py --task dna_k562 --start 1 --steps 60
echo; echo "### multi-start, multi-seed, aggregate"
python scripts/optimize.py --task protein_canya --start all --seeds 42,0 --steps 40 --quiet --out /tmp/gall2
python scripts/reproduce.py --aggregate /tmp/gall2 2>&1 | tail -5
echo; echo "### examples"
bash examples/03_compare_methods.sh 2>&1 | tail -6
echo "### DONE"
