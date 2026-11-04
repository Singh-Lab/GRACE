#!/usr/bin/env python3
"""Redesign a peptide of your own, then check what the constraint did."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grace import load_task, optimize

PEPTIDE = "GSKTAWFVGQMNALYRPHDE"          # 20 aa, the amyloid task's length

task = load_task("protein_canya")
print(task.describe(), "\n")

for scale in (1.0, 0.5):
    d = optimize("protein_canya", sequence=PEPTIDE, tolerance_scale=scale, quiet=True)
    print(f"tolerance x{scale}: gain {d.gain:+.3f}  "
          f"{d.edits} substitutions  feasible={d.feasible}")
    print(f"  {d.sequence}")

d.to_fasta("my_design.fasta")
print("\nwrote my_design.fasta")
