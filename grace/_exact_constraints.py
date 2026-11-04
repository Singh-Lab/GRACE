"""Exact, non-learned constraint scorers used by the benchmark.

These are computed directly from the sequence, so there is no trained model and
no training data. Values match the implementations used during optimization.
"""

CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.5}
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
      "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
      "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}


def net_charge(seq):
    """Sum of residue charges at neutral pH."""
    return sum(CHARGE.get(a, 0.0) for a in seq)


def hydrophobicity(seq):
    """Mean Kyte-Doolittle hydropathy."""
    return sum(KD.get(a, 0.0) for a in seq) / len(seq)


def gc_fraction(seq):
    """Fraction of G or C bases."""
    return sum(1.0 for a in seq.upper() if a in "GC") / len(seq)


def edit_distance(seq, reference):
    """Number of substituted positions relative to the reference."""
    return sum(a != b for a, b in zip(seq, reference))
