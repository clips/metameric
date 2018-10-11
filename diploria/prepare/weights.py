"""Utilities specifically for IA models."""

# Dict format
IA_WEIGHTS = {("letters", "orthography"): [.28, -.01],
              ("orthography", "letters"): [1.2, .0],
              ("orthography", "orthography"): [.0, -.21],
              ("letters-features", "letters"): [.005, -.15]}
