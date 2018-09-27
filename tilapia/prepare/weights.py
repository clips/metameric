"""Utilities specifically for IA models."""
import numpy as np
from itertools import chain


# Dict format
IA_WEIGHTS = {("letters", "orthography"): [.28, -.01],
              ("orthography", "letters"): [1.2, .0],
              ("orthography", "orthography"): [.0, -.21],
              ("letters-features", "letters"): [.005, -.15]}


def weights_to_matrix(weights):
    """Convert a dictionary to a matrix."""
    all_keys = set(chain.from_iterable(weights.keys()))
    mtr = np.zeros((len(all_keys), len(all_keys), 2))
    main_key = {k: idx for idx, k in enumerate(all_keys)}
    for (a, b), value in weights.items():
        mtr[main_key[a], main_key[b]] = value

    k, _ = zip(*sorted(main_key.items(), key=lambda x: x[1]))

    return mtr, k
