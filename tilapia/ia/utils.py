"""Utilities specifically for IA models."""
import numpy as np

from wordkit.orthography.features import fourteen
from copy import copy, deepcopy
from itertools import chain

# Dict format
IA_WEIGHTS = {("letters", "orthography"): [.28, -.01],
              ("orthography", "letters"): [1.2, .0],
              ("orthography", "orthography"): [.0, -.21],
              ("features", "letters"): [.005, -.15],
              ("features_neg", "letters"): [.005, -.15]}


fourteen[" "] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def convert_fourteen(fourteen):
    """Convert the fourteen feature set to indices."""
    return {k: np.flatnonzero(np.array(v)).tolist()
            for k, v in fourteen.items()}


def convert_fourteen_neg(fourteen):
    """Convert the fourteen features set to negative indices."""
    return {k: np.flatnonzero(np.array(v) == 0).tolist()
            for k, v in fourteen.items()}


pos_fourteen = convert_fourteen(fourteen)
neg_fourteen = convert_fourteen_neg(fourteen)


def ia_weights(word_length, weights=IA_WEIGHTS):
    """Rescale weights for the IA dependent on length."""
    weights = deepcopy(weights)
    weights[('letters', 'orthography')][0] /= word_length
    weights[('letters', 'orthography')][1] *= word_length
    weights[('orthography', 'letters')][0] /= word_length
    # Is 0 in standard, but could be different.
    weights[('orthography', 'letters')][1] *= word_length

    return weights


def weights_to_matrix(weights):
    """Convert a dictionary to a matrix."""
    mtr = np.zeros((4, 4, 2))

    all_keys = set(chain.from_iterable(weights.keys()))
    main_key = {k: idx for idx, k in enumerate(all_keys)}
    for (a, b), value in weights.items():
        mtr[main_key[a], main_key[b]] = value

    k, _ = zip(*sorted(main_key.items(), key=lambda x: x[1]))

    return mtr, k


def prep_words(words):
    """Add features and letters to words."""
    words = copy(words)
    for x in words:
        x['letters'] = []
        x['features'] = []
        x['features_neg'] = []
        for idx, l in enumerate(x['orthography']):
            x["letters"].append("{}-{}".format(l, idx))
            x["features"].extend(["{}-{}".format(x, idx)
                                  for x in pos_fourteen[l]])
            x["features_neg"].extend(["{}-{}".format(x, idx)
                                      for x in neg_fourteen[l]])

    return words
