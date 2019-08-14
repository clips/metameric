"""Load subtlex corpus and RTs."""
import numpy as np
from wordkit.corpora import lexiconproject
from string import ascii_lowercase


LETTERS = set(ascii_lowercase)


def orth_func(x):
    letters = LETTERS
    a = not set(x) - letters
    return a


def read_elp_format(filename, lengths=()):
    """Read RT data from the ELP."""
    w = lexiconproject(filename, fields=("orthography", "rt", "SUBTLWF"))
    lengths = set(lengths)

    w = w[w['orthography'].apply(orth_func)]
    if lengths:
        w = w[w['length'].isin(lengths)]

    freq = w['SUBTLWF'].values
    m = np.min(freq[freq > 0])

    w['frequency'] = w['SUBTLWF'] + m
    return w.to_dict('records')
