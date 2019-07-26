"""Load subtlex corpus and RTs."""
import numpy as np
from wordkit.corpora import LexiconProject
from string import ascii_lowercase


LETTERS = set(ascii_lowercase)


def orth_func(x):
    letters = LETTERS
    a = not set(x) - letters
    return a


def read_elp_format(filename, lengths=()):
    """Read RT data from the ELP."""
    lex = LexiconProject(filename, fields=("orthography", "rt", "SUBTLWF"))
    lengths = set(lengths)
    w = lex.transform(orthography=orth_func, filter_nan=("rt", "SUBTLWF"))
    if lengths:
        w = w.filter(orthography=lambda x: len(x) in lengths)

    freq = w.get('SUBTLWF')
    m = np.min(freq[freq > 0])

    for x in w:
        x['frequency'] = x['SUBTLWF'] + m

    return w
