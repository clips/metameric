"""Load subtlex corpus and RTs."""
from wordkit.corpora import elp
from string import ascii_lowercase


LETTERS = set(ascii_lowercase)


def orth_func(x):
    letters = LETTERS
    a = not set(x) - letters
    return a


def read_elp_format(filename, lengths=()):
    """Read RT data from the ELP."""
    w = elp(filename, fields=("orthography", "rt", "frequency"))
    lengths = set(lengths)

    w = w[w['orthography'].apply(orth_func)]
    if lengths:
        w = w[w['length'].apply(lambda x: x in lengths)]

    return w.to_dict('records')
