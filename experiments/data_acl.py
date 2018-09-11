"""Load words and such."""
import numpy as np
import pandas as pd

from wordkit.corpora import Subtlex
from string import ascii_lowercase
from functools import partial
from random import shuffle


def rt_data_eng(filename):
    """Load rt data."""
    f = pd.read_csv(filename)
    w = f.loc[:, ("Word", "I_Mean_RT")].to_dict("records")
    d = {x["Word"]: x["I_Mean_RT"] for x in w}
    return {k: v for k, v in d.items() if not np.isnan(v)}


def rt_data_dut(filename):
    """Load rt data."""
    f = pd.read_csv(filename, sep="\t")
    w = f.loc[:, ("spelling", "rt")].to_dict("records")
    d = {x["spelling"]: x["rt"] for x in w}
    return {k: v for k, v in d.items() if not np.isnan(v)}


def filter_words(x, length):
    """Filter words."""
    a = length is None or len(x['orthography']) == length
    b = not set(x['orthography']) - set(ascii_lowercase)
    return all([a, b])


def load_words(wordlist=None,
               num_types=1000,
               num_tokens=None,
               length=None,
               num_slices=1):
    """Load and featurize words for experiments."""
    fields = ["orthography", "log_frequency", "frequency"]
    prefix = "/Users/stephantulkens/Documents/corpora/"
    # prefix = "/home/tulkens/corpora/"

    types = num_types is not None

    if not types and num_slices > 1:
        raise ValueError("You set num_types to None, but still requested "
                         "{} slices. Set num_slices to 1".format(num_slices))

    c_d = Subtlex(prefix + "subtlex/SUBTLEX-NL.cd-above2.txt",
                  merge_duplicates=True,
                  fields=fields,
                  language="nld",
                  scale_frequencies=False)

    filterer = partial(filter_words,
                       length=length)

    words = c_d.transform(filter_function=filterer)
    if c_d.language == "eng-us":
        rts = rt_data_eng(prefix + "lexicon_projects/elp-items.csv")
    elif c_d.language == "nld":
        rts = rt_data_dut(prefix + "lexicon_projects/dlp-items.txt")
    else:
        raise ValueError("No RT corpus.")

    for x in range(num_slices):

        shuffle(words)

        new_words = []
        new_rts = []

        for x in words:
            if x['orthography'] in rts:
                new_rts.append(rts[x['orthography']])
                new_words.append(x)
                if types and len(new_words) == num_types:
                    break
        else:
            if types and len(new_words) <= num_types and num_slices > 1:
                raise ValueError("You requested more types per slice than "
                                 "types in the corpus. Every slice will "
                                 "be the same with these settings.")

        new_rts = np.array(new_rts)

        yield new_words, new_rts
