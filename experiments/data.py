"""Load subtlex corpus and RTs."""
import pandas as pd
import numpy as np
from string import ascii_lowercase


def read_elp_format(filename, lengths=()):
    """Read RT data from the ELP."""
    temp = set()
    lengths = set(lengths)
    df = pd.read_csv(filename)
    df = df.dropna(subset=["Word", "I_Mean_RT", "SUBTLWF"])
    for idx, line in df.iterrows():
        if line['Word'] in temp:
            continue
        if np.isnan(line['I_Mean_RT']):
            continue
        if set(line['Word']) - set(ascii_lowercase):
            continue
        if lengths and len(line['Word']) not in lengths:
            continue
        temp.add(line['Word'])
        yield {"orthography": line["Word"],
               "rt": line["I_Mean_RT"],
               "frequency": line["SUBTLWF"]}
