import numpy as np
import pandas as pd

from metameric.builder import Builder
from metameric.prepare.weights import IA_WEIGHTS
from metameric.prepare.data import process_data
from experiments.data import read_elp_format


def accuracy(words, results, threshold=.7):
    """Compute accuracy."""
    score = []

    for w, result in zip(words, results):
        result = result['orthography']
        if not result[-1]:
            score.append(False)
            continue
        keys, values = zip(*result[-1].items())
        if np.max(values) < threshold:
            score.append(False)
            continue
        if keys[np.argmax(values)] == w:
            score.append(True)
            continue
        else:
            score.append(False)

    return np.sum(score) / len(score), score


if __name__ == "__main__":

    header = ["word", "rt", "freq", "cycles"]
    results = []

    path = "../../corpora/lexicon_projects/elp-items.csv"

    words = np.array(list(read_elp_format(path, lengths=list(range(3, 11)))))

    freqs = [x['frequency'] + 1 for x in words]
    freqs = np.log10(freqs)

    n_cyc = 1000

    rt = np.array([x['rt'] for x in words])
    w = process_data(words,
                     decomposable=('orthography',),
                     decomposable_names=('letters',),
                     feature_layers=('letters',),
                     feature_sets=('fourteen',),
                     negative_features=True,
                     length_adaptation=True)

    rla = {k: 'global' for k in ["letters", "letters-features"]}
    rla['orthography'] = 'frequency'

    s = Builder(IA_WEIGHTS,
                rla,
                -.05,
                outputs=('orthography',),
                monitors=('orthography',),
                step_size=.1,
                weight_adaptation=True)

    m = s.build_model(w)
    result = list(m.activate(w,
                             max_cycles=n_cyc,
                             threshold=.7,
                             strict=True,
                             shallow_run=True))

    cycles = np.array([len(x['orthography']) for x in result])
    right = cycles == n_cyc
    cycles[right] = -1
    for word, c in zip(w, cycles):
        results.append([word['orthography'][0],
                        word['rt'],
                        word['frequency'],
                        c])

    df = pd.DataFrame(results, columns=header)
    df.to_csv("metameric_experiment_3.csv", sep=",", index=False)
