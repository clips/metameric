import numpy as np
import random

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

    header = ["word", "iteration", "rt", "freq", "cycles"]
    results = []
    random.seed(44)

    path = "../../corpora/lexicon_projects/elp-items.csv"

    w = read_elp_format(path, lengths=[4])
    np.random.seed(44)

    n_cyc = 1000

    rt = np.array([x['rt'] for x in w])

    inputs = ('letters-features',)

    w = process_data(w,
                     decomposable=('orthography',),
                     decomposable_names=('letters',),
                     feature_layers=('letters',),
                     feature_sets=('fourteen',),
                     negative_features=True,
                     length_adaptation=True)
    rla = {k: 'global' for k in {'letters-features', 'letters'}}
    rla['orthography'] = 'frequency'

    s = Builder(IA_WEIGHTS,
                rla,
                -.05,
                outputs=('orthography',),
                monitors=('orthography',),
                step_size=1.0,
                weight_adaptation=True)

    m = s.build_model(w)
    result = m.activate(w,
                        max_cycles=n_cyc,
                        threshold=.7,
                        strict=False)

    cycles = np.array([len(x['orthography']) for x in result])
    right = cycles == n_cyc
    cycles[right] = -1
