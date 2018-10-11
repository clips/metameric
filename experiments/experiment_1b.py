import numpy as np
import random
import pandas as pd

from diploria.builder import Builder
from diploria.prepare.weights import weights_to_matrix, IA_WEIGHTS
from diploria.prepare.data import process_data
from experiments.data import read_elp_format
from copy import deepcopy
from tqdm import tqdm
from binningsampler import BinnedSampler


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

    words = np.array(list(read_elp_format(path, lengths=[4])))

    freqs = [x['frequency'] + 1 for x in words]
    freqs = np.log10(freqs)

    sampler = BinnedSampler(words, freqs)
    np.random.seed(44)

    n_cyc = 1000

    for idx in tqdm(range(100)):
        w = deepcopy(sampler.sample(1000))
        rt = np.array([x['rt'] for x in w])

        inputs = ('letters-features',)

        w = process_data(w,
                         decomposable=('orthography',),
                         decomposable_names=('letters',),
                         feature_layers=('letters',),
                         feature_sets=('fourteen',),
                         negative_features=True,
                         length_adaptation=False)
        matrix, names = weights_to_matrix(IA_WEIGHTS)

        rla = {k: 'global' for k in names}
        rla['orthography'] = 'frequency'

        s = Builder(names,
                    matrix,
                    rla,
                    -.05,
                    outputs=('orthography',),
                    monitors=('orthography',),
                    step_size=.5,
                    weight_adaptation=True)

        m = s.build_model(w)
        result = m.activate(w,
                            max_cycles=n_cyc,
                            threshold=.7,
                            strict=False)

        cycles = np.array([len(x['orthography']) for x in result])
        right = cycles == n_cyc
        cycles[right] = -1
        for word, c in zip(w, cycles):
            results.append([word['orthography'][0],
                            idx,
                            word['rt'],
                            word['frequency'],
                            c])

    df = pd.DataFrame(results, columns=header)
    df.to_csv("diploria_experiment_1b.csv", sep=",", index=False)
