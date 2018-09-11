import numpy as np
import random

from tilapia.builder import build_model
from tilapia.ia.utils import ia_weights, weights_to_matrix, prep_words
from experiments.data import read_elp_format
from scipy.stats import spearmanr, pearsonr
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

    acc = []
    corr = []
    random.seed(44)

    path = "../../corpora/lexicon_projects/elp-items.csv"

    words = np.array(list(read_elp_format(path, lengths=[4])))

    freqs = [x['frequency'] + 1 for x in words]
    freqs = np.log10(freqs)

    sampler = BinnedSampler(words, freqs)
    np.random.seed(44)

    n_cyc = 350

    for idx in tqdm(range(100)):
        w = deepcopy(sampler.sample(1000))
        rt = np.array([x['rt'] for x in w])

        max_len = 4
        inputs = ['features']

        w = prep_words(w)
        matrix, names = weights_to_matrix(ia_weights(max_len))

        rla = {k: 'global' for k in names}
        rla['orthography'] = 'frequency'

        s = build_model(w,
                        names,
                        matrix,
                        rla,
                        -.05,
                        outputs=('orthography',),
                        step_size=.5,
                        inputs=inputs)

        result = s.activate_bunch(w,
                                  num_cycles=n_cyc,
                                  threshold=.7,
                                  strict=False)

        cycles = np.array([len(x['orthography']) for x in result])
        right = cycles < n_cyc
        cycles = cycles[right]
        result_filter = np.array(result)[right]
        ortho = np.array([x['orthography'] for x in w])[right]

        corr.append((spearmanr(cycles, rt[right])[0],
                     pearsonr(cycles, rt[right])[0]))
        a = len(np.flatnonzero(np.array(accuracy(ortho,
                                                 result_filter,
                                                 threshold=.7)[1])))
        acc.append(a / len(w))
