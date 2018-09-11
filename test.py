"""Test."""
from skeleton.builder import build_model
from skeleton.ia.utils import ia_weights, weights_to_matrix, prep_words
from experiments.data import read_elp_format


if __name__ == "__main__":

    words = list(read_elp_format("/Users/stephantulkens/Documents/corpora/lexicon_projects/elp-items.csv", (4,)))

    inputs = ("features", "features_neg")

    words = prep_words(words)
    max_len = max([len(x['orthography']) for x in words])
    matrix, names = weights_to_matrix(max_len, ia_weights(max_len))

    rla = {k: 'global' for k in names}
    rla['orthography'] = 'frequency'

    s = build_model(words,
                    names,
                    matrix,
                    rla,
                    -.05,
                    outputs=('orthography',),
                    inputs=inputs,
                    excluded_fields=('frequency', 'language', 'rt'))
    res = s.activate_bunch(words)
    print(res[0]['orthography'])
