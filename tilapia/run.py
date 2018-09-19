"""Make an IA run."""
import numpy as np

from tilapia.prepare.weights import weight_adaptation, weights_to_matrix
from tilapia.prepare.weights import IA_WEIGHTS
from tilapia.builder import build_model
from itertools import chain
from collections import Counter


def read_input_file(f):
    """Read an input file."""
    items = []
    header = next(f)
    decode = False
    if isinstance(header, bytes):
        header = header.decode('utf-8').strip().split(",")
        decode = True
    else:
        header = header.strip().split(",")

    for line in (l.decode('utf-8') if decode else l for l in f):
        item = {}
        for k, v in zip(header, line.strip().split(",")):
            item[k] = v.split()
        items.append(item)

    for k in header:
        if all([len(i[k]) == 1 for i in items]):
            for i in items:
                i[k] = i[k][0]
    return items


def parse_parameter_file(f):
    """Parse a parameter file."""
    weights = {}
    if f.mode == 'rb':
        f = (x.decode('utf-8') for x in f)
    for line in f:
        orig, dest, pos, neg = line.lower().strip().split("\t")
        weights[(orig, dest)] = [float(pos), float(neg)]

    return weights


def get_model(words_file,
              parameters,
              rla_variable,
              rla_layers,
              output_layers,
              global_rla,
              step_size,
              decay_rate,
              minimum_activation):
    if parameters is None:
        print("Defaulting to standard IA parameters.")
        weights = IA_WEIGHTS

    words = read_input_file(words_file)
    max_length = max([len(w['orthography']) for w in words])

    weights = weight_adaptation(max_length, weights)
    matrix, names = weights_to_matrix(weights)

    for w in words:
        for idx in range(len(w['orthography']), max_length):
            w['letters'].append(" -{}".format(idx))
        w['orthography'] = w['orthography'].ljust(max_length)

    rla = {k: 'global' if k not in rla_layers
           else rla_variable for k in names}

    m = build_model(words,
                    names,
                    matrix,
                    rla,
                    global_rla,
                    outputs=output_layers,
                    step_size=step_size,
                    decay_rate=decay_rate,
                    minimum=minimum_activation)

    return m


def make_run(words_file,
             test_words_file,
             output_path,
             parameters,
             threshold,
             rla_variable,
             rla_layers,
             output_layers,
             global_rla,
             step_size,
             max_cycles,
             decay_rate,
             minimum_activation):
    """Method for running."""
    test_words = read_input_file(test_words_file)
    m = get_model(words_file,
                  parameters,
                  rla_variable,
                  rla_layers,
                  output_layers,
                  global_rla,
                  step_size,
                  decay_rate,
                  minimum_activation)

    keys_words = Counter(chain.from_iterable(test_words))
    keys_in_all = [k for k, v in keys_words.items() if v == len(test_words)]

    results = m.activate_bunch(test_words,
                               max_cycles=max_cycles,
                               threshold=threshold,
                               strict=False)

    cycles = [len(x[output_layers[0]]) for x in results]
    cycles = np.array(cycles)
    right = cycles < max_cycles
    cycles[~right] = -1

    with open(output_path, 'w') as f:
        k = ",".join([o for o in keys_in_all])
        f.write("{},cycles\n".format(k))
        for a, b in zip(test_words, cycles):
            a = ",".join([x if isinstance(x, str) else " ".join(x)
                          for x in [a[k] for k in keys_in_all]])
            f.write("{},{}\n".format(a, b))
