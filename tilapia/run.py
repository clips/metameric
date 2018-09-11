import os
import numpy as np

from tilapia.ia.utils import IA_WEIGHTS, prep_words
from tilapia.ia.utils import ia_weights, weights_to_matrix
from tilapia.builder import build_model
from csv import reader


def read_input_file(path):
    """Read an input file."""
    items = []
    f = reader(open(path))
    header = next(f)
    for line in f:
        item = {}
        for k, v in zip(header, line):
            item[k] = v.split()
        items.append(item)

    for k in header:
        if all([len(i[k]) == 1 for i in items]):
            for i in items:
                i[k] = i[k][0]

    return items


def parse_parameter_file(path):
    """Parse a parameter file."""
    weights = {}
    for line in open(path):
        orig, dest, pos, neg = line.lower().strip().split("\t")
        weights[(orig, dest)] = [float(pos), float(neg)]

    return weights


def make_run(input_file,
             test_file,
             output_file,
             parameter_file,
             weight,
             space,
             threshold,
             rla,
             rla_layers,
             input_layers,
             output_layers,
             global_rla,
             step_size,
             max_cycles,
             decay_rate,
             minimum_activation):
    """Method for running."""
    if parameter_file is None:
        print("Defaulting to standard IA parameters.")
        weights = IA_WEIGHTS
    else:
        if not os.path.exists(parameter_file):
            raise ValueError("Parameter file {} does not exist. "
                             "Aborting.".format(parameter_file))
        weights = parse_parameter_file(parameter_file)

    if os.path.exists(output_file):
        raise ValueError("Output file {} already exists. "
                         "Aborting.".format(output_file))
    if not os.path.exists(input_file):
        raise ValueError("Input file {} does not exist. "
                         "Aborting.".format(input_file))

    words = read_input_file(input_file)
    test_words = read_input_file(test_file)

    max_length = max([len(w['orthography']) for w in words])

    max_length_test = max([len(w['orthography']) for w in test_words])

    if max_length_test > max_length:
        raise ValueError("A word from the test set if longer than the longest "
                         "word in your training set.")

    if weight:
        weights = ia_weights(max_length, weights)

    if space:
        for w in words:
            w['orthography'] = w['orthography'].ljust(max_length)

    matrix, names = weights_to_matrix(weights)

    if 'features' in names and 'features' not in words[0]:
        words = prep_words(words)
        test_words = prep_words(test_words)
    for word in test_words:
        if set(input_layers) - set(word):
            raise ValueError("{} does not contain all input layers."
                             "".format(word))

    rla = {k: 'global' if k not in rla_layers
           else 'frequency' for k in names}

    s = build_model(words,
                    names,
                    matrix,
                    rla,
                    global_rla,
                    outputs=output_layers,
                    step_size=step_size,
                    inputs=input_layers,
                    decay_rate=decay_rate,
                    minimum=minimum_activation)

    results = s.activate_bunch(test_words,
                               num_cycles=max_cycles,
                               threshold=threshold,
                               strict=False)

    cycles = [len(x[output_layers[0]]) for x in results]
    cycles = np.array(cycles)
    right = cycles < max_cycles
    cycles[~right] = -1

    with open(output_file, 'w') as f:
        k = ",".join([o for o in output_layers])
        f.write("{},cycles\n".format(k))
        for a, b in zip(words, cycles):
            a = ",".join([a[o] for o in output_layers])
            f.write("{},{}\n".format(a, b))
