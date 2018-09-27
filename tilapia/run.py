"""Make an IA run."""
import numpy as np
import pandas as pd

from tilapia.prepare.weights import weights_to_matrix
from tilapia.prepare.weights import IA_WEIGHTS
from tilapia.builder import Builder
from itertools import chain
from collections import Counter


def make_slot(x):
    """Turn a list of slot features into slots."""
    for value in x:
        value = value.split("-")
        yield((value[0], int(value[1])))


def is_slot(x):
    """Check whether a feature is slot-based."""
    for value in x:
        s = value.split("-")
        if len(s) != 2:
            return False
        try:
            int(s[1])
        except ValueError:
            return False
    return True


def read_input_file(f):
    """Read an input file."""
    df = pd.read_csv(f)
    dtypes = [col for col, dtype in zip(df.columns, df.dtypes)
              if dtype == object]
    items = df.to_dict('records')
    for d in dtypes:
        slot_feature = []
        for i in items:
            i[d] = i[d].split()
            slot_feature.append(is_slot(i[d]))
        if all(slot_feature):
            for i in items:
                i[d] = list(make_slot(i[d]))

    return items


def write_output_file(path, items, columns):
    """Write the output."""
    for item in items:
        for k, v in item.items():
            try:
                try:
                    if isinstance(item[k][0], tuple):
                        item[k] = ["-".join([x[0], str(x[1])])
                                   for x in item[k]]
                except IndexError:
                    pass
                item[k] = " ".join(item[k])
            except TypeError:
                pass

    d = pd.DataFrame(items)
    d[columns].to_csv(path, index=False)


def parse_parameter_file(f):
    """Parse a parameter file."""
    weights = {}
    if f.mode == 'rb':
        f = (x.decode('utf-8') for x in f)
    for line in f:
        orig, dest, pos, neg = line.lower().strip().split(",")
        weights[(orig, dest)] = [float(pos), float(neg)]

    return weights


def get_model(items_file,
              parameters,
              rla_variable,
              rla_layers,
              output_layers,
              monitor_layers,
              global_rla,
              step_size,
              decay_rate,
              minimum_activation,
              adapt_weights):
    if parameters is None:
        print("Defaulting to standard IA parameters.")
        weights = IA_WEIGHTS

    items = read_input_file(items_file)
    matrix, names = weights_to_matrix(weights)

    rla = {k: 'global' if k not in rla_layers
           else rla_variable for k in names}

    m = Builder(names,
                matrix,
                rla,
                global_rla,
                outputs=output_layers,
                monitors=monitor_layers,
                step_size=step_size,
                decay_rate=decay_rate,
                minimum=minimum_activation,
                weight_adaptation=adapt_weights).build_model(items)

    return m


def make_run(items_file,
             test_items_file,
             output_path,
             parameters,
             threshold,
             rla_variable,
             rla_layers,
             output_layers,
             monitor_layers,
             global_rla,
             step_size,
             max_cycles,
             decay_rate,
             minimum_activation,
             adapt_weights):
    """Method for running."""
    test_items = read_input_file(test_items_file)
    m = get_model(items_file,
                  parameters,
                  rla_variable,
                  rla_layers,
                  output_layers,
                  monitor_layers,
                  global_rla,
                  step_size,
                  decay_rate,
                  minimum_activation,
                  adapt_weights)

    keys_items = Counter(chain.from_iterable(test_items))
    columns = [k for k, v in keys_items.items() if v == len(test_items)]
    columns.append("cycles")

    results = m.activate_bunch(test_items,
                               max_cycles=max_cycles,
                               threshold=threshold,
                               strict=False)

    cycles = [len(x[output_layers[0]]) for x in results]
    cycles = np.array(cycles)
    right = cycles < max_cycles
    cycles[~right] = -1

    for i, c in zip(test_items, cycles):
        i["cycles"] = c

    write_output_file(output_path, test_items, columns)
