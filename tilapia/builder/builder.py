"""Interface for building monomodels."""
import numpy as np

from ..core import Network
from itertools import chain, product
from collections import defaultdict


def build_model(items,
                layer_names,
                weight_matrix,
                rla,
                global_rla,
                outputs=(),
                monitors=(),
                minimum=-.2,
                step_size=1.0,
                decay_rate=.07):
    """
    Build an interactive activation model.

    Parameters
    ----------
    items : list
        A list of dictionaries containing the characteristics of each item.
    layer_names : list
        A list of strings, containing the names of the layers to be created.
    weight_matrix : np.array
        The weights from each layer to each other layer.
    rla : dict
        A dictionary specifying which fields are used to establish the variable
        RLA for each layer that has a variable RLA.
    global_rla : float
        The global RLA, used to scale the variable RLA.
    outputs : tuple
        The names of the layers which are used as output.
    monitors : tuple
        The names of the layers which are monitored for convergence.
    minimum : float
        The minimum activation.
    step_size : float
        The step size by which to scale the activation.
    decay_rate : float
        The decay rate.

    Returns
    -------
    instance : Network
        An initialized network.

    """
    if weight_matrix.shape[0] != weight_matrix.shape[1]:
        raise ValueError("Weight matrix must be square, is {}"
                         "".format(weight_matrix.shape))
    if weight_matrix.shape[0] != len(layer_names):
        raise ValueError("Weight matrix shape must be the same as the number"
                         " of names passed into the builder. The matrix has"
                         " {} rows, and you passed {} names"
                         "".format(weight_matrix.shape[0], len(layer_names)))

    # Gather all unique items.
    _check(items, layer_names)
    out_layers = set(outputs) - set(layer_names)
    if out_layers:
        raise ValueError("Not all outputs were in your layer names."
                         "".format(out_layers))

    unique_items = defaultdict(set)

    slot_layers = set()

    for key in layer_names:
        for item in items:
            try:
                try:
                    unique_items[key].add(item[key])
                except TypeError:
                    unique_items[key].update(item[key])
                    slot_layers.add(key)
            except KeyError:
                pass

        unique_items[key] = {k: idx for idx, k in
                             enumerate(unique_items[key])}

    # Initialize the Skeleton.
    s = Network(minimum=minimum,
                step_size=step_size,
                decay_rate=decay_rate)

    # Iterate over all unique items.
    for key, local_items in unique_items.items():

        # Determine resting level activation.
        # If the resting is denoted as "global", every
        # node has the same rla.
        if rla[key] == 'global':
            resting = np.ones(len(local_items)) * global_rla
        else:
            # Determine resting level activation
            resting_variable = rla[key]
            resting = np.zeros(len(local_items))
            for item in items:
                value = item[key]
                idx = local_items[value]
                resting[idx] += float(item[resting_variable])
            if resting.min() < 1:
                resting += (1 - resting.min())
            resting = np.log10(resting)
            resting /= max(resting)
            resting = global_rla * (1.0 - resting)

        node_names, _ = zip(*sorted(local_items.items(), key=lambda x: x[1]))

        s.create_layer(key,
                       resting,
                       node_names,
                       key in outputs,
                       key in monitors)

    # Transfer matrix is a N * N * 2 matrix.
    for a, b in product(layer_names, layer_names):
        pos, neg = weight_matrix[layer_names.index(a),
                                 layer_names.index(b)]

        if not pos and not neg:
            continue

        lookup_1 = unique_items[a]
        lookup_2 = unique_items[b]

        mtr = np.zeros((len(lookup_1), len(lookup_2))) + neg

        for i in items:
            indices_a = []
            indices_b = []
            if a in slot_layers:
                for x in i[a]:
                    indices_a.append(lookup_1[x])
            else:
                indices_a.append(lookup_1[i[a]])
            if b in slot_layers:
                for x in i[b]:
                    indices_b.append(lookup_2[x])
            else:
                indices_b.append(lookup_2[i[b]])

            x, y = zip(*tuple(product(indices_a, indices_b)))
            mtr[x, y] = pos

        if a in slot_layers and b in slot_layers:
            for name, idx in s.layers[a].name2idx.items():
                n = name.split("-")[-1]
                for name_2, idx_2 in s.layers[b].name2idx.items():
                    if n != name_2.split("-")[-1]:
                        mtr[idx, idx_2] = 0

        s.connect_layers(a, b, mtr)

    s.compile()

    return s


def _check(items, layer_names):
    """Check whether the items are valid."""
    all_keys = set(chain.from_iterable([i.keys() for i in items]))
    if set(layer_names) - all_keys:
        raise ValueError("Not all layer names were present in the items: "
                         "{}".format(set(layer_names) - all_keys))
