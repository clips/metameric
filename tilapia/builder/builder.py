"""Interface for building monomodels."""
import numpy as np

from ..core import Network
from itertools import chain, product
from collections import Counter


class Item(object):
    """An item class."""
    def __init__(self):
        self.data = {}
        self.meta = {}

    def __getitem__(self, key):
        return self.data[key]

    def add_meta(self, key, value):
        self.meta[key] = value

    def add_data(self, key, value):
        self.data[key] = value

    def get_correspondence(self, source, dest):
        """Get the correspondence between items."""
        if source not in self.data or dest not in self.data:
            raise ValueError("{} or {} not present".format(source, dest))
        s = self.data[source]
        d = self.data[dest]
        if self.is_sequence(source) and self.is_sequence(dest):
            return [(x[0], y[0]) for x, y in
                    filter(lambda x: x[0][1] == x[1][1], product(s, d))]
        else:
            return product(s, d)

    def is_decomposed(self, key):
        """Check whether the layer is decomposed."""
        return not self.is_feature(key) and self.is_sequence(key)

    def is_sequence(self, key):
        """Check whether a key is a sequence."""
        return all([len(x) == 2 and isinstance(x[1], int)
                    for x in self.data[key]])

    def is_feature(self, key):
        """Check whether a key is a feature."""
        if not self.is_sequence(key):
            return False
        a, b = zip(*self.data[key])
        return any([x > 1 for x in Counter(b).values()])


class Builder(object):
    """
    Parameters
    ----------
    store : Store
        A store object containing the data on which to build the model.
    layer_names : list
        A list of strings, containing the names of the layers to be
        created.
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

    """

    def __init__(self,
                 items,
                 layer_names,
                 weight_matrix,
                 rla,
                 global_rla,
                 outputs=(),
                 monitors=(),
                 minimum=-.2,
                 step_size=1.0,
                 decay_rate=.07):
        """Build a model out of a set of items."""
        if weight_matrix.shape[0] != weight_matrix.shape[1]:
            raise ValueError("Weight matrix must be square, is {}"
                             "".format(weight_matrix.shape))
        if weight_matrix.shape[0] != len(layer_names):
            raise ValueError("Weight matrix shape must be the same as the "
                             "number of names passed into the builder. The "
                             "matrix has {} rows, and you passed {} names"
                             "".format(weight_matrix.shape[0],
                                       len(layer_names)))

        # Gather all unique items.
        self._check(items, layer_names)
        out_layers = set(outputs) - set(layer_names)
        if out_layers:
            raise ValueError("Not all outputs were in your layer names."
                             "".format(out_layers))

        self.layer_names = layer_names
        self.weight_matrix = weight_matrix
        self.rla = rla
        self.global_rla = global_rla
        self.outputs = outputs
        self.monitors = monitors
        self.minimum = minimum
        self.step_size = step_size
        self.decay_rate = decay_rate

        self.items = []
        for i in items:
            it = Item()
            for k, v in i.items():
                if k in self.layer_names:
                    it.add_data(k, v)
                else:
                    it.add_meta(k, v)
            self.items.append(it)

        layers = [x.data.keys() for x in self.items]
        self.num_slots = {}
        self.layers = set(chain.from_iterable(layers))
        self.sequence = {k for k in self.layers
                         if all([i.is_sequence(k) for i in self.items])}
        self.feature = {k for k in self.layers
                        if any([i.is_feature(k) for i in self.items])}
        self.decomposed = {k for k in self.layers
                           if any([i.is_decomposed(k) for i in self.items])}
        self.unique_items = {k: list(set(self.item_sequence(k)))
                             for k in self.layers}

    def item_sequence(self, key):
        """Get items as a sequence."""
        d = chain.from_iterable([i[key] for i in self.items])
        if key not in self.sequence:
            return sorted(d)
        else:
            d, idx = zip(*d)
            self.num_slots[key] = max(idx) + 1
            res = []
            for x in range(self.num_slots[key]):
                res.extend([(d_, x) for d_ in d])
            return res

    def get_correspondence(self, key, key_b):
        """Get correspondence between a and b."""
        k_1 = {k: idx for idx, k in enumerate(self.unique_items[key])}
        k_2 = {k: idx for idx, k in enumerate(self.unique_items[key_b])}

        mtr = np.zeros((len(k_1), len(k_2))) - 1

        for i in self.items:
            for a, b in i.get_correspondence(key, key_b):
                if key in self.sequence and key_b in self.sequence:
                    k = key if key in self.decomposed else key_b
                    for l in range(self.num_slots[k]):
                        mtr[k_1[(a, l)], k_2[(b, l)]] = 1
                else:
                    mtr[k_1[a], k_2[b]] = 1

        if key in self.sequence and key_b in self.sequence:
            for (a, slot_1) in self.unique_items[key]:
                for (b, slot_2) in self.unique_items[key_b]:
                    if slot_1 != slot_2:
                        mtr[k_1[(a, slot_1)], k_2[(b, slot_2)]] = 0

        return mtr

    def get_node_names(self, key):
        """Get the node names of a layer."""
        return self.unique_items[key]

    def sum_over(self, key, field_to_sum):
        """Sum over a field for a given key."""
        k_1 = {k: idx for idx, k in enumerate(self.unique_items[key])}
        sums = np.zeros(len(k_1))
        for i in self.items:
            f = i.meta[field_to_sum]
            for x in i[key]:
                sums[k_1[x]] += f

        return sums

    def build_model(self):
        """

        Returns
        -------
        instance : Network
            An initialized network.

        """
        # Initialize the TilapIA.
        m = Network(minimum=self.minimum,
                    step_size=self.step_size,
                    decay_rate=self.decay_rate)

        # Iterate over all unique items.
        for k in self.layer_names:

            # Determine resting level activation.
            # If the resting is denoted as "global", every
            # node has the same rla.
            if self.rla[k] == 'global':
                resting = np.ones(len(self.unique_items[k])) * self.global_rla
            else:
                resting = self.sum_over(k, self.rla[k])
                if resting.min() < 1:
                    resting += (1 - resting.min())
                resting = np.log10(resting)
                resting /= max(resting)
                resting = self.global_rla * (1.0 - resting)

            node_names = self.unique_items[k]

            m.create_layer(k,
                           resting,
                           node_names,
                           k in self.outputs,
                           k in self.monitors)

        # Transfer matrix is a N * N * 2 matrix.
        for a, b in product(self.layer_names, self.layer_names):
            pos, neg = self.weight_matrix[self.layer_names.index(a),
                                          self.layer_names.index(b)]

            if not pos and not neg:
                continue

            mtr = self.get_correspondence(a, b)
            mtr[mtr == 1] = pos
            mtr[mtr == -1] = neg

            m.connect_layers(a, b, mtr)

        m.compile()
        return m

    def _check(self, items, layer_names):
        """Check whether the items are valid."""
        all_keys = set(chain.from_iterable([i.keys() for i in items]))
        if set(layer_names) - all_keys:
            raise ValueError("Not all layer names were present in the items: "
                             "{}".format(set(layer_names) - all_keys))
