"""Interface for building monomodels."""
import numpy as np

from ..core import Network
from itertools import chain, product
from collections import Counter, defaultdict


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
                 layer_names,
                 weight_matrix,
                 rla,
                 global_rla,
                 outputs=(),
                 monitors=(),
                 minimum=-.2,
                 step_size=1.0,
                 decay_rate=.07,
                 weight_adaptation=True):
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

        self.layer_names = layer_names
        self.weight_matrix = weight_matrix
        self.rla = rla
        self.global_rla = global_rla
        self.outputs = outputs
        self.monitors = monitors
        self.minimum = minimum
        self.step_size = step_size
        self.decay_rate = decay_rate
        self.weight_adaptation = weight_adaptation

    def is_sequence(self, item):
        """Check whether a key is a sequence."""
        for x in item:
            if len(x) != 2 or not isinstance(x[1], int):
                return False
        return True

    def is_feature(self, item):
        """Check whether a key is a feature."""
        if not self.is_sequence(item):
            return False
        a, b = zip(*item)
        return any([x > 1 for x in Counter(b).values()])

    def item_sequence(self, items, key):
        """Get items as a sequence."""
        d = chain.from_iterable([i[key] for i in items])
        if key not in self.slot_layers:
            return sorted(d)
        else:
            d, idx = zip(*d)
            self.num_slots[key] = max(max(idx)+1, self.num_slots[key])
            return sorted(d)

    def get_node_names(self, key):
        """Get the node names of a layer."""
        return self.unique_items[key]

    def sum_over(self, items, key, field_to_sum):
        """Sum over a field for a given key."""
        k_1 = {k: idx for idx, k in enumerate(self.unique_items[key])}
        sums = np.zeros(len(k_1))
        for i in items:
            f = i[field_to_sum]
            for x in i[key]:
                sums[k_1[x]] += f

        return sums

    def build_model(self, items):
        """

        Returns
        -------
        instance : Network
            An initialized network.

        """
        # Gather all unique items.
        self._check(items, self.layer_names)
        out_layers = set(self.outputs) - set(self.layer_names)
        if out_layers:
            raise ValueError("Not all outputs were in your layer names."
                             "".format(out_layers))

        # Initialize the TilapIA.
        m = Network(minimum=self.minimum,
                    step_size=self.step_size,
                    decay_rate=self.decay_rate)

        self.num_slots = defaultdict(int)

        self.feature_layers = set()
        self.slot_layers = set()
        for k in self.layer_names:
            for i in items:
                if self.is_feature(i[k]):
                    self.feature_layers.add(k)
                    break
            for i in items:
                if self.is_sequence(i[k]):
                    self.slot_layers.add(k)
                    break

        self.unique_items = {k: set(self.item_sequence(items, k))
                             for k in self.layer_names}

        for k in self.slot_layers:
            u = self.unique_items[k]
            if k not in self.feature_layers:
                u.add(" ")
                pass
            self.unique_items[k] = sorted(u)

        self.unique_items = {k: {x: idx for idx, x in enumerate(v)}
                             for k, v in self.unique_items.items()}

        # Iterate over all unique items.
        for k in sorted(self.layer_names):

            # Determine resting level activation.
            # If the resting is denoted as "global", every
            # node has the same rla.
            if self.rla[k] == 'global':
                resting = np.ones(len(self.unique_items[k])) * self.global_rla
            else:
                resting = self.sum_over(items, k, self.rla[k])
                if resting.min() < 1:
                    resting += (1 - resting.min())
                resting = np.log10(resting)
                resting /= max(resting)
                resting = self.global_rla * (1.0 - resting)

            node_names = self.unique_items[k]
            if k in self.slot_layers:
                resting = np.concatenate([resting] * self.num_slots[k])
                n = []
                for idx in range(self.num_slots[k]):
                    n.extend([(x, idx) for x in node_names])
                node_names = n

            m.create_layer(k,
                           resting,
                           node_names,
                           k in self.outputs,
                           k in self.monitors,
                           k in self.feature_layers)

        # Transfer matrix is a N * N * 2 matrix.
        for a, b in product(self.layer_names, self.layer_names):
            pos, neg = self.weight_matrix[self.layer_names.index(a),
                                          self.layer_names.index(b)]

            if not pos and not neg:
                continue

            a_slot = a in self.slot_layers
            b_slot = b in self.slot_layers

            d = a_slot or b_slot
            f = a in self.feature_layers or b in self.feature_layers

            if d and not f:
                true_num_slots = max(self.num_slots.get(a, 0),
                                     self.num_slots.get(b, 0))
                pos = pos / true_num_slots
                neg = neg * true_num_slots

            u_a = self.unique_items[a]
            u_b = self.unique_items[b]
            num_u_a = len(u_a)
            num_u_b = len(u_b)

            if a_slot and not b_slot:
                dim_a = num_u_a * self.num_slots.get(a, 1)
            else:
                dim_a = num_u_a
            if not a_slot and b_slot:
                dim_b = num_u_b * self.num_slots.get(b, 1)
            else:
                dim_b = num_u_b
            mtr = np.zeros((dim_a,
                            dim_b))
            mtr = mtr + neg
            for i in items:
                a_values = i[a]
                b_values = i[b]
                if a_slot and b_slot:
                    for ((x, i1), (y, i2)) in product(a_values, b_values):
                        if i1 != i2:
                            continue
                        mtr[(u_a[x], u_b[y])] = pos

                    if a not in self.feature_layers:
                        for x, y in b_values:
                            if x.endswith("neg"):
                                mtr[u_a[" "], u_b[x]] = pos

                    if b not in self.feature_layers:
                        for x, y in a_values:
                            if x.endswith("neg"):
                                mtr[u_a[x], u_b[" "]] = pos
                else:
                    if a_slot:
                        idx_a = [u_a[x] + (num_u_a * y) for x, y in a_values]
                    else:
                        idx_a = [u_a[x] for x in a_values]
                    if b_slot:
                        idx_b = [u_b[x] + (num_u_b * y) for x, y in b_values]
                    else:
                        idx_b = [u_b[x] for x in b_values]
                    mtr[np.ix_(idx_a, idx_b)] = pos

            if a_slot and b_slot:
                x, y = mtr.shape
                new_mtr = np.zeros((x * self.num_slots[a],
                                    y * self.num_slots[b]))
                for idx in range(self.num_slots[a]):
                    s_a, e_a = x * idx, x * (idx + 1)
                    s_b, e_b = y * idx, y * (idx + 1)
                    new_mtr[s_a:e_a, s_b:e_b] = mtr
                mtr = new_mtr

            m.connect_layers(a, b, mtr)

        m.compile()
        return m

    def _check(self, items, layer_names):
        """Check whether the items are valid."""
        all_keys = set(chain.from_iterable([i.keys() for i in items]))
        if set(layer_names) - all_keys:
            raise ValueError("Not all layer names were present in the items: "
                             "{}".format(set(layer_names) - all_keys))
