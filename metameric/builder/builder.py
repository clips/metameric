"""Interface for building monomodels."""
import numpy as np

from ..core import Network
from itertools import chain, product
from collections import Counter, defaultdict


class MetaMericError(Exception):

    pass


class Builder(object):
    """
    A factory class that builds networks.

    A Builder takes as input a bunch of basic parameters in the constructor,
    and then creates a network based on the data which is put into the model.

    In machine learning terms, the builder can be thought of as a trainer,
    which returns a fully trained model based on the input data.

    Parameters
    ----------
    weights : dict
        A dictionary with tuples as keys. The first item of the tuple is the
        from layer, the second the to layer. Each value is also a tuple, the
        first value of which is the positive weight, and the second value of
        which is the negative weight.

        Example:
            {("orthography", "letters"): [.01, -.01]}
    rla : dict
        A dictionary specifying which fields are used to establish the variable
        RLA for each layer that has a variable RLA.
    global_rla : float
        The global RLA. This value is assigned to all non-variable RLA layers,
        and is used to scale the global RLA layers.
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
                 weights,
                 rla=None,
                 global_rla=-.05,
                 outputs=(),
                 monitors=(),
                 minimum=-.2,
                 step_size=1.0,
                 decay_rate=.07,
                 weight_adaptation=True):
        """Build a model out of a set of items."""
        self.layer_names = sorted(set(chain.from_iterable(weights.keys())))
        self.weights = weights
        self.rla = defaultdict(lambda: "global")
        if rla:
            self.rla.update(rla)
        self.global_rla = global_rla
        if isinstance(outputs, str):
            outputs = (outputs,)
        if isinstance(monitors, str):
            monitors = (monitors,)

        self.outputs = outputs
        if not monitors:
            self.monitors = outputs
        else:
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

    def sum_over(self, items, key, field_to_sum):
        """Sum over a field for a given key."""
        k_1 = self.unique_items[key]
        sums = np.zeros(len(k_1))
        for i in items:
            try:
                f = i[field_to_sum]
            except KeyError:
                raise MetaMericError("The RLA variable {} was not in "
                                     "all of your items".format(field_to_sum))
            try:
                for x in i[key]:
                    sums[k_1[x]] += f
            except KeyError:
                raise MetaMericError("The RLA field {} was not in all of your "
                                     "items.".format(key))

        return sums

    def _check(self, items, layer_names):
        """Check whether the items are valid."""
        all_keys = set(chain.from_iterable([i.keys() for i in items]))
        diff = set(layer_names) - all_keys
        if diff:
            z = ",".join(diff)
            raise MetaMericError("{} were selected as layer names, but not "
                                 "present in your items".format(z))

    def build_model(self, items):
        """
        Builds a network by iterating over all items and building layers.

        Parameters
        ----------
        items : list
            A list of dicts, where each dictionary has all the layers of the
            model as keys.

        Returns
        -------
        instance : Network
            An initialized network.

        """
        # Gather all unique items.
        self._check(items, self.layer_names)
        out_layers = set(self.outputs) - set(self.layer_names)
        if out_layers:
            raise MetaMericError("{} were selected as output layers, but were "
                                 "not in the layer names: {}"
                                 "".format(out_layers, self.layer_names))

        rla_layers = {k for k, v in self.rla.items() if v != "global"}
        rla_layers -= set(self.layer_names)
        if rla_layers:
            raise MetaMericError("{} were selected as rla layers, but were "
                                 "not in the layer names: {}"
                                 "".format(rla_layers, self.layer_names))

        # Initialize the metameric.
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

        # Take care of sorting
        self.unique_items = {k: {x: idx for idx, x in enumerate(sorted(v))}
                             for k, v in self.unique_items.items()}

        # Iterate over all unique items.
        for k in self.layer_names:

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

            node_names, _ = zip(*sorted(self.unique_items[k].items(),
                                        key=lambda x: x[1]))
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

        # a and b are keys.
        for a, b in product(self.layer_names, self.layer_names):
            try:
                pos, neg = self.weights[(a, b)]
            except KeyError:
                continue

            # This prevents the creation of layers with all zero weights.
            if not pos and not neg:
                continue

            # Check whether the layers are slot-based layers.
            a_slot = a in self.slot_layers
            b_slot = b in self.slot_layers

            f = a in self.feature_layers or b in self.feature_layers

            # If one or both are slots, and none are feature layers, adapt
            # the weights to the length of the longest input.
            # Gets overridden by the weight adaptation switch.
            if self.weight_adaptation and (a_slot or b_slot) and not f:
                true_num_slots = max(self.num_slots.get(a, 1),
                                     self.num_slots.get(b, 1))
                pos = pos / true_num_slots
                neg = neg * true_num_slots

            # Note all unique items and their number.
            u_a = self.unique_items[a]
            u_b = self.unique_items[b]
            num_u_a = len(u_a)
            num_u_b = len(u_b)

            # Form the matrices.
            if a_slot and not b_slot:
                dim_a = num_u_a * self.num_slots.get(a, 1)
            else:
                dim_a = num_u_a
            if not a_slot and b_slot:
                dim_b = num_u_b * self.num_slots.get(b, 1)
            else:
                dim_b = num_u_b

            # Create the matrix.
            mtr = np.zeros((dim_a,
                            dim_b))

            # By default, connections are negative.
            mtr = mtr + neg
            # Iterate over items to set the weights
            for i in items:
                a_values = i[a]
                b_values = i[b]
                if a_slot and b_slot:
                    # If both layers are slot layers, we can only link
                    # items with the same slot index together.
                    for ((x, i1), (y, i2)) in product(a_values, b_values):
                        if i1 != i2:
                            continue
                        mtr[(u_a[x], u_b[y])] = pos

                    # Explicitly add the space character.
                    # and set its weights
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

            # If both layers are slot-based, only items with the same slot
            # number can be connected.
            # So cells of unconnected items have to be explicitly set to 0.
            # if we don't do this, every item would have inhibitory connections
            # to other items in other slots.
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

        # Check whether the model is valid
        m.check()
        return m
