"""Base class for IA models."""
import numpy as np

from collections import defaultdict
from .layer import Layer
from tqdm import tqdm


class Network(object):
    """
    Interactive activation.

    Parameters
    ----------
    minimum : float
        A number specifying the minimum activation nodes in this network can
        have.
    step_size : float, optional, default 1.0
        The step size this network uses. Every update in the network is
        multiplied by this number. Can be used to rescale the updates.
        Note that the network is not guaranteed to give the same output under
        different step sizes; changing the step size not only changes the
        granularity of the output, but can cause it to converge to different
        results.
    decay_rate : float, optional, default .07
        The decay rate used in the update equations. The decay rate specifies
        the rate at which nodes decay back to their resting state.

    Attributes
    ----------
    layers : dict
        A dictionary of layers. Can be used to look up layers and check
        their activations by name.
    outputs : dict
        Output layers are the layers which are output when a run ends.
    monitors : dict
        Monitor layers are layers which are monitored for convergence.
    inputs : dict
        Input layers are the layers which are clamped when an input is
        presented. The input layers are automatically determined: if a layer
        has no incoming connections, we automatically assume it is an input
        layer.
    feature : set
        The names of the layers which are feature set layers. Feature set
        layers are layers which have a many-to-one mapping onto a slot-based
        layer.
    checked : bool
        Whether the model has been successfully checked.

    """

    def __init__(self,
                 minimum=-.2,
                 step_size=1.0,
                 decay_rate=.07):
        """Init function."""
        self.layers = {}
        self.minimum = np.float64(minimum)
        if not .0 < step_size <= 1.0:
            raise ValueError("Step size should be greater than 0 and smaller "
                             "or equal to 1.0, is now {}".format(step_size))
        if not -1.0 <= minimum <= .0:
            raise ValueError("Minimum should be equal to or greater than 1.0"
                             ", and smaller than or equal to 0, is now"
                             "".format(minimum))
        self.step_size = step_size
        if decay_rate <= .0:
            raise ValueError("Decay rate should be a positive number, is now"
                             "".format(decay_rate))
        self.decay_rate = np.float64(decay_rate)
        self.outputs = {}
        self.monitors = {}
        self.inputs = {}
        self.feature = {}
        self.checked = False

    def __getitem__(self, k):
        """Get a single layer by name."""
        return self.layers[k]

    def check(self):
        """Check the network by checking whether all settings are valid."""
        if not self.outputs:
            raise ValueError("You did not specify any outputs.")

        for k, v in self.layers.items():
            if v.static:
                self.inputs[k] = v

        self.checked = True

    @property
    def rla(self):
        """Summarize the resting level activations."""
        rla = {}
        for k, v in self.layers.items():
            r = v.resting
            if np.all(r == r[0]):
                rla[k] = r[0]
            else:
                rla[k] = (v.resting.min(), v.resting.mean(), v.resting.max())

        return rla

    def create_layer(self,
                     layer_name,
                     resting_activation,
                     node_names,
                     is_output=False,
                     is_monitor=False,
                     is_feature=False):
        """
        Add a layer to the network.

        Parameters
        ----------
        layer_name : str
            The name of the current layer.
        resting_activation : np.array
            The resting activation of the current layer.
        node_names : list of str
            The names of the nodes.
        is_output : bool
            Whether the layer which is added is an output layer.

        """
        layer = Layer(resting_activation,
                      node_names,
                      self.minimum,
                      self.step_size,
                      self.decay_rate,
                      name=layer_name)

        self.layers[layer_name] = layer
        if is_feature:
            self.feature[layer_name] = layer
        if is_output:
            self.outputs[layer_name] = layer
        if is_monitor:
            self.monitors[layer_name] = layer

    def _create_mask(self, x):
        """Create a valid mask given a prime."""
        mask = defaultdict(list)
        for k, v in x.items():
            if k in self.feature:
                continue
            try:
                _, idxes = zip(*v)
                max_idx = max(idxes) + 1
                for idx in range(max_idx):
                    mask[k].append(("#", idx))
            except (ValueError, TypeError):
                pass

        return self.expand(dict(mask))

    def prime(self,
              X,
              primes,
              max_cycles=30,
              prime_cycles=5,
              mask_cycles=5,
              threshold=.7,
              strict=True):
        """Priming experiment."""
        outputs = []
        if prime_cycles <= 0:
            raise ValueError("Your number of prime cycles is 0, please "
                             "raise it or use the regular activate() function")
        for x, prime in tqdm(zip(X, primes)):

            # Use "#" as mask.
            mask = self._create_mask(prime)
            out = self.activate([prime],
                                prime_cycles,
                                True,
                                threshold=1.0,
                                strict=False)[0]
            interm = self.activate([mask],
                                   mask_cycles,
                                   False,
                                   threshold=threshold,
                                   strict=False)[0]
            result = self.activate([x],
                                   max_cycles,
                                   False,
                                   threshold=threshold,
                                   strict=strict)[0]

            for x in out:
                if interm:
                    out[x] = np.concatenate([out[x], interm[x], result[x]])
                else:
                    out[x] = np.concatenate([out[x], result[x]])

            outputs.append(out)

        return outputs

    def activate(self,
                 X,
                 max_cycles=30,
                 clamp_cycles=None,
                 reset=True,
                 threshold=.7,
                 strict=True,
                 inputs=None,
                 shallow_run=False):
        """
        Activate the model by clamping an input and letting it oscillate.

        Parameters
        ----------
        X : list dictionaries.
            The inputs to the model. The dictionaries have layer names as their
            keys, and tuples of symbols as their values.
        max_cycles : int, optional, default 30
            The maximum number of cycles to run the activation for.
        clamp_cycles : int or float, optional, default None
            The number of cycles to clamp the input for. After the number of
            clamp cycles has been exceeded, the layers are unclamped, and free
            to oscillate.
            If clamp_cycles is between 0 and 1, it will be interpreted as a
            proportion of max_cycles, rounded down.
        reset : bool
            Whether to reset the activations of the network to 0 before
            clamping and activation.
        threshold : float, optional, default .7
            The activation threshold. Once one of the output layers reaches
            this activation level, the network stops running and returns the
            current activation levels as output.
        strict : bool
            Whether to halt execution if the threshold is not reached when
            max_cycles have passed.
        inputs : tuple of strings
            Use this field to override the behavior of the network and to
            specify your own inputs.
        shallow_run : bool, optiodefault False
            If a run is shallow, only the final activations are returned.

        """
        if not self.checked:
            raise ValueError("Your model is not checked.")
        if max_cycles <= 0:
            raise ValueError("max_cycles must be > 0, is now "
                             "{}".format(max_cycles))
        if threshold > 1.0 or threshold <= .0:
            raise ValueError("Threshold should be 0 < x <= 1.0, is now "
                             "{}".format(threshold))
        if clamp_cycles is None:
            clamp_cycles = max_cycles
        if clamp_cycles <= 0:
            raise ValueError("Clamp cycles should be > 0, is now "
                             "{}".format(clamp_cycles))

        if 0 < clamp_cycles < 1.0:
            clamp_cycles = max_cycles // clamp_cycles

        if inputs:
            input_layers = {k: self.layers[k] for k in inputs}
        else:
            input_layers = self.inputs

        for x in tqdm(X):

            # Reset all layers to their resting levels.
            if reset:
                self._reset()

            # Clamp the inputs
            for name, layer in input_layers.items():
                data = x[name]
                # Can be necessary if someone wants to clamp orthography
                if not isinstance(data, (tuple, set, list)):
                    data = [data]
                # Reset only the input layer to 0
                layer.reset()
                layer.activations[[layer.name2idx[p] for p in data]] = 1
                layer.clamped = True

            # Prepare the activations
            activations = defaultdict(list)

            for idx in range(max_cycles):

                if clamp_cycles is not None and idx == clamp_cycles:
                    for name, layer in self.layers.items():
                        layer.clamped = False

                # Let the network oscillate once.
                self._single_cycle()

                # Copy to the output buffer.
                for k, l in self.outputs.items():
                    if shallow_run:
                        activations[k].append(list(l.active()))
                    else:
                        act = np.copy(l.activations)
                        activations[k].append(act)

                # Check the monitor layers for convergence
                if self.monitors:
                    if np.all([np.any(l.activations > threshold)
                               for l in self.monitors.values()]):
                        break
            else:

                # If the maximum number of cycles has been reached, we
                # might throw an error, depending on the value of the strict
                # flag.
                if strict:
                    max_activation = max([max(x.activations)
                                          for x in self.monitors.values()])
                    raise ValueError("Maximum cycles reached, maximum "
                                     "activation was {}, input was {}"
                                     "".format(max_activation, x))

            if shallow_run:
                yield activations
            else:
                yield {k: np.array(v) for k, v in activations.items()}

    def _single_cycle(self):
        """Perform a single pass through the network."""
        updates = {}

        # The updates are synchronous, so all updates are first calculated,
        # and then applied simultaneously.
        for k, layer in self.layers.items():
            # Static layers don't get updated.
            if layer.static or layer.clamped:
                continue
            updates[k] = layer.activate()
        for k, v in updates.items():
            self.layers[k].activations[:] += v
            self.layers[k].activations = np.clip(self.layers[k].activations,
                                                 a_min=self.minimum,
                                                 a_max=1.0)

    def _collect_net(self):
        """Convenience function for diagnostic."""
        net = {}
        for k, layer in self.layers.items():
            # Static layers don't have net input.
            if layer.static or layer.clamped:
                continue
            net[k] = layer.net_input()

        return net

    def _reset(self):
        """Reset the activation of all nodes back to their resting levels."""
        for layer in self.layers.values():
            self.clamped = False
            layer.reset()

    def connect_layers(self, from_name, to_name, weights):
        """
        Connect 2 layers to each other using a weight matrix.

        Parameters
        ----------
        from_name : str
            The name of the originating layer.
        to_name : str
            The name of the terminating layer.
        weights : np.array
            The weight matrix used to connect both layers.

        """
        to_layer = self.layers[to_name]
        from_layer = self.layers[from_name]
        to_layer.add_from_connection(from_layer, weights)
        from_layer.add_to_connection(to_layer)

    def __repr__(self):
        """Print the metameric."""
        string = "Network with {} layers\n".format(len(self.layers))
        string += "\n".join(["{}:\t {}".format(a, str(b))
                             for a, b in sorted(self.layers.items())])

        return string

    def diagnostic_run(self,
                       X,
                       max_cycles=30,
                       threshold=.7):
        """Do a run while tracking all positive and negative connections."""
        # Make a simpler thing, we are not interested in error checking.
        strengths = []
        for x in X:
            # Always reset
            s = []
            self._reset()
            # Clamp the inputs
            for name, layer in self.inputs.items():
                data = x[name]
                # Can be necessary if someone wants to clamp orthography
                if not isinstance(data, (tuple, set, list)):
                    data = [data]
                # Reset only the input layer to 0
                layer.reset()
                layer.activations[[layer.name2idx[p] for p in data]] = 1
                layer.clamped = True

            for idx in range(max_cycles):

                self._single_cycle()
                # After each cycle, collect the incoming and outgoing
                # connections.
                s.append(self._collect_net())
                # Check the monitor layers for convergence
                if self.monitors:
                    if np.all([np.any(l.activations > threshold)
                               for l in self.monitors.values()]):
                        break

            strengths.append(s)

        return strengths

    def expand(self, item, overwrite=False):
        """Expands an item for which we only have partial data."""
        for k, v in self.layers.items():
            # tracks whether # is a mask.
            mask = None
            if k in item and not overwrite:
                continue
            for c in v._to_connections:
                k2 = c.name
                if k2 not in item:
                    continue
                i = []
                for x in item[k2]:
                    try:
                        i.append(c.name2idx[x])
                    except KeyError as e:
                        if x[0] == "#":
                            # assign current index to mask
                            mask = x[1]
                            continue
                        else:
                            raise e
                mtr = c.weight_matrices[k]
                if k not in self.feature and k2 not in self.feature:
                    idxes = defaultdict(set)
                    a, b = np.nonzero(mtr[:, i] > 0)
                    for x, y in zip(a, b):
                        idxes[y].add(x)
                    i = list(idxes.values())
                    if not i:
                        idxes = i
                    else:
                        idxes = set.intersection(*list(idxes.values()))
                else:
                    idxes = np.nonzero(mtr[:, i] > 0)[0]
                item[k] = {v.idx2name[x] for x in idxes}
                if mask is not None and k in self.feature:
                    feats = [(x, y) for x, y in v.node_names
                             if y == mask and x.endswith("neg")]
                    if not feats:
                        raise ValueError("No data for {} at layer {}"
                                         "".format(mask, k))
                    item[k].update(feats)

                item[k] = sorted(item[k], key=lambda x: x[-1])

        return item
