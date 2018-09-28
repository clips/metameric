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
        Output layers are the layers which are checked for convergence when
        the network is run.

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
        self.compiled = False

    def __getitem__(self, k):
        """Get a single layer by name."""
        return self.layers[k]

    def compile(self):
        """Compile the network by checking whether all settings are valid."""
        if not self.outputs:
            raise ValueError("You did not specify any outputs.")
        if not self.monitors:
            raise ValueError("You did not specify any layers to monitor.")

        for k, v in self.layers.items():
            if v.static:
                self.inputs[k] = v

        if not self.inputs:
            raise ValueError("You did not specify any inputs.")

        self.compiled = True

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
                     is_monitor=False):
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
        if is_output:
            self.outputs[layer_name] = layer
        if is_monitor:
            self.monitors[layer_name] = layer

    def prime(self,
              X,
              primes,
              max_cycles=30,
              prime_cycles=5,
              threshold=.7,
              strict=True):
        """Priming experiment."""
        outputs = []
        for x, prime in tqdm(zip(X, primes)):

            out = self.activate([prime],
                                prime_cycles,
                                True,
                                threshold=1.0,
                                strict=False)[0]

            result = self.activate([x],
                                   max_cycles,
                                   False,
                                   threshold=threshold,
                                   strict=strict)[0]

            for x in out:
                out[x] = np.concatenate([out[x], result[x]])

            outputs.append(out)

        return outputs

    def activate(self,
                 X,
                 max_cycles=30,
                 reset=True,
                 threshold=.7,
                 strict=True):
        """
        Activate the model by clamping an input and letting it oscillate.

        Parameters
        ----------
        x : list dictionaries.
            The inputs to the model. The dictionaries have layer names as their
            keys, and tuples of symbols as their values.
        max_cycles : int, optional, default 30
            The maximum number of cycles to run the activation for.
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

        """
        if not self.compiled:
            raise ValueError("Your model is not compiled.")

        outputs = []

        for x in tqdm(X):

            if reset:
                self._reset()

            for name, layer in self.inputs.items():
                data = x[name]
                if not isinstance(data, (tuple, set, list)):
                    data = [data]
                layer.reset()
                layer.activations[[layer.name2idx[p] for p in data]] = 1

            activations = defaultdict(list)

            for idx in range(max_cycles):

                self._single_cycle()

                for k, l in self.outputs.items():

                    act = np.copy(l.activations)
                    activations[k].append(act)

                if np.all([np.any(l.activations > threshold)
                           for l in self.monitors.values()]):
                    break
            else:
                if strict:
                    max_activation = np.max([x.activations
                                             for x in self.monitors.values()],
                                            1)
                    raise ValueError("Maximum cycles reached, maximum "
                                     "activation was {}, input was {}"
                                     "".format(max_activation, x))

            outputs.append({k: np.array(v) for k, v in activations.items()})

        return outputs

    def _single_cycle(self):
        """Perform a single pass through the network."""
        updates = {}

        # The updates are synchronous, so all updates are first calculated,
        # and then applied simultaneously.
        for k, layer in self.layers.items():
            # Static layers don't get updated.
            if layer.static:
                continue
            updates[k] = layer.activate()
        for k, v in updates.items():
            self.layers[k].activations[:] += v

    def _reset(self):
        """Reset the activation of all nodes back to their resting levels."""
        for layer in self.layers.values():
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
        """Print the TilapIA."""
        string = "tilapIA with {} layers\n".format(len(self.layers))
        string += "\n".join(["{}:\t {}".format(a, str(b))
                             for a, b in sorted(self.layers.items())])

        return string

    def prepare(self, item):
        """Prepare an item to feature layers."""
        # TODO: add exception for mask character
        for k, v in self.inputs.items():
            # checks whether # is a mask.
            mask = None
            if k in item:
                continue
            for c in v.to_connections:
                k2 = c.name
                if k2 not in item:
                    continue
                if isinstance(item[k2], (tuple, list, set)):
                    i = []
                    for x in item[k2]:
                        try:
                            i.append(c.name2idx[x])
                        except KeyError as e:
                            if x[0] == "#":
                                mask = x[1]
                                continue
                            else:
                                raise e
                else:
                    i = [c.name2idx[item[k2]]]
                mtr = c.weight_matrices[k]
                idxes = np.nonzero(mtr[:, i] > 0)[0]
                item[k] = {v.idx2name[x] for x in idxes}
                if mask is not None:
                    item[k].update([(x, y) for x, y in v.node_names
                                    if y == mask and x.endswith("neg")])

                item[k] = sorted(item[k], key=lambda x: x[-1])

        return item
