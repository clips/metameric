"""Base class for IA models."""
import numpy as np

from collections import defaultdict, Iterable
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
    outputs : list
        A list of layers. This lists the output layers. Output layers are
        the only layers which are checked for convergence when the algorithm
        is run.

    """

    def __init__(self,
                 minimum=-.2,
                 step_size=1.0,
                 decay_rate=.07,
                 layertype=Layer):
        """Init function."""
        self.layers = {}
        self.minimum = np.float64(minimum)
        assert .0 < step_size <= 1.0
        assert -1.0 <= minimum < .0
        self.step_size = step_size
        assert decay_rate >= .0
        self.decay_rate = np.float64(decay_rate)
        self.outputs = {}
        self.inputs = {}
        self.layertype = layertype
        self.compiled = False

    def __getitem__(self, k):
        """Get a single layer by name."""
        return self.layers[k]

    def compile(self):
        """Compile the network by checking whether all settings are valid."""
        if not self.outputs:
            raise ValueError("You did not specify any outputs.")

        for k, v in self.layers.items():
            if v.static:
                self.inputs[k] = v

        if not self.inputs:
            raise ValueError("You did not specify any inputs.")

        self.compiled = True

    @property
    def rla(self):
        """Get the resting level activations."""
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
                     is_output=False):
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
        layer = self.layertype(len(node_names),
                               resting_activation,
                               node_names,
                               self.minimum,
                               self.step_size,
                               self.decay_rate,
                               name=layer_name)

        self.layers[layer_name] = layer
        if is_output:
            self.outputs[layer_name] = layer

    def activate_bunch(self,
                       x,
                       max_cycles=30,
                       threshold=.7,
                       strict=True):
        """Activate a list of items."""
        a = []
        for item in tqdm(x):
            a.append(self.activate(item,
                                   max_cycles,
                                   True,
                                   threshold,
                                   strict))

        return a

    def activate(self,
                 x,
                 max_cycles=30,
                 reset=True,
                 threshold=.7,
                 strict=True):
        """
        Activate the model by clamping an input and letting it oscillate.

        Parameters
        ----------
        x : list of np.arrays, optional, default ()
            The inputs which are clamped in the first timestep.
        input_layers : list of string, optional, default ()
            The names of the layers which are clamped in the first timestep.
        max_cycles : int, optional, default 30
            The maximum number of cycles to run the activation for.
        reset : bool
            Whether to reset the activations of the network to 0 before
            clamping and activation.
        threshold : float, optional, default .7
            The activation threshold. Once one of the output layers reaches
            this activation level, the network stops running and returns the
            current activation levels as output.

        """
        if not self.compiled:
            raise ValueError("Your model is not compiled.")

        if reset:
            self._reset()

        for name, layer in self.inputs.items():
            data = x[name]
            if not isinstance(data, Iterable):
                data = [data]
            layer.activations[[layer.name2idx[p] for p in data]] = 1

        activations = defaultdict(list)

        for idx in range(max_cycles):

            self._single_cycle()

            for k, l in self.outputs.items():

                act = dict(l.active())
                # If anything is active
                activations[k].append(act)

            if self.outputs:
                if np.all([np.any(x.activations > threshold)
                           for x in self.outputs.values()]):
                    break
        else:
            if strict:
                max_activation = np.max([x.activations
                                         for x in self.outputs.values()], 1)
                raise ValueError("Maximum cycles reached, maximum activation "
                                 "was {}".format(max_activation))

        return dict(activations)

    def _single_cycle(self):
        """Perform a single pass through the network."""
        updates = {}

        for k, layer in self.layers.items():
            if layer.static:
                continue
            a = layer.activate()
            updates[k] = np.copy(a)
        for k, layer in self.layers.items():
            if layer.static:
                continue
            layer.activations[:] += updates[k]
            layer.activations[:] = np.clip(layer.activations,
                                           a_min=self.minimum,
                                           a_max=1.0)

    def _reset(self):
        """Reset the activation of all nodes back to 0."""
        for layer in self.layers.values():
            layer.activations[:] = layer.resting

    def connect_layers(self, from_name, to_name, weights):
        """
        Connect 2 layers to each other using a transfer matrix.

        Parameters
        ----------
        from_name : str
            The name of the originating layer.
        to_name : str
            The name of the terminating layer.
        weights : np.array
            The transfer matrix used to connect both layers.

        """
        to_layer = self.layers[to_name]
        from_layer = self.layers[from_name]
        to_layer.add_connection(from_layer, weights)

    def __repr__(self):
        """Print the TilapIA."""
        string = "tilapIA with {} layers\n".format(len(self.layers))
        string += "\n".join(["{}:\t {}".format(a, str(b))
                             for a, b in sorted(self.layers.items())])

        return string
