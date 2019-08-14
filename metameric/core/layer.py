"""Layers in competitive networks."""
import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()}) # noqa
from .metric import strength


class Layer(object):
    """
    A single layer in a competitive network.

    Parameters
    ----------
    resting : np.array
        The resting level activations of the nodes in the current network
    node_names : list of string
        The names of the nodes in the network. These are only used in
        identifying the nodes, not in computation of values.
    minimum : float
        The minimum activation of nodes in this layer.
    step_size : float
        A constant with which all activations are multiplied. Making this
        number smaller results in more fine-grained results, but also
        intensifies the computation required.
    decay_rate : float
        The rate at which activations decay back to their resting state.
        In general, this number should be small (.07) to obtain interesting
        effects.

    Attributes
    ----------
    connections : list of Layer
        A list of Layers with which this layer is connected.
    weights : np.array
        A tmatrix which details how the incoming connections influence the
        activation of the current layer.
    activations : np.array
        The activation of the current layer at the current time.
    name : string
        The name of the current layer.
    name2idx : dict
        Lookup from name to neuron index.
    idx2name : dict
        Lookup from neuron index to item name.

    """

    def __init__(self,
                 resting,
                 node_names,
                 minimum,
                 step_size,
                 decay_rate,
                 name=""):
        """Init function."""
        if len(resting) != len(node_names):
            raise ValueError("Node names and resting level activations do "
                             "not have the same length: {} and {}"
                             "".format(len(resting), len(node_names)))
        self.activations = np.zeros_like(resting, dtype=np.float64)
        self.name2idx = {k: idx for idx, k in enumerate(node_names)}
        self.idx2name = {v: k for k, v in self.name2idx.items()}
        self._from_connections = []
        self._to_connections = []
        self.weights = []
        self.resting = np.copy(resting).astype(np.float64)
        self.minimum = minimum
        self.decay_rate = decay_rate
        self.name = name
        self.step_size = step_size
        self.clamped = False
        self.ext_input = np.zeros_like(resting, dtype=np.float64)

    @property
    def connections(self):
        """Get all connections and their names."""
        return {l.name: l for l in self._from_connections}

    def active(self):
        """Get all currently active nodes and their names."""
        for x in np.flatnonzero(self.activations > 0):
            yield self.idx2name[x], self.activations[x]

    @property
    def node_names(self):
        """Return all node names."""
        _, names = zip(*sorted(self.idx2name.items()))
        return names

    def add_from_connection(self, layer, weights):
        """
        Add a connection to the layer.

        Parameters
        ----------
        layer : Layer
            The layer to be connected to the current layer.
        weights : np.array
            A M * N matrix, where M is the dimensionality of the incoming
            connection, and N is the dimensionality of the current layer's
            activations.

        """
        if weights.shape[0] != layer.activations.shape[0]:
            raise ValueError("Transfer matrix is not correct shape.")
        if weights.shape[1] != self.activations.shape[0]:
            raise ValueError("Transfer matrix is not correct shape.")

        self._from_connections.append(layer)
        self.weights.append(weights)

    def add_to_connection(self, layer):
        """
        Add a connection to the layer.

        _to_connections do not have a functional purpose, and serve
        as a way of tracking which layer is connected to which.

        Parameters
        ----------
        layer : Layer
            The layer to be connected to the current layer.

        """
        self._to_connections.append(layer)

    @property
    def weight_matrices(self):
        """Return each weights matrix individually."""
        mtrs = {}
        for x, w in zip(self._from_connections, self.weights):
            mtrs[x.name] = w

        return mtrs

    @property
    def static(self):
        """Whether the activations should be updated."""
        return not self.weights

    def reset(self):
        """Reset the activations to resting level."""
        self.activations = np.copy(self.resting)
        self.ext_input *= 0

    def net_input(self):
        """
        Get the net input for diagnostic purposes.

        NOTE: This function should not be used in a serious manner, as it is
        much slower than the cythonized functions.

        Use the core.metric functions for anything involving speed.
        """
        net = {}
        for mtr, layer in zip(self.weights, self._from_connections):
            a = layer.activations.clip(.0, 1.0)
            net[layer.name] = a.dot(mtr)

        return net

    def activate(self):
        """
        Activate the layer.

        This function calculates the net input based on the state of the
        layers to which it is connected, and then calculates the updated state
        of the current layer based on the net input and the current activation.

        The update of a single neuron is given by the following equation.

            net_i = sum([x_j * w_ij if x_j > 0])
            add_i = (1.0 - net_i) if net_i > 0 else (net_i - minimum)
            delta_i = add_i - (decay * (activation_i - resting_i))

        Returns
        -------
        delta : np.array
            The change in activation for each neuron.

        """
        if not self._from_connections:
            return np.zeros_like(self.activations)
        return strength(np.copy(self.ext_input),
                        self.activations,
                        self.resting,
                        [x.activations for x in self._from_connections],
                        self.weights,
                        self.minimum,
                        self.decay_rate,
                        self.step_size)

    def __repr__(self):
        """Return a description of the layer."""
        return "Layer object with {} nodes, {} "\
               "connections.".format(len(self.activations),
                                     len(self.connections))
