"""Layers in competitive networks."""
import numpy as np
from .metric import strength_new


class Layer(object):
    """
    A single layer in a competitive network.

    Parameters
    ----------
    num_nodes : int
        The number of nodes in the current network.
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
    connections : list of np.array
        A list of arrays with which this layer is connected.
    weights : np.array
        A transfer matrix which details how the incoming connections influence
        the activation of the current layer.
    activations : np.array
        The activation of the current layer at the current time.

    """

    def __init__(self,
                 num_nodes,
                 resting,
                 node_names,
                 minimum,
                 step_size,
                 decay_rate,
                 static,
                 name=""):
        """Init function."""
        self.activations = np.zeros(num_nodes, dtype=np.float64)
        self.node_names = node_names
        self.name2idx = {k: idx for idx, k in enumerate(node_names)}
        self.idx2name = {v: k for k, v in self.name2idx.items()}
        self.connections = []
        self.weights = None
        self.resting = np.copy(resting).astype(np.float64)
        self.minimum = minimum
        self.decay_rate = decay_rate
        self.name = name
        self._static = static
        self.step_size = step_size

    def active(self):
        """Get all currently active nodes and their names."""
        for x in np.flatnonzero(self.activations > 0):
            yield self.idx2name[x], self.activations[x]

    def add_connection(self, layer, weights):
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

        self.connections.append(layer)
        if self.weights is None:
            self.weights = weights
        else:
            self.weights = np.concatenate([self.weights,
                                           weights])

    @property
    def weight_matrices(self):
        """Return each weights matrix individually."""
        prev = 0
        mtrs = []
        for x in self.connections:
            num_act = len(x.activations)
            mtrs.append(self.weights[prev:prev+num_act])
            prev += num_act

        return mtrs

    @property
    def static(self):
        """Whether the activations should be updated."""
        return self.weights is None or self._static

    def activate(self):
        """
        Activate the layer.

        This function calculates the net input based on the state of the
        layers to which it is connected, and then calculates the updated state
        of the current layer based on the net input and the current activation.

        The update of a IA layer is given by the following equation.

        delta = net_input - (decay * (activation - resting))

        where decay is a scalar, activation and resting are vectors.
        net_input is the net amount of stimulation each neuron receives.

        Returns
        -------
        delta : np.array
            The change in activation of the current layer.

        """
        if not self.connections:
            return np.zeros_like(self.activations)
        p = np.concatenate([x.activations for x in self.connections])

        return strength_new(self.activations,
                            self.resting,
                            p,
                            self.weights,
                            self.minimum,
                            self.decay_rate,
                            self.step_size)

    def __repr__(self):
        """Return a description of the layer."""
        return "Layer object with {} nodes, {} "\
               "connections.".format(len(self.activations),
                                     len(self.connections))
