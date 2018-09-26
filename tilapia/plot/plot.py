"""Plot activations for a model."""
import matplotlib.pyplot as plt
import numpy as np


def plot_result(result, node_names, max_cycles=None, minimum=-.2):
    """Plot the activations of a single word, and show the plot."""
    result_plot(result, max_cycles=max_cycles).show()


def result_plot(result, node_names, max_cycles=None, minimum=-.2):
    """
    Plot the activations of a single word.

    Parameters
    ----------
    result : dict of np.arrays
        The result of a single call to activate of a tilapia model.
        The keys of the dictionary are layer names, and the arrays are
        activations over time for each node in that layer.
    node_names : dict
        The names of all nodes in each layers in result.
        The key is again the layer name, and each dict is a sorted list of
        names for each node
    max_cycles : int or None, default None
        The maximum number of cycles. If set to None, the maximum number of
        cycles is decided by the data.
    minimum : float
        The minimum activation of the model.
        Used to make sure the graph prints nicely.

    """
    keys = list(result.keys())
    if max_cycles is None:
        max_cycles = max([len(v) for v in result.values()])

    f, plots = plt.subplots(1, len(keys))

    # Necessary because subplots has a weird contract.
    if not isinstance(plots, np.ndarray):
        plots = np.array([plots])

    for idx, (key, plot) in enumerate(zip(keys, plots)):
        data = result[key]
        names = node_names[key]

        idxes = np.max(data, 0) > .0

        data = data[:, idxes]
        names = [names[idx] for idx in np.flatnonzero(idxes)]

        for k, v in zip(names, data.T):
            plot.plot(v)
            plot.annotate(k, (max_cycles * np.random.uniform(.5, .9), v[-1]))
        plot.set_title(key)
        plot.set_ylim(minimum, 1.0)
        if idx == 0:
            plot.set_ylabel("Activation")
        plot.set_xlabel("Cycles")

    return f
