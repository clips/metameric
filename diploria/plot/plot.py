"""Plot activations for a model."""
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict


REDDISH = (.82, .1, .12)


def get_cmap(n, name='hsv'):
    """
    from stack overflow:
    https://stackoverflow.com/
    questions/14720331/how-to-generate-random-colors-in-matplotlib
    """
    return plt.cm.get_cmap(name, n)


def _convert_to_str(x):
    """Helper function to convert an item to a string."""
    if not x:
        return " "
    if isinstance(x[0], tuple):
        return " ".join(["{}-{}".format(x[0], x[1]) for x in x])
    else:
        return " ".join([str(x) for x in x])


def plot_result(result, node_names, max_cycles=None, minimum=-.2):
    """Plot the activations of a single word, and show the plot."""
    result_plot(result, max_cycles=max_cycles).show()


def result_plot(word,
                result,
                node_names,
                max_cycles=None,
                minimum=-.2,
                threshold=.7,
                monitors=(),
                **fig_kwargs):
    """
    Plot the activations of a single word.

    Parameters
    ----------
    result : dict of np.arrays
        The result of a single call to activate of a network.
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

    f, plots = plt.subplots(1, len(keys), **fig_kwargs)
    div = max_cycles // 4

    # Necessary because subplots has a weird contract.
    if not isinstance(plots, np.ndarray):
        plots = np.array([plots])

    for idx, (key, plot) in enumerate(zip(keys, plots)):
        data = result[key]
        names = node_names[key]

        # If the data is a list, we need to construct a new matrix.
        # This only happens if we are dealing with reduced results.
        if isinstance(data, list):
            mtr = np.zeros((len(data), len(names)))
            name2idx = {k: idx for idx, k in enumerate(names)}
            for idx, x in enumerate(data):
                for k, v in x:
                    mtr[idx, name2idx[k]] = v

            data = mtr

        # Get index of all columns which have a positive element.
        idxes = np.max(data, 0) > .0

        # Remove everything but the non-positive elements
        data = data[:, idxes]
        # Get the right names for these data columns
        names = [names[idx] for idx in np.flatnonzero(idxes)]

        intervals = defaultdict(int)
        # Get the activations at the last time step, and multiply them by 10.
        bins = np.floor(data.T[:, -1] * 10)
        # Create bins.
        for k, b in zip(names, bins):
            intervals[b] += 1
        intervals = {k: list(.8 - np.arange(0, .85, .85 / v))
                     for k, v in intervals.items()}

        cmap = get_cmap(len(names)+1, name='viridis')

        if not monitors or key in monitors:
            plot.plot(np.ones(data.shape[0]) * threshold, color=REDDISH)
        for idx, (k, v, b) in enumerate(zip(names, data.T, bins)):
            plot.plot(v, color=cmap(idx))
            interval = intervals[b].pop()
            position = int(np.floor(max_cycles * interval))
            v = v[max(0, position-div):position+div]
            if len(v) > 0:
                ypos = max(v)
            else:
                ypos = 1.0
            plot.annotate(k,
                          (position, ypos),
                          color=np.array(cmap(idx)[:3]) / 4)
        plot.set_title("{}: {}".format(key, _convert_to_str(word[key])))
        plot.set_ylim(minimum, 1.0)
        plot.set_xlim(0, max_cycles-1)
        if idx == 0:
            plot.set_ylabel("Activation")
        plot.set_xlabel("Cycles")

    return f
