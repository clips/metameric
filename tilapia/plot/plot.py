"""Plot activations for a model."""
import matplotlib.pyplot as plt
import numpy as np

from itertools import chain


def plot_result(result, max_cycles=None):
    """Plot the activations of a single word, and show the plot."""
    result_plot(result, max_cycles=max_cycles).show()


def result_plot(result, max_cycles=None):
    """Plot the activations of a single word."""
    keys = list(result.keys())
    if max_cycles is None:
        max_cycles = max([len(v) for v in result.values()])

    f, plots = plt.subplots(1, len(keys))

    # Necessary because subplots has a weird contract.
    if not isinstance(plots, np.ndarray):
        plots = np.array([plots])

    for idx, (key, plot) in enumerate(zip(keys, plots)):

        data = result[key]
        all_words = set(chain.from_iterable(data))
        data_dict = {w: [0] for w in all_words}

        for x in range(max_cycles):
            for w in all_words:
                data_dict[w].append(data[x].get(w, -1))

        for k, v in data_dict.items():
            plot.plot(v)
            plot.annotate(k, (max_cycles * np.random.uniform(.5, .9), v[-1]))
        plot.set_title(key)
        plot.set_ylim(.0, 1.0)
        if idx == 0:
            plot.set_ylabel("Activation")
        plot.set_xlabel("Cycles")

    return f
