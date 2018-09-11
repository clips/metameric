"""Samples according to a binned frequency."""
import numpy as np
from collections import defaultdict


class BinnedSampler(object):
    """Sample by binning."""
    def __init__(self, items, frequencies, bin_width=1.0):
        """init the sampler."""
        # Frequencies are assumed to be logged.
        # Bin words by their frequency.
        self.total = len(items)
        w = defaultdict(list)
        for x, freq in zip(items, frequencies):
            w[int(freq // bin_width)].append(x)
        self.words = [np.array(w[idx]) for idx in range(len(w))]
        self.lengths = np.array([len(x) for x in self.words])

    def sample(self, n):
        """Sample."""
        num_to_sample = n / self.total
        sample_bins = (num_to_sample * self.lengths).astype(np.int32)
        items = []
        for num, v in zip(sample_bins, self.words):
            idxes = np.random.choice(len(v), num, replace=False)
            items.extend(v[idxes])

        return items
