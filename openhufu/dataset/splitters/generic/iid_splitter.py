import abc
from openhufu.dataset.splitters.base_splitter import BaseSplitter
import numpy as np

class IIDSplitter(BaseSplitter):
    def __init__(self, n_clients, seed=None):
        super(IIDSplitter, self).__init__(n_clients, seed)
        if seed is not None:
            self.seed = seed
            np.random.seed(seed)

    @abc.abstractmethod
    def subset(self, dataset, idxs: list):
        raise NotImplementedError

    def __call__(self, dataset):
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        idx_slices = np.array_split(indices, self.n_clients)
        subsets = [self.subset(dataset, [int(idx) for idx in idxs]) for idxs in idx_slices]
        return subsets