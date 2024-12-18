import abc
from collections import defaultdict

from openhufu.dataset.splitters.base_splitter import BaseSplitter
import numpy as np

class LDASplitter(BaseSplitter):
    def __init__(self, n_clients, seed=None):
        super(LDASplitter, self).__init__(n_clients, seed)
        if seed is not None:
            self.seed = seed
            np.random.seed(seed)

    @abc.abstractmethod
    def subset(self, dataset, idxs):
        raise NotImplementedError

    """
    Splits the dataset into subsets for each client using LDA.

    Args:
        dataset: The dataset to be split.
        get_labels_fn (function): A function to get labels from the dataset.
        **kwargs: Additional keyword arguments.

    Returns:
        list: A list of subsets, one for each client.
    """
    def __call__(self, dataset, get_labels_fn, alpha=0.5, **kwargs):
        labels = get_labels_fn(dataset)

        uniques = sorted(list(set(labels)))
        label_mapping = {label: idx for idx, label in enumerate(uniques)}
        labels = [label_mapping[label] for label in labels]

        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)

        class_dist = np.random.dirichlet([alpha] * self.n_clients, num_classes)

        label_indices = defaultdict(list)

        for idx, label in enumerate(labels):
            label_indices[label].append(idx)

        client_indices = [[] for _ in range(self.n_clients)]
        for label, indices in label_indices.items():
            np.random.shuffle(indices)
            split_indices = np.cumsum(class_dist[label] * len(indices)).astype(int)[:-1]
            splits = np.split(indices, split_indices)
            for client_id, split in enumerate(splits):
                client_indices[client_id].extend(split)

        subsets = [self.subset(dataset, [int(idx) for idx in idxs]) for idxs in client_indices]
        return subsets
