# coding=utf-8
"""
PyCharm Editor
Author @git cyrille-kone & geoffroyO
"""
import torch as th
import numpy as np
import pandas as pd
from pathlib import Path
import torch.utils.data as thd
from sklearn.preprocessing import StandardScaler as SC
from sklearn.datasets import make_classification as mkc


class ToyDataset(thd.Dataset):
    r'''Implement a custom dataloader'''

    def __init__(self, n_samples=50000, n_classes=2, n_features=2, n_clusters_per_class=1, n_redundant=0, **kw) -> None:
        self.data, self.targets = mkc(n_samples=n_samples, n_classes=n_classes, n_features=n_features,
                                      n_clusters_per_class=n_clusters_per_class, n_redundant=n_redundant, **kw)
        self.targets = self.targets.astype(int)
        self.data = SC().fit_transform(self.data.astype(np.float32))

    def __getitem__(self, idx) -> tuple:
        data, target = self.data[idx], self.targets[idx]
        return data, target

    # override
    def __len__(self) -> int:
        return len(self.data)

    def random_split(self, lengths=None):
        r'''Split the data set into parts '''
        lengths = [0.8, 0.2] if lengths is None else lengths
        assert sum(lengths) == 1, " Lengths should sum up to one"
        lengths = (len(self.data) * th.tensor(lengths)).int()
        if sum(lengths) != len(self.data):
            lengths[0] += len(self.data) - sum(lengths)
        return thd.random_split(self, lengths)
