# coding=utf-8
"""
PyCharm Editor
Author @git cyrille-kone & geoffroyO
"""
import torch as th
import numpy as np
from pathlib import Path
import torch.utils.data as thd
from sklearn.datasets import fetch_covtype

# data dir
DATA_DIR = Path("./data/.covtype")
DATA_DIR.mkdir(exist_ok=True)


class ForestCoverType(thd.Dataset):
    r'''Implement a custom dataloader'''

    def __init__(self) -> None:
        self.data, self.targets = fetch_covtype(data_home=DATA_DIR, return_X_y=True)

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


if __name__ == "__main__":
    print("Import Forest CoverType dataset")
    fct = ForestCoverType()
    fct_train, fct_val = fct.random_split()
    fct_loader = thd.DataLoader(fct, batch_size=10)
