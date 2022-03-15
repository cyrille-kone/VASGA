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
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler as SC

CONTINUOUS_FEATURES = [
    'Elevation',
    'Aspect',
    'Slope',
    'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways',
    'Hillshade_9am',
    'Hillshade_Noon',
    'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points',
]
CATEGORICAL_FEATURES = [
    'Wilderness_Area_0',
    'Wilderness_Area_1',
    'Wilderness_Area_2',
    'Wilderness_Area_3',
    'Soil_Type_0',
    'Soil_Type_1',
    'Soil_Type_2',
    'Soil_Type_3',
    'Soil_Type_4',
    'Soil_Type_5',
    'Soil_Type_6',
    'Soil_Type_7',
    'Soil_Type_8',
    'Soil_Type_9',
    'Soil_Type_10',
    'Soil_Type_11',
    'Soil_Type_12',
    'Soil_Type_13',
    'Soil_Type_14',
    'Soil_Type_15',
    'Soil_Type_16',
    'Soil_Type_17',
    'Soil_Type_18',
    'Soil_Type_19',
    'Soil_Type_20',
    'Soil_Type_21',
    'Soil_Type_22',
    'Soil_Type_23',
    'Soil_Type_24',
    'Soil_Type_25',
    'Soil_Type_26',
    'Soil_Type_27',
    'Soil_Type_28',
    'Soil_Type_29',
    'Soil_Type_30',
    'Soil_Type_31',
    'Soil_Type_32',
    'Soil_Type_33',
    'Soil_Type_34',
    'Soil_Type_35',
    'Soil_Type_36',
    'Soil_Type_37',
    'Soil_Type_38',
    'Soil_Type_39']

# data dir
# TODO
DATA_DIR = Path("/Users/cyrille/PycharmProjects/VASGA/data/.covtype")
DATA_DIR.mkdir(exist_ok=True)

def preprocess(covtype):
    r"""
    Preprocess the data set
    Parameters
    ----------
    :param covtype: Bunch instance dataset
    Returns
    -------
    data, target
    """
    sc = SC()
    df = pd.DataFrame(covtype.data, columns=covtype.feature_names)
    data = [sc.fit_transform(df[CONTINUOUS_FEATURES].values), df[CATEGORICAL_FEATURES].values]
    data = np.concatenate(data, axis=1)
    return data, covtype.target - 1

class ForestCoverType(thd.Dataset):
    r'''Implement a custom dataloader'''

    def __init__(self) -> None:
        covtype = fetch_covtype(data_home=DATA_DIR)
        self.data, self.targets = preprocess(covtype)
        self.targets = self.targets.astype(int)
        self.data = self.data.astype(np.float32)

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
