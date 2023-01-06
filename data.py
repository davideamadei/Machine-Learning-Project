from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

@dataclass
class Dataset:
    ids: np.array
    labels: Optional[np.array]
    data: np.array
    @property
    def shape(self):
        if self.labels is None:
            labels_shape = 0
        elif len(self.labels.shape) == 1:
            labels_shape = 1
        else:
            labels_shape = self.labels.shape[1]
        return (self.data.shape[0], [self.data.shape[1], labels_shape])
    

def train_valid_test_split(dataset : Dataset, splits=(.5, .2, .3), seed=123, keep_original=False)-> (Dataset, Dataset, Dataset):
    if len(splits) != 3 or (1-1e-2 >= sum(splits) or sum(splits) >= 1+1e-2):
        raise ValueError(f"Invalid splits: {splits}")

    total_size = dataset.shape[0]
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(total_size)
    
    offsets = np.cumsum([int(splits[0]*total_size), int(splits[1]*total_size)])
    
    train_ids = permutation[:offsets[0]]
    valid_ids = permutation[offsets[0]:offsets[1]]
    test_ids  = permutation[offsets[1]:]
    
    # arrays generated are copies (due to advanced indexing)
    train_dataset = Dataset(ids=dataset.ids[train_ids], labels=dataset.labels[train_ids], data=dataset.data[train_ids])
    valid_dataset = Dataset(ids=dataset.ids[valid_ids], labels=dataset.labels[valid_ids], data=dataset.data[valid_ids])
    test_dataset  = Dataset(ids=dataset.ids[test_ids], labels=dataset.labels[test_ids], data=dataset.data[test_ids])
    
    if not keep_original:
        del dataset
    return train_dataset, valid_dataset, test_dataset
    
    
def read_ML_cup(dname, basedir="./ML_cup"):
    basedir = basedir if basedir[-1] != "/" else basedir[:-1]
    TRAIN_PATH = f"{basedir}/ML-CUP22-TR.csv"
    TEST_PATH  = f"{basedir}/ML-CUP22-TS.csv"
    
    if dname == "train":
        # pandas doesn't work for me here
        raw = np.genfromtxt(TRAIN_PATH, delimiter=',')
        dataset = Dataset(
            ids   =raw.T[0].copy(),
            labels=raw.T[-2:].T.copy(),
            data  =raw.T[1:-2].T.copy()
        )
    elif dname == "test":
        raw = np.genfromtxt(TEST_PATH, delimiter=',')
        dataset = Dataset(
            ids   =raw.T[0].copy(),
            labels=None,
            data  =raw.T[1:].T.copy()
        )
    else:
        raise ValueError(f"Invalid Dataset Name: {dname}")
    return dataset

def read_monks(number, dname, basedir="./monks"):
    basedir = basedir if basedir[-1] != "/" else basedir[:-1]
    if number not in (1,2,3):
        raise ValueError(f"Invalid Dataset Number: {number}")
    if dname not in ("train", "test"):
        raise ValueError(f"Invalid Dataset Type: {dname}")
    
    path = f"{basedir}/monks-{number}.{dname}"
    raw = pd.read_csv("monks/monks-1.train", sep=" ", header=None).drop(columns=0)
    raw[8] = raw[8].str.removeprefix("data_").astype(int)
    
    dataset = Dataset(
        ids   =raw[8].to_numpy(),
        labels=raw[1].to_numpy(),
        data  =raw[[2,3,4,5,6,7]].to_numpy()
    )
    return dataset