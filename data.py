from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

@dataclass
class Dataset:
    ids: np.ndarray
    labels: Optional[np.ndarray]
    data: np.ndarray
    @property
    def shape(self):
        if self.labels is None:
            labels_shape = 0
        elif len(self.labels.shape) == 1:
            labels_shape = 1
        else:
            labels_shape = self.labels.shape[1]
        return (self.data.shape[0], [self.data.shape[1], labels_shape])
    
    
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