# python libraries
import numpy as np
import pandas as pd
from typing import List

# local libraries
from ..utils import Dataset

# all functions and classes implemented in this file
__all__ = ["train_valid_split", "train_valid_test_split", "read_ML_cup", "read_monks", "onehot_encoding"]


def train_valid_split(
    dataset: Dataset,
    splits: tuple[int, int] = (0.7, 0.3),
    seed=123,
    keep_original=False,
) -> tuple[Dataset, Dataset]:
    """Splits dataset into training and validation.

    Parameters
    ----------
    dataset : Dataset
        dataset to partition
    splits : tuple, optional
        split proportions, by default (.7, .3)
    seed : int, optional
        to permuting the data, by default 123
    keep_original : bool, optional
        if False the original dataset is deleted, by default False

    Returns
    -------
    tuple[Dataset, Dataset]
        training, validation

    Raises
    ------
    ValueError
        in case splits is not a valid 2-partition
    """
    if len(splits) != 2 or (1 - 1e-2 >= sum(splits) or sum(splits) >= 1 + 1e-2):
        raise ValueError(f"Invalid splits: {splits}")

    total_size = dataset.shape[0]
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(total_size)

    offset = int(splits[0] * total_size)

    train_ids = permutation[:offset]
    valid_ids = permutation[offset:]

    # arrays generated are copies (due to advanced indexing)
    train_dataset = Dataset(
        ids=dataset.ids[train_ids],
        labels=dataset.labels[train_ids],
        data=dataset.data[train_ids],
    )
    valid_dataset = Dataset(
        ids=dataset.ids[valid_ids],
        labels=dataset.labels[valid_ids],
        data=dataset.data[valid_ids],
    )

    if not keep_original:
        del dataset
    return train_dataset, valid_dataset


def train_valid_test_split(
    dataset: Dataset,
    splits: tuple[int, int, int] = (0.5, 0.2, 0.3),
    seed=123,
    keep_original=False,
) -> tuple[Dataset, Dataset, Dataset]:
    """Splits dataset into training, validation and testing.

    Parameters
    ----------
    dataset : Dataset
        dataset to partition
    splits : tuple, optional
        split proportions, by default (.5, .2, .3)
    seed : int, optional
        to permuting the data, by default 123
    keep_original : bool, optional
        if False the original dataset is deleted, by default False

    Returns
    -------
    tuple[Dataset, Dataset, Dataset]
        training, validation, testing

    Raises
    ------
    ValueError
        if splits is not a valid 3-partition
    """
    if len(splits) != 3 or (1 - 1e-2 >= sum(splits) or sum(splits) >= 1 + 1e-2):
        raise ValueError(f"Invalid splits: {splits}")

    total_size = dataset.shape[0]
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(total_size)

    offsets = np.cumsum([int(splits[0] * total_size), int(splits[1] * total_size)])

    train_ids = permutation[: offsets[0]]
    valid_ids = permutation[offsets[0] : offsets[1]]
    test_ids = permutation[offsets[1] :]

    # arrays generated are copies (due to advanced indexing)
    train_dataset = Dataset(
        ids=dataset.ids[train_ids],
        labels=dataset.labels[train_ids],
        data=dataset.data[train_ids],
    )
    valid_dataset = Dataset(
        ids=dataset.ids[valid_ids],
        labels=dataset.labels[valid_ids],
        data=dataset.data[valid_ids],
    )
    test_dataset = Dataset(
        ids=dataset.ids[test_ids],
        labels=dataset.labels[test_ids],
        data=dataset.data[test_ids],
    )

    if not keep_original:
        del dataset
    return train_dataset, valid_dataset, test_dataset


def read_ML_cup(dname: str, basedir="./ML_cup") -> Dataset:
    """Loads ML cup dataset and generates a Dataset class.

    Parameters
    ----------
    dname : str
        Either "train" to load the training dataset, or "test"
        to load the testing dataset (without labels)
    basedir : str, optional
        path to ML cup directory, by default "./ML_cup"

    Returns
    -------
    Dataset
        dataset containing ids, data and labels (if available)

    Raises
    ------
    ValueError
        if dname is invalid
    """
    basedir = basedir if basedir[-1] != "/" else basedir[:-1]
    TRAIN_PATH = f"{basedir}/ML-CUP22-TR.csv"
    TEST_PATH = f"{basedir}/ML-CUP22-TS.csv"

    if dname == "train":
        # pandas doesn't work for me here
        raw = np.genfromtxt(TRAIN_PATH, delimiter=",")
        dataset = Dataset(
            ids=raw.T[0].copy(), labels=raw.T[-2:].T.copy(), data=raw.T[1:-2].T.copy()
        )
    elif dname == "test":
        raw = np.genfromtxt(TEST_PATH, delimiter=",")
        dataset = Dataset(ids=raw.T[0].copy(), labels=None, data=raw.T[1:].T.copy())
    else:
        raise ValueError(f"Invalid Dataset Name: {dname}")
    return dataset


def read_monks(number: int, dname: str, basedir="./monks") -> Dataset:
    """Loads monks dataset and generates a Dataset class.

    Parameters
    ----------
    number : int
        which monk dataset, either 1, 2 or 3
    dname : str
        which type of dataset, either "train" or "test"
    basedir : str, optional
        path to monk directory, by default "./monks"

    Returns
    -------
    Dataset
        dataset containing ids, data and labels (if available)

    Raises
    ------
    ValueError
        if number is invalid (not in [1,2,3])
    ValueError
        if dname is invalid (not in ["train", "test"])
    """

    basedir = basedir if basedir[-1] != "/" else basedir[:-1]
    if number not in (1, 2, 3):
        raise ValueError(f"Invalid Dataset Number: {number}")
    if dname not in ("train", "test"):
        raise ValueError(f"Invalid Dataset Type: {dname}")

    path = f"{basedir}/monks-{number}.{dname}"
    raw = pd.read_csv("monks/monks-1.train", sep=" ", header=None).drop(columns=0)
    raw[8] = raw[8].str.removeprefix("data_").astype(int)

    dataset = Dataset(
        ids=raw[8].to_numpy(),
        labels=raw[1].to_numpy(),
        data=raw[[2, 3, 4, 5, 6, 7]].to_numpy(),
    )
    return dataset



def onehot_encoding(data: Dataset, is_categorical:List[int] = None, transform_label:bool=False) -> Dataset:
    """Encode a dataset with onehot encoding. Columns can be select with is_categorical.
    label can also be transformed.

    Parameters
    ----------
    data : Dataset
        dataset to encode
    is_categorical : List[int], optional
        boolean map encoding which column to transform, by default None
    transform_label : bool, optional
        encodes label if True, by default False

    Returns
    -------
    _type_
        _description_

    Raises
    -------
    ValueError
        if is_categorical is invalid
    """
    if is_categorical is None:
        is_categorical = np.full(data.data.shape[1], True)
    elif len(is_categorical) != data.data.shape[1]:
        raise ValueError(f"shape of is_categorical does not match: {is_categorical.shape}!={data.data.shape[1]}")
    elif not all(isinstance(x, bool) for x in is_categorical):
        raise ValueError(f"is_categorical is not a bitmap")

    def onehot(column):
        min, max = column.min(axis=0), column.max(axis=0)
        onehot = np.empty((max-min+1, column.size))
        for i in range(onehot.shape[0]):
            onehot[i] = 1*(column == i+min)
        return onehot.T
    
    out = []
    for transform, column in zip(is_categorical, data.data.T):
        if transform:
            out.append(onehot(column))
        else:
            out.append(column.reshape(column.size, 1))

    dataset = Dataset(
        ids=data.ids,
        data=np.concatenate(out, axis=1),
        labels=onehot(data.labels) if transform_label else data.labels
    )
    
    return dataset