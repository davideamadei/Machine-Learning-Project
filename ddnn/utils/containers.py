# python libraries
from dataclasses import dataclass
from typing import Optional

# external libraries
import numpy as np


@dataclass
class Parameter:
    """A simple dataclass holding Parameters.
    its currently used for multiple purposes:
    - hold weights and biases
    - hold gradients with respect to weights and biases
    - hold updates that should (or have been applied) to weights and biases
    """

    weights: np.array
    bias: np.array

    def __iadd__(self, other):  # += operator
        """support for += operator"""
        self.weights += other.weights
        self.bias += other.bias
        return self

    def shape(self):
        return self.weights.shape
    
    def randomize(self, rng: np.random.Generator):
        """random initialization of weights and biases

        Parameters
        ----------
        rng : np.random.Generator
            random number generator used.
        """
        self.weights[:] = 2*(rng.random(self.weights.shape) - 0.5)
        self.bias[:] = 2*(rng.random(self.bias.shape) - 0.5)


@dataclass
class Dataset:
    """A simple dataclass holding a dataset.
    A dataset is composed of:
    - data: also denoted as X
    - Optional[labels]: also denoted as y
    - ids: unique identifies of data and labels.
    All three of these should have same 0-dimension.
    """

    ids: np.array
    labels: Optional[np.array]
    data: np.array

    @property
    def shape(self) -> tuple[int, tuple[int, Optional[int]]]:
        """Returns the shape of the dataset.

        Returns
        -------
        tuple[int,tuple[int,Optional[int]]]
            (#samples, [#features, dim(labels)])
        """
        if self.labels is None:
            labels_shape = 0
        elif len(self.labels.shape) == 1:
            labels_shape = 1
        else:
            labels_shape = self.labels.shape[1]
        return (self.data.shape[0], (self.data.shape[1], labels_shape))
