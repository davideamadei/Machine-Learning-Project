import numpy as np
from dataclasses import dataclass
from typing import Optional, List

# DATACLASSES
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
    def __iadd__(self, other): # += operator
        """support for += operator
        """
        self.weights += other.weights
        self.bias += other.bias
        return self
    def randomize(self, rng: np.random.Generator):
        """random initialization of weights and biases

        Parameters
        ----------
        rng : np.random.Generator
            random number generator used.
        """
        self.weights[:] = rng.random(self.weights.shape)
        self.bias[:] = rng.random(self.bias.shape)

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
    def shape(self)-> tuple[int,List[int,Optional[int]]]:
        """Returns the shape of the dataset.

        Returns
        -------
        tuple[int,List[int,Optional[int]]]
            (#samples, [#features, dim(labels)])
        """
        if self.labels is None:
            labels_shape = 0
        elif len(self.labels.shape) == 1:
            labels_shape = 1
        else:
            labels_shape = self.labels.shape[1]
        return (self.data.shape[0], [self.data.shape[1], labels_shape])