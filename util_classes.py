import numpy as np
from dataclasses import dataclass
from typing import Optional

# DATACLASSES
@dataclass
class Parameter:
    weights: np.array
    bias: np.array
    def __iadd__(self, other): # += operator
        self.weights += other.weights
        self.bias += other.bias
        return self
    def randomize(self, rng: np.random.Generator):
        self.weights[:] = rng.random(self.weights.shape)
        self.bias[:] = rng.random(self.bias.shape)

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