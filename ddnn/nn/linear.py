# python libraries


# external libraries
import numpy as np

# local libraries
from .abstract import UpdatableLayer
from ..utils import Parameter

__all__ = ["LinearLayer"]


class LinearLayer(UpdatableLayer):
    """Defines a linear layer."""

    def __init__(self, shape: tuple[int, int]):
        """Intializes a new linear layer.

        Parameters
        ----------
        shape : tuple[int,int]
            Layer input and output dimension.
        seed : int or None, optional
            seed used to initialize weights, by default None
        """
        self.params = Parameter(weights=np.empty(shape[::-1]), bias=np.empty(shape[1]))
        self.grads = Parameter(weights=np.empty(shape[::-1]), bias=np.empty(shape[1]))

    def foward(self, input: np.ndarray) -> np.ndarray:
        """Foward call of this Layer.
        x -> Wx+b, where W are the weights and b the biases.

        Parameters
        ----------
        input : np.ndarray
            input

        Returns
        -------
        np.ndarray
            output
        """
        self.cache = input
        output = input @ self.params.weights.T + self.params.bias.T
        return output

    def backward(self, ograds: np.ndarray) -> np.ndarray:
        """Backward call of this Layer.
        Internally also computes gradients with respect to weights and biases.

        Parameters
        ----------
        ograds : np.ndarray
            gradient with respect to the output.

        Returns
        -------
        np.ndarray
            gradient with respect to the input
        """
        self.grads.bias[:] = ograds.sum(axis=0)
        self.grads.weights[:] = ograds.T @ self.cache
        igrads = ograds @ self.params.weights
        return igrads
