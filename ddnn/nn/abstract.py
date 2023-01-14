# python libraries
from abc import ABC, abstractmethod
from typing import Any

# external libraries
import numpy as np

# local libraries
from ..utils import Parameter
from .optimizer import Optimizer
from .initializer import Initializer

__all__ = ["Layer", "UpdatableLayer"]


class Layer(ABC):
    """Layer without parameters"""

    @property
    def cache(self) -> Any:
        """Property that manages caching of input values required
        for backpropagation. A call to get before set is undefined. 
        """
        return self._cache
    @cache.setter
    def cache(self, value: Any):
        self._cache = value

    @abstractmethod
    def foward(self, input: np.ndarray) -> np.ndarray:
        """Foward call of a Layer.

        Parameters
        ----------
        input : np.ndarray
            Layer input

        Returns
        -------
        np.ndarray
            Layer output
        """
        pass

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward call of a Layer.

        Parameters
        ----------
        grad : np.ndarray
            gradient with respect to the output.

        Returns
        -------
        np.ndarray
            gradient with respect to the input
        """
        pass



class UpdatableLayer(Layer):
    """Layer with parameters"""

    @property
    def state(self) -> Any:
        """Property that manages caching of optimization values required
        for updates. State should be set in any subclass.
        """
        if not hasattr(self, "_state"):
            self._state = None
        return self._state
    @state.setter
    def state(self, value: Any):
        self._state = value

    @property
    def grads(self) -> Parameter:
        """Property holding gradients with respect to parameters.
        """
        return self._grads
    @grads.setter
    def grads(self, value) -> Parameter:
        self._grads = value

    @property
    def params(self) -> Parameter:
        """Property holding parameters (weights and biases).
        """
        return self._params
    @params.setter
    def params(self, value) -> Parameter:
        self._params = value
    
    def update(self, optimizer: Optimizer) -> None:
        """Function that updates current Parameters.
        Function should call optimizer(params, grads, state)
        which will return (delta, state). This values have to then be
        used to compute the update as [self.params += delta] and save the
        state as [self._state = state].

        Parameters
        ----------
        optimizer : Optimizer
            optimization method
        """
        delta, state = optimizer(self.params, self.grads, self.state)
        self.params += delta
        self.state = state

    def initialize(self, initializer: Initializer) -> None:
        """Function that initializes Parameters (or resets them).
        Function should call initializer(params.shape) which will return params.
        This value has to then be set with [self.params = params]

        Parameters
        ----------
        initializer : Initializer
            initializer method
        """
        self.params = initializer(self.params.shape)
