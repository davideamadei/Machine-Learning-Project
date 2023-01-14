# python libraries
from typing import Callable

# external libraries
import numpy as np

# local libraries
from .abstract import Layer


class ActivationFunction(Layer):
    """Defines a generic activation function (Layer)."""

    def __init__(self, fname="ReLU"):
        """Initializes a new instance.

        Parameters
        ----------
        fname : str, optional
            name describing an activation function, by default "ReLU"
        """
        self._foward, self._backward = ActivationFunction.get_functions(fname)
        self._buffer = None

    def foward(self, input: np.ndarray) -> np.ndarray:
        """Foward call of the Layer

        Parameters
        ----------
        input : np.ndarray
            Layer input

        Returns
        -------
        np.ndarray
            Layer output
        """
        self.cache = input
        return self._foward(input)

    def backward(self, ograds: np.ndarray) -> np.ndarray:
        """Backward call of a Layer.

        Parameters
        ----------
        ograds : np.ndarray
            gradient with respect to the output.

        Returns
        -------
        np.ndarray
            gradient with respect to the input
        """
        return ograds * self._backward(self.cache)

    @staticmethod
    def get_functions(
        fname: str,
    ) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
        """Given a name identifing a function. Returns said function and its derivative.

        Parameters
        ----------
        fname : str
            Name of a supported fuction. Currently: ReLU, logistic, tanh.

        Returns
        -------
        tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]
            function and its derivative

        Raises
        ------
        ValueError
            if fname is not supported
        """
        if fname == "ReLU":
            return (
                lambda x: x * (x > 0),  # function
                lambda x: 1 * (x > 0),  # gradient
            )
        if fname == "logistic":
            def logistic_func(x):
                return 1 / (1 + np.exp(-x))

            def logistic_grad(x):
                y = logistic_func(x)
                return y * (1 - y)

            return (logistic_func, logistic_grad)
        if fname == "tanh":
            def tanh_func(x):
                y = np.exp(x)
                z = np.exp(-x)
                return (y - z) / (y + z)

            def tanh_grad(x):
                y = tanh_func(x)
                return 1 - (y*y)

            return (tanh_func, tanh_grad)
        else:
            raise ValueError(f"Invalid Activation Function: {fname}")