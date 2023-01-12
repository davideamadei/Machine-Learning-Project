# python libraries
import numpy as np
from typing import Callable

# LOSS FUNCTIONS
class LossFunction:
    """Defines a generic loss function."""

    def __init__(self, fname: str = "MSE"):
        """Initializes a new instance.

        Parameters
        ----------
        fname : str, optional
            name describing a loss function, by default "MSE".
            currently supported functions are: MSE.
        """
        self._foward, self._backward = LossFunction.get_functions(fname)
        self._buffer = None

    def foward(self, pred: np.ndarray, label: np.ndarray) -> np.ndarray:
        """Returns loss of predictions and labels.

        Parameters
        ----------
        pred : np.ndarray
            predictions
        label : np.ndarray
            labels

        Returns
        -------
        np.ndarray
            loss value
        """
        # one dimensional labels cause issues
        if len(label.shape) == 1:
            label = label.reshape(label.shape[0], 1)

        self._buffer = (pred, label)
        return self._foward(pred, label)

    def backward(self) -> np.ndarray:
        """Returns gradients of the loss with respects to the predictions.

        Returns
        -------
        np.ndarray
            gradient
        """
        delta = self._backward(*self._buffer)
        self._buffer = None
        return delta

    @staticmethod
    def get_functions(
        fname: str,
    ) -> tuple[
        Callable[[np.ndarray, np.ndarray], float],
        Callable[[np.ndarray, np.ndarray], np.ndarray],
    ]:
        """Given a name identifing a function. Returns said function and its derivative.

        Parameters
        ----------
        fname : str
            Name of a supported fuction. Currently: MSE.

        Returns
        -------
        tuple[Callable[[np.ndarray, np.ndarray], float], Callable[[np.ndarray, np.ndarray], np.ndarray]]
            function and derivative

        Raises
        ------
        ValueError
            if fname is not a supported function
        """
        if fname == "MSE":
            return (
                lambda o, y: np.sum((o - y) ** 2) / (2 * o.shape[0]),  # function
                lambda o, y: (o - y) / o.shape[0],  # gradient
            )
        if fname == "binary_accuracy":
            def accuracy_func(o, y):
                if len(o.shape) > 1:
                    raise ValueError("output not one-dimensional")
                return (np.round(o) == y).sum() / y.size
            def accuracy_grad(o, y):
                raise NotImplementedError()
            return (
                accuracy_func, accuracy_grad
            )
        else:
            raise ValueError(f"Invalid Activation Function: {fname}")
