# python libraries
from typing import Callable, Any, Tuple, Dict

# external libraries
import numpy as np

# local libraries
from ..utils import Parameter

__all__ = ["Optimizer"]


class Optimizer:
    """Class defining parameter updates. Currently WIP."""

    def __init__(self, fname: str = "SGD", **kwargs):
        """Returns a new instance of an optimizer

        Parameters
        ----------
        fname : str, optional
            Name of a supported optimizer algorithm, by default "SGD".
            Currently supported functions are: SGD.
        **kwargs
            Arguments of the optimizer
        """
        self._opt = Optimizer.get_functions(fname, kwargs)

    def __call__(
        self, params: Parameter, grads: Parameter, state: Parameter
    ) -> Tuple[Parameter, Any]:
        return self._opt(params, grads, state)

    @staticmethod
    def get_functions(
        fname: str, kwargs: Dict
    ) -> Callable[[Parameter, Parameter, Any], Tuple[Parameter, Any]]:
        """Given a name identifing an optimizer. Returns an optimizer with
        given hyperparamters.

        Parameters
        ----------
        fname : str
            Name of a supported optimizer algorithm

        Returns
        -------
        Callable[[Parameter, Parameter, Any], Tuple[Parameter, Any]]
            function that given (params, grads, state) returns (update, state)

        Raises
        ------
        ValueError
            if fname is not supported
        """
        if fname == "SGD":

            class SGD:
                def __init__(
                    self,
                    *,
                    learning_rate=1e-3,
                    l2_coefficient=1e-4,
                    momentum_coefficient=0.8,
                ):
                    self._eta = learning_rate
                    self._l2 = l2_coefficient
                    self._m = momentum_coefficient

                def __call__(
                    self, params: Parameter, grads: Parameter, state: Parameter
                ) -> Tuple[Parameter, Any]:
                    # first iteration set momentum to gradient
                    if state == None:
                        state = grads

                    delta = Parameter(
                        -self._eta * grads.weights
                        + self._m * state.weights
                        - 2 * self._l2 * params.weights,
                        -self._eta * grads.bias + self._m * state.bias,
                    )
                    return (delta, delta)

            return SGD(**kwargs)
        if fname == "Adam":

            class Adam:
                pass

            return Adam(**kwargs)
        else:
            raise ValueError(f"Invalid Optimizer: {fname}")
