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
        self, t: int
    ) -> Callable[[Parameter, Parameter, Parameter], Tuple[Parameter, Any]]:
        return self._opt(t)

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
                    self._t = 0

                def __call__(
                    self, t: int
                ) -> Callable[[Parameter, Parameter, Parameter], Tuple[Parameter, Any]]:
                    self._t = t

                    def call_t(
                        params: Parameter, grads: Parameter, state: Parameter
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

                    return call_t

            return SGD(**kwargs)

        if fname == "Adam":

            class Adam:
                def __init__(
                    self,
                    *,
                    learning_rate=1e-3,
                    l2_coefficient=1e-4,
                    beta1=0.9,
                    beta2=0.999,
                    eps=1e-8,
                ) -> None:
                    self._eta = learning_rate
                    self._l2 = l2_coefficient
                    self._beta1 = beta1
                    self._beta2 = beta2
                    self._eps = eps
                    self._t = 0

                def __call__(
                    self, t: int
                ) -> Callable[[Parameter, Parameter, Parameter], Tuple[Parameter, Any]]:
                    self._t = t

                    def call_t(
                        params: Parameter, grads: Parameter, state: Parameter
                    ) -> Tuple[Parameter, Any]:
                        if self._t == 0:
                            raise ValueError()
                        if state == None:
                            old_m = Parameter(
                                np.zeros_like(grads.weights), np.zeros_like(grads.bias)
                            )
                            old_v = Parameter(
                                np.zeros_like(grads.weights), np.zeros_like(grads.bias)
                            )
                        else:
                            old_m = state[0]
                            old_v = state[1]

                        temp = grads.weights
                        if self._l2 != 0:
                            temp += self._l2 * params.weights

                        m_w = self._beta1 * old_m.weights + (1 - self._beta1) * temp
                        m_b = self._beta1 * old_m.bias + (1 - self._beta1) * grads.bias

                        v_w = (
                            self._beta2 * old_v.weights
                            + (1 - self._beta2) * temp * temp
                        )
                        v_b = (
                            self._beta2 * old_v.bias
                            + (1 - self._beta2) * grads.bias * grads.bias
                        )

                        old_m = Parameter(m_w, m_b)
                        old_v = Parameter(v_w, v_b)

                        adj = (1 - self._beta2**self._t) ** 0.5 / (
                            1 - self._beta1**self._t
                        )
                        delta_w = (-self._eta * adj) * m_w / (np.sqrt(v_w) + self._eps)
                        delta_b = (-self._eta * adj) * m_b / (np.sqrt(v_b) + self._eps)

                        delta = Parameter(delta_w, delta_b)

                        return (delta, (old_m, old_v))

                    return call_t

            return Adam(**kwargs)
        else:
            raise ValueError(f"Invalid Optimizer: {fname}")
