# python libraries
import numpy as np

# local libraries
from util_classes import Parameter


class Optimizer:
    """Class defining weight updates. Currently WIP."""

    def __init__(self, eta: float = 1e-3, l2_coeff: float = 0.01, alpha: float = 0.3):
        """Returns a new instance of an optimizer

        Parameters
        ----------
        eta : float, optional
            learning rate, by default 1e-3
        l2_coeff : float, optional
            L2 regression constant, by default 0.01
        alpha : float, optional
            momentum constant, by default 0.3
        """
        # TODO fix momentum for iteration #0 with trainer class
        def optimize(
            old: Parameter, grad: Parameter, old_delta: Parameter
        ) -> Parameter:
            return Parameter(
                -eta * grad.weights
                + alpha * old_delta.weights
                - 2 * l2_coeff * old.weights,
                -eta * grad.bias + alpha * old_delta.bias - old.bias,
            )

        self.optimize = optimize
