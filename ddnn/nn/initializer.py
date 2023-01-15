# python libraries
from typing import Callable, Any, Tuple, Dict

# external libraries
import numpy as np

# local libraries
from ..utils import Parameter

__all__ = ["Initializer"]


class Initializer:
    """Class defining parameter initialization. Currently WIP."""

    def __init__(self, fname: str = "random_uniform", **kwargs):
        """Returns a new instance of an initializer

        Parameters
        ----------
        fname : str, optional
            Name of a supported initializer algorithm, by default "random_uniform".
            currently supported functions are: random_uniform.
        **kwargs
            Arguments of the initializer
        """
        self.rng = np.random.default_rng(None)
        self._initializer = Initializer.get_functions(fname, kwargs)

    def __call__(self, shape: Tuple):
        """Call interface for initializer"""
        return self._initializer(self, shape)

    @staticmethod
    def get_functions(fname: str, kwargs: Dict) -> Callable[[tuple], Parameter]:
        """Given a name identifing an initializer. Returns an initializer with
        given hyperparamters.

        Parameters
        ----------
        fname : str
            Name of a supported initializer algorithm

        Returns
        -------
        Callable[[tuple], Parameter]
            function and its derivative

        Raises
        ------
        ValueError
            if fname is not supported
        """
        if fname == "random_uniform":

            def initializer(self, shape):
                a = 1
                return Parameter(
                    weights=self.rng.uniform(-a, a, size=shape),
                    bias=self.rng.uniform(-a, a, size=shape[0]),
                )

            return initializer
        
        if fname == "random_normal":

            def initializer(self, shape):
                std = 1
                return Parameter(
                    weights=self.rng.normal(0, std, size=shape),
                    bias=self.rng.normal(0, std, size=shape[0])
                )

        if fname == "glorot_uniform":

            def initializer(self, shape):
                a = np.sqrt(2) * np.sqrt(6 / sum(shape))
                return Parameter(
                    weights=self.rng.uniform(-a, a, size=shape),
                    bias=self.rng.uniform(-a, a, size=shape[0])
                )
        if fname == "glorot_normal":

            def initializer(self, shape):
                std = np.sqrt(2) * np.sqrt(2 / sum(shape))
                var = std*std
                return Parameter(
                    weights=self.rng.normal(0, var, size=shape),
                    bias=self.rng.normal(0, var, size=shape[0])
                )

        if fname == "he_uniform":
            is_fan_in = None
            if hasattr(kwargs, "fan_mode"):
                if kwargs["fan_mode"] == "fan_in":
                    is_fan_in = True
                elif kwargs["fan_mode"] == "fan_out":
                    is_fan_in = False
            if is_fan_in is None:
                raise ValueError("Invalid fan_mode")
            

            def initializer(self, shape):
                if is_fan_in:
                    a = np.sqrt(2) * np.sqrt(3 / shape[1])
                else:
                    a = np.sqrt(2) * np.sqrt(3 / shape[0])
                return Parameter(
                    weights=self.rng.uniform(-a, a, size=shape),
                    bias=self.rng.uniform(-a, a, size=shape[0])
                )
        
        if fname == "he_normal":
            is_fan_in = None
            if hasattr(kwargs, "fan_mode"):
                if kwargs["fan_mode"] == "fan_in":
                    is_fan_in = True
                elif kwargs["fan_mode"] == "fan_out":
                    is_fan_in = False
            if is_fan_in is None:
                raise ValueError("Invalid fan_mode")
            

            def initializer(self, shape):
                if is_fan_in:
                    std = np.sqrt(2) / np.sqrt(shape[1])
                else:
                    std = np.sqrt(2) / np.sqrt(shape[0])
                var = std*std
                return Parameter(
                    weights=self.rng.normal(0, var, size=shape),
                    bias=self.rng.normal(0, var, size=shape[0])
                )
        else:
            raise ValueError(f"Invalid Activation Function: {fname}")
