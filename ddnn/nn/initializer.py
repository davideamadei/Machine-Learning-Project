# python libraries
from typing import Callable, Any, Tuple, Dict

# external libraries
import numpy as np

# local libraries
from ..utils import Parameter


class Initializer:
    """Class defining parameter initialization. Currently WIP."""

    def __init__(self, fname, **kwargs):
        """Returns a new instance of an initializer

        Parameters
        ----------
        fname : str
            Name of a supported initializer algorithm
        **kwargs
            Arguments of the initializer
        """
        self.rng = np.random.default_rng(None)
        self.__call__ = Initializer.get_functions(fname, kwargs)

    @staticmethod
    def get_functions(
        fname: str, kwargs: Dict
    ) -> Callable[[tuple], Parameter]:
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
                params = Parameter(
                    weights= 2 * (self.rng.random(shape) - 0.5),
                    bias= 2 * (self.rng.random(shape[0]) - 0.5)
                )
            return initializer
        else:
            raise ValueError(f"Invalid Activation Function: {fname}")