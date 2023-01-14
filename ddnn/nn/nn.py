# python libraries
from typing import List

# external libraries
import numpy as np

# local libraries
from .optimizer import Optimizer
from .abstract import Layer, UpdatableLayer
from .activation import ActivationFunction
from .initializer import Initializer

__all__ = ["NeuralNetwork"]


class NeuralNetwork:
    """NeuralNetwork"""

    def __init__(self, net: List[Layer]):
        """Intializes a new NeuralNetwork.

        Parameters
        ----------
        net : List[Layer]
            a sequence of layer, in the order in which they should be applied.
        seed : int or None, optional
            seed to initialize weights in all UpdatableLayer, by default None

        Raises
        ------
        ValueError
            in case net has bad structure.
        """
        NeuralNetwork.check_network(net)
        self._net = net
    
    def __getitem__(self, index):
        """Returns the index-th layer."""
        return self._net[index]
    def __setitem__(self, index, value):
        """Sets the index-th layer to the passed input"""
        self._net[index] = value
    def __len__(self):
        """Returns the number of layers"""
        return len(self._net)

    def foward(self, input: np.ndarray) -> np.ndarray:
        """Foward call of the network. Propagates outputs of each layer as inputs
        to the next.

        Parameters
        ----------
        input : np.ndarray
            input

        Returns
        -------
        np.ndarray
            output
        """
        out = input
        for layer in self[:]:
            out = layer.foward(out)
        return out

    def backward(self, ograd: np.ndarray) -> np.ndarray:
        """Backward call of the network. Propagates gradient with respect to
        the output layer across all layes and returns gradient with respect to inputs.

        Parameters
        ----------
        grad : np.ndarray
            gradient with respect to the output.

        Returns
        -------
        np.ndarray
            gradient with respect to the input.
        """
        for layer in self[::-1]:
            ograd = layer.backward(ograd)
        return ograd

    def update(self, optimizer: Optimizer):
        """updates all layers using the given optimizer. Gradients
        have to be computed before this call.

        Parameters
        ----------
        optimizer : Optimizer
            an optimizer that selects how weights and biases are updated
        """
        for layer in self[:]:
            if isinstance(layer, UpdatableLayer):
                layer.update(optimizer)

    def initialize(self, initializer: Initializer):
        """intializes all layers using the given initializer. Training
        will be voided after this call.

        Parameters
        ----------
        optimizer : Optimizer
            an optimizer that selects how weights and biases are updated
        """
        for layer in self[:]:
            if isinstance(layer, UpdatableLayer):
                layer.initialize(initializer)

    @staticmethod
    def check_network(net):
        """(WIP) function that check whenever layers follow a structure that makes sense.
        Currently checks that net consists of the alternating of UpdatableLayer and ActivationFunction.

        Parameters
        ----------
        net : List[Layer]
            a sequence of layer

        Raises
        ------
        ValueError
            in case net has bad structure.
        """
        expected_layer_type = UpdatableLayer
        for i, layer in enumerate(net):
            if not isinstance(layer, expected_layer_type):
                raise ValueError(
                    f"layer #{i} is of type {type(layer)} expected type is {expected_layer_type}"
                )
            expected_layer_type = (
                ActivationFunction
                if expected_layer_type == UpdatableLayer
                else UpdatableLayer
            )
