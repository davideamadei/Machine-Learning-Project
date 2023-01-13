# python libraries
from abc import ABC, abstractmethod
from typing import Callable, List

# external libraries
import numpy as np

# local libraries
from optimizer import Optimizer
from util_classes import Parameter

# TODO: split into parameter, gradient, ...
# Gradient = Parameter

# ABSTRACT BASE CLASSES (INTERFACES)
class Layer(ABC):
    """Layer without parameters"""

    @abstractmethod
    def foward(self, data: np.ndarray) -> np.ndarray:
        """Foward call of a Layer.

        Parameters
        ----------
        data : np.ndarray
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

    def _set_rng(self, anrng: np.random.Generator) -> None:
        """Property that manages randomness. Property should be set only.
        And setting this property should cause all weights to be reset.

        Parameters
        ----------
        anrng : np.random.Generator
            random number generator used to initialize weights.
        """
        pass

    rng = property(fset=abstractmethod(_set_rng))

    @property
    @abstractmethod
    def gradients(self) -> Parameter:
        """Gradients with respect to the Layer Parameters. Property should be get only.

        Returns
        -------
        Parameter
            gradients with respect to weights and biases
        """
        pass

    @property
    @abstractmethod
    def deltas(self) -> Parameter:
        """Deltas of the Layer Parameters (previous update to said Parameters).
        Property should be get only.

        Returns
        -------
        Parameter
            last change of weights and biases
        """
        pass

    @property
    @abstractmethod
    def parameters(self) -> Parameter:
        """Parameters of a Layer. Property should be get only.

        Returns
        -------
        Parameter
            weights and biases
        """
        pass

    @abstractmethod
    def update(self, deltas: Parameter) -> None:
        """Function that should update current Parameters with new delta.
        update should follow the formula:
        new = old + delta

        Parameters
        ----------
        deltas : Parameter
            update quantities of weights and biases
        """
        pass


# ACTIVATION FUNCTIONS (Layer with no parameters)
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

    def foward(self, data: np.ndarray) -> np.ndarray:
        """Foward call of the Layer

        Parameters
        ----------
        data : np.ndarray
            Layer input

        Returns
        -------
        np.ndarray
            Layer output
        """
        self._buffer = data
        return self._foward(data)

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
        delta = grad * self._backward(self._buffer)
        self._buffer = None
        return delta

    @staticmethod
    def get_functions(
        fname: str,
    ) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
        """Given a name identifing a function. Returns said function and its derivative.

        Parameters
        ----------
        fname : str
            Name of a supported fuction. Currently: ReLU.

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


# LAYERS: LINEAR (Layer with parameters)
class LinearLayer(UpdatableLayer):
    """Defines a linear layer."""

    def __init__(self, shape: tuple[int, int], seed: int = None):
        """Intializes a new linear layer.

        Parameters
        ----------
        shape : tuple[int,int]
            Layer input and output dimension.
        seed : int or None, optional
            seed used to initialize weights, by default None
        """
        self._params = Parameter(weights=np.empty(shape[::-1]), bias=np.empty(shape[1]))
        # gradient
        self._grad = Parameter(weights=np.empty(shape[::-1]), bias=np.empty(shape[1]))
        # old weight delta (used in momentum)
        self._delta = Parameter(weights=np.zeros(shape[::-1]), bias=np.zeros(shape[1]))
        # input buffer (used for backprop)
        self._buffer = None
        # set rng and initialize weights
        self.rng = np.random.default_rng(seed)

    # set-only property
    def _set_rng(self, rng: np.random.Generator):
        """Sets a new random number generator and reinitalizes all weights and biases.

        Parameters
        ----------
        rng : np.random.Generator
            new random number generator
        """
        self._rng = rng
        self._params.randomize(self._rng)

    rng = property(fset=_set_rng)

    @property
    def gradients(self) -> Parameter:
        return self._grad

    @property
    def deltas(self) -> Parameter:
        return self._delta

    @property
    def parameters(self) -> Parameter:
        return self._params

    def foward(self, data: np.ndarray) -> np.ndarray:
        """Foward call of this Layer.
        x -> Wx+b, where W are the weights and b the biases.

        Parameters
        ----------
        data : np.ndarray
            input

        Returns
        -------
        np.ndarray
            output
        """
        self._buffer = data
        output = data @ self._params.weights.T + self._params.bias.T
        return output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """Backward call of this Layer.
        Internally also computes gradients with respect to weights and biases.

        Parameters
        ----------
        output_gradient : np.ndarray
            gradient with respect to the output.

        Returns
        -------
        np.ndarray
            gradient with respect to the input
        """
        self._grad.bias[:] = output_gradient.sum(axis=0)
        self._grad.weights[:] = output_gradient.T @ self._buffer
        self._buffer = None
        input_gradient = output_gradient @ self._params.weights
        return input_gradient

    def update(self, deltas: Parameter):
        self._params += deltas
        self._delta = deltas


# NEURAL NETWORK CLASS (handles method chaining)
class NeuralNetwork:
    """NeuralNetwork"""

    def __init__(self, net: List[Layer], seed: int = None):
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
        self.net = net
        self._buffer = None
        self._rng = np.random.default_rng(seed)

    def _set_rng(self, rng: np.random.Generator):
        """Sets a new random number generator and reinitalizes all weights and biases.
        The new rng is also propagated on all instances of UpdatableLayer.

        Parameters
        ----------
        rng : np.random.Generator
            new random number generator
        """
        self._rng = rng
        for layer in self.net:
            if isinstance(layer, UpdatableLayer):
                # rng is a property hence all layers are regenerated
                layer.rng = self._rng

    rng = property(fset=_set_rng)

    def foward(self, data: np.ndarray) -> np.ndarray:
        """Foward call of the network. Propagates outputs of each layer as inputs
        to the next.

        Parameters
        ----------
        data : np.ndarray
            input

        Returns
        -------
        np.ndarray
            output
        """
        out = data
        for layer in self.net:
            out = layer.foward(out)
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward call of the network. Propagates gradient with respect to
        the output layer across all layes and returns gradient with respect to inputs.

        Parameters
        ----------
        grad : np.ndarray
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        for layer in self.net[::-1]:
            grad = layer.backward(grad)
        return grad

    def optimize(self, optimizer: Optimizer):
        """updates all layers using the given optimizer. Gradients
        have to be computed before this call.

        Parameters
        ----------
        optimizer : Optimizer
            an optimizer that selects how weights and biases are updated
        """
        for layer in self.net:
            if isinstance(layer, UpdatableLayer):
                layer.update(
                    optimizer.optimize(layer.parameters, layer.gradients, layer.deltas)
                )

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
