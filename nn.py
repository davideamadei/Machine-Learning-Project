# python libraries
from abc import ABC, abstractmethod
from dataclasses import dataclass
# external libraries
import numpy as np
#local libraries
from optimizer import Optimizer
from util_classes import Parameter

# TODO: split into parameter, gradient, ...
# Gradient = Parameter

# ABSTRACT BASE CLASSES (INTERFACES)
class Layer(ABC):
    """Layer without parameters"""
    @abstractmethod
    def foward(self, data):
        pass
    @abstractmethod
    def backward(self, grad):
        pass
    
class UpdatableLayer(Layer):
    """Layer with parameters"""
    @property
    @abstractmethod
    def rng(self, rng):
        pass
    @property
    @abstractmethod
    def gradients(self)-> Parameter:
        pass
    @property
    @abstractmethod
    def deltas(self)-> Parameter:
        pass
    @property
    @abstractmethod
    def parameters(self)-> Parameter:
        pass
    @abstractmethod
    def update(self, deltas: Parameter)-> None:
        pass
    

# ACTIVATION FUNCTIONS (Layer with no parameters)
class ActivationFunction(Layer):
    def __init__(self, fname="ReLU"):
        self._foward, self._backward = ActivationFunction.get_functions(fname)
        self._buffer = None
    
    def foward(self, data):
        if self._buffer is not None:
            print("No call to backward after previous foward call.")
        self._buffer = data
        return self._foward(data)
    
    def backward(self, grad):
        delta = grad * self._backward(self._buffer)
        self._buffer = None
        return delta
    
    @staticmethod
    def get_functions(fname):
        if fname == "ReLU":
            return (
                lambda x: x*(x>0), # function
                lambda x: 1*(x>0)  # gradient
            )
        else:
            raise ValueError(f"Invalid Activation Function: {fname}")


# LAYERS: LINEAR (Layer with parameters)
class LinearLayer(UpdatableLayer):
    def __init__(self, shape, seed=None):     
        self._params = Parameter(
            weights= np.empty(shape[::-1]), 
            bias   = np.empty(shape[1])
        )
        # gradient
        self._grad = Parameter(
            weights= np.empty(shape[::-1]), 
            bias   = np.empty(shape[1])
        )
        # old weight delta (used in momentum)
        self._delta = Parameter(
            weights= np.zeros(shape[::-1]),
            bias   = np.zeros(shape[1])
        )
        # input buffer (used for backprop)
        self._buffer = None
        # set rng and initialize weights
        self.rng = np.random.default_rng(seed)

    # set-only property
    def _set_rng(self, rng):
        self._rng = rng
        self._params.randomize(self._rng)
    rng = property(fset=_set_rng)
    @property
    def gradients(self)-> Parameter:
        return self._grad
    @property
    def deltas(self)-> Parameter:
        return self._delta
    @property
    def parameters(self)-> Parameter:
        return self._params
        
    def foward(self, data):
        if self._buffer is not None:
            print("No call to backward after previous foward call.")
        self._buffer = data
        output = data @ self._params.weights.T + self._params.bias.T
        return output
    
    def backward(self, output_gradient):
        self._grad.bias[:] = output_gradient.sum(axis=0)
        self._grad.weights[:] = (output_gradient.T @ self._buffer).sum(axis=0)
        self._buffer = None
        input_gradient = (output_gradient @ self._params.weights)
        return input_gradient
        
    def update(self, deltas: Parameter):
        self._params += deltas
        self._delta = deltas
        

# NEURAL NETWORK CLASS (handles method chaining)
class NeuralNetwork:
    def __init__(self, net, seed=None):
        NeuralNetwork.check_network(net)
        self.net = net
        self._buffer = None
        self._rng = np.random.default_rng(seed)

    def _set_rng(self, rng):
        self._rng = rng
        for layer in self.net:
            if isinstance(layer, UpdatableLayer):
                # rng is a property hence all layers are regenerated
                layer.rng = self._rng
    rng = property(fset=_set_rng)
            
    def foward(self, data):
        if self._buffer is not None:
            print("No call to backward after previous foward call.") 
        out = data
        for layer in self.net:
            out = layer.foward(out)
        return out
    
    def backward(self, grad):
        for layer in self.net[::-1]:
            grad = layer.backward(grad)
        return grad
    
    def optimize(self, optimizer: Optimizer):
        for layer in self.net:
            if isinstance(layer, UpdatableLayer):
                layer.update(optimizer.optimize(
                    layer.parameters, layer.gradients, layer.deltas
                ))

    @staticmethod
    def check_network(net):
        expected_layer_type = UpdatableLayer
        for i, layer in enumerate(net):
            if not isinstance(layer, expected_layer_type):
                raise ValueError(f"layer #{i} is of type {type(layer)} expected type is {expected_layer_type}")
            expected_layer_type = ActivationFunction if expected_layer_type == UpdatableLayer else UpdatableLayer