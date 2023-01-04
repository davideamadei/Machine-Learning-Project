# python libraries
from abc import ABC, abstractmethod
from dataclasses import dataclass
# external libraries
import numpy as np


# DATACLASSES
@dataclass
class Parameter:
    weights: np.array
    bias: np.array
    def __iadd__(self, other): # += operator
        self.weights += other.weights
        self.bias += other.bias
        return self


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
        if self._buffer:
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
    def __init__(self, shape):
        self._params = Parameter(
            np.random.rand(*shape[::-1]), 
            np.random.rand(shape[1])
        )
        # gradient
        self._grad = Parameter(
            np.empty(shape[::-1]), 
            np.empty(shape[1])
        )
        # old weight delta (used in momentum)
        self._delta = Parameter(
            np.empty(shape[::-1]),
            np.empty(shape[1])
        )
        # input buffer (used for backprop)
        self._buffer = None
    
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
        if self._buffer:
            print("No call to backward after previous foward call.")
        self._buffer = data
        output = data @ self._params.weights.T + self._params.bias.T
        return output
    
    def backward(self, output_gradient):
        self._grad.bias = output_gradient.sum(axis=0)
        self._grad.weights[:] = output_gradient.T @ self._buffer
        self._buffer = None
        input_gradient = (output_gradient @ self._params.weights)
        return input_gradient
        
    def update(self, deltas: Parameter):
        self._params += deltas
        self._delta = deltas


# LOSS FUNCTIONS
class LossFunction:
    def __init__(self, fname="MSE"):
        self._foward, self._backward = LossFunction.get_functions(fname)
        self._buffer = None
        
    def foward(self, pred, label):
        if self._buffer:
            print("No call to backward after previous foward call.")
        self._buffer = (pred, label)
        return self._foward(pred, label)
    
    def backward(self):
        delta = self._backward(*self._buffer)
        self._buffer = None
        return delta
        
    @staticmethod
    def get_functions(fname):
        if fname == "MSE":
            return (
                lambda o, y: np.sum((o - y)**2) / (2*o.shape[0]), # function
                lambda o, y: (o - y) / o.shape[0]                 # gradient
            )
        else:
            raise ValueError(f"Invalid Activation Function: {fname}")


# OPTIMIZER
class Optimizer:
    def __init__(self, eta=1e-3, l2_coeff=0.01, alpha=0.3):
        # TODO fix momentum for iteration #0 with trainer class
        def optimize(old: Parameter, grad: Parameter, old_delta: Parameter)-> Parameter:
            return Parameter(
                -eta*grad.weights + alpha*old_delta.weights - 2*l2_coeff*old.weights,
                -eta*grad.bias    + alpha*old_delta.bias    - 2*l2_coeff*old.bias
            )
        self.optimize = optimize
        

# NEURAL NETWORK CLASS (handles method chaining)
class NeuralNetwork:
    def __init__(self, net, *, loss=LossFunction(), optimizer=Optimizer()):
        NeuralNetwork.check_network(net)
        self.net = net
        self.loss = loss
        self.optimizer = optimizer
        self._buffer = None
        
    def foward(self, data, label):
        if self._buffer:
            print("No call to backward after previous foward call.")
        out = data
        for layer in self.net:
            out = layer.foward(out)
        return self.loss.foward(out, label), out
    
    def backward(self):
        grad = self.loss.backward()
        for layer in self.net[::-1]:
            grad = layer.backward(grad)
        return grad
    
    def optimize(self):
        for layer in self.net:
            if isinstance(layer, UpdatableLayer):
                layer.update(self.optimizer.optimize(
                    layer.parameters, layer.gradients, layer.deltas
                ))

    @staticmethod
    def check_network(net):
        expected_layer_type = UpdatableLayer
        for i, layer in enumerate(net):
            if not isinstance(layer, expected_layer_type):
                raise ValueError(f"layer #{i} is of type {type(layer)} expected type is {expected_layer_type}")
            expected_layer_type = ActivationFunction if expected_layer_type == UpdatableLayer else UpdatableLayer