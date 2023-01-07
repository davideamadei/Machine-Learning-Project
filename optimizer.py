#python libraries
import numpy as np
#local libraries
from util_classes import Parameter

class Optimizer:
    def __init__(self, eta=1e-3, l2_coeff=0.01, alpha=0.3):
        # TODO fix momentum for iteration #0 with trainer class
        def optimize(old: Parameter, grad: Parameter, old_delta: Parameter)-> Parameter:
            return Parameter(
                -eta*grad.weights + alpha*old_delta.weights - 2*l2_coeff*old.weights,
                -eta*grad.bias    + alpha*old_delta.bias    - 2*l2_coeff*old.bias
            )
        self.optimize = optimize