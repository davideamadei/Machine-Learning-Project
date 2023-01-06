#python libraries
import numpy as np

# LOSS FUNCTIONS
class LossFunction:
    def __init__(self, fname="MSE"):
        self._foward, self._backward = LossFunction.get_functions(fname)
        self._buffer = None
        
    def foward(self, pred, label):
        if self._buffer is not None:
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