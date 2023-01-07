#python libraries
import numpy as np
#local libraries
from loss import LossFunction
from optimizer import Optimizer

class Estimator:
    def __init__(self, net, *,
        loss=LossFunction(), optimizer=Optimizer(),
        batchsize=1, start_it=0, seed=None
    ):
        self.net = net
        self.t = start_it
        self.loss = loss
        self.optimizer = optimizer
        self.batchsize = batchsize
        self.rng = np.random.default_rng(seed)
        if seed != None:
            # re-randomize all layers with new rng
            self.net.rng = self.rng
    
    def update_params(net=None, loss=None, optimizer=None, batchsize=None, seed=None):
        self.t = 0
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if net is not None:
            self.net = net
            self.net.rng = self.rng
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if batchsize is not None:
            self.batchsize = batchsize
    
    @staticmethod
    def get_minibatches(x, y, batchsize):
        size = x.shape[0]
        batchtotal, remainder = divmod(size, batchsize)
        for i in range(batchtotal):
            mini_x = x[i*batchsize:(i+1)*batchsize]
            mini_y = y[i*batchsize:(i+1)*batchsize]
            yield mini_x, mini_y
        if remainder > 0:
            yield (
                x[batchtotal*batchsize:],
                y[batchtotal*batchsize:]
            )
        
    def train(self, dataset, *, n_epochs=1, callback=print, mb_callback=None):
        for i in range(n_epochs):
            # permute dataset
            permutation = self.rng.permutation(dataset.shape[0])
            x = dataset.data[permutation]
            y = dataset.labels[permutation]
            # iterate minibatches
            avg_loss, batchcount = 0., np.ceil(x.shape[0] / self.batchsize)
            for b, (mini_x, mini_y) in enumerate(Estimator.get_minibatches(x, y, self.batchsize)):
                pred = self.net.foward(mini_x)
                loss = self.loss.foward(pred, mini_y)
                if mb_callback is not None:
                    record = {"epoch": self.t, "batch": b, "loss": loss}
                    mb_callback(self.t, b, loss)
                avg_loss += loss
                loss_grad = self.loss.backward()
                self.net.backward(loss_grad)
                self.net.optimize(self.optimizer)
            avg_loss /= batchcount
            self.t += 1
            record = {"epoch": self.t, "loss": avg_loss}
            callback(record)