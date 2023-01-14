# python libraries
from typing import Iterator, Callable, List, Dict

# external libraries
import numpy as np

# local libraries
from .loss import LossFunction
from .optimizer import Optimizer
from .nn import NeuralNetwork
from .initializer import Initializer
from ..utils import Dataset


class Estimator:
    """Utility class defining generic training behaviour."""

    def __init__(
        self,
        net: NeuralNetwork,
        *,
        loss: LossFunction = LossFunction(),
        optimizer: Optimizer = Optimizer(),
        initializer: Initializer = Initializer(),
        batchsize: int = 1,
        seed: int = None
    ):
        """Initializes a new Estimator.

        Parameters
        ----------
        net : NeuralNetwork
            NeuralNetwork to train.
        loss : LossFunction, optional
            loss function used during training, by default LossFunction()
        optimizer : Optimizer, optional
            optimizer used to update weights and biases, by default Optimizer()
        batchsize : int, optional
            batch size used during training, by default 1
        seed : int or None, optional
            seed used to initialize weights of the NeuralNetwork, by default None
        """
        self.net = net
        self.t = 0
        self.loss = loss
        self.optimizer = optimizer
        self.initializer = initializer
        self.batchsize = batchsize
        self.initializer.rng = np.random.default_rng(seed)
        self.net.initialize(self.initializer)

    def reset(self):
        """Resets the model to its initial conditions."""
        self.t = 0
        self.net.initialize(self.initializer)

    def update_params(
        self, *,
        net: NeuralNetwork = None,
        loss: LossFunction = None,
        optimizer: Optimizer = None,
        initializer: Initializer = None,
        batchsize: int = None,
        seed: int = None,
    ) -> None:
        """Updates current Estimator with new parameters. This is equivalent to creating a
        new estimator. In case the net is not modified the same memory will be used.

        Parameters
        ----------
        net : NeuralNetwork or None, optional
            If not None the net will be updated (weights will be reset according to seed), by default None
        loss : LossFunction or None, optional
            If not None the loss will be updated, by default None
        optimizer : Optimizer or None, optional
            If not None the optimizer will be updated, by default None
        batchsize : int or None, optional
            If not None the batch size will be updated, by default None
        seed : int or None, optional
            If not None the seed will be changed (and net updated), by default None
        """
        # update parameters
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if net is not None:
            self.net = net
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if initializer is not None:
            self.initializer = initializer
        if batchsize is not None:
            self.batchsize = batchsize
        # reset state
        self.reset()

    @staticmethod
    def get_minibatches(
        x: np.ndarray, y: np.ndarray, batchsize: int
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Returns minibatches of given size over (x, y).

        Parameters
        ----------
        x : np.ndarray
            data array
        y : np.ndarray
            label array
        batchsize : int
            batch size of yielded minibatches

        Yields
        ------
        Iterator[tuple[np.ndarray,np.ndarray]]
            iterator over minibatches
        """
        size = x.shape[0]
        batchtotal, remainder = divmod(size, batchsize)
        for i in range(batchtotal):
            mini_x = x[i * batchsize : (i + 1) * batchsize]
            mini_y = y[i * batchsize : (i + 1) * batchsize]
            yield mini_x, mini_y
        if remainder > 0:
            yield (x[batchtotal * batchsize :], y[batchtotal * batchsize :])

    def train(
        self,
        dataset: Dataset,
        *,
        n_epochs: int = 1,
        callback: Callable[[dict], None] = print,
        mb_callback: Callable[[dict], None] = None
    ) -> None:
        """Trains the net with passed dataset.

        Parameters
        ----------
        dataset : Dataset
            dataset to train on
        n_epochs : int, optional
            number of epoch the training should continue, by default 1
        callback : Callable[[dict],None], optional
            callback after an epoch has finished, by default print
        mb_callback : Callable[[dict],None], optional
            callback after a minibatch has finihed, by default None
        """
        for i in range(n_epochs):
            # permute dataset
            permutation = self.rng.permutation(dataset.shape[0])
            x = dataset.data[permutation]
            y = dataset.labels[permutation]
            # iterate minibatches
            avg_loss, batchcount = 0.0, np.ceil(x.shape[0] / self.batchsize)
            for b, (mini_x, mini_y) in enumerate(
                Estimator.get_minibatches(x, y, self.batchsize)
            ):
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

    def evaluate(self, losses:List[str], dataset: Dataset) -> Dict[str, float]:
        """Evaluate current model on dataset with given list of losses.

        Parameters
        ----------
        losses : List[str]
            list of valid loss names
        dataset : Dataset
            dataset to evaluate on

        Returns
        -------
        Dict[str, float]
            for each loss value of result
        """
        res = {}
        pred = self.net.foward(dataset.data)
        for loss in losses:
            loss_fn = LossFunction(loss)
            # move predictions outside loop
            res[loss] = loss_fn.foward(pred, dataset.labels)
        return res
