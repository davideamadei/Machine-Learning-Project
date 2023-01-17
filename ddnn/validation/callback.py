from ..utils import Dataset
from ..nn import Estimator

__all__ = ["EarlyStopping", "TrainingThresholdStopping"]


class EarlyStopping:
    """implements early stopping callback"""

    def __init__(
        self,
        estimator: Estimator,
        losses: list[str] = ["MSE"],
        checks_to_stop: int = 10,
        check_frequency: int = 10,
        verbose : bool = False
    ) -> None:
        """init method

        Parameters
        ----------
        estimator : Estimator
            estimator to use for evaluation and to notify when training has to be stopped
        losses : list[str], optional
            list of losses on which to evaluate the validation set, first one is used to check, by default ["MSE"]
        checks_to_stop : int, optional
            number of checks without a decrease in loss needed to stop training, by default 10
        check_frequency : int, optional
            how often to do a check in terms of epochs, by default 10
        """
        self._estimator = estimator
        self._losses = losses
        self._n_worse_checks = 0
        self._checks_to_stop = checks_to_stop
        self._check_frequency = check_frequency
        self._current_epoch = 0
        self._best_vl_loss = dict.fromkeys(losses, float("inf"))
        self._best_tr_loss = 0
        self._best_epoch = 0
        self._verbose = verbose

    def __call__(self, record: dict) -> None:
        """call method

        Parameters
        ----------
        record : dict
            record containing the epoch number and the loss on the training set at that epoch
        """
        current_epoch = record["epoch"]
        self._current_epoch = current_epoch
        if (current_epoch - 1) % self._check_frequency == 0:
            validation_set = self._validation_set
            estimator = self._estimator
            losses = self._losses
            validation_loss = self._estimator.evaluate(
                losses=losses, dataset=validation_set
            )
            if validation_loss[losses[0]] < self._best_vl_loss[losses[0]]:
                self._n_worse_checks = 0
                self._best_vl_loss = validation_loss
                self._best_epoch = current_epoch
                self._best_tr_loss = record["loss"]
            else:
                self._n_worse_checks += 1
                if self._n_worse_checks == self._checks_to_stop:
                    if self._verbose:
                        print(
                            f"Stopped early at epoch {current_epoch} after"
                            f" {self._n_worse_checks} check(s) had a validation loss worse"
                            " than the current best one."
                        )
                    estimator.stop_training = True

    def reset(self) -> None:
        """method to reset the early stopping class"""
        self._n_worse_checks = 0
        self._best_epoch = 0
        self._best_vl_loss = dict.fromkeys(self._losses, float("inf"))
        self._best_tr_loss = 0

    def set_validation_set(self, validation_set: Dataset) -> None:
        """method to set the validation set on which checks must be done, also resets the class

        Parameters
        ----------
        validation_set : Dataset
            dataset to use for checks
        """
        self.reset()
        self._validation_set = validation_set


class TrainingThresholdStopping:
    """class implementing a threshold stopping callback on the training set loss"""

    def __init__(self, estimator: Estimator, threshold_loss: float) -> None:
        """init method

        Parameters
        ----------
        estimator : Estimator
            estimator to use for evaluation and to notify when training has to be stopped
        threshold_loss : float
            threshold under which the training loss has to go for training to be stopped
        """
        self._estimator = estimator
        self._threshold_loss = threshold_loss

    def __call__(self, record: dict) -> None:
        """call method

        Parameters
        ----------
        record : dict
            record containing the epoch number and the loss on the training set at that epoch
        """
        if record["loss"] < self._threshold_loss:
            print(
                f"Stopped training early at epoch {record['epoch']} as threshold loss"
                f" of {self._threshold_loss} on training set was reached."
            )
            self._estimator.stop_training = True

    def update_threshold(self, threshold_loss: float) -> None:
        """method to update the trheshold stopping with a new threshold

        Parameters
        ----------
        threshold_loss : float
            new threshold under which the training loss has to go for training to be stopped
        """
        self._threshold_loss = threshold_loss
