from ..utils import Dataset
from ..nn import Estimator

__all__ = ["EarlyStopping", "TrainingThresholdStopping", "Logger"]


class EarlyStopping:
    """implements early stopping callback"""

    def __init__(
        self,
        estimator: Estimator,
        losses: list[str] = ["MSE"],
        checks_to_stop: int = 10,
        check_frequency: int = 10,
        eps: int = 10 - 8,
        verbose: bool = False,
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
        eps : int
            tolerance for the decrease of the validation loss
        verbose : bool
            if True adds some prints
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
        self._eps = eps
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
        # only check every given number number of epochs
        if (current_epoch - 1) % self._check_frequency == 0:
            validation_set = self._validation_set
            estimator = self._estimator
            losses = self._losses
            # evaluate the validation set
            validation_loss = self._estimator.evaluate(
                losses=losses, dataset=validation_set
            )
            # current loss is less than (best loss - eps)
            if validation_loss[losses[0]] < self._best_vl_loss[losses[0]] - self._eps:
                self._n_worse_checks = 0
                self._best_vl_loss = validation_loss
                self._best_epoch = current_epoch
                self._best_tr_loss = record["loss"]
            # no significant increase in loss
            else:
                self._n_worse_checks += 1
                # early stopping condition was reached
                if self._n_worse_checks == self._checks_to_stop:
                    if self._verbose:
                        print(
                            f"Stopped early at epoch {current_epoch} after"
                            f" {self._n_worse_checks} check(s) had a validation loss"
                            " worse than the current best one."
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


class Logger:
    """class implementing a callback to log results during training for a training set and a validation set.
    Mainly used during cross validation, can also be used for holdout
    by passing a training set and validation set at creation time.
    For every fold, saves the training loss and validation loss for each epoch and for each loss function that was given.
    """

    def __init__(
        self,
        estimator: Estimator,
        losses: list[str],
        training_set: Dataset = None,
        every: int = 1,
        validation_set: Dataset = None,
    ):
        """init method

        Parameters
        ----------
        estimator : Estimator
            estimator to use for evaluation
        losses : list[str]
            list of loss functions to use for evaluation
        every : int
            how many epochs need to pass between evaluations
        training_set : Dataset, optional
            training set to evaluate, by default None
        validation_set : Dataset, optional
            validation set to evaluate, by default None
        """
        self._estimator = estimator
        self._losses = losses
        # logs are a list of dictionaries
        self._scores = []
        self._every = every
        # if training set and validation set are given, initialize for a single training
        if training_set != None and validation_set != None:
            self._scores.append({"folds": []})
            self.update_fold(fold_dict={"train": training_set, "test": validation_set})

    def update_hp(self, hp_dict: dict):
        """function to update the logger after changing hyperparameters

        Parameters
        ----------
        hp_dict : dict
            dictionary containing the current combination of hyperparameters
        """
        # append a new dictionary with keys 'hp' and 'folds'. The value for 'hp' is the current combination of hyperparamaters
        # while 'folds' is initialized as an empty list
        self._scores.append({"hp": hp_dict.copy(), "folds": []})

    def update_fold(self, fold_dict: dict):
        """function to update the logger after changing folds

        Parameters
        ----------
        fold_dict : dict
            dictionary representing a fold, contains a value 'test' for the validation set
            and a value 'train' for the training set
        """
        # get the current fold with the split on training and test set
        self._vfold = fold_dict["test"]
        self._tfold = fold_dict["train"]

        # append a new dictionary corresponding to the current fold to the list of folds for the current combination of hyperparameters
        # 'train' and 'valid' contain a dictionary where the keys are the names of the loss functions and the values are a list of
        # loss values, one for each evaluation made by the logger
        current_hp_config = self._scores[-1]
        current_hp_config["folds"].append({"train": {}, "valid": {}})

        # initialize the dictionaries for 'train' and 'valid'
        current_fold = current_hp_config["folds"][-1]
        for loss in self._losses:
            current_fold["train"][loss] = []
            current_fold["valid"][loss] = []

    def __call__(self, record: dict):
        """call method

        Parameters
        ----------
        record : dict
            dictionary containing the current epoch
        """
        # only update logs every given number of epochs
        if (record["epoch"] - 1) % self._every == 0:
            # evaluate the training and validation set on the given list of loss functions
            dt = self._estimator.evaluate(self._losses, self._tfold)
            dv = self._estimator.evaluate(self._losses, self._vfold)

            current_train = self._scores[-1]["folds"][-1]["train"]
            current_valid = self._scores[-1]["folds"][-1]["valid"]

            # append evaluations to the corresponding lists
            for loss in self._losses:
                current_train[loss].append(dt[loss])
                current_valid[loss].append(dv[loss])
