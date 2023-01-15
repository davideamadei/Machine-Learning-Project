# TODO EarlyStopping and TrainingThresholdStopping docs

from ddnn.utils import Dataset
from ddnn.nn import Estimator

__all__ = ['EarlyStopping', 'TrainingThresholdStopping']

class EarlyStopping:
    def __init__(
        self,
        estimator: Estimator,
        losses: list[str] = ["MSE"],
        checks_to_stop: int = 10,
        check_frequency: int = 10,
    ) -> None:
        self._estimator = estimator
        self._losses = losses
        self._n_worse_checks = 0
        self._checks_to_stop = checks_to_stop
        self._check_frequency = check_frequency
        self._current_epoch = 0
        self._best_vl_loss = dict.fromkeys(losses, float("inf"))
        self._best_tr_loss = 0
        self._best_epoch = 0

    def __call__(self, record: dict) -> None:
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
                self._best_tr_loss = record['loss']
            else:
                self._n_worse_checks += 1
                if self._n_worse_checks == self._checks_to_stop:
                    print(f"Stopped early at epoch {current_epoch} after {self._n_worse_checks} check(s) had a validation loss worse than the current best one.")
                    estimator.stop_training = True

    def reset(self) -> None:
        self._n_worse_checks = 0
        self._best_epoch = 0
        self._best_vl_loss = dict.fromkeys(self._losses, float("inf"))
        self._best_tr_loss = 0

    def set_validation_set(self, validation_set: Dataset) -> None:
        self.reset()
        self._validation_set = validation_set

class TrainingThresholdStopping:
    def __init__(self, estimator: Estimator, threshold_loss: float) -> None:
        self._estimator = estimator
        self._threshold_loss = threshold_loss
    
    def __call__(self, record: dict) -> None:
        if record['loss'] < self._threshold_loss:
            print(f'Stopped training early at epoch {record["epoch"]} as threshold loss of {self._threshold_loss} on training set was reached.')
            self._estimator.stop_training = True
    
    def update_threshold(self, threshold_loss: float) -> None:
        self._threshold_loss = threshold_loss