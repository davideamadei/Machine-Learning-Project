# TODO maybe change ValueError exceptions to show wrong value
# TODO add comments to nested k-fold and check those in k-fold
# TODO EarlyStopping docs

# python libraries
from typing import Callable
import itertools
from numbers import Number

# external libraries
import numpy as np

# local libraries
from ddnn.utils import Dataset
from ddnn.nn import (
    Estimator,
    LossFunction,
    Optimizer,
    LinearLayer,
    ActivationFunction,
    Initializer,
    NeuralNetwork,
)
from .callback import EarlyStopping, TrainingThresholdStopping

__all__ = ["GridSearch"]


# TODO maybe change ValueError exceptions to show wrong value
# TODO add comments to nested k-fold and check those in k-fold


class GridSearch:
    _optional_keys = ["fan_mode"]
    _net_keys = ["layers"]
    _weight_initializer_keys = ["weight_initializer"]
    _optimizer_keys = [
        "optimizer",
        "learning_rate",
        "l2_coefficient",
        "momentum_coefficient",
    ]
    _loss_keys = ["loss"]
    _estimator_keys = ["batchsize"]
    _global_keys = (
        _net_keys
        + _optimizer_keys
        + _weight_initializer_keys
        + _loss_keys
        + _estimator_keys
    )

    # check param_grid validity
    @staticmethod
    def _check_param_grid(
        param_list: list[str], optional_params: list[str], hyper_grid: dict
    ) -> None:
        """Checks that the given grid of hyperparameters is correct

        Parameters
        ----------
        param_list : list
            list of accepted hyperparameters
        hyper_grid : dict
            dictionary containing the grid of hyperparameters on which to perform the validity check

        Returns
        -------
        None

        Raises
        ------
        ValueError
            when an hyperparameter is missing or one of its possible values is not valid
        TypeError
            when an hyperparameter has a value with the wrong type or is not a list of values
        """

        weight_initializers = [
            "random_uniform",
            "random_normal",
            "glorot_uniform",
            "glorot_normal",
            "he_uniform",
            "he_normal",
        ]
        optimizers = ["SGD", "ADAM"]
        activation_functions = ["ReLU", "logistic", "tanh", "linear"]
        loss_functions = ["MSE", "binary_accuracy"]

        for key in param_list:
            if not key in hyper_grid.keys():
                raise ValueError(
                    (
                        "All the following parameters must be present in the"
                        " hyperparameter grid: "
                    ),
                    list(param_list),
                )
            if not isinstance(hyper_grid[key], list) or len(hyper_grid[key]) == 0:
                raise ValueError(
                    "Each parameter must have an associated not empty list of"
                    " parameters"
                )

        # check that only accepted parameters were passed
        for key in hyper_grid.keys():
            if key not in param_list and key not in optional_params:
                raise ValueError(
                    f"{key} not accepted as hyperparameter. Only the following hyperparameters are accepted. Mandatory: {param_list}. Optional: {optional_params}"
                )

        # check layers
        for net in hyper_grid["layers"]:
            for layer in net:
                if (
                    not isinstance(layer, tuple)
                    or not len(layer) == 2
                    or not isinstance(layer[0], int)
                    or not isinstance(layer[1], str)
                ):
                    raise TypeError(
                        "The layers parameter accepts a list of tuples of length 2 with"
                        " the first element being an integer that is the number of"
                        " units in that layer, and the second element is a string"
                        " containing the name of the activation function for that layer"
                    )
                if layer[0] <= 0:
                    raise ValueError("The number of units must be greater than 0")
                if not layer[1] in activation_functions:
                    raise ValueError(
                        (
                            "Only the following values are accepted for activation"
                            " function: "
                        ),
                        activation_functions,
                    )

        # check l2
        for l2_coeff in hyper_grid["l2_coefficient"]:
            if not isinstance(l2_coeff, Number):
                raise TypeError("The L2 coefficient must be a number")
            if l2_coeff < 0:
                raise ValueError("The L2 coefficient must be at least 0")

        # check momentum
        for momentum in hyper_grid["momentum_coefficient"]:
            if not isinstance(momentum, Number):
                raise TypeError("The momentum parameter must be a number")
            if momentum < 0:
                raise ValueError("The momentum parameter must be at least 0")

        # check learning rate
        for eta in hyper_grid["learning_rate"]:
            if not isinstance(eta, Number):
                raise TypeError("The learning rate must be a number")
            if eta <= 0:
                raise ValueError("The learning rate must be greater than 0")

        # check optimizers
        for optimizer in hyper_grid["optimizer"]:
            if not isinstance(optimizer, str):
                raise TypeError(
                    "The optimizer must be a string corresponding to the required optimizer"
                )
            if optimizer not in optimizers:
                raise ValueError(
                    "Only the following values are accepted for the optimizer: ",
                    optimizers,
                )

        # check if 'fan_mode' was passed as hyperparameter
        fan_mode_flag = "fan_mode" in hyper_grid.keys()
        # check weight initialization functions
        for weight_init in hyper_grid["weight_initializer"]:
            if not isinstance(weight_init, str):
                raise TypeError(
                    "The weight initialization must be a string corresponding to the required weight initialization function"
                )
            if weight_init not in weight_initializers:
                raise ValueError(
                    "Only the following values are accepted for the weight initialization function: ",
                    weight_initializers,
                )
            if (
                weight_init == "he_uniform" or weight_init == "he_normal"
            ) and not fan_mode_flag:
                raise ValueError(
                    "He initializer requires fan_mode hyperparameter to also be passed with values fan_in or fan_out"
                )
            if (
                weight_init != "he_uniform"
                and weight_init != "he_normal"
                and fan_mode_flag
            ):
                raise ValueError(
                    "fan_mode hyperparameter can only be set in conjunction with he_uniform or he_normal weight initializers"
                )
        # check fan_mode values
        if fan_mode_flag:
            for fan_mode in hyper_grid["fan_mode"]:
                if fan_mode != "fan_in" and fan_mode != "fan_out":
                    raise ValueError(
                        "fan_mode for He weight initialization can only be set to fan_in or fan_out"
                    )

        # check loss functions
        for loss in hyper_grid["loss"]:
            if not isinstance(loss, str):
                raise TypeError(
                    "The loss must be a string corresponding to the required loss"
                )
            if loss not in loss_functions:
                raise ValueError(
                    "Only the following values are accepted for the loss: ",
                    loss_functions,
                )

        # check batchsize
        for batchsize in hyper_grid["batchsize"]:
            if not isinstance(batchsize, int):
                raise TypeError("The batch size must be an integer")
            if batchsize <= 0 and batchsize != -1:
                raise ValueError(
                    "The batch size must be greater than 0. If -1 is passed as a value"
                    " the size of the dataset will be used"
                )

    def __init__(self, estimator: Estimator, hyper_grid: dict):
        """Initializes a new instance

        Parameters
        ----------
        estimator : Estimator
            the estimator to use for training and evaluation
        hyper_grid : dict
            grid of hyperparameters

        Raises
        ------
        TypeError
            when parameter types are incorrect
        """
        if estimator == None or type(estimator) != Estimator:
            raise TypeError
        self._estimator = estimator
        if hyper_grid == None or type(hyper_grid) != dict:
            raise TypeError

        # check for wrong values
        GridSearch._check_param_grid(self._global_keys, self._optional_keys, hyper_grid)
        self._hyper_grid = hyper_grid

    # returns a list of data folds through indexes
    def _generate_folds(
        self, dataset: Dataset, n_folds: int
    ) -> list[tuple[Dataset, Dataset]]:
        """function to generate the folds to use during grid search

        Parameters
        ----------
        dataset : Dataset
            dataset to split in folds
        n_folds : int
            number of folds

        Returns
        -------
        list(tuple[Dataset, Dataset])
            returns a list containing tuples of Dataset classes. Each tuple is of the form (Training set, Test set)

        """

        data_size = dataset.ids.shape[0]
        indices = np.arange(data_size)

        # TODO maybe shuffle not needed if we assume dataset has already been shuffled
        np.random.shuffle(indices)

        folds = []

        for index_lists in np.array_split(indices, n_folds):
            # make mask to split test and training set indices
            mask = np.zeros(data_size, dtype=bool)
            mask[index_lists] = True
            test_indices = indices[mask]
            train_indices = indices[~mask]
            # initialize test set and training set
            test_set = Dataset(
                ids=dataset.ids[test_indices],
                labels=dataset.labels[test_indices],
                data=dataset.data[test_indices],
            )
            train_set = Dataset(
                ids=dataset.ids[train_indices],
                labels=dataset.labels[train_indices],
                data=dataset.data[train_indices],
            )
            folds.append((train_set, test_set))
        return folds

    def _create_estimator_params(self, combination: dict, input_dim: int) -> dict:
        """function to create the dictionary to pass to estimator for update

        Parameters
        ----------
        combination : dict
            combination of hyperparameters to use
        input_dim : int
            number of features of dataset

        Returns
        -------
        dict
            a dictionary to pass to the estimator's update function
        """
        # filter parameters for various classes
        loss_params = {key: combination[key] for key in self._loss_keys}
        loss_params["fname"] = loss_params.pop("loss")
        weight_init_params = {
            key: combination[key] for key in self._weight_initializer_keys
        }
        if (
            combination["weight_initializer"] == "he_uniform"
            or combination["weight_initializer"] == "he_normal"
        ):
            weight_init_params["fan_mode"] = combination["fan_mode"]
        weight_init_params["fname"] = weight_init_params.pop("weight_initializer")
        estimator_params = {key: combination[key] for key in self._estimator_keys}
        optimizer_params = {key: combination[key] for key in self._optimizer_keys}
        optimizer_params["fname"] = optimizer_params.pop("optimizer")
        net_params = {key: combination[key] for key in self._net_keys}

        # create dictionary of params to pass to constructors
        print(weight_init_params, loss_params, optimizer_params)
        estimator_params["initializer"] = Initializer(**weight_init_params)
        estimator_params["loss"] = LossFunction(**loss_params)
        estimator_params["optimizer"] = Optimizer(**optimizer_params)

        # create list of layers to create NN
        old_units = input_dim
        layer_list = []
        for layer in net_params["layers"][:-1]:
            layer_list.append(LinearLayer((old_units, layer[0])))
            # TODO maybe linear layers can be removed
            layer_list.append(ActivationFunction(fname=layer[1]))
            old_units = layer[0]
        last_layer = net_params["layers"][-1]
        layer_list.append(LinearLayer((old_units, last_layer[0])))
        if last_layer[1] != "linear":
            layer_list.append(ActivationFunction(fname=last_layer[1]))
        estimator_params["net"] = NeuralNetwork(layer_list)
        return estimator_params

    # returns the best set of hyperparameters
    def k_fold(
        self,
        dataset: Dataset,
        n_folds: int,
        n_epochs: int,
        callback: Callable[[dict], None] = print,
        loss_list: list[str] = ["MSE"],
        early_stopping: tuple[int, int] = None,
    ) -> list:
        """function to execute a k-fold cross-validation on the given dataset

        Parameters
        ----------
        dataset : Dataset
            dataset to use for k-fold cross-validation
        n_folds : int
            number of folds to use in cross-validation
        n_epochs : int
            number of epochs to run training
        callback : Callable[[dict], None], optional
            callback function to use during training, by default print
        loss_list: list[str]
            list of loss functions to evaluate the test set on
        early_stopping: tuple
            dictionary containing two values, respectively how many checks have to fail before stopping training and how many epochs need to pass between checks

        Returns
        -------
        list
            list containing results of the cross-validation ordered by increasing average loss on the test set.
            Every element is a list of dictionaries containing the
            combination of hyperparameters, the average of the test loss and the standard deviation on the test loss

        Raises
        ------
        ValueError
            when values of some parameters are incorrect
        """

        hyper_grid = self._hyper_grid
        estimator = self._estimator

        data_size = dataset.shape[0]
        input_dim = dataset.shape[1][0]
        output_dim = dataset.shape[1][1]

        # check n_folds value
        if n_folds > data_size:
            raise ValueError(
                "The number of folds cannot be greater than the number of samples in"
                " the dataset"
            )
        # check if output layer is correct for all combinations
        for layers in hyper_grid["layers"]:
            if layers[-1][0] != output_dim:
                raise ValueError(
                    "Number of units in last layer must be equal to the output"
                    " dimension of the data"
                )
        # check values for batchsize
        for batchsize in hyper_grid["batchsize"]:
            if batchsize > data_size:
                raise ValueError(
                    "The batchsize cannot be greater than the number of samples"
                )

        # creates folds
        folds = self._generate_folds(dataset=dataset, n_folds=n_folds)

        # creates early stopping if it was passed to the function
        if early_stopping != None:
            early_stopper = EarlyStopping(
                estimator=estimator,
                losses=loss_list,
                checks_to_stop=early_stopping[0],
                check_frequency=early_stopping[1],
            )

            def new_callback(record: dict) -> None:
                callback(record)
                early_stopper(record)

        # generates all combinations of hyperparameters
        keys, values = zip(*hyper_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        combination_results = []

        # iterates all combinations of hyperparameters
        for combination in param_combinations:

            estimator_params = self._create_estimator_params(combination, input_dim)
            if estimator_params["batchsize"] == -1:
                estimator_params["batchsize"] = data_size
            estimator.update_params(**estimator_params)

            test_loss_list = []
            epoch_list = []
            train_loss_list = []
            print(combination)

            # iterates folds of dataset
            for train_set, test_set in folds:
                if early_stopping != None:
                    early_stopper.set_validation_set(test_set)

                    estimator.train(
                        dataset=train_set, n_epochs=n_epochs, callback=new_callback
                    )
                    epoch_list.append(early_stopper._best_epoch)
                    test_loss_list.append(early_stopper._best_vl_loss)
                    train_loss_list.append(early_stopper._best_tr_loss)
                else:
                    estimator.train(
                        dataset=train_set, n_epochs=n_epochs, callback=callback
                    )
                    test_loss_list.append(
                        estimator.evaluate(losses=loss_list, dataset=test_set)
                    )
                estimator.reset()

            test_loss_avg = {}
            test_loss_std = {}
            for loss in loss_list:
                test_loss_avg[loss] = sum(d[loss] for d in test_loss_list) / len(
                    test_loss_list
                )
                test_loss_std[loss] = np.std([d[loss] for d in test_loss_list])

            # test_loss_avg = sum(test_loss_list) / n_folds
            # test_loss_std = np.std(test_loss_list)

            combination_results.append(
                {
                    "parameters": combination,
                    "test_loss_avg": test_loss_avg,
                    "test_loss_std": test_loss_std,
                }
            )

            if early_stopping != None:
                combination_results[-1]["n_epoch_avg"] = np.average(epoch_list)
                combination_results[-1]["n_epoch_std"] = np.std(epoch_list)
                combination_results[-1]["train_loss_avg"] = np.average(train_loss_list)
                combination_results[-1]["train_loss_std"] = np.std(train_loss_list)

        if loss_list[0] == "binary_accuracy":
            combination_results.sort(
                key=lambda x: x["test_loss_avg"][loss_list[0]], reverse=True
            )
        else:
            combination_results.sort(key=lambda x: x["test_loss_avg"][loss_list[0]])
        return combination_results

    # returns an estimation of the risk for the model, average +- standard deviation
    def nested_k_fold(
        self,
        dataset: Dataset,
        inner_n_folds: int,
        outer_n_folds: int,
        n_epochs: int,
        inner_callback: Callable[[dict], None] = print,
        outer_callback: Callable[[dict], None] = print,
        loss_list: list[str] = ["MSE"],
        early_stopping: tuple[int, int] = None,
    ) -> dict:
        """function implementing nested k-fold cross validation

        Parameters
        ----------
        dataset : Dataset
            dataset to run cross validation on
        inner_n_folds : int
            number of folds for the inner cross validation
        outer_n_folds : int
            number of folds for the outer cross validation
        n_epochs : int
            number of epochs to run training for
        inner_callback : Callable[[dict], None], optional
            callback function to use during training for the inner cross validation, by default print
        outer_callback : Callable[[dict], None], optional
            callback function to use during training for the outer cross validation, by default print
        loss_list: list[str]
            list of loss functions to evaluate the test set on
        early_stopping: dict
            dictionary containing 'checks_to_stop' and 'check_frequency', respectively how many checks have to fail before stopping training and how many epochs need to pass between checks

        Returns
        -------
        dict
            a dictionary containing a list of tuples each containing the best combination of hyperparameters
            on that fold and the corresponding loss on the test set for that fold, the average loss on the test sets across the folds
            and their standard deviation

        Raises
        ------
        ValueError
            when values are incorrect
        """

        estimator = self._estimator
        data_size = dataset.shape[0]
        folds = self._generate_folds(dataset=dataset, n_folds=outer_n_folds)
        input_dim = dataset.shape[1][0]

        # check outer_n_folds value
        if outer_n_folds > data_size:
            raise ValueError(
                f"The number of folds cannot be greater than the number of samples in"
                f" the dataset: {outer_n_folds} > {data_size}"
            )

        # check inner_n_folds value
        if inner_n_folds > data_size:
            raise ValueError(
                f"The number of folds cannot be greater than the number of samples in"
                f" the dataset: {inner_n_folds} > {data_size}"
            )

        if early_stopping != None:
            threshold_stopper = TrainingThresholdStopping(
                estimator=estimator, threshold_loss=0
            )

            def new_callback(record: dict) -> None:
                outer_callback(record)
                threshold_stopper(record)

        test_loss_list = []
        param_combination_list = []

        for train_set, test_set in folds:
            train_results = self.k_fold(
                dataset=train_set,
                n_folds=inner_n_folds,
                n_epochs=n_epochs,
                callback=inner_callback,
                early_stopping=early_stopping,
            )
            params = train_results[0]["parameters"]
            estimator_params = self._create_estimator_params(params, input_dim)
            if estimator_params["batchsize"] == -1:
                estimator_params["batchsize"] = data_size

            estimator.update_params(**estimator_params)
            if early_stopping != None:
                early_epochs = int(train_results[0]["n_epoch_avg"])
                threshold_stopper.update_threshold(train_results[0]["train_loss_avg"])
                print(f'threshold = {train_results[0]["train_loss_avg"]}')
                estimator.train(
                    dataset=train_set, n_epochs=early_epochs, callback=new_callback
                )
            else:
                estimator.train(
                    dataset=train_set, n_epochs=n_epochs, callback=outer_callback
                )
            test_loss_list.append(
                estimator.evaluate(losses=loss_list, dataset=test_set)
            )
            param_combination_list.append(params)

        test_loss_avg = {}
        test_loss_std = {}
        for loss in loss_list:
            test_loss_avg[loss] = sum(d[loss] for d in test_loss_list) / len(
                test_loss_list
            )
            test_loss_std[loss] = np.std([d[loss] for d in test_loss_list])

        results = {
            "test_loss_list": list(zip(param_combination_list, test_loss_list)),
            "test_loss_avg": test_loss_avg,
            "test_loss_std": test_loss_std,
        }
        return results
