#TODO maybe change ValueError exceptions to show wrong value
#TODO add comments to nested k-fold and check those in k-fold
#TODO EarlyStopping docs

# python libraries
import numpy as np
from typing import Iterator, Callable
import itertools
from numbers import Number

# local libraries
from estimator import Estimator
from util_classes import Dataset
from optimizer import Optimizer
from nn import NeuralNetwork, LinearLayer, ActivationFunction

from loss import LossFunction

class EarlyStopping:
    def __init__(self, estimator: Estimator, losses: list[str] = ['MSE'], checks_to_stop: int = 10, check_frequency: int = 10) -> None:
        self._estimator = estimator
        self._losses = losses
        self._n_worse_checks = 0
        self._checks_to_stop = checks_to_stop
        self._check_frequency = check_frequency
        self._current_epoch = 0
        self._current_best = dict.fromkeys(losses, float('inf'))
        self._best_epoch = 0
    def __call__(self, record) -> None:
        current_epoch = record['epoch']
        self._current_epoch = current_epoch
        if (current_epoch - 1) % self._check_frequency == 0:
            validation_set = self._validation_set
            estimator = self._estimator
            losses = self._losses
            validation_loss = self._estimator.evaluate(losses=losses, dataset=validation_set)
            if validation_loss[losses[0]] < self._current_best[losses[0]]:
                self._n_worse_checks = 0
                self._current_best = validation_loss
                self._best_epoch = current_epoch
            else:
                self._n_worse_checks += 1
                if self._n_worse_checks == self._checks_to_stop:
                    print(f'Stopped early at epoch {current_epoch}.')
                    estimator.stop_training = True
    
    def reset(self) -> None:
        self._n_worse_checks = 0
        self._best_epoch = 0
        self._current_best = dict.fromkeys(self._losses, float('inf'))

    def set_validation_set(self, validation_set: Dataset) -> None:
        self.reset()
        self._validation_set = validation_set

class GridSearch:
    _net_keys = ["layers"]
    _optimizer_keys = ["eta", "l2_coeff", "alpha"]
    _loss_keys = ["fname"]
    _estimator_keys = ["batchsize"]
    # _global_keys = _net_keys + _optimizer_keys + _loss_keys + _estimator_keys

    # dictionary containing translations from exposed names to names to pass to functions internally
    _param_name_translations = {
        "layers": "layers",
        "l2": "l2_coeff",
        "momentum": "alpha",
        "eta": "eta",
        "loss": "fname",
        "batchsize": "batchsize",
    }

    # check param_grid validity
    @staticmethod
    def _check_param_grid(param_dict: dict, hyper_grid: dict) -> None:
        """Checks that the given grid of hyperparameters is correct

        Parameters
        ----------
        param_dict : dict
            dictionary of accepted hyperparameters
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

        activation_functions = ["ReLU", 'logistic', 'tanh', "linear"]
        loss_functions = ["MSE"]

        for key in param_dict.keys():
            if not key in hyper_grid.keys():
                raise ValueError(
                    (
                        "All the following parameters must be present in the"
                        " hyperparameter grid: "
                    ),
                    list(param_dict.keys()),
                )
            if not isinstance(hyper_grid[key], list) or len(hyper_grid[key]) == 0:
                raise ValueError(
                    "Each parameter must have an associated not empty list of"
                    " parameters"
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
        for l2_coeff in hyper_grid["l2"]:
            if not isinstance(l2_coeff, Number):
                raise TypeError("The L2 coefficient must be a number")
            if l2_coeff < 0:
                raise ValueError("The L2 coefficient must be at least 0")

        # check momentum
        for momentum in hyper_grid["momentum"]:
            if not isinstance(momentum, Number):
                raise TypeError("The momentum parameter must be a number")
            if momentum < 0:
                raise ValueError("The momentum parameter must be at least 0")

        # check eta
        for eta in hyper_grid["eta"]:
            if not isinstance(eta, Number):
                raise TypeError("The learning rate must be a number")
            if eta <= 0:
                raise ValueError("The learning rate must be greater than 0")

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
        GridSearch._check_param_grid(self._param_name_translations, hyper_grid)

        # translate names of parameters and sort by key for better efficiency
        new_grid = {}
        for key in self._param_name_translations.keys():
            if key in hyper_grid:
                new_grid[self._param_name_translations[key]] = hyper_grid[key]
        self._hyper_grid = new_grid

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
        estimator_params = {key: combination[key] for key in self._estimator_keys}
        optimizer_params = {key: combination[key] for key in self._optimizer_keys}
        net_params = {key: combination[key] for key in self._net_keys}

        # create dictionary of params to pass to constructors
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
        loss_list: list[str] = ['MSE'],
        early_stopping: tuple[int, int] = None
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

        #creates folds
        folds = self._generate_folds(dataset=dataset, n_folds=n_folds)

        #creates early stopping if it was passed to the function
        if early_stopping != None:
            early_stopper = EarlyStopping(estimator=estimator, losses=loss_list, checks_to_stop=early_stopping[0], check_frequency=early_stopping[1])


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

            print(combination)

            # iterates folds of dataset
            for train_set, test_set in folds:
                if early_stopping != None:
                    early_stopper.set_validation_set(test_set)
                    def my_callback(record: dict) -> None:
                        callback(record)
                        early_stopper(record)
                    estimator.train(dataset=train_set, n_epochs=n_epochs, callback=my_callback)
                    epoch_list.append(early_stopper._best_epoch)
                    test_loss_list.append(early_stopper._current_best)
                else:
                    estimator.train(dataset=train_set, n_epochs=n_epochs, callback=callback)
                    test_loss_list.append(estimator.evaluate(losses = loss_list, dataset = test_set))
                estimator.reset()

            test_loss_avg = {}
            test_loss_std = {}
            for loss in loss_list:
                test_loss_avg[loss] = sum(d[loss] for d in test_loss_list) / len(test_loss_list)
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
                combination_results[-1]['epoch_avg'] = np.average(epoch_list)
                combination_results[-1]['epoch_std'] = np.std(epoch_list)


            print(combination_results)
        if(loss_list[0] == 'binary_accuracy'):
            combination_results.sort(key=lambda x: x["test_loss_avg"][loss_list[0]], reverse = True)
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
        loss_list: list[str] = ['MSE'],
        early_stopping: tuple[int, int] = None
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
                f'The number of folds cannot be greater than the number of samples in'
                f' the dataset: {outer_n_folds} > {data_size}'
            )

        # check inner_n_folds value
        if inner_n_folds > data_size:
            raise ValueError(
                f"The number of folds cannot be greater than the number of samples in"
                f" the dataset: {inner_n_folds} > {data_size}"
            )

        test_loss_list = []
        param_combination_list = []

        for train_set, test_set in folds:
            train_results = self.k_fold(
                dataset=train_set,
                n_folds=inner_n_folds,
                n_epochs=n_epochs,
                callback=inner_callback,
                early_stopping = early_stopping
            )
            params = train_results[0]["parameters"]
            estimator_params = self._create_estimator_params(params, input_dim)
            if estimator_params["batchsize"] == -1:
                estimator_params["batchsize"] = data_size

            estimator.update_params(**estimator_params)
            if early_stopping != None:
                early_epochs = int(train_results[0]['epoch_avg'])
                estimator.train(dataset=train_set, n_epochs=early_epochs, callback=outer_callback)
            else:
                estimator.train(dataset=train_set, n_epochs=n_epochs, callback=outer_callback)
            test_loss_list.append(estimator.evaluate(losses = loss_list, dataset = test_set))
            param_combination_list.append(params)

        test_loss_avg = {}
        test_loss_std = {}
        for loss in loss_list:
            test_loss_avg[loss] = sum(d[loss] for d in test_loss_list) / len(test_loss_list)
            test_loss_std[loss] = np.std([d[loss] for d in test_loss_list])

        results = {
            "test_loss_list": list(zip(param_combination_list, test_loss_list)),
            "test_loss_avg": test_loss_avg,
            "test_loss_std": test_loss_std,
        }
        return results