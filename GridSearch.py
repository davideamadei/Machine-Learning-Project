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

# def filter_dict_by_key(dictionary: dict, keys: list[str]) -> dict:
#     filtered_dict = {}
#     for key in keys:
#         if key in dictionary.keys():
#             filtered_dict[key] = dictionary[key]
#     return filtered_dict


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
    def _check_param_grid(param_dict, hyper_grid) -> bool:

        activation_functions = ["ReLU", "linear"]
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

        return True

    def __init__(self, estimator: Estimator, hyper_grid: dict):
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
    ) -> Iterator[tuple[Dataset, Dataset]]:

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
            layer_list.append(ActivationFunction(fname=layer_list[1]))
        estimator_params["net"] = NeuralNetwork(layer_list)
        return estimator_params

    # returns the best set of hyperparameters
    def k_fold(
        self,
        dataset: Dataset,
        n_folds: int,
        n_epochs: int,
        callback: Callable[[dict], None] = print,
    ) -> dict:

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

        folds = self._generate_folds(dataset=dataset, n_folds=n_folds)

        # generates all combinations of hyperparameters
        keys, values = zip(*hyper_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        combination_loss_list = []

        # iterates all combinations of hyperparameters
        for combination in param_combinations:

            estimator_params = self._create_estimator_params(combination, input_dim)
            if estimator_params["batchsize"] == -1:
                estimator_params["batchsize"] = data_size
            estimator.update_params(**estimator_params)

            test_loss_list = []

            print(combination)

            # iterates folds of dataset
            for train_set, test_set in folds:
                estimator.train(dataset=train_set, n_epochs=n_epochs, callback=callback)
                test_loss_list.append(estimator.evaluate(test_set))
                estimator.reset()

            test_loss_avg = sum(test_loss_list) / n_folds
            test_loss_std = np.std(test_loss_list)

            combination_loss_list.append(
                {
                    "parameters": combination,
                    "test_loss_avg": test_loss_avg,
                    "test_loss_std": test_loss_std,
                }
            )

        combination_loss_list.sort(key=lambda x: x["test_loss_avg"])
        return combination_loss_list

    # returns an estimation of the risk for the model, average +- standard deviation
    def nested_k_fold(
        self,
        dataset: Dataset,
        inner_n_folds: int,
        outer_n_folds: int,
        n_epochs: int,
        inner_callback: Callable[[dict], None] = print,
        outer_callback: Callable[[dict], None] = print,
    ) -> dict:

        estimator = self._estimator
        data_size = dataset.shape[0]
        folds = self._generate_folds(dataset=dataset, n_folds=outer_n_folds)
        input_dim = dataset.shape[1][0]

        # check outer_n_folds value
        if outer_n_folds > data_size:
            raise ValueError(
                "The number of folds cannot be greater than the number of samples in"
                " the dataset"
            )

        test_loss_list = []

        for train_set, test_set in folds:
            train_results = self.k_fold(
                dataset=train_set,
                n_folds=inner_n_folds,
                n_epochs=n_epochs,
                callback=inner_callback,
            )
            params = train_results[0]["parameters"]
            estimator_params = self._create_estimator_params(params, input_dim)
            if estimator_params["batchsize"] == -1:
                estimator_params["batchsize"] = data_size

            estimator.update_params(**estimator_params)
            estimator.train(
                dataset=train_set, n_epochs=n_epochs, callback=outer_callback
            )
            test_loss_list.append(estimator.evaluate(test_set))

        test_loss_avg = sum(test_loss_list) / outer_n_folds
        test_loss_std = np.std(test_loss_list)

        results = {
            "test_loss_list": test_loss_list,
            "test_loss_avg": test_loss_avg,
            "test_loss_std": test_loss_std,
        }
        return results
