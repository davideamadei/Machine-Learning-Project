# external libraries
import numpy as np
from typing import Iterator, Callable
import itertools

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

    # check param_grid and remove invalid values
    # TODO implement
    @staticmethod
    def _check_param_grid(hyper_grid) -> bool:
        return True

    def __init__(self, estimator: Estimator, hyper_grid: dict):
        if estimator == None or type(estimator) != Estimator:
            raise TypeError
        self._estimator = estimator
        if hyper_grid == None or type(hyper_grid) != dict:
            raise TypeError

        # check for wrong values
        if not GridSearch._check_param_grid(hyper_grid):
            raise ValueError

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

        data_size = dataset.ids.shape[0]
        if n_folds > data_size:
            raise ValueError

        hyper_grid = self._hyper_grid
        estimator = self._estimator
        input_dim = dataset.shape[1][0]
        output_dim = dataset.shape[1][1]

        # check if output layer is correct for all combinations
        for layers in hyper_grid["layers"]:
            if layers[-1][0] != output_dim:
                raise ValueError(
                    "Number of units in last layer must be equal to the output dimension of the data"
                )

        folds = self._generate_folds(dataset=dataset, n_folds=n_folds)

        # generates all combinations of hyperparameters
        keys, values = zip(*hyper_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        combination_loss_list = []

        # iterates all combinations of hyperparameters
        for combination in param_combinations:

            estimator_params = self._create_estimator_params(combination, input_dim)
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
        folds = self._generate_folds(dataset=dataset, n_folds=outer_n_folds)
        input_dim = dataset.shape[1][0]
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
