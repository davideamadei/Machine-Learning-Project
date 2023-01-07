# Machine-Learning-Project

Current Code Structure is as follows:

data.py : contains interface to Datasets
- [dataclass] Dataset: holder of (data, labels, ids)
- [functions] train_valid_test_split, read_ML_cup, read_monks

loss.py : contains loss functions
- [class] LossFunction: defines generic loss function behavioud
  - corrently implements: MSE

estimator.py : contains model logic (esp. training utilities)
- [class] Estimator: defines a trainable model
  - uses a LossFunction, Optimizer, NeuralNetwork

nn.py : contains neural network layers
- [dataclass] Parameter: holder of weights and biases
  - currently overloaded also holdes gradients and deltas
- [abstract class] Layer: defines functions a Layer should implement
  - a concrete implementation is: ActivationFunction, (Dropout)
- [abstract class] UpdatableLayer: defines functions a Layer with parameters should implement
  - a concrete implementation is: LinearLayer, (ConvolutionalLayer)
- [class] ActivationFunction: defines a generic activation function behaviour
  - currently implements: ReLU
- [class] LinearLayer: defines a linear layer
- [class] NeuralNetwork: defines a NeuralNetwork (sequence of layers)
