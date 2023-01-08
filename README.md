# Machine-Learning-Project

General conventions:
- documentation generated with ([vscode-plugin](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)).
  - set the plugin to numpy mode (with typing)
- [maybe] we could use a code formatter [black](https://github.com/psf/black)

Current Code Structure is as follows:

data.py : contains interface to Datasets
- *[dataclass]* **Dataset**:<br>holder of (data, labels, ids)
- *[functions]* train_valid_test_split, read_ML_cup, read_monks

loss.py : contains loss functions
- *[class]* **LossFunction**:<br> defines generic loss function behavioud
  - corrently implements: MSE

estimator.py : contains model logic (esp. training utilities)
- *[class]* **Estimator**:<br> defines a trainable model
  - uses a LossFunction, Optimizer, NeuralNetwork

nn.py : contains neural network layers
- *[dataclass]* **Parameter**:<br> holder of weights and biases
  - currently overloaded also holdes gradients and deltas
- *[abstract class]* **Layer**:<br> defines functions a Layer should implement
  - a concrete implementation is: ActivationFunction, (Dropout)
- *[abstract class]* **UpdatableLayer**:<br> defines functions a Layer with parameters should implement
  - a concrete implementation is: LinearLayer, (ConvolutionalLayer)
- *[class]* **ActivationFunction**:<br> defines a generic activation function behaviour
  - currently implements: ReLU
- *[class]* **LinearLayer**:<br> defines a linear layer
- *[class]* **NeuralNetwork**:<br> defines a NeuralNetwork (sequence of layers)
