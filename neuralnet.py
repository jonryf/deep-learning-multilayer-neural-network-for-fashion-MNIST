################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Manjot Bilkhu
# Winter 2020
################################################################################
# We've provided you with the dataset in PA2.zip
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

import os, gzip
import yaml
import numpy as np

train_std = None
train_mean = None


def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open('config.yaml', 'r'), Loader=yaml.SafeLoader)


def normalize_data(img):
    """
    Normalize your inputs here and return them.
    """

    return (img - train_mean) / train_std


def one_hot_encoding(labels, num_classes=10):
    """
    Encode labels using one hot encoding and return them.
    """
    one_hot_labels = []
    for label in labels:
        ohe = [0] * num_classes
        ohe[int(label)] = 1
        one_hot_labels.append(ohe)
    return np.array(one_hot_labels)


def load_data(path, mode='train'):
    """
    Load Fashion MNIST data.
    Use mode='train' for train and mode='t10k' for test.
    """
    labels_path = os.path.join(path, f'{mode}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{mode}-images-idx3-ubyte.gz')
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    if mode == 'train':
        train_mean = np.mean(images)
        train_std = np.std(images)

    normalized_images = normalize_data(images)
    one_hot_labels = one_hot_encoding(labels, num_classes=10)

    return normalized_images, one_hot_labels


def softmax(x):
    """
    Implement the softmax function here.
    Remember to take care of the overflow condition.
    """
    return np.exp(x) / np.sum(np.exp(x), axis=len(x.shape) - 1, keepdims=True)


class Activation:
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type="sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type
        # Placeholder for input. This will be used for computing gradients.
        self.x = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        Implement the sigmoid activation here.
        """
        self.x = 1 / (1 + np.exp(-x))
        return self.x

    def tanh(self, x):
        """
        Implement tanh here.
        """
        self.x = np.tanh(x)
        return self.x

    def ReLU(self, x):
        """
        Implement ReLU here.
        """
        self.x = np.maximum(0, x)
        return self.x

    def grad_sigmoid(self):
        """
        Compute the gradient for sigmoid here.
        """
        return self.x * (1 - self.x)

    def grad_tanh(self):
        """
        Compute the gradient for tanh here.
        """
        return 1 - self.x ** 2

    def grad_ReLU(self):
        """
        Compute the gradient for ReLU here.
        """
        grad = self.x
        grad[self.x <= 0] = 0
        grad[self.x > 0] = 1
        return grad


class Layer:
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(784, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        self.w = np.random.randn(in_units, out_units)  # Declare the Weight matrix
        self.b = np.array(np.zeros((1, out_units)))  # Create a placeholder for Bias
        self.x = None  # Save the input to forward in this
        self.a = None  # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        Compute the forward pass through the layer here.
        Do not apply activation here.
        Return self.a
        """
        self.x = x
        self.a = self.x.dot(self.w) + self.b
        return self.a

    def backward(self, delta):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        self.d_x = delta.dot(self.w.T)
        self.d_w = self.x.T.dot(delta)
        self.d_b = np.mean(delta, axis=0).reshape((1, -1))
        return self.d_x


class Neuralnetwork():
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []  # Store all layers in this list.
        self.x = None  # Save the input to forward in this
        self.y = None  # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable
        self.config = config
        self.training_loss = []
        self.validation_loss = []
        self.training_acc = []
        self.validation_acc = []

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        self.x = x

        for layer in self.layers:
            self.x = layer.forward(self.x)

        self.y = softmax(self.x)

        self.targets = targets
        if targets is not None:
            return self.loss(self.x, targets), self.y

        return None, self.y

    def loss(self, logits, targets):
        '''
        compute the categorical cross-entropy loss and return it.
        '''
        loss = 1 / 2 * np.sum(targets.reshape(-1, 1) * np.log(logits + 0.0000000001), axis=0) / logits.shape[0]

        # L2 loss:
        l2_penalty = config['L2_penalty']
        for layer in self.layers:
            if isinstance(layer, Layer):
                loss += l2_penalty / 2 * np.sum(layer.w ** 2)

        return loss

    def backward(self):
        '''
        Implement backpropagation here.
        Call backward methods of individual layer's.
        '''
        delta = self.targets - self.y

        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def log_metrics(self, training_loss, validation_loss, training_acc, validation_acc):
        self.training_loss.append(training_loss)
        self.validation_loss.append(validation_loss)
        self.training_acc.append(training_acc)
        self.validation_acc.append(validation_acc)



def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """
    for epoch in range(config['epochs']):
        model.forward(x_train, y_train)
        model.backward()

    # update weights:
    v = 0

    for layer in model.layers:
        if isinstance(layer, Layer):
            learning_rate = config['learning_rate']
            if config['momentum']:
                momentum_gamma = config['momentum_gamma']
                v = momentum_gamma * v + (1 - momentum_gamma) * layer.d_w
                layer.w += learning_rate * v * config['L2_penalty']
            else:
                layer.w += learning_rate * layer.d_w * config['L2_penalty']
            layer.b += learning_rate * layer.d_b


def test(model, X_test, y_test):
    """
    Calculate and return the accuracy on the test set.
    """
    loss, predictions = model.forward(X_test, y_test)
    targets = np.argmax(y_test, axis=-1)
    predictions = np.argmax(predictions, axis=-1)
    return np.sum(targets == predictions) / len(y_test)


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./")

    # Create the model
    model = Neuralnetwork(config)

    # Load the data
    x_train, y_train = load_data(path="./", mode="train")
    x_test, y_test = load_data(path="./", mode="t10k")

    # Create splits for validation data here.
    size = len(x_train)
    validation_size = 0.9
    x_valid, y_valid = x_train[int(size * validation_size):], y_train[int(size * validation_size):]
    x_train, y_train = x_train[:int(size * validation_size)], y_train[:int(size * validation_size)]

    # train the model
    train(model, x_train, y_train, x_valid, y_valid, config)

    test_acc = test(model, x_test, y_test)
