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
import random

import yaml
import numpy as np
import math
import timeit
import io

from utils import plot, numerical_approximation
from matplotlib import pyplot as plt
from utils import plot

train_std = None
train_mean = None


def load_config(version):
    """
    Load the configuration from config.yaml.
    """
    #
    return yaml.load(open('config' + version + '.yaml', 'r'), Loader=yaml.SafeLoader)


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


def mini_batch(x, y, size):  # returns data split into 128-max batches (leaves out remainder)
    if len(x) < size:
        return x, y
    num_batches = math.floor(len(x) / size)
    x_batches = []
    y_batches = []

    data = list(zip(x, y))
    random.shuffle(data)

    x, y = zip(*data)
    x = np.array(x)
    y = np.array(y)

    for i in range(num_batches - 1):
        x_batches.append(x[i * size: (i + 1) * size])
        y_batches.append(y[i * size: (i + 1) * size])
    return x_batches, y_batches


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
        global train_mean, train_std
        train_mean = np.mean(images)
        train_std = np.std(images)

    normalized_images = normalize_data(images)
    one_hot_labels = one_hot_encoding(labels, num_classes=10)

    data = list(zip(normalized_images, one_hot_labels))

    random.shuffle(data)

    normalized_images, one_hot_labels = zip(*data)
    normalized_images = np.array(normalized_images)
    one_hot_labels = np.array(one_hot_labels)

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

        self.v = None  # Momentum

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
        self.validation_increments = 0

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
            return self.loss(self.y, targets), self.y

        return None, self.y

    def loss(self, logits, targets):
        '''
        compute the categorical cross-entropy loss and return it.
        '''
        loss = -np.sum(np.sum(targets * np.log(logits + 0.0000000001), axis=0, keepdims=True) / logits.shape[0])

        # L2 loss:
        l2_penalty = self.config['L2_penalty']
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
        if len(self.validation_loss) > 0 and validation_loss > self.validation_loss[-1]:
            self.validation_increments += 1
        else:
            self.validation_increments = 0

        self.training_loss.append(training_loss)
        self.validation_loss.append(validation_loss)
        self.training_acc.append(training_acc)
        self.validation_acc.append(validation_acc)

    def update_weights(self):
        """
        Update the weights, after back-propagation has happen

        @param model: model to update weights on
        """
        for layer in self.layers:
            if isinstance(layer, Layer):
                learning_rate = self.config['learning_rate']
                if self.config['momentum']:
                    if layer.v is None:
                        layer.v = layer.d_w
                    momentum_gamma = self.config['momentum_gamma']
                    layer.v = momentum_gamma * layer.v + (1 - momentum_gamma) * layer.d_w
                    layer.w += learning_rate * (layer.v - self.config['L2_penalty'] * layer.w)
                else:
                    layer.w += learning_rate * (layer.d_w - self.config['L2_penalty'] * layer.w)

                layer.b += learning_rate * layer.d_b


def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.

    @param model:
    @param x_train:
    @param y_train:
    @param x_valid:
    @param y_valid:
    @param config:
    """

    # number of times the error can increase before ending training
    threshold = config['early_stop_epoch']

    # store current-best model
    # best_model = model

    for epoch in range(config['epochs']):
        # for each batch in one epoch
        # break data into 128-size batches for small batch gradient descent
        x_batches, y_batches = mini_batch(x_train, y_train, config['batch_size'])

        for i in range(len(x_batches)):
            model.forward(x_batches[i], y_batches[i])
            model.backward()
            model.update_weights()  # implements momentum and regularization

        # track metrics across each epoch
        tl, ta = test(model, x_train, y_train)
        vl, va = test(model, x_valid, y_valid)
        model.log_metrics(tl, vl, ta, va)

        # early stopping condition
        if config['early_stop'] == 'True' and epoch >= 4 and model.validation_increments > threshold:
            break
        if epoch % 10 == 0:
            print("Running epoch {}".format(epoch + 1))


def test(model, X_test, y_test):
    """
    Calculate and return the accuracy on the test set.
    """
    loss, predictions = model.forward(X_test, y_test)
    targets = np.argmax(y_test, axis=-1)
    predictions = np.argmax(predictions, axis=-1)
    accuracy = np.sum(targets == predictions) / len(y_test)
    return loss, accuracy


def task_b():
    ###############################
    # Load the configuration.
    config = load_config("b")

    num_of_examples = 12
    # load the test data
    x_train, y_train = load_data(path="./", mode="train")
    x_train = x_train[:num_of_examples]
    y_train = y_train[:num_of_examples]

    # Create the model
    model = Neuralnetwork(config)

    # bias output weight
    numerical_approximation(x_train, y_train, model, 4, 0, 0, bias=True)

    # Create splits for validation data here.
    num_examples = len(x_train)
    # print("# examples:", num_examples)

    # create validation set
    size = len(x_train)
    validation_size = 0.9
    x_valid, y_valid = x_train[int(size * validation_size):], y_train[int(size * validation_size):]
    x_train, y_train = x_train[:int(size * validation_size)], y_train[:int(size * validation_size)]

    # train the model
    # train(model, x_train, y_train, x_valid, y_valid, config)

    test_acc = test(model, x_test, y_test)


def task_c():
    ###############################
    # Load the configuration.
    config = load_config("c")

    # Create the model
    model = Neuralnetwork(config)
    # Load the data
    x_train, y_train = load_data(path="./", mode="train")
    x_test, y_test = load_data(path="./", mode="t10k")

    num_examples = len(x_train)
    print("# examples:", num_examples)

    # Hold the values from each model that we will graph
    ten_training_losses = []
    ten_training_accuracies = []
    ten_validation_losses = []
    ten_validation_accuracies = []

    # store best model for test set
    best_model = model
    # hack to fix error
    best_model.validation_loss.append(100)

    # perform k fold 10 times:
    K = 10
    for k in range(K):  # for each fold
        print("K value: ", k)
        model = Neuralnetwork(config)
        X = []
        y = []
        k_size = int(len(x_train) / K)
        for i in range(k_size):  # get folds of size 1/K)
            r = np.random.randint(num_examples)
            # if not(x_train[r] in X): #insert random pairs if not already present
            X.append(x_train[r])
            y.append(y_train[r])
        X = np.array(X)
        y = np.array(y)
        # create validation set
        size = len(X)
        validation_size = 0.9
        Xv, yv = X[int(size * validation_size):], y[int(size * validation_size):]
        Xt, yt = X[:int(size * validation_size)], y[:int(size * validation_size)]
        # train the model
        train(model, Xt, yt, Xv, yv, config)

        # append the values of each model to their lists
        ten_training_accuracies.append(model.training_acc)
        ten_training_losses.append(model.training_loss)
        ten_validation_accuracies.append(model.validation_acc)
        ten_validation_losses.append(model.validation_loss)

        # Store best model
        if model.validation_loss[-1] < best_model.validation_loss[-1]:
            best_model = model
    ### Report training and validation accuracy and loss for each K
    #plot(model)

    #find the minimum epoch number to convergence:
    lengths = []
    for item in ten_validation_losses:
        lengths.append(len(item))
    cutoff = min(lengths)

    #align the losses and accuracy by cutoff value
    aligned_training_losses = np.array([x[:cutoff] for x in ten_training_losses])
    aligned_training_accuracies = np.array([x[:cutoff] for x in ten_training_accuracies])
    aligned_validation_losses = np.array([x[:cutoff] for x in ten_validation_losses])
    aligned_validation_accuracies = np.array([x[:cutoff] for x in ten_validation_accuracies])

    #plot the data
    aligned_data = [aligned_training_losses, aligned_training_accuracies, aligned_validation_losses, aligned_validation_accuracies]
    aligned_titles = ["Aligned Training Losses", "Aligned Training Accuracies", "Aligned Validation Losses", "Aligned Validation Accuracies"]
    aligned_ylables = ["Cross Entropy Error", "Accuracy", "Cross Entropy Error", "Accuracy"]
    for i in range(len(aligned_data)):
        print(aligned_titles[i])
        data = aligned_data[i]
        print("Data shape:", data.shape)
        mean = np.mean(data, axis=0)  # divide sum of columns by num of folds
        print("mean shape:", mean.shape)
        print(mean)
        std = np.std(data, axis=0)
        print("Std shape:", std.shape)
        print(std)
        '''
        y_std = []
        for i in range(len(mean)):
            if (i + 1) % 10 == 0 or i == 0:
                y_std.append(np.std(data[:, i]))
            else:
                y_std.append(0)
        '''
        plt.errorbar(np.arange(1, len(mean)+1), mean, yerr=std)
        plt.title(aligned_titles[i])
        plt.ylabel(aligned_ylables[i])
        plt.xlabel("Epoch")
        #plt.legend(["Training data", "Validation data"])
        plt.show()

    ### Report Test Accuracy of best model
    test_loss, test_acc = test(best_model, x_test, y_test)
    print("Test accuracy of best model: ", test_acc)

def task_d():
    config = load_config("d")
    model = Neuralnetwork(config)
    x_train, y_train = load_data(path="./", mode="train")
    x_test, y_test = load_data(path="./", mode="t10k")

    train_size = 0.9
    size = len(x_train)
    x_valid, y_valid = x_train[int(size * train_size):], y_train[int(size * train_size):]
    x_train, y_train = x_train[:int(size * train_size)], y_train[:int(size * train_size)]

    train(model, x_train, y_train, x_valid, y_valid, config)
    plot(model)
    test_acc = test(model, x_test, y_test)
    print("Test accuracy: {}".format(test_acc))


def task_e():
    pass


def task_f():
    pass


if __name__ == "__main__":
    task = ''
    while task != 'q':
        task = input("Choose your task - lowercase letter: ")
        if task == 'b':
            task_b()
        elif task == 'c':
            task_c()
        elif task == 'd':
            task_d()
        elif task == 'e':
            task_e()
        elif task == 'f':
            task_f()
        elif task == 'q':
            print("Ending Program")
        else:
            print("invalid entry - select  from b,c,d,e,f")

