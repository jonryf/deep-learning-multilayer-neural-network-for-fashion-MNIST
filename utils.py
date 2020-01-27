import matplotlib.pyplot as plt
import numpy as np


def graph_plot(data, labels, legends):
    """
    Plot multiple graphs in same plot

    @param data: data of the graphs to be plotted
    @param labels: x- and y-label
    @param legends: legends for the graphs
    """
    x = np.arange(1, len(data[0]) + 1)
    for to_plot in data:
        plt.plot(x, to_plot)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(legends)
    plt.show()

def plot(model):
    """
    Plot loss and accuracy graphs

    @param model: trained model to plot
    """
    # plot the loss
    graph_plot([model.training_loss, model.validation_loss],
               ["Epoch", "Cross-entropy loss"], ["Training loss", "Validation loss"])

    # plot the accuracy
    graph_plot([model.training_acc * 100, model.validation_acc * 100],
               ["Epoch", "Accuracy"], ["Training accuracy", "Validation accuracy"])


def numerical_approximation(x_data, y_data, model, layer_idx, node_idx):
    """

    @param x_data: some example data
    @param y_data: labels for the example data
    @param model: model used
    @param layer_idx: index of layer to calculate approximation
    @param node_idx: index of node in layer to calculate approximation
    """
    eps = 0.001
    layer = model.layers[layer_idx]
    # extract modify weights
    # w + eps
    #layer.w[node_idx] += eps

    # forward pass and calculate loss
    loss_1, _ = model.forward(x_data, y_data)
    # undo
    layer.w[node_idx] -= eps

    # w - eps
    layer.w[node_idx] -= eps

    # forward pass and calculate loss
    loss_2, _ = model.forward(x_data, y_data)
    # undo
    layer.w[node_idx] += eps

    # backprop
    model.forward(x_data, y_data)
    model.backward()

    numerical_grad = (loss_1 - loss_2) / (2 * eps)
    backprop_grad = layer.d_w[node_idx]
    print("Gradient difference: {}".format(numerical_grad-backprop_grad))
    print("Numerical approximation: {}".format(numerical_grad))
    print("Backprop: {}".format(backprop_grad))



