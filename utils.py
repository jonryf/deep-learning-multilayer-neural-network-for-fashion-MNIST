import matplotlib.pyplot as plt
import numpy as np


def graph_plot(data, labels, legends, title=""):
    """
    Plot multiple graphs in same plot

    @param data: data of the graphs to be plotted
    @param labels: x- and y-label
    @param legends: legends for the graphs
    """
    x = np.arange(1, len(data[0]) + 1)
    for to_plot in data:
        plt.plot(x, to_plot)
    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(legends)
    plt.show()

def plot(model, title=""):
    """
    Plot loss and accuracy graphs

    @param model: trained model to plot
    """
    # plot the loss
    graph_plot([model.training_loss, model.validation_loss],
               ["Epoch", "Cross-entropy loss"], ["Training loss", "Validation loss"], title)

    # plot the accuracy
    graph_plot([model.training_acc, model.validation_acc],
               ["Epoch", "Accuracy"], ["Training accuracy", "Validation accuracy"], title)


def numerical_approximation(x_data, y_data, model, layer_idx, node_idx, col, bias=False):
    """

    @param x_data: some example data
    @param y_data: labels for the example data
    @param model: model used
    @param layer_idx: index of layer to calculate approximation
    @param node_idx: index of node in layer to calculate approximation
    """
    eps = 0.000001
    layer = model.layers[layer_idx]

    vec = layer.b if bias else layer.w

    # extract modify weights
    # w + eps
    vec[node_idx][col] += eps

    # forward pass and calculate loss
    loss_1, _ = model.forward(x_data, y_data)
    # undo
    vec[node_idx][col] -= eps

    # w - eps
    vec[node_idx][col] -= eps

    # forward pass and calculate loss
    loss_2, _ = model.forward(x_data, y_data)
    # undo
    vec[node_idx][col] += eps

    # backprop
    model.forward(x_data, y_data)
    model.backward()

    numerical_grad = ((loss_1 - loss_2) / (2 * eps))
    # Attention: Loss is divided by number of images, therefore multiplied by it here
    if not bias:
        numerical_grad *=len(x_data)

    backprop_grad = layer.d_b[node_idx][col] if bias else layer.d_w[node_idx][col]
    print("\nLayer: {}, node: {} pixel: {}".format(layer_idx, node_idx, col))
    print("Gradient difference: {}".format(abs(abs(numerical_grad)-abs(backprop_grad))))
    print("Numerical approximation: {}".format(numerical_grad))
    print("Backprop: {}".format(backprop_grad))



