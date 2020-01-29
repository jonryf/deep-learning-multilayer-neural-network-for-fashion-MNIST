import matplotlib.pyplot as plt
import numpy as np


def graph_error(means, stds, labels, legends, title, show=True):
    x = np.arange(1, len(means[0]) + 1)
    for i in range(2):
        plt.errorbar(np.arange(1, len(means[i])+1), means[i], yerr=stds[i])
    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(legends)
    if show:
        plt.show()


def graph_plot(data, labels, legends, title, show=True):
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
    if show:
        plt.show()


def plot_loss(model, title, show=True):
    """
    Plot loss and accuracy graphs

    @param model: trained model to plot
    """
    # plot the loss
    graph_plot([model.training_loss, model.validation_loss],
               ["Epoch", "Cross-entropy loss"], ["Training loss", "Validation loss"], title, show)


def plot_acc(model, title, show=True):
    # plot the accuracy
    graph_plot([model.training_acc, model.validation_acc],
               ["Epoch", "Accuracy"], ["Training accuracy", "Validation accuracy"], title, show)


def plot(model, title=""):
    plot_loss(model, title, show=True)
    plot_acc(model, title, show=True)


def multi_plots(models, names):
    """
    Plot loss and accuracy graphs

    @param model: trained model to plot
    """
    # plot the loss
    losses = []
    loss_labels = []

    acc = []

    for model, name in zip(models, names):
        losses += model.training_loss
        losses += model.validation_loss
        loss_labels += "Training loss - " + name
        loss_labels += "validation loss - " + name

        acc += model.training_acc
        acc += model.validation_acc

    graph_plot(losses,
               ["Epoch", "Cross-entropy loss"], loss_labels)

    # plot the accuracy
    graph_plot([model.training_acc, model.validation_acc],
               ["Epoch", "Accuracy"], ["Training accuracy", "Validation accuracy"])


def numerical_approximation(x_data, y_data, model, layer_idx, node_idx, col, bias=False):
    """

    @param x_data: some example data
    @param y_data: labels for the example data
    @param model: model used
    @param layer_idx: index of layer to calculate approximation
    @param node_idx: index of node in layer to calculate approximation
    """
    eps = 0.01
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

    numerical_grad = -((loss_1 - loss_2) / (2 * eps))
    # Attention: Loss is divided by number of images, therefore multiplied by it here
    if not bias:
        numerical_grad *= len(x_data)

    backprop_grad = layer.d_b[node_idx][col] if bias else layer.d_w[node_idx][col]
    print("\nLayer: {}, node: {} pixel: {}".format(layer_idx, node_idx, col))
    print("Gradient difference: {}".format(abs(abs(numerical_grad) - abs(backprop_grad))))
    print("Numerical approximation: {}".format(numerical_grad))
    print("Backprop: {}".format(backprop_grad))



