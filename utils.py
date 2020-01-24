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
