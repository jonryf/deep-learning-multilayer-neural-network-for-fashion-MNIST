from checker import load_config
from neuralnet import Neuralnetwork, load_data
from utils import numerical_approximation


def task_2b():
    config = load_config("./")

    # load the test data
    x_train, y_train = load_data(path="./", mode="train")
    print(x_train.shape)
    print(y_train.shape)

    x_train = x_train[:12]
    y_train = y_train[:12]

    print(x_train.shape)
    print(y_train.shape)



    # Create the model
    model = Neuralnetwork(config)

    numerical_approximation(x_train, y_train, model, 0, 0)


task_2b()
