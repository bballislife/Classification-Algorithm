import numpy as np
import os
from pathlib import Path


def createdataset(filepath):
    """

    :param filepath:
    :return:
    """
    home = str(Path.home())
    my_data = np.genfromtxt(home+filepath, delimiter=',', dtype=int)
    my_data = augmentvector(my_data)
    return my_data[0:10000], my_data[10000:len(my_data)]


def augmentvector(dataset):
    """
    This function is to augment the dataset for normalization
    :param dataset: Dataset to be normalized
    :return: normalized dataset
    """
    dataset[:, [0, len(dataset[0])-1]] = dataset[:, [len(dataset[0])-1, 0]]
    return np.insert(dataset, 0, 1, axis=1)


def singlesampleperceptron(dataset):
    """
    build a classifier for the dataset based on the single sample perceptron model
    :param dataset: dataset to be classified
    :return: the classification results
    """
    featurecount = len(dataset[0])-1
    for row in dataset:
        if row[featurecount] == 0:
            row *= -1


def main():
    """

    :return:
    """
    (training_data, test_data) = createdataset("/Documents/sem_7/SMAI/assignment1/q1_mnist_train.csv")
    singlesampleperceptron(training_data)


if __name__ == "__main__":
    main()