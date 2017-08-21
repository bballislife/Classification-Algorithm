import numpy as np
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


def train_sample(dataset):
    lendataset = len(dataset[0])-1
    for row in dataset:
        if row[lendataset] == 0:
            row *= -1
    solution_vector_batch = batchperceptron(dataset)
    solution_vector_single_sample = singlesampleperceptron(dataset)


def singlesampleperceptron(dataset):
    """
    build a classifier for the dataset based on the single sample perceptron model
    :param dataset: dataset to be classified
    :return: the classification results
    """
    print("single sample perceptron training started")
    print(dataset)
    lendataset = len(dataset[0])-1
    solutionvector = np.random.rand(lendataset-1)
    count_of_rightly_classified = 0
    epochs = 0
    while count_of_rightly_classified < np.shape(dataset)[0]:
        epochs += 1
        count_of_rightly_classified = 0
        for row in dataset:
            dotproduct = np.dot(solutionvector, row[0:lendataset-1])
            if dotproduct <= 0.0:
                solutionvector = np.add(solutionvector, row[0:lendataset-1])
            else:
                count_of_rightly_classified += 1
    print("epochs = ", epochs)
    return solutionvector


def batchperceptron(dataset):
    """
    build a classifier for the dataset based on the single sample perceptron model
    :param dataset: dataset to be classified
    :return: the classification results
    """
    print("batch perceptron training started")
    lendataset = len(dataset[0])-1
    solutionvector = np.random.rand(lendataset-1)
    count_of_rightly_classified = 0
    epochs = 0
    learning_rate = 0.6
    while count_of_rightly_classified < np.shape(dataset)[0]:
        epochs += 1
        batch_vector_to_be_added = np.zeros(lendataset-1)
        count_of_rightly_classified = 0
        for row in dataset:
            dotproduct = np.dot(solutionvector, row[0:lendataset-1])
            if dotproduct <= 0.0:
                batch_vector_to_be_added = np.add(batch_vector_to_be_added, row[0:lendataset-1])
            else:
                count_of_rightly_classified += 1
        solutionvector = np.add(solutionvector, learning_rate*batch_vector_to_be_added)
    print("epoch = ", epochs)
    return solutionvector


def run_on_test_data(dataset, single_sample, batch):
    accuracy = 0.0
    return accuracy


def main():
    """

    :return:
    """
    (training_data, test_data) = createdataset("/Documents/sem_7/SMAI/assignment1/mnist_train.csv")
    train_sample(training_data)


if __name__ == "__main__":
    main()