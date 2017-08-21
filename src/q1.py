import numpy as np
import os, sys


def createdataset(filepath, status):
    """

    :param filepath:
    :return:
    """
    home = os.path.expanduser('~')                      #when submitting, comment this and return only one value
    my_data = np.genfromtxt(home+filepath, delimiter=',')
    my_data = augmentvector(my_data)
    if status in ["devlopment"]:
        return my_data[0:10000], my_data[10000:len(my_data)]
    return my_data


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
    # return single_sample_perceptron(dataset), single_sample_perceptron_with_margin(dataset), batch_perceptron(dataset), batch_perceptron_with_margin(dataset)
    return single_sample_perceptron_with_margin(dataset)


def margin_multiplier(a, y, b):
    return (b - np.dot(a, y))/np.linalg.norm(y)


def single_sample_perceptron(dataset):
    """
    build a classifier for the dataset based on the single sample perceptron model
    :param dataset: dataset to be classified
    :return: the classification results
    """
    print("single sample perceptron training started")
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
    return solutionvector


def single_sample_perceptron_with_margin(dataset):
    """
    build a classifier for the dataset based on the single sample perceptron model
    :param dataset: dataset to be classified
    :return: the classification results
    """
    print("single sample perceptron with margin training started")
    lendataset = len(dataset[0])-1
    solutionvector = np.random.rand(lendataset-1)
    count_of_rightly_classified = 0
    epochs = 0
    margin = 10000
    while count_of_rightly_classified < np.shape(dataset)[0]:
        epochs += 1
        count_of_rightly_classified = 0
        for row in dataset:
            dotproduct = np.dot(solutionvector, row[0:lendataset-1])
            if dotproduct <= margin:
                multiplication_factor = margin_multiplier(solutionvector, row[0:lendataset-1], margin)
                print(multiplication_factor)
                solutionvector = np.add(solutionvector, np.multiply(np.full(lendataset-1, multiplication_factor), row[0:lendataset-1]))
            else:
                count_of_rightly_classified += 1
    return solutionvector


def batch_perceptron(dataset):
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
    learning_rate = 1.5
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
    return solutionvector


def batch_perceptron_with_margin(dataset):
    """
    build a classifier for the dataset based on the single sample perceptron model
    :param dataset: dataset to be classified
    :return: the classification results
    """
    print("batch perceptron with margin training started")
    lendataset = len(dataset[0])-1
    solutionvector = np.random.rand(lendataset-1)
    count_of_rightly_classified = 0
    epochs = 0
    learning_rate = 1.5
    margin = 2
    while count_of_rightly_classified < np.shape(dataset)[0]:
        epochs += 1
        batch_vector_to_be_added = np.zeros(lendataset-1)
        count_of_rightly_classified = 0
        for row in dataset:
            dotproduct = np.dot(solutionvector, row[0:lendataset-1])
            if dotproduct <= 0.0:
                multiplication_factor = margin_multiplier(solutionvector, row[0:lendataset-1], margin)
                batch_vector_to_be_added = np.add(batch_vector_to_be_added, row[0:lendataset-1])
            else:
                count_of_rightly_classified += 1
        solutionvector = np.add(solutionvector, learning_rate*batch_vector_to_be_added)
    return solutionvector


def run_on_test_data(dataset, solution_vector_set):
    lendataset = len(dataset[0])-1
    accuracy = np.full(len(solution_vector_set), 0.0)
    for index, vector in enumerate(solution_vector_set):
        for row in dataset:
            dotproduct = np.dot(row[0:lendataset-1], vector)
            if dotproduct < 0 and row[lendataset] == 0:
                accuracy[index] += 1
            elif dotproduct > 0 and row[lendataset] == 1:
                accuracy[index] += 1
        accuracy[index] = np.divide(accuracy[index], np.shape(dataset)[0])
        accuracy[index] = np.multiply(accuracy[index], 100)
    print(accuracy)
    return accuracy


def main():
    """

    :return:
    """
    (training_data, test_data) = createdataset("/Documents/sem_7/SMAI/assignment1/mnist_train.csv", "development")
    # uncomment them while submitting the code and change the home path from createdataset function
    # training_data = createdataset(sys.argv[0], "production")
    # test_data = createdataset(sys.argv[1], "production")
    solution_vector_set = train_sample(training_data)
    run_on_test_data(test_data, solution_vector_set)

if __name__ == "__main__":
    main()