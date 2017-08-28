import numpy as np
import sys


class Datasets(object):
    def __init__(self, filepath, train=1):
        self.filepath = filepath
        self.train = train
        self.my_data = np.genfromtxt(self.filepath, delimiter=',', dtype=float)
        self.createdataset()

    def createdataset(self):
        self.my_data = self.normalize(self.my_data)
        self.my_data = self.augment_vector(self.my_data)

    @staticmethod
    def normalize(dataset):
        max_array = dataset.max(axis=0)
        for index, value in enumerate(max_array):
            if value == 0.0:
                max_array[index] = 0.1
        dataset_normed = dataset / max_array  # this function will normalise the dataset to values between 0 and 1
        return dataset_normed

    @staticmethod
    def augment_vector(dataset):
        dataset[:, [0, len(dataset[0]) - 1]] = dataset[:, [len(dataset[0]) - 1, 0]]
        return np.insert(dataset, 0, 1, axis=1)


class PerceptronTraining(object):
    def __init__(self, dataset):
        self.len_dataset = len(dataset[0]) - 1
        self.dataset = dataset
        for row in self.dataset:
            if row[self.len_dataset] == 0:
                row *= -1
        self.solution_vector_set = []

    def start_perceptron_training(self):
        self.single_sample_perceptron()
        self.single_sample_perceptron_with_margin()
        self.batch_perceptron()
        self.batch_perceptron_with_margin()

    def single_sample_perceptron(self):
        print("single sample perceptron training started")
        solution_vector = np.random.uniform(0.0, 1.0, self.len_dataset - 1)
        count_of_rightly_classified = 0
        epochs = 0
        while count_of_rightly_classified < np.shape(self.dataset)[0]:
            epochs += 1
            count_of_rightly_classified = 0
            for row in self.dataset:
                dot_product = np.dot(solution_vector, row[0:self.len_dataset - 1])
                if dot_product <= 0.0:
                    solution_vector = np.add(solution_vector, row[0:self.len_dataset - 1])
                else:
                    count_of_rightly_classified += 1
        self.solution_vector_set.append(solution_vector)

    def single_sample_perceptron_with_margin(self):
        print("single sample perceptron with margin training started")
        solution_vector = np.random.uniform(0.0, 1.0, self.len_dataset - 1)
        count_of_rightly_classified = 0
        epochs = 0
        margin = 0.3
        while count_of_rightly_classified < np.shape(self.dataset)[0]:
            epochs += 1
            count_of_rightly_classified = 0
            for row in self.dataset:
                dot_product = np.dot(solution_vector, row[0:self.len_dataset - 1])
                if dot_product <= margin:
                    solution_vector = np.add(solution_vector, row[0:self.len_dataset - 1])
                else:
                    count_of_rightly_classified += 1
        self.solution_vector_set.append(solution_vector)

    def batch_perceptron(self):
        print("batch perceptron training started")
        solution_vector = np.random.uniform(0.0, 1.0, self.len_dataset - 1)
        count_of_rightly_classified = 0
        epochs = 0
        while count_of_rightly_classified < np.shape(self.dataset)[0] and epochs < 100:
            epochs += 1
            batch_vector_to_be_added = np.zeros(self.len_dataset - 1)
            count_of_rightly_classified = 0
            for row in self.dataset:
                dot_product = np.dot(solution_vector, row[0:self.len_dataset - 1])
                if dot_product <= 0.0:
                    batch_vector_to_be_added = np.add(batch_vector_to_be_added, row[0:self.len_dataset - 1])
                else:
                    count_of_rightly_classified += 1
            solution_vector = np.add(solution_vector, batch_vector_to_be_added)
        self.solution_vector_set.append(solution_vector)

    def batch_perceptron_with_margin(self):
        print("batch perceptron with margin training started")
        solution_vector = np.random.uniform(0.0, 1.0, self.len_dataset - 1)
        count_of_rightly_classified = 0
        epochs = 0
        learning_rate = 1.5
        margin = 0.3
        while count_of_rightly_classified < np.shape(self.dataset)[0] and epochs < 100:
            epochs += 1
            batch_vector_to_be_added = np.zeros(self.len_dataset - 1)
            count_of_rightly_classified = 0
            for row in self.dataset:
                dot_product = np.dot(solution_vector, row[0:self.len_dataset - 1])
                if dot_product <= margin:
                    batch_vector_to_be_added = np.add(batch_vector_to_be_added, row[0:self.len_dataset - 1])
                else:
                    count_of_rightly_classified += 1
            solution_vector = np.add(solution_vector, learning_rate * batch_vector_to_be_added)
        self.solution_vector_set.append(solution_vector)


def run_on_test_data(dataset, solution_vector_set):
    len_dataset = len(dataset[0]) - 1
    accuracy = np.full(len(solution_vector_set), 0.0)
    for index, vector in enumerate(solution_vector_set):
        for row in dataset:
            dot_product = np.dot(row[0:len_dataset - 1], vector)
            if dot_product < 0 and int(row[len_dataset]) == 0:
                accuracy[index] += 1
            elif dot_product > 0 and int(row[len_dataset]) == 1:
                accuracy[index] += 1
        accuracy[index] = np.divide(accuracy[index], np.shape(dataset)[0])
        accuracy[index] = np.multiply(accuracy[index], 100)
    print(accuracy)
    # return accuracy


def main():
    datadir = "./datasets/q1/"
    training_dataset = Datasets(sys.argv[1], 1)
    test_dataset = Datasets(sys.argv[2], 0)
    solution_class = PerceptronTraining(training_dataset.my_data)
    solution_class.start_perceptron_training()
    run_on_test_data(test_dataset.my_data, solution_class.solution_vector_set)


if __name__ == "__main__":
    main()
