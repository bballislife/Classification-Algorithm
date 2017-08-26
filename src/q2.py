import numpy as np
import sys


class Dataset(object):
    def __init__(self, filepath, status):
        """

        :rtype: object
        """
        self.home = "datasets/q2/"
        self.my_data = np.genfromtxt(self.home + filepath, delimiter=',', dtype=float)
        indices_with_nan = []
        for index, row in enumerate(self.my_data):
            if np.isnan(row).any():
                indices_with_nan.append(index)
        self.my_data = np.delete(self.my_data, indices_with_nan, axis=0)
        self.my_data = Dataset.clean_data(self.my_data, status)
        self.my_data = Dataset.normalize(self.my_data)

    @classmethod
    def clean_data(cls, dataset, status):
        new_dataset = np.delete(dataset, 0, 1)
        for row in new_dataset:
            assert isinstance(row, object)
            if row[-1] == 4:
                row[-1] = 1
            elif row[-1] == 2:
                row[-1] = 0
                if status in ['training']:
                    row *= -1
        return new_dataset

    @classmethod
    def normalize(cls, dataset):
        dataset_normed = dataset / dataset.max(
            axis=0)  # this function will normalise the dataset to values between 0 and 1
        return dataset_normed


class PerceptronTraining(object):
    def __init__(self, train, test):
        self.training_set = train
        self.testing_set = test
        self.accuracy = []

    def batch_perceptron_with_relaxation(self, max_epochs, learning_rate, margin):
        correctly_classified = 0
        current_epoch = 0
        length_of_sample = len(self.training_set[0])
        solution_vector = np.full(length_of_sample-1, 0.1)
        while correctly_classified < np.shape(self.training_set)[0] and current_epoch < max_epochs:
            current_epoch += 1
            batch_vector_to_be_added = np.zeros(length_of_sample - 1)
            count_of_rightly_classified = 0
            for row in self.training_set:
                dot_product = np.dot(solution_vector, row[0:length_of_sample - 1])
                if dot_product <= margin:
                    relaxation_parameter = (margin - dot_product)/ np.linalg.norm(solution_vector)
                    batch_vector_to_be_added = np.add(batch_vector_to_be_added, relaxation_parameter*row[0:length_of_sample - 1])
                else:
                    count_of_rightly_classified += 1
            solution_vector = np.add(solution_vector, learning_rate * batch_vector_to_be_added)
        self.run_test_for_batch(solution_vector)

    def voted_perceptron(self, max_epochs):
        length_of_sample = len(self.training_set[0])
        k = 0
        solution_vector = [np.zeros(length_of_sample-1)]
        c = [0]
        current_epoch = 0
        while current_epoch < max_epochs:
            current_epoch += 1
            for row in self.training_set:
                dot_product = np.dot(solution_vector[-1], row[0:length_of_sample-1])
                if dot_product > 0:
                    c[-1] += 1
                else:
                    solution_vector.append(np.add(solution_vector[-1], row[0:length_of_sample-1]))
                    c.append(1)
                    k += 1
        self.run_test_for_voted(solution_vector, k, c)

    def run_test_for_batch(self, solution):
        weight_vector = solution
        rightly_classified_samples = 0
        length_of_sample = len(self.training_set[0])
        for row in self.testing_set:
            dot_product = np.dot(weight_vector, row[0:length_of_sample-1])
            if dot_product < 0 and int(row[length_of_sample-1]) == 0:
                rightly_classified_samples += 1
            elif dot_product > 0 and int(row[length_of_sample-1]) == 1:
                rightly_classified_samples += 1
        self.accuracy.append(float(rightly_classified_samples)*100/float(np.shape(self.testing_set)[0]))

    def run_test_for_voted(self, solution, k, c):
        length_of_sample = len(self.training_set[0])
        rightly_classified_samples = 0
        for row in self.testing_set:
            s = 0
            for i in range(k):
                s += c[i]*np.sign(np.dot(row[0:length_of_sample-1], solution[i]))
            s = np.sign(s)
            if s == -1 and int(row[length_of_sample-1]) == 0:
                rightly_classified_samples += 1
            elif s == 1 and int(row[length_of_sample-1]) == 1:
                rightly_classified_samples += 1
        self.accuracy.append(float(rightly_classified_samples)*100/float(np.shape(self.testing_set)[0]))


def main():
    training_dataset = Dataset(sys.argv[1], "training")
    test_dataset = Dataset(sys.argv[2], "test")
    perceptron_sample = PerceptronTraining(training_dataset.my_data, test_dataset.my_data)
    perceptron_sample.batch_perceptron_with_relaxation(500, 1, 0)
    perceptron_sample.voted_perceptron(50)
    print perceptron_sample.accuracy

if __name__ == "__main__":
    main()
