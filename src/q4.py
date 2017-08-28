import os
import numpy as np


class FeatureVector(object):
    def __init__(self, vocabsize, numdata):
        self.vocabsize = vocabsize
        self.X = np.zeros((numdata, self.vocabsize), dtype=np.int)
        self.Y = np.zeros((numdata,), dtype=np.int)
        self.cur_index = 0

    def make_featurevector(self, inputlist, classid):
        np.copyto(self.X[self.cur_index], inputlist)
        self.Y[self.cur_index] = classid
        if np.linalg.norm(self.X[self.cur_index]) == 0.0:
            print classid, self.X[self.cur_index]
            exit()
        self.cur_index += 1


class KNN(object):
    def __init__(self, train_vec, test_vec):
        self.X_train = train_vec.X
        self.Y_train = train_vec.Y
        self.X_test = test_vec.X
        self.Y_test = test_vec.Y
        self.metric = Metrics('f1', 10)
        self.class_array = []

    def classify(self, nn=1):
        """
        Takes input X_train, Y_train, X_test and Y_test and displays the accuracies.
        """
        for test_point in self.X_test:
            temp_distances = []
            if np.linalg.norm(test_point) == 0.0:
                self.class_array.append(0)
                continue
            class_count = np.zeros(11)
            for (data_point, classid) in zip(self.X_train, self.Y_train):
                if classid == 0:
                    break
                temp_distances.append((self.cosine_distance(data_point, test_point), classid))
            temp_distances.sort(key=lambda x: x[0])
            for i in range(nn):
                class_count[temp_distances[i][1]] += float(1)/(temp_distances[i][0]**2)
            self.class_array.append(np.argmax(class_count))
        print (self.class_array == self.Y_test).sum()

    @staticmethod
    def cosine_distance(data_point_1, data_point_2):
        return 1 - np.dot(data_point_1, data_point_2) / (np.linalg.norm(data_point_2) * np.linalg.norm(data_point_1))

    def print_metrics(self):
        self.metric.get_confmatrix(self.class_array, self.Y_test)


class Metrics(object):
    def __init__(self, metric, number_of_classes):
        self.metric = metric
        self.number_of_classes = number_of_classes
        self.confusion_matrix = np.zeros([number_of_classes+1, number_of_classes+1], dtype=np.int)

    def get_confmatrix(self, y_pred, y_test):
        for i in range(len(y_pred)):
            self.confusion_matrix[y_pred[i]][y_test[i]] += 1
        print "accuracy = ", self.accuracy(y_pred, y_test)
        print "f1_score = ", self.f1_score()

    @staticmethod
    def accuracy(y_pred, y_test):
        y_pred = np.array(y_pred)
        return float((y_pred == y_test).sum())*100/y_pred.size

    def f1_score(self):
        """
        Implements the f1-score function
        """
        FP = np.zeros(self.number_of_classes+1)
        FN = np.zeros(self.number_of_classes+1)
        TP = np.zeros(self.number_of_classes+1)
        for i in range(self.number_of_classes+1):
            for j in range(self.number_of_classes+1):
                if i == j:
                    TP[i] += self.confusion_matrix[i][j]
                else:
                    FP[i] += self.confusion_matrix[i][j]
                    FN[j] += self.confusion_matrix[i][j]
        precision = np.zeros(self.number_of_classes+1)
        recall = np.zeros(self.number_of_classes+1)
        f1_score = 0.0
        for i in range(1, self.number_of_classes+1):
            precision[i] = TP[i] / (TP[i] + FP[i])
            recall[i] = TP[i] / (TP[i] + FN[i])
            f1_score += (2*precision[i]*recall[i]) / (precision[i] + recall[i])

        f1_score /= self.number_of_classes
        return f1_score

if __name__ == '__main__':
    datadir = './datasets/q4/'
    classes = ['galsworthy/', 'galsworthy_2/', 'mill/', 'shelley/',
               'thackerey/', 'thackerey_2/', 'wordsmith_prose/',
               'cia/', 'johnfranklinjameson/', 'diplomaticcorr/']
    inputdir = ['train/', 'test/']

    vocabulory = {}
    datasz = []
    index = 0
    common_words = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'it', 'for']
    for idir in inputdir:
        files = 0
        classid = 1
        for c in classes:
            listing = os.listdir(datadir + idir + c)
            path = datadir + idir + c
            for filename in listing:
                files += 1
                with open(path + filename, 'r') as f:
                    for line in f:
                        if idir in ['train/']:
                            for word in line.split():
                                word = word.lower()
                                if word not in common_words:
                                    if word not in vocabulory:
                                        vocabulory[word] = index
                                        index += 1
            classid += 1
        datasz.append(files)
    vocabulory["UKN"] = index
    vocab = len(vocabulory)
    trainsz = datasz[0]
    testsz = datasz[1]

    print('Making the feature vectors.')
    trainVec = FeatureVector(vocab, trainsz)
    testVec = FeatureVector(vocab, testsz)

    for idir in inputdir:
        classid = 1
        for c in classes:
            listing = os.listdir(datadir+idir+c)
            path = datadir + idir + c
            for filename in listing:
                inputs = np.zeros(vocab, dtype=np.int)
                with open(path + filename, 'r') as f:
                    for line in f:
                        for word in line.split():
                            word = word.lower()
                            if word in vocabulory:
                                inputs[vocabulory[word]] += 1
                            else:
                                inputs[vocabulory["UKN"]] += 1
                if np.linalg.norm(inputs) != 0.0:
                    if idir == 'train/':
                        trainVec.make_featurevector(inputs, classid)
                    else:
                        testVec.make_featurevector(inputs, classid)
            classid += 1

    print('Finished making features.')
    print('Statistics ->')
    print(trainVec.X.shape, trainVec.Y.shape, testVec.X.shape, testVec.Y.shape)

    knn = KNN(trainVec, testVec)
    knn.classify(1)
    knn.print_metrics()