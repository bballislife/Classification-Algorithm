import csv
import os
from math import log
from pathlib import Path  # to access the home directory


def createdataset(filepath):
    """

    :param filepath:
    :return:
    """
    home = str(Path.home())
    with open(os.path.abspath(home + filepath), 'r') as f:
        reader = csv.reader(f)
        dataset = []
        for row in reader:
            dataset.append(row)
        for index, title in enumerate(dataset[0]):
            if title.lower() in ['left']:
                classindex = index
        trainingindex = len(dataset) * 4 / 5
        trainingdata = dataset[1:int(trainingindex)]
        testdata = dataset[int(trainingindex):len(dataset)]
        return trainingdata, testdata, classindex


def findentropy(classlist):
    """
    find the entropy for the attribute
    :param classlist: list based on division of data on classes
    :return: entropy for the class
    """
    datasetsize = 0
    for value in classlist:
        datasetsize += value
    if datasetsize is 0:
        return 0
    finalentropy = 0.0
    for value in classlist:
        finalentropy += (value / datasetsize) * log((value / datasetsize), 2)
    return finalentropy


def informationgain(attributeindex, dataset, classindex, typeofattribute):
    """

    :param attributeindex:
    :param dataset:
    :param classindex:
    :param typeofattribute:
    :return:
    """
    attributeproperties = {}
    classproperties = [0, 0]
    if typeofattribute.lower() in ['discreet']:
        for row in dataset:
            row[classindex] = int(row[classindex])
            classproperties[row[classindex]] += 1
            if row[attributeindex] in attributeproperties.keys():
                attributeproperties[row[attributeindex]][row[classindex]] += 1
            else:
                attributeproperties[row[attributeindex]] = [0, 0]
                attributeproperties[row[attributeindex]][row[classindex]] += 1
    else:
        print(1)
    entropyofparent = findentropy(classproperties)
    entropyofchildren = 0.0
    for key in attributeproperties:
        entropyofchildren += findentropy(attributeproperties[key])
    return entropyofparent - entropyofchildren


def build_tree(trainingdataset, classindex):
    """
    building the decision tree based on the training dataset
    :param trainingdataset: the training dataset
    :param classindex: column index defining the class in dataset
    :return:
    """
    for x in range(0, 9):
        if x is not classindex:
            informationgain(x, trainingdataset, classindex, "continuous" if x < 5 else "discreet")


def main():
    (trainingdataset, testdataset, classindex) = createdataset("/Downloads/decision_tree_train.csv")
    build_tree(trainingdataset, classindex)

if __name__ == "__main__":
    main()