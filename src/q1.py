import numpy as np
import os
from pathlib import path


def createdataset(filepath):
    home = str(Path.home())
    my_data = np.genfromtxt(home+filepath, delimiter=',')
    print(my_data)


def main():
    createdataset("/Documents/sem_7/SMAI/assignment1/q1_mnist_train.csv")

if __name__ == "__main__":
    main()