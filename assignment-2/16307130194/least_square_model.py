import os
os.sys.path.append('..')

from matplotlib import pyplot as plt
from handout import get_linear_seperatable_2d_2c_dataset
import numpy as np


class LSM:
    def __init__(self, dataset):
        self.X = dataset.X
        self.y = dataset.y
        self.w = [0, 0]
        self.d = dataset

    def train(self):
        self.w[0], self.w[1], self.b = np.linalg.pinv(np.c_[self.X, np.ones(len(self.X))]) @ self.y
        print("w =", self.w)
        print("b =", self.b)

    def predict(self, X):
        result = X @ self.w + self.b
        result[result<0.5]=False
        result[result>=0.5]=True
        return result

    def plot(self):
        x_min, x_max = -1.5, 1.5
        plt.plot([x_min, x_max], [(0.5-self.b-self.w[0]*x_min)/self.w[1], (0.5-self.b-self.w[0]*x_max)/self.w[1]], label="least square model")
        plt.legend()
        self.d.plot(plt).show()

    def accuracy(self):
        print("Accuracy: %f" % (sum(self.y == self.predict(self.X)) / len(self.y)))

    def run(self):
        self.train()
        self.accuracy()
        self.plot()

print("Least square model:")
lsm = LSM(get_linear_seperatable_2d_2c_dataset())
lsm.run()
