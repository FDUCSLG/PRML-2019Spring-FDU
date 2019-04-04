import os
os.sys.path.append('..')

from matplotlib import pyplot as plt
from handout import get_linear_seperatable_2d_2c_dataset
import numpy as np


class Perceptron:
    def __init__(self, dataset):
        self.X = dataset.X
        self.y = self.preprocess_y(dataset.y * 1)
        self.d = dataset

    def preprocess_y(self, y_data):
        y_data[y_data == False] = -1
        return y_data

    def train(self, eta=0.1):
        w = np.ones(3)
        X3 = np.c_[self.X, np.ones(len(self.X))]
        while True:
            last_w = w.copy()
            for X, y in zip(X3, self.y):
                if w @ X * y < 0:
                    w = w + eta * X * y
            if (last_w == w).all():
                break

        self.w = w[0:2]
        self.b = w[2]
        print("w =", self.w)
        print("b =", self.b)

    def predict(self, X):
        result = X @ self.w + self.b
        result[result<0]=-1
        result[result>=0]=1
        return result

    def plot(self):
        x_min, x_max = -1.5, 1.5
        plt.plot([x_min, x_max], [(-self.b-self.w[0]*x_min)/self.w[1], (-self.b-self.w[0]*x_max)/self.w[1]], label="perceptron")
        plt.legend()
        self.d.plot(plt).show()

    def accuracy(self):
        print("Accuracy: %f" % (sum(self.y == self.predict(self.X)) / len(self.y)))

    def run(self):
        self.train()
        self.accuracy()
        self.plot()

print("Perceptron:")
perceptron = Perceptron(get_linear_seperatable_2d_2c_dataset())
perceptron.run()
