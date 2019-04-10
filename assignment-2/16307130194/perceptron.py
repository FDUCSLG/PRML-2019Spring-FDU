import os
os.sys.path.append('..')

from matplotlib import pyplot as plt
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
        X = np.c_[self.X, np.ones(len(self.X))]
        while True:
            last_w = w.copy()
            for x, y in zip(X, self.y):
                if w @ x * y < 0:
                    w = w + eta * x * y
            if (last_w == w).all():
                break

        self.w = w[0:2]
        self.b = w[2]
        print("w =", self.w)
        print("b =", self.b)

    def predict(self, X):
        y = X @ self.w + self.b
        y[y<0]=-1
        y[y>=0]=1
        return y

    def plot(self):
        x_min, x_max = -1.5, 1.5
        plt.plot([x_min, x_max], [(-self.b-self.w[0]*x_min)/self.w[1], (-self.b-self.w[0]*x_max)/self.w[1]], label="perceptron")
        plt.legend()
        self.d.plot(plt).show()

    def accuracy(self):
        print("Accuracy: %f" % (sum(self.y == self.predict(self.X)) / len(self.y)))

    def run(self):
        print("Perceptron:")
        self.train()
        self.accuracy()
        self.plot()
