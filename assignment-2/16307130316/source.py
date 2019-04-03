import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import *
import numpy as np
import math
import matplotlib.pyplot as plt
import re
import string
import pickle
np.random.seed(1)

# Part 1
data_set = get_linear_seperatable_2d_2c_dataset()
# split the data set to training set and test set
train, test = data_set.split_dataset()


class LeastSquare:
    def __init__(self, data_set):
        self.W = self.least_square(data_set)

    def least_square(self, d):
        # compute X~
        X = np.array(d.X)
        b = np.ones(len(d.X))
        X = np.mat(np.insert(X, 0, values=b, axis=1))
        X_trans = X.T
        X_dot = np.dot(X_trans, X)
        X_pseudo_inverse = np.dot(np.mat(X_dot).I, X_trans)
        # turn T into one-hot representation
        T = np.array([[0, 1] if yy else [1, 0] for yy in d.y])

        W = np.dot(X_pseudo_inverse, T)
        return W

    def classify(self, X):  # classify a sample point
        X = np.mat([1, X[0], X[1]]).reshape(3, 1)
        result = np.dot(self.W.T, X)
        if result[0] > result[1]:
            return False
        else:
            return True

    def draw(self, plt):
        W = self.W.tolist()
        k = (W[1][0]-W[1][1])/(W[2][1]-W[2][0])
        b = (W[0][0]-W[0][1])/(W[2][1]-W[2][0])
        plt.plot([-1.5, 1.0], [-1.5 * k + b, 1.0 * k + b])
        return plt

    def compute_accuracy(self, d):  # calculate accuracy over test set
        pred_y = [self.classify(x) for x in d.X]
        return d.acc(pred_y)

"""
M = LeastSquare(train)
fig = plt.figure()
# plt.title("Training set, accuracy=%f" % M.compute_accuracy(train))
# plt = train.plot(plt)
plt.title("Test set, accuracy=%f" % M.compute_accuracy(test))
plt = test.plot(plt)  # d.X 为二维坐标，d.y是类别true or false,黄色是true，紫色是false
plt = M.draw(plt)
plt.show()
# print("Accuracy over training set:", M.compute_accuracy(train))
print("Accuracy over test set", M.compute_accuracy(test))
"""


class Perceptron:
    def __init__(self, data_set):
        self.w = self.perceptron(data_set)

    def perceptron(self, d):
        X = np.array(d.X)
        b = np.ones(len(d.X))
        X = np.mat(np.insert(X, 0, values=b, axis=1))

        # initial w
        w = np.random.rand(3, 1)

        # compute t
        t = [1 if yy else -1 for yy in d.y]

        for i in range(len(X)):
            if np.dot(X[i], w)*t[i] < 0:  # misclassification
                w = w + X[i].T*t[i]

        # print(w)
        return w

    def draw(self, plt):
        w = self.w.tolist()
        k = -w[1][0]/w[2][0]
        b = -w[0][0]/w[2][0]
        plt.plot([-1.5, 1.0], [-1.5 * k + b, 1.0 * k + b])
        return plt

    def classify(self, X):
        X = np.mat([1, X[0], X[1]]).reshape(3, 1)
        result = np.dot(self.w.T, X)
        if result[0] < 0:
            return False
        else:
            return True

    def compute_accuracy(self, d):  # accuracy over test set
        pred_y = [self.classify(x) for x in d.X]
        return d.acc(pred_y)

"""
MP = Perceptron(train)
fig = plt.figure()
# plt.title("Training set, accuracy=%f" % MP.compute_accuracy(train))
# plt = train.plot(plt)
plt.title("Test set, accuracy=%f" % MP.compute_accuracy(test))
plt = test.plot(plt)  # d.X 为二维坐标，d.y是类别true or false,黄色是true，紫色是false
plt = MP.draw(plt)
plt.show()
# print("Accuracy over training set:", MP.compute_accuracy(train))
print("Accuracy over test set", MP.compute_accuracy(test))
"""


# Part 2
# softmax with offset
def softmax(z):
    z = z - np.max(z)
    return np.exp(z)/np.sum(np.exp(z))


# Softmax for a matrix
def Softmax(z):
    if z.ndim > 1:
        return [softmax(zz) for zz in z]
    return softmax(z)


class LogisticTextClassification:
    def __init__(self):
        self.text_train, self.text_test = get_text_classification_datasets()
        self.train_vec = []
        self.test_vec = []
        self.vocab = []
        self.categories = self.text_train.target_names
        self.X, self.target_vec = self.preprocessing()
        # self.W = self.logistic()
        self.W = self.logistic_batched(5)
        # self.W = self.logistic_stochastic()

    # vector representation of input and target
    def preprocessing(self):
        for line in self.text_train.data:
            line = line.lower()
            for c in string.punctuation:
                line = line.replace(c, "")
            for w in string.whitespace:
                line = line.replace(w, " ")
            line = line.split()
            self.train_vec.append(line)

        self.vocab = sorted(list(set([w for line in self.train_vec for w in line])))
        self.vocab_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        multi_hot_vec = np.zeros((len(self.train_vec), len(self.vocab)))
        for i in range(len(self.train_vec)):
            for j in range(len(self.train_vec[i])):
                multi_hot_vec[i][self.vocab_to_idx[self.train_vec[i][j]]] = 1.0

        target_vec = np.zeros((len(self.text_train.target), 4))
        for i in range(len(target_vec)):
            target_vec[i][self.text_train.target[i]] = 1.0

        X = multi_hot_vec
        b = np.ones(len(X))
        X = np.insert(X, 0, values=b, axis=1)
        return X, target_vec

    def compute_partial(self, W):  # W:(1+D)*4, x:n*(1+D) n training samples，1+D dimensions
        error = Softmax(np.dot(self.X, W.T)) - self.target_vec
        N = len(self.X)
        return 1/N * np.dot(error.T, self.X)

    def compute_batched_partial(self, W, start, end):
        error = Softmax(np.dot(self.X[start:end], W.T)) - self.target_vec[start:end]
        N = end - start
        return 1/N * np.dot(error.T, self.X[start:end])

    def logistic(self):  # full batch
        if not os.path.exists('W.pkl'):  # store the model for reuse
            W = np.random.rand(4, len(self.X[0].tolist()))
            lr = 1  # exponential decay learning rate
            step = [i for i in range(10000)]
            loss = [self.compute_loss(W)]
            for i in range(len(step)):
                if i > 50:  # minimal lr is 0.1
                    W = W - 0.1 * self.compute_partial(W)
                else:
                    W = W - lr * (max(0.95 ** i, 0.1)) * self.compute_partial(W)
                new_loss = self.compute_loss(W)
                if new_loss > loss[-1]:  # loss不再下降
                    break
                loss.append(new_loss)

            plt.plot(step[:min(len(loss), 10000)], loss[:min(len(loss), 10000)])
            # print("final loss", loss[-1])
            # print("steps", i)
            plt.show()
            with open('W.pkl', 'wb') as of1:
                pickle.dump(W, of1)
        else:
            with open('W.pkl', 'rb') as if1:
                W = pickle.load(if1)
        return W

    def logistic_stochastic(self):  # stochastic gradient descent
        if not os.path.exists('W_stochastic.pkl'):
            W = np.random.rand(4, len(self.X[0].tolist()))
            N = len(self.X)
            lr = 1
            step = [i for i in range(10000)]
            loss = [self.compute_loss(W)]
            for i in range(len(step)):
                n = np.random.randint(0, N-1)  # select a sample randomly to update W
                if i > 50:  # minimal lr is 0.1
                    W = W - 0.1 * self.compute_batched_partial(W, n, n+1)
                else:
                    W = W - lr * (max(0.95 ** i, 0.1)) * self.compute_batched_partial(W, n, n+1)

                new_loss = self.compute_loss(W)
                loss.append(new_loss)
                if new_loss < 0.00001:
                    print("loss,", new_loss)
                    print("steps ", i+1)
                    break

            plt.plot(step[:min(len(loss), 10000)], loss[:min(len(loss), 10000)])
            # print("final loss", loss[-1])
            # print("steps", i)
            plt.show()
            with open('W_stochastic.pkl', 'wb') as of1:
                pickle.dump(W, of1)
        else:
            with open('W_stochastic.pkl', 'rb') as if1:
                W = pickle.load(if1)
        return W

    def logistic_batched(self, batch_size):  # batched gradient descent
        if not os.path.exists('W_batched.pkl'):
            W = np.random.rand(4, len(self.X[0].tolist()))
            N = len(self.X)
            lr = 1
            step = [i for i in range(2000)]
            loss = [self.compute_loss(W)]
            for k in range(3):
                for i in range(0, N, batch_size):
                    if i+batch_size > N:  # in case N is not sufficient large
                        batch_size = N - i
                    if i > 50:  # minimal lr is 0.1
                        W = W - 0.1 * self.compute_batched_partial(W, i, i + batch_size)
                    else:
                        W = W - lr * (max(0.95 ** i, 0.1)) * self.compute_batched_partial(W, i, i + batch_size)
                    new_loss = self.compute_loss(W)
                    loss.append(new_loss)
                    if new_loss < 0.00001:
                        print("loss,", new_loss)
                        print("steps ", i+1)
                        break

            plt.plot(step[:min(len(loss), 2000)], loss[:min(len(loss), 2000)])
            # print("final loss", loss[-1])
            # print("steps", i)
            plt.show()
            with open('W_batched.pkl', 'wb') as of1:
                pickle.dump(W, of1)
        else:
            with open('W_batched.pkl', 'rb') as if1:
                W = pickle.load(if1)
        return W

    def classify(self, W, xn):
        prob = Softmax(np.dot(xn, W.T))
        return prob.tolist().index(max(prob))

    def compute_loss(self, W):
        N = len(self.X)
        return -1/N * sum(math.log(Softmax(np.dot(self.X[i], W.T))[self.text_train.target[i]]) for i in range(N))

    """
    def compute_batched_loss(self, W, start, end):
        N = end - start
        return -1/N * sum(math.log(Softmax(np.dot(self.X[i], W.T))[self.text_train.target[i]]) for i in range(start,end))
    """
    def compute_accuracy(self):
        for line in self.text_test.data:
            line = line.lower()
            for c in string.punctuation:
                line = line.replace(c, "")
            for w in string.whitespace:
                line = line.replace(w, " ")
            line = line.split()
            self.test_vec.append(line)
        multi_hot_vec = np.zeros((len(self.test_vec), len(self.vocab)))
        for i in range(len(self.test_vec)):
            for j in range(len(self.test_vec[i])):
                if self.test_vec[i][j] in self.vocab:  # 如果这个词在vocab中能找到
                    # print("yes")
                    multi_hot_vec[i][self.vocab_to_idx[self.test_vec[i][j]]] = 1.0

        test_target_vec = np.zeros((len(self.text_test.target), 4))
        for i in range(len(test_target_vec)):
            test_target_vec[i][self.text_test.target[i]] = 1.0

        test_X = multi_hot_vec
        b = np.ones(len(test_X))
        test_X = np.insert(test_X, 0, values=b, axis=1)
        right = 0
        for i in range(len(test_X)):
            right = right + (self.classify(self.W, test_X[i]) == self.text_test.target[i])

        # accuracy = sum(self.classify(self.W, test_X[i]) == self.text_test.target[i] for i in range(len(test_X)))/ len(self.text_test.data)
        return right/len(self.text_test.data)

"""
ML = LogisticTextClassification()
print(ML.compute_loss(ML.W))
print(ML.compute_accuracy())
"""






