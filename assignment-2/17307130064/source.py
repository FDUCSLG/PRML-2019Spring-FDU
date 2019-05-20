import os
os.sys.path.append('..')

from handout import get_linear_seperatable_2d_2c_dataset
from handout import get_text_classification_datasets
import numpy as np
from matplotlib import pyplot as plt
import string


def least_squares_classify(X:np.ndarray, y:np.ndarray):
    W = np.linalg.pinv(X) @ y
    return W

d1 = get_linear_seperatable_2d_2c_dataset()
y1 = np.eye(2)[d1.y.astype(np.int)]

W1 = least_squares_classify(d1.X, y1)

x1_test, x2_test = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
x_test = np.array([x1_test, x2_test]).reshape(2, -1).T
y_pred = np.argmax(x_test @ W1, axis = -1)

plt.scatter(d1.X[:, 0], d1.X[:, 1], c = d1.y)
plt.contourf(x1_test, x2_test, y_pred.reshape(100, 100), alpha = 0.2, levels = np.linspace(0, 1, 3))
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.gca().set_aspect('equal', adjustable = 'box')
plt.show()

y_pred_2 = np.argmax(d1.X @ W1, axis = -1)
print(y_pred_2)
print(d1.y.astype(np.int))

cnt = np.sum(np.abs(y_pred_2 - d1.y.astype(np.int)), 0)
print(cnt)
print(1 - cnt/len(d1.y))
# 0.905


def perceptron(X:np.ndarray, y:np.ndarray, max_epoch = 100):
    W = np.zeros(np.size(X, 1))
    for _ in range(max_epoch):
        N = len(y)
        index = np.random.permutation(N)
        X = X[index]
        y = y[index]
        for x, label in zip(X, y):
            W += x * label
            if (X @ W * y > 0).all():
                break
        else:
            continue
        break
    return W

y2 = d1.y.astype(np.int) * 2 - 1

W2 = perceptron(d1.X, y2)

x1_test, x2_test = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
x_test = np.array([x1_test, x2_test]).reshape(2, -1).T
y_pred = (np.sign(x_test @ W2).astype(np.int) + 1) / 2

plt.scatter(d1.X[:, 0], d1.X[:, 1], c = d1.y)
plt.contourf(x1_test, x2_test, y_pred.reshape(100, 100), alpha = 0.2, levels = np.linspace(0, 1, 3))
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.gca().set_aspect('equal', adjustable = 'box')
plt.show()

y_pred_2 = (np.sign(d1.X @ W2).astype(np.int) + 1) / 2
print(y_pred_2)
print(d1.y.astype(np.int))

cnt = np.sum(np.abs(y_pred_2 - d1.y.astype(np.int)), 0)
print(cnt)
print(1 - cnt/len(d1.y))
# 0.84


def create_vocabulary(data):
    re = []
    for s in data.data[0:200]:
        s = s.lower()
        for i in s:
            if i in string.punctuation:
                s = s.replace(i, " ")
        re += s.split()

    re = list(set(re))
    re.sort()

    vocabulary = {}
    for i in range(len(re)):
        vocabulary[re[i]] = i
    return vocabulary

def create_X(data, vocabulary):
    K = len(vocabulary)
    re = np.empty([1, K], dtype = int)
    for s in data.data[0:200]:
        s = s.lower()
        for i in s:
            if i in string.punctuation:
                s = s.replace(i, " ")
        s = s.split()
        ss = np.zeros(K)
        for i in s:
            if i in vocabulary:
                ss[vocabulary[i]] = 1
        re = np.vstack((re, ss))
    re = np.delete(re, 0, axis = 0)
    return re

def create_y(data):
    K = 4
    re = np.empty([1, K], dtype = int)
    for i in data.target[0:200]:
        one_hot = np.eye(K)[i]
        re = np.vstack((re, one_hot))
    re = np.delete(re, 0, axis = 0)
    return re

d_train, d_test = get_text_classification_datasets()

def softmax(a):
    a_max = np.max(a, axis = -1, keepdims = True)
    exp_a = np.exp(a - a_max)
    return exp_a / np.sum(exp_a, axis = -1, keepdims = True)

def logistic_regression(X, y, max_iter : int = 100, learning_rate : float = 0.1):
    W = np.zeros((np.size(X, 1), np.size(y, 1)))
    for _ in range(max_iter):
        W_prev = np.copy(W)
        y_pred = softmax(X @ W)
        grad = X.T @ (y_pred - y)
        W -= learning_rate * grad
        if np.allclose(W, W_prev):
            break
    return W

def logistic_regression_2(X, y, max_iter : int = 100, learning_rate : float = 0.1):
    W = np.zeros((np.size(X, 1), np.size(y, 1)))
    for _ in range(max_iter):
        N = len(y)
        index = np.random.permutation(N)
        X = X[index]
        y = y[index]
        W_prev = np.copy(W)
        y_pred = softmax(X @ W)
        grad = X.T @ (y_pred - y)
        W -= learning_rate * grad
        if np.allclose(W, W_prev):
            break
    return W

def logistic_regression_3(X, y, max_iter : int = 100, learning_rate : float = 0.1):
    W = np.zeros((np.size(X, 1), np.size(y, 1)))
    for _ in range(max_iter):
        N = len(y)
        index = np.random.permutation(N)
        X = X[index]
        y = y[index]
        W_prev = np.copy(W)
        y_pred = softmax(X[0:10][:] @ W)
        grad = X[0:10][:].T @ (y_pred - y[0:10])
        W -= learning_rate * grad
        if np.allclose(W, W_prev):
            break
    return W

X = create_X(d_train, create_vocabulary(d_train))
y = create_y(d_train)
W = logistic_regression_3(X, y)
X_test = create_X(d_test, create_vocabulary(d_train))
p1 = np.argmax(softmax(X_test @ W), axis = -1)
p2 = d_test.target[0:200]
print(np.mean(p1 == p2))