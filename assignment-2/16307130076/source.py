import os
import random
import string
from collections import Counter

os.sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np
import argparse

from handout import (get_linear_seperatable_2d_2c_dataset,
                     get_text_classification_datasets)


def lsm(dataset):
    X = np.insert(dataset.X, 2, values=np.ones(dataset.X.shape[0]), axis=1)
    y = np.array(dataset.y, dtype=np.int8)
    y[y == 0] = -1
    W = np.matmul(np.linalg.inv(
        np.matmul(X.transpose(), X)), np.matmul(X.transpose(), y))
    x1 = np.linspace(-2, 2, 1000)
    x2 = -(W[2]/W[1]) - (W[0]/W[1])*x1
    plt.plot(x1, x2)
    dataset.plot(plt).show()
    pred_y = np.matmul(W, X.transpose())
    pred_y[pred_y >= 0] = 1
    pred_y[pred_y <= 0] = 0
    print("acc:",dataset.acc(pred_y))


def perceptron(dataset, n, it):
    w = [1, 1]
    b = 0
    X = dataset.X
    y = np.array(dataset.y, dtype=np.int8)
    y[y == 0] = -1
    for lr in range(it):
        flag = False
        for i in range(X.shape[0]):
            if y[i]*(np.dot(w, X[i])+b) <= 0:
                flag = True
                w = w + n*y[i]*X[i]
                b = b + n*y[i]
        if not flag:
            x1 = np.linspace(-2, 2, 1000)
            x2 = -(b/w[1]) - (w[0]/w[1])*x1
            plt.plot(x1, x2)
            dataset.plot(plt).show()
            pred_y = np.matmul(w, X.transpose())+b
            pred_y[pred_y >= 0] = 1
            pred_y[pred_y <= 0] = 0
            print("acc:", dataset.acc(pred_y),", steps:",lr)
            break
，

def convert_data(train_data, test_data, min_count):
    cnt = Counter()
    processed_train_data = []
    processed_test_data = []
    data = train_data + test_data
    for item in train_data:
        processed_item = item.lower().replace(string.punctuation, '').split()
        cnt.update(processed_item)
        processed_train_data.append(processed_item)
    for item in test_data:
        processed_item = item.lower().replace(string.punctuation, '').split()
        processed_test_data.append(processed_item)
    for item in cnt.items():
        if item[1] < min_count:
            cnt[item[0]] = 0
    cnt = cnt - Counter()
    plist = sorted(cnt)
    print('[NOTE]create dictionary complete!')
    train_vec = np.zeros((len(train_data), len(plist)))
    test_vec = np.zeros((len(test_data), len(plist)))
    for idx, item in enumerate(processed_train_data):
        for word in item:
            if word in plist:
                train_vec[idx][plist.index(word)] = 1
        print('[INFO]preprocessing train data...',
              idx, '/', len(processed_train_data))
    for idx, item in enumerate(processed_test_data):
        for word in item:
            if word in plist:
                test_vec[idx][plist.index(word)] = 1
        print('[INFO]preprocessing test data...',
              idx, '/', len(processed_test_data))
    return train_vec, test_vec


def convert_category(data, num):
    vec = np.zeros((len(data), num))
    for idx in range(len(data)):
        vec[idx][data[idx]] = 1
        print('[INFO]preprocessing category...', idx, '/', len(data))
    return vec


def softmax(x):
    if len(x.shape) > 1:
        # Matrix
        def exp_minmax(x): return np.exp(x - np.max(x))

        def denom(x): return 1.0 / np.sum(x)
        x = np.apply_along_axis(exp_minmax, 1, x)
        denominator = np.apply_along_axis(denom, 1, x)

        if len(denominator.shape) == 1:
            denominator = denominator.reshape((denominator.shape[0], 1))

        x = x * denominator
    else:
        # Vector
        x_max = np.max(x)
        x = x - x_max
        numerator = np.exp(x)
        denominator = 1.0 / np.sum(numerator)
        x = numerator.dot(denominator)

    return x


def optimizer(w, b, c, X, Y):
    n = X.shape[0]
    A = softmax(np.dot(X, w)+b)
    cost = -(1/n)*np.sum(Y*np.log(A)) + 0.5*c*np.sum(w*w)
    dw = (1/n) * np.dot(X.T, (A-Y)) + c*w
    db = (1/n) * np.dot(np.ones(n), (A-Y))
    return dw, db, cost


def predict(X, w, b):
    return softmax(np.dot(X, w)+b)


def accuracy(y_hat, y):
    accuracy = np.sum(np.argmax(y_hat, axis=1) == np.argmax(y, axis=1))
    accuracy = accuracy*1.0/y.shape[0]
    return accuracy


def train(X, y, batch_size, c, learning_rate=0.1, iterations=1000):
    costs = []
    w = np.zeros((X.shape[1], y.shape[1]))
    b = np.zeros(y.shape[1])

    for i in range(iterations):
        rdm_idx = random.randint(0, X.shape[0]-batch_size)
        bX = X[rdm_idx:rdm_idx+batch_size]
        by = y[rdm_idx:rdm_idx+batch_size]
        dw, db, cost = optimizer(w, b, c, bX, by)
        w -= learning_rate * dw
        b -= learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
            print('[INFO]iteration %d, acc: %.6f, cost: %.6f' % (i,
                                                                 accuracy(predict(X, w, b), y), cost))

    y_hat = predict(X, w, b)
    acc = accuracy(y_hat, y)
    print("After %d iterations,the total accuracy is %f" %
          (iterations, acc))
    xs = np.linspace(1, iterations, iterations/100)
    plt.plot(xs, costs)
    plt.show()

    return w, b


def test(X, y, w, b):
    y_hat_test = predict(X, w, b)
    acc = accuracy(y_hat_test, y)
    print(acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', type=str, choices=[
                        'lsm', 'perceptron', 'full_batch', 'stochastic', 'batched'], help='tell me what you want to do')
    parser.add_argument('--c', '-c', type=float,
                        help='para c in regularization')
    parser.add_argument('--learning_rate', '-lr', type=float)
    parser.add_argument('--batch_size', '-bs', type=int)
    parser.add_argument('--iterations', '-i', type=int)
    args = parser.parse_args()
    if args.operation == 'lsm':
        lsm(get_linear_seperatable_2d_2c_dataset())
    elif args.operation == 'perceptron':
        perceptron(get_linear_seperatable_2d_2c_dataset(),
                   args.learning_rate, args.iterations)
    else:
        try:
            train_vec_x = np.loadtxt('train_vec_x.txt', dtype=np.bool)
            train_vec_y = np.loadtxt('train_vec_y.txt', dtype=np.bool)
            print('[NOTE]train data load complete!')
            test_vec_x = np.loadtxt('test_vec_x.txt', dtype=np.bool)
            test_vec_y = np.loadtxt('test_vec_y.txt', dtype=np.bool)
            print('[NOTE]test data load complete!')
        except OSError:
            train_data, test_data = get_text_classification_datasets()
            train_vec_x, test_vec_x = convert_data(
                train_data.data, test_data.data, 10)
            train_vec_y = convert_category(train_data.target, 4)
            test_vec_y = convert_category(test_data.target, 4)
            np.savetxt('train_vec_x.txt', train_vec_x, fmt='%d')
            np.savetxt('train_vec_y.txt', train_vec_y, fmt='%d')
            np.savetxt('test_vec_x.txt', test_vec_x, fmt='%d')
            np.savetxt('test_vec_y.txt', test_vec_y, fmt='%d')
        if args.operation == 'full_batch':
            w, b = train(train_vec_x, train_vec_y,
                         train_vec_x.shape[0], args.c, args.learning_rate, args.iterations)
        elif args.operation == 'stochastic':
            w, b = train(train_vec_x, train_vec_y, 1, args.c,
                         args.learning_rate, args.iterations)
        elif args.operation == 'batched':
            w, b = train(train_vec_x, train_vec_y, args.batch_size,
                         args.c, args.learning_rate, args.iterations)
        test(test_vec_x, test_vec_y, w, b)
