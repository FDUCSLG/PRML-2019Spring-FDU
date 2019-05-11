import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import *
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from functools import reduce
import string
import re
import collections
import random
import sys
import argparse

dataset_a1 = get_linear_seperatable_2d_2c_dataset()
dataset_train, dataset_test = get_text_classification_datasets()


def least_square_1(X, y):

    y_tf = [1 if x else -1 for x in y]
    X_tf = np.hstack((np.ones((X.shape[0], 1)), X))

    weight = reduce(np.dot, [np.linalg.inv(np.dot(X_tf.T, X_tf)), X_tf.T, y_tf])
    test_output = np.dot(X_tf, weight)
    test_output = test_output >= 0
    accuracy = sum(y == test_output) / test_output.size
    print("accuracy of least_square_model =", accuracy)

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.005), np.arange(x2_min, x2_max, 0.005))
    yy = np.dot((np.array([xx1.ravel(), xx2.ravel()])).T, weight[1:]) + weight[0]
    yy = (yy >= 0).reshape(xx1.shape)
    plt.contourf(xx1, xx2, yy, alpha=0.3)

    colors = np.where(y, 'red', 'blue')
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Decision Region by Least Square')
    plt.show()


class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for i in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                if(self.predict(xi) * target <= 0):
                    update = self.eta * target
                    self.w_[1:] += update * xi
                    self.w_[0] += update
                    errors += 1
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) > 0.0, 1, -1)


def decision_region(X, y, classifier):

    colors = ('red', 'blue', 'yellow', 'green', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.005), np.arange(x2_min, x2_max, 0.005))
    yy = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    yy = yy.reshape(xx1.shape)
    plt.contourf(xx1, xx2, yy, alpha=0.3)

    for idx, t in enumerate(np.unique(y)):
        plt.scatter(x=X[y == t, 0], y=X[y == t, 1], alpha=0.8, c=colors[idx], label=t)


def accuracy_of_model(X, y, classifier):
    test_output = classifier.predict(X)
    return sum(test_output == y) / X.shape[0]


def assignment1(dataset):

    least_square_1(dataset.X, dataset.y)

    ppn_a1 = Perceptron(0.0005, 10)
    dataset_y_tf = [1 if x else -1 for x in dataset.y]
    ppn_a1.fit(dataset.X, dataset_y_tf)
    plt.plot(range(1, len(ppn_a1.errors_) + 1), ppn_a1.errors_)
    plt.title('Process of Fitting with Perceptron')
    plt.xlabel('Epochs')
    plt.ylabel('Errors')
    plt.show()

    print('accuracy of perceptron = ', accuracy_of_model(dataset.X, dataset_y_tf, ppn_a1))
    decision_region(dataset.X, dataset_y_tf, ppn_a1)
    plt.title('Decision Region by Perceptron')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# assignment1(dataset_a1)


def vocabulary_init(data_text):
    total_word_list = []

    for piece in data_text:
        piece = re.sub('[' + string.punctuation + ']', '', piece)
        piece = re.sub('[' + string.whitespace + '\u200b]+', ' ', piece)
        piece = piece.strip().lower()
        word_list = piece.split(' ')
        total_word_list += word_list

    word_f_dict = collections.Counter(total_word_list)

    word_set = [word for word in word_f_dict if word_f_dict[word] >= 10]
    vocab = {word:i for i, word in enumerate(sorted(word_set))}

    return vocab


def pre_processing(data_text, targets, cate_num, vocab):
    text_word_lists = []

    for piece in data_text:
        piece = re.sub('[' + string.punctuation + ']', '', piece)
        piece = re.sub('[' + string.whitespace + '\u200b]', ' ', piece)
        piece = piece.strip().lower()
        word_list = piece.split(' ')
        text_word_lists.append(word_list)

    multi_hot = []
    for word_list in text_word_lists:
        hot_vec = np.zeros(len(vocab))
        for word in word_list:
            if word in vocab:
                hot_vec[vocab[word]] = 1
        multi_hot.append(hot_vec)

    one_hot = np.zeros((len(targets), cate_num))
    for i, t in enumerate(targets):
        one_hot[i, t] = 1

    return np.mat(multi_hot), one_hot


def softmax(x):
    return np.exp(x) / (np.sum(np.exp(x), axis=1))


def cross_entropy_loss(X, y, W, b, lamda):
    N = X.shape[0]
    return lamda * np.sum(np.square(W)) - np.trace(np.dot(y.T, np.log(softmax(np.dot(X, W) + b.T)))) / N


def loss_grad(X, y, W, b, lamda):
    y_hat = softmax(np.dot(X, W) + b.T)
    grad_w = 2 * lamda * W - np.dot(X.T, y - y_hat) / X.shape[0]
    grad_b = (- np.sum((y - y_hat), axis=0) / X.shape[0]).T
    return grad_w, grad_b


def grad_check(X, y, num_check=10, lamda=0.1, h=1e-5):
    N, D, P = X.shape[0], X.shape[1], y.shape[1]
    for _ in range(num_check):
        W = np.random.rand(D, P)
        b = np.random.rand(P, 1)
        grad_w, grad_b = loss_grad(X, y, W, b, lamda)
        iw = tuple([random.randint(0, D - 1), random.randint(0, P - 1)])
        ib = tuple([random.randint(0, P - 1), 0])
        W[iw] += h
        fwph = cross_entropy_loss(X, y, W, b, lamda)
        W[iw] -= 2 * h
        fwmh = cross_entropy_loss(X, y, W, b, lamda)
        W[iw] += h
        grad_w_num = (fwph - fwmh) / (2 * h)
        grad_w_iw = grad_w[iw]
        w_error = abs(grad_w_num - grad_w_iw) / (abs(grad_w_num) + abs(grad_w_iw))

        b[ib] += h
        fbph = cross_entropy_loss(X, y, W, b, lamda)
        b[ib] -= 2 * h
        fbmh = cross_entropy_loss(X, y, W, b, lamda)
        b[ib] += h
        grad_b_num = (fbph - fbmh) / (2 * h)
        grad_b_ib = grad_b[ib]
        b_error = abs(grad_b_num - grad_b_ib) / (abs(grad_b_num) + abs(grad_b_ib))

        print('grad_W: numerical: %f analytic: %f, relative error: %e' % (grad_w_num, grad_w_iw, w_error))
        print('grad_b: numerical: %f analytic: %f, relative error: %e' % (grad_b_num, grad_b_ib, b_error))
        if (w_error > 1e-5 or b_error > 1e-5):
            print('relative error of grad is too large!')
            return

    print('In the test, all the relative errors < 1e-5')


class SoftmaxPpn(object):
    def __init__(self, eta=0.01, lamda=0.001, batch=1, var_size = 20, stable_var=1e-6):
        self.eta = eta
        self.lamda = lamda
        self.batch = batch
        self.var_size = var_size
        self.stb_var = stable_var

    def fit(self, X, y, n_iter=50):
        char_num = X.shape[1]
        cate_num = y.shape[1]
        self.w_ = np.zeros((char_num, cate_num))
        self.b_ = np.zeros((cate_num, 1))
        self.loss_ = []
        self.accuracy_ = []
        N = X.shape[0]
        flag = False

        for _ in range(n_iter):
            for i in range(0, N, self.batch):
                grad_W, grad_b = loss_grad(X[i:min(i + self.batch, N)], y[i:min(i + self.batch, N)],
                                           self.w_, self.b_, self.lamda)
                self.w_ -= self.eta * grad_W
                self.b_ -= self.eta * grad_b
                self.loss_.append(cross_entropy_loss(X, y, self.w_, self.b_, self.lamda))
                self.accuracy_.append(self.accuracy(X, y))
                if (len(self.loss_) % self.var_size == 0 and np.var(self.loss_[-self.var_size:]) < self.stb_var):
                    flag = True
                    break
            if(flag):
                break
            if(self.batch < N):
                connected_xy = np.hstack((X, y))
                np.random.shuffle(connected_xy)
                X = connected_xy[0:N, 0:char_num].copy()
                y = connected_xy[0:N, char_num:char_num + cate_num].copy()
        return self

    def fit_cv(self, X, y, vali_X, vali_y, n_iter=50):
        char_num = X.shape[1]
        cate_num = y.shape[1]
        self.w_ = np.zeros((char_num, cate_num))
        self.b_ = np.zeros((cate_num, 1))
        self.loss_ = []
        self.loss_vali = []
        self.accuracy_ = []
        self.accuracy_vali = []
        N = X.shape[0]
        flag = False

        for _ in range(n_iter):
            for i in range(0, N, self.batch):
                grad_W, grad_b = loss_grad(X[i:min(i + self.batch, N)], y[i:min(i + self.batch, N)],
                                           self.w_, self.b_, self.lamda)
                self.w_ -= self.eta * grad_W
                self.b_ -= self.eta * grad_b
                self.loss_.append(cross_entropy_loss(X, y, self.w_, self.b_, self.lamda))
                self.loss_vali.append(self.predict(vali_X, vali_y))
                self.accuracy_.append(self.accuracy(X, y))
                self.accuracy_vali.append(self.accuracy(vali_X, vali_y))
                if (len(self.loss_) % self.var_size == 0 and np.var(self.loss_[-self.var_size:]) < self.stb_var):
                    flag = True
                    break

            if(flag):
                break
            if(self.batch < N):
                connected_xy = np.hstack((X, y))
                np.random.shuffle(connected_xy)
                X = connected_xy[0:N, 0:char_num].copy()
                y = connected_xy[0:N, char_num:char_num + cate_num].copy()

        print("eta=%.3f, lamda=%.2e, validation_loss=%e, validation_accuracy=%.2f%%, epochs=%d" %
              (self.eta, self.lamda, self.loss_vali[-1], self.accuracy_vali[-1] * 100, len(self.loss_)))
        return self

    def predict(self, X, y):
        return cross_entropy_loss(X, y, self.w_, self.b_, self.lamda)

    def accuracy(self, X, y):
        y_hat = np.argmax(softmax(np.dot(X, self.w_) + self.b_.T), axis=1)
        sum_num = 0
        for i, j in zip(range(y.shape[0]), y_hat):
            sum_num += y[i, j]
        return float(sum_num / X.shape[0])


vocab = vocabulary_init(dataset_train.data)
# print(vocab)
cate_num = len(dataset_train.target_names)
train_multi_hot, train_one_hot = pre_processing(dataset_train.data, dataset_train.target, cate_num, vocab)
test_multi_hot, test_one_hot = pre_processing(dataset_test.data, dataset_test.target, cate_num, vocab)
# print(len(vocab))
# grad_check(train_multi_hot, train_one_hot, 100)


def cross_validation_fb(X, y, lamda, eta_min, eta_max, eta_step):
    N1 = int(X.shape[0] * 0.7)
    N2 = X.shape[0] - N1
    eta_range = np.arange(eta_min, eta_max, eta_step)
    vali_loss = []
    vali_accuracy = []

    for eta in eta_range:
        ppn = SoftmaxPpn(eta, lamda=lamda, batch=N1, var_size=10, stable_var=1e-6)
        ppn.fit_cv(X[:N1], y[:N1], X[N1:], y[N1:], 2000)
        # print(np.var(ppn.loss_[-ppn.var_size:]))

        vali_loss.append(ppn.loss_vali[-1])
        plt.plot(range(1, len(ppn.loss_)+1), ppn.loss_, label='train_set')
        plt.plot(range(1, len(ppn.loss_vali)+1), ppn.loss_vali, label='validation_set', alpha=0.5)
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.title('Full Batch GD\nlearning_rate=%.3f, lamda=%.2e' % (ppn.eta, ppn.lamda))
        plt.show()
        vali_accuracy.append(ppn.accuracy_vali[-1])
        plt.plot(range(1, len(ppn.accuracy_) + 1), ppn.accuracy_, label='train_set')
        plt.plot(range(1, len(ppn.accuracy_vali) + 1), ppn.accuracy_vali, label='validation_set', alpha=0.5)
        plt.xlabel('epochs')
        plt.ylabel('Accuracy')
        plt.title('Full Batch GD\nlearning_rate=%.3f, lamda=%.2e' % (ppn.eta, ppn.lamda))
        plt.legend()
        plt.show()
    plt.plot(eta_range, vali_loss)
    plt.title('Full Batch GD\nlamda =%.2e' % lamda)
    plt.xlabel('learning_rate')
    plt.ylabel('Validation_Loss')
    plt.show()
    plt.plot(eta_range, vali_accuracy)
    plt.title('Full Batch GD\nlamda =%.2e' % lamda)
    plt.xlabel('learning_rate')
    plt.ylabel('Validation_Accuracy')
    plt.show()


def cross_validation_sb(X, y, lamda, eta_min, eta_max, eta_step):
    N1 = int(X.shape[0] * 0.7)
    N2 = X.shape[0] - N1
    eta_range = np.arange(eta_min, eta_max, eta_step)
    vali_loss = []
    vali_accuracy = []

    for eta in eta_range:
        ppn = SoftmaxPpn(eta, lamda=lamda, batch=1, var_size=100)
        ppn.fit_cv(X[:N1], y[:N1], X[N1:], y[N1:], 10)
        vali_loss.append(ppn.loss_vali[-1])
        plt.plot(range(1, len(ppn.loss_)+1), ppn.loss_, label='train_set')
        plt.plot(range(1, len(ppn.loss_vali)+1), ppn.loss_vali, label='validation_set', alpha=0.5)
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.title('Stochastic Batch GD\nlearning_rate=%.3f, lamda=%.2e' % (ppn.eta, ppn.lamda))
        plt.legend()
        plt.show()
        vali_accuracy.append(ppn.accuracy_vali[-1])
        plt.plot(range(1, len(ppn.accuracy_) + 1), ppn.accuracy_, label='train_set')
        plt.plot(range(1, len(ppn.accuracy_vali) + 1), ppn.accuracy_vali, label='validation_set', alpha=0.5)
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('Accuracy')
        plt.title('Stochastic Batch GD\nlearning_rate=%.3f, lamda=%.2e' % (ppn.eta, ppn.lamda))
        plt.show()
    plt.plot(eta_range, vali_loss)
    plt.title('Stochastic Batch GD\nlamda =%.2e' % lamda)
    plt.xlabel('learning_rate')
    plt.ylabel('Validation_Loss')
    plt.show()
    plt.plot(eta_range, vali_accuracy)
    plt.title('Stochastic Batch GD\nlamda =%.2e' % lamda)
    plt.xlabel('learning_rate')
    plt.ylabel('Validation_Accuracy')
    plt.show()


def cross_validation_mb(X, y, lamda, batch_size, eta_min, eta_max, eta_step):
    N1 = int(X.shape[0] * 0.7)
    N2 = X.shape[0] - N1
    eta_range = np.arange(eta_min, eta_max, eta_step)
    vali_loss = []
    vali_accuracy = []

    for eta in eta_range:
        ppn = SoftmaxPpn(eta, lamda=lamda, batch=batch_size, var_size=20, stable_var=1e-6)
        ppn.fit_cv(X[:N1], y[:N1], X[N1:], y[N1:], 50)
        vali_loss.append(ppn.loss_vali[-1])
        plt.plot(range(1, len(ppn.loss_)+1), ppn.loss_, label='train_set')
        plt.plot(range(1, len(ppn.loss_vali)+1), ppn.loss_vali, label='validation_set', alpha=0.5)
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.title('Mini-batch GD\nlearning_rate=%.3f, batch=%d, lambda=%.2e' % (ppn.eta, ppn.batch, ppn.lamda))
        plt.legend()
        plt.show()
        vali_accuracy.append(ppn.accuracy_vali[-1])
        plt.plot(range(1, len(ppn.accuracy_) + 1), ppn.accuracy_, label='train_set')
        plt.plot(range(1, len(ppn.accuracy_vali) + 1), ppn.accuracy_vali, label='validation_set', alpha=0.5)
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('Accuracy')
        plt.title('Mini-batch GD\nlearning_rate=%.3f, batch=%d, lambda=%.2e' % (ppn.eta, ppn.batch, ppn.lamda))
        plt.show()
    plt.plot(eta_range, vali_loss)
    plt.title('Mini-batch GD\nbatch=%d, lambda =%.2e' % (batch_size, lamda))
    plt.xlabel('learning_rate')
    plt.ylabel('Validation_Loss')
    plt.show()
    plt.plot(eta_range, vali_accuracy)
    plt.title('Mini-batch GD\nbatch=%d, lambda =%.2e' % (batch_size, lamda))
    plt.xlabel('learning_rate')
    plt.ylabel('Validation_Accuracy')
    plt.show()


def k_cross_validation(k, X, y, lamda, batch_size, eta_min, eta_max, eta_step, n_iter=500, var_size=20):
    N = X.shape[0]
    vali_len = int(N / k) + 1
    vali_sequence = range(0, N, vali_len)
    eta_range = np.arange(eta_min, eta_max, eta_step)
    vali_loss = []
    vali_accuracy = []

    char_num = X.shape[1]
    cate_num = y.shape[1]
    connected_xy = np.hstack((X, y))
    np.random.shuffle(connected_xy)
    X = connected_xy[0:N, 0:char_num].copy()
    y = connected_xy[0:N, char_num:char_num + cate_num].copy()

    for eta in eta_range:
        ppn = SoftmaxPpn(eta, lamda=lamda, batch=batch_size, var_size=var_size, stable_var=1e-6)
        ppn_loss = np.empty(0)
        ppn_loss_vali = np.empty(0)
        ppn_accuracy_ = np.empty(0)
        ppn_accuracy_vali = np.empty(0)
        min_epochs = 1e+10
        mean_loss = 0
        mean_ac = 0
        for vali_begin in vali_sequence:
            ppn.fit_cv(np.vstack((X[0:vali_begin], X[min(N, vali_begin+vali_len):N])),
                       np.vstack((y[0:vali_begin], y[min(N, vali_begin + vali_len):N])),
                       X[vali_begin:vali_begin+vali_len], y[vali_begin:vali_begin+vali_len], n_iter)
            mean_loss += ppn.loss_vali[-1]
            mean_ac += ppn.accuracy_vali[-1]
            min_epochs = min(min_epochs, len(ppn.loss_))
            if(vali_begin == 0):
                ppn_loss = np.array(ppn.loss_)
                ppn_loss_vali = np.array(ppn.loss_vali)
                ppn_accuracy_ = np.array(ppn.accuracy_)
                ppn_accuracy_vali = np.array(ppn.accuracy_vali)
            else:
                ppn_loss[:min_epochs] += np.array(ppn.loss_)[:min_epochs]
                ppn_loss_vali[:min_epochs] += np.array(ppn.loss_vali)[:min_epochs]
                ppn_accuracy_[:min_epochs] += np.array(ppn.accuracy_)[:min_epochs]
                ppn_accuracy_vali[:min_epochs] += np.array(ppn.accuracy_vali)[:min_epochs]

        ppn_loss /= k
        ppn_loss_vali /= k
        ppn_accuracy_ /= k
        ppn_accuracy_vali /= k
        mean_loss /= k
        mean_ac /= k
        vali_loss.append(mean_loss)
        vali_accuracy.append(mean_ac)
        print('GD: learning_rate=%.3f, batch=%d, lambda=%.2e\n'
              'mean_validation_loss=%.3f, mean_validation_accuracy=%.2f%%' %
              (ppn.eta, ppn.batch, ppn.lamda, mean_loss, 100 * mean_ac))
        plt.plot(range(1, min_epochs + 1), ppn_loss[:min_epochs], label='train_set')
        plt.plot(range(1, min_epochs + 1), ppn_loss_vali[:min_epochs], label='validation_set', alpha=0.5)
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.title('GD\nlearning_rate=%.3f, batch=%d, lambda=%.2e' % (ppn.eta, ppn.batch, ppn.lamda))
        plt.legend()
        plt.show()
        plt.plot(range(1, min_epochs + 1), ppn_accuracy_[:min_epochs], label='train_set')
        plt.plot(range(1, min_epochs + 1), ppn_accuracy_vali[:min_epochs], label='validation_set', alpha=0.5)
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('Accuracy')
        plt.title('GD\nlearning_rate=%.3f, batch=%d, lambda=%.2e' % (ppn.eta, ppn.batch, ppn.lamda))
        plt.show()

    plt.plot(eta_range, vali_loss)
    plt.title('GD\nbatch=%d, lambda =%.2e' % (batch_size, lamda))
    plt.xlabel('learning_rate')
    plt.ylabel('Validation_Loss')
    plt.show()
    plt.plot(eta_range, vali_accuracy)
    plt.title('GD\nbatch=%d, lambda =%.2e' % (batch_size, lamda))
    plt.xlabel('learning_rate')
    plt.ylabel('Validation_Accuracy')
    plt.show()


def train_and_test(train_X, train_y, test_X, test_y, lamda, batch_size, eta, n_iter, var_size=20, stable_var=1e-6):
    ppn = SoftmaxPpn(eta, lamda=lamda, batch=batch_size, var_size=var_size, stable_var=stable_var)
    ppn.fit(train_X, train_y, n_iter)
    print('learning_rate=%.3f, batch=%d, lambda=%.2e, epochs=%d, test_accuracy=%.3f%%' %
          (ppn.eta, ppn.batch, ppn.lamda, len(ppn.loss_), 100 * ppn.accuracy(test_X, test_y)))
    plt.plot(range(1, len(ppn.loss_) + 1), ppn.loss_, label='train_set')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.title('learning_rate=%.3f, batch=%d, lambda=%.2e' % (ppn.eta, ppn.batch, ppn.lamda))
    plt.legend()
    plt.show()
    plt.plot(range(1, len(ppn.accuracy_) + 1), ppn.accuracy_, label='train_set')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.title('learning_rate=%.3f, batch=%d, lambda=%.2e' % (ppn.eta, ppn.batch, ppn.lamda))
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default='mb', help="train model and test it",
                        choices=['part1', 'grad_check', 'fb', 'sb', 'mb'])
    parser.add_argument("--kcv", help="using k-cross-validation, k=5", action="store_true")
    parser.add_argument("--iter", "-n", help="max epochs of training", default=3000, type=int)
    parser.add_argument("--learning_rate", "-l", default=0.15, type=float)
    parser.add_argument("--lamda", default=1e-3, type=float)
    parser.add_argument("--batch", "-b", help="batch size, valid only when method=mb", default=64, type=int)
    parser.add_argument("--lmin", help="parameter of cross validaiton: min learning_rate", default=0.05, type=float)
    parser.add_argument("--lmax", help="parameter of cross validaiton: max learning_rate", default=0.50, type=float)
    parser.add_argument("--lstep", help="parameter of cross validaiton: step_length of learning_rate test",
                        default=0.05, type=float)
    args = parser.parse_args()

    if(args.method == 'part1'):
        assignment1(dataset_a1)
    elif(args.method == 'fb'):
        if(args.kcv):
            k_cross_validation(5, train_multi_hot, train_one_hot, batch_size=train_one_hot.shape[0],
                               lamda=args.lamda, eta_min=args.lmin, eta_max=args.lmax + args.lstep, eta_step=args.lstep,
                               n_iter=args.iter, var_size=10)
        else:
            train_and_test(train_multi_hot, train_one_hot, test_multi_hot, test_one_hot,
                           lamda=args.lamda, batch_size=train_one_hot.shape[0], eta=args.learning_rate,
                           n_iter=args.iter, var_size=10)
    elif(args.method == 'sb'):
        if (args.kcv):
            k_cross_validation(5, train_multi_hot, train_one_hot, batch_size=1,
                               lamda=args.lamda, eta_min=args.lmin, eta_max=args.lmax + args.lstep, eta_step=args.lstep,
                               n_iter=int(args.iter / train_one_hot.shape[0]) + 1, var_size=100)
        else:
            train_and_test(train_multi_hot, train_one_hot, test_multi_hot, test_one_hot,
                           lamda=args.lamda, batch_size=1, eta=args.learning_rate,
                           n_iter=int(args.iter / train_one_hot.shape[0]) + 1, var_size=100)
    elif(args.method == 'mb'):
        if (args.kcv):
            k_cross_validation(5, train_multi_hot, train_one_hot, batch_size=args.batch,
                               lamda=args.lamda, eta_min=args.lmin, eta_max=args.lmax + args.lstep, eta_step=args.lstep,
                               n_iter=int(args.iter * args.batch / train_one_hot.shape[0]) + 1, var_size=20)
        else:
            train_and_test(train_multi_hot, train_one_hot, test_multi_hot, test_one_hot,
                           lamda=args.lamda, batch_size=args.batch, eta=args.learning_rate,
                           n_iter=int(args.iter * args.batch / train_one_hot.shape[0]) + 1, var_size=20)
    elif(args.method == 'grad_check'):
        grad_check(train_multi_hot, train_one_hot, 100)


if(__name__ == "__main__"):
    main()


# train_and_test(train_multi_hot, train_one_hot, test_multi_hot, test_one_hot,
#                lamda=1e-3, batch_size=64, eta=0.4, n_iter=500, var_size=20)
# k_cross_validation_mb(5, train_multi_hot, train_one_hot, lamda=1e-3, batch_size=256,
#                       eta_min=0.50, eta_max=0.55, eta_step=0.05)
# cross_validation_mb(train_multi_hot, train_one_hot, lamda=1e-3, batch_size=64,
#                     eta_min=0.35, eta_max=0.40, eta_step=0.05)
# cross_validation_sb(train_multi_hot, train_one_hot, lamda=1e-3, eta_min=0.15, eta_max=0.20, eta_step=0.05)
# cross_validation_fb(train_multi_hot, train_one_hot, lamda=1e-3, eta_min=0.15, eta_max=0.20, eta_step=0.05)
# ppn_a2_full = SoftmaxPpn(eta=0.25, batch=train_multi_hot.shape[0], var_size=20)
# ppn_a2_full.fit(train_multi_hot, train_one_hot, 500)
# plt.plot(range(1, len(ppn_a2_full.loss_)+1), ppn_a2_full.loss_)
# plt.xlabel('iterations')
# plt.ylabel('Loss')
# plt.title('Full Batch GD\nlearning_rate =%.3f' % ppn_a2_full.eta)
# plt.show()

# ppn_a2_mb = SoftmaxPpn(eta=0.25, batch=25, var_size=10)
# ppn_a2_mb.fit(train_multi_hot, train_one_hot, 100)
# plt.plot(range(1, len(ppn_a2_mb.loss_)+1), ppn_a2_mb.loss_)
# plt.xlabel('iterations')
# plt.ylabel('Loss')
# plt.title('Mini-Batch GD\nlearning_rate =%.3f batch_size=%d' % (ppn_a2_mb.eta, ppn_a2_mb.batch))
# plt.show()

# ppn_a2_mb = SoftmaxPpn(eta=0.005, batch=1, var_size=5)
# ppn_a2_mb.fit(train_multi_hot, train_one_hot, 50)
# plt.plot(range(1, len(ppn_a2_mb.loss_)+1), ppn_a2_mb.loss_)
# plt.xlabel('iterations')
# plt.ylabel('Loss')
# plt.title('Stochastic GD\nlearning_rate =%.3f' % ppn_a2_mb.eta)
# plt.show()
