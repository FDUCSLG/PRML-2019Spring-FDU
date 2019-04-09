import os
os.sys.path.append('..')

from matplotlib import pyplot as plt
from handout import get_text_classification_datasets
import numpy as np
import string
import re


class Logistic:
    alpha = 0.1
    lamb = 1e-4
    train_validate_ratio = 4 / 5
    epoch = 1000
    min_count = 10

    def __init__(self, dataset_train, dataset_test):
        self.categories = len(dataset_train.target_names)
        self.vocabulary = self.get_vocabulary(dataset_train.data)

        X_train, y_train = self.shuffle_dataset(self.preprocess_X(dataset_train.data), self.preprocess_y(dataset_train.target))
        divide = int(len(X_train)*self.train_validate_ratio)
        self.X_train, self.X_validate = X_train[0:divide], X_train[divide:]
        self.y_train, self.y_validate = y_train[0:divide], y_train[divide:]

        self.X = self.preprocess_X(dataset_test.data)
        self.y = self.preprocess_y(dataset_test.target)

        self.batch = len(self.X_train)
        # self.batch = 1
        # self.batch = 200

    def get_vocabulary(self, X_data):
        vocabulary = {}

        for text in X_data:
            text = text.translate(str.maketrans("", "", string.punctuation))
            text = re.sub('[' + string.whitespace + ']', ' ', text)
            words = text.lower().split(' ')
            for word in words:
                if word in vocabulary.keys():
                    vocabulary[word] = vocabulary[word] + 1
                else:
                    vocabulary[word] = 1

        index = 0
        new_vocabulary = {}
        for key in vocabulary:
            if vocabulary[key] >= self.min_count:
                new_vocabulary[key] = index
                index = index + 1

        return new_vocabulary

    def preprocess_X(self, X_data):
        documents = []
        for text in X_data:
            text = text.translate(str.maketrans("", "", string.punctuation))
            text = re.sub('[' + string.whitespace + ']', ' ', text)
            words = text.lower().split(' ')
            documents.append(words)

        dataset = np.zeros([len(X_data), len(self.vocabulary)])
        vocabulary_list = self.vocabulary.keys()
        for i, words in enumerate(documents):
            for word in words:
                if word in vocabulary_list:
                    dataset[i, self.vocabulary[word]] = 1

        return dataset

    def preprocess_y(self, y_data):
        dataset = np.zeros([len(y_data), self.categories])
        dataset[np.arange(len(y_data)), y_data] = 1
        return dataset

    def softmax(self, z):
        z = np.exp(z - np.max(z, axis=1).reshape(z.shape[0], 1))
        z = z / np.sum(z, axis=1).reshape(z.shape[0], 1)
        return z

    def loss(self, X_train, y_train, w, b):
        return - np.sum(y_train * np.log(self.softmax(X_train @ w + b))) / len(y_train) + self.lamb * np.sum(w * w)

    def gradient(self, X_train, y_train, w, b):
        y_pred = self.softmax(X_train @ w + b)
        w_gradient = 2 * self.lamb * w - X_train.T @ (y_train - y_pred) / X_train.shape[0]
        b_gradient = - np.sum(y_train - y_pred, axis=0) / X_train.shape[0]
        return w_gradient, b_gradient

    def check_gradient(self, epoch=7, amount=8, delta=1e-6, epsilon=1e-7):
        print("Checking gradient...")

        X_train, y_train = self.shuffle_dataset(self.X_train, self.y_train)
        X_train, y_train = X_train[0:amount], y_train[0:amount]
        w, b = np.zeros([X_train.shape[1], self.categories]), np.zeros(self.categories)

        error = []
        for k in range(epoch):
            w_gradient, b_gradient = self.gradient(X_train, y_train, w, b)
            # w gradient error
            for i in range(w_gradient.shape[0]):
                for j in range(w_gradient.shape[1]):
                    w_delta = w.copy()
                    w_delta[i, j] = w_delta[i, j] + delta
                    error.append(np.abs(w_gradient[i, j] - (self.loss(X_train, y_train, w_delta, b) - self.loss(X_train, y_train, w, b)) / delta))
            # b gradient error
            for i in range(len(b_gradient)):
                b_delta = b.copy()
                b_delta[i] = b_delta[i] + delta
                error.append(np.abs(b_gradient[i] - (self.loss(X_train, y_train, w, b_delta) - self.loss(X_train, y_train, w, b)) / delta))
            # update w & b
            w = w - self.alpha * w_gradient
            b = b - self.alpha * b_gradient

        print("Max error %f" % np.max(error))
        if np.max(error) < epsilon:
            print("Right gradient!")
        else:
            print("Wrong gradient!")

    def shuffle_dataset(self, X, y):
        dataset = np.hstack((y, X))
        np.random.shuffle(dataset)
        return dataset[:, self.categories:], dataset[:, 0:self.categories]

    def train(self, lamb, alpha, batch):
        self.lamb, self.alpha, self.batch = lamb, alpha, batch
        N, M, K = self.X_train.shape[0], self.X_train.shape[1], self.categories

        print("Start training...")
        w, b = np.zeros([M, K]), np.zeros(K)
        w_best, b_best = w.copy(), b.copy()
        loss, train_accuracy, validate_accuracy = [], [], []

        for epoch in range(self.epoch):
            print("Epoch %d" % epoch, end=": ")
            X_train, y_train = self.shuffle_dataset(self.X_train, self.y_train)
            for i in range(0, N, self.batch):
                end = i + self.batch
                if end > N:
                    end = N
                w_gradient, b_gradient = self.gradient(X_train[i:end, :], y_train[i:end, :], w, b)
                w = w - self.alpha * w_gradient
                b = b - self.alpha * b_gradient
                loss.append(self.loss(X_train[i:end, :], y_train[i:end, :], w, b))

            # accuracy
            accuracy = self.accuracy(self.X_train, self.y_train, w, b)
            print("train_acc %f" % accuracy, end=" ")
            train_accuracy.append(accuracy)
            accuracy = self.accuracy(self.X_validate, self.y_validate, w, b)
            print("validate_acc %f" % accuracy)
            if epoch and accuracy > max(validate_accuracy):
                w_best, b_best = w.copy(), b.copy()
            validate_accuracy.append(accuracy)

        print("Test accuracy %f" % self.accuracy(self.X, self.y, w_best, b_best))

        return loss, train_accuracy, validate_accuracy

    def show(self):
        loss, train_accuracy, validate_accuracy = self.train(self.lamb, self.alpha, self.batch)

        plt.subplot(121)
        plt.plot(np.arange(len(loss)), loss)
        plt.xlabel("batch")
        plt.title("loss")

        plt.subplot(122)
        plt.plot(np.arange(len(train_accuracy)), train_accuracy, label="train accuracy")
        plt.plot(validate_accuracy, label="validate accuracy")
        max_index = np.argmax(validate_accuracy)
        plt.plot(max_index, validate_accuracy[max_index], '^', label="max validate_acc %f" % validate_accuracy[max_index])
        plt.xlabel("epoch")
        plt.title("accuracy")
        plt.legend()
        plt.show()

    def show_batch_diff(self):
        loss0, train_accuracy0, validate_accuracy0 = self.train(self.lamb, self.alpha, 1)
        loss1, train_accuracy1, validate_accuracy1 = self.train(self.lamb, self.alpha, 20)
        loss2, train_accuracy2, validate_accuracy2 = self.train(self.lamb, self.alpha, 200)
        loss3, train_accuracy3, validate_accuracy3 = self.train(self.lamb, self.alpha, len(self.X_train))

        loss_0 = loss0[::int(len(self.X_train) / 1)]
        loss_1 = loss1[::int(len(self.X_train) / 20)]
        loss_2 = loss2[::int(len(self.X_train) / 200)]
        loss_3 = loss3[::int(len(self.X_train) / len(self.X_train))]

        plt.subplot(131)
        plt.plot(np.arange(len(loss0)), loss0, label="batch size 1")
        plt.plot(np.arange(len(loss1)), loss1, label="batch size 20")
        plt.plot(np.arange(len(loss2)), loss2, label="batch size 200")
        plt.plot(np.arange(len(loss3)), loss3, label="batch size full")
        plt.xlabel("batch")
        plt.title("loss")
        plt.legend()

        plt.subplot(132)
        plt.plot(np.arange(len(loss_0)), loss_0, label="batch size 1")
        plt.plot(np.arange(len(loss_1)), loss_1, label="batch size 20")
        plt.plot(np.arange(len(loss_2)), loss_2, label="batch size 200")
        plt.plot(np.arange(len(loss_3)), loss_3, label="batch size full")
        plt.xlabel("epoch")
        plt.title("loss")
        plt.legend()

        plt.subplot(133)

        plt.plot(np.arange(len(train_accuracy0)), train_accuracy0, label="train accuracy 1")
        plt.plot(validate_accuracy0, label="validate accuracy 1")
        max_index = np.argmax(validate_accuracy0)
        plt.plot(max_index, validate_accuracy0[max_index], '^', label="max validate_acc 1 %f" % validate_accuracy0[max_index])

        plt.plot(np.arange(len(train_accuracy2)), train_accuracy2, label="train accuracy 200")
        plt.plot(validate_accuracy2, label="validate accuracy 200")
        max_index = np.argmax(validate_accuracy2)
        plt.plot(max_index, validate_accuracy2[max_index], '^',
                 label="max validate_acc 200 %f" % validate_accuracy2[max_index])

        plt.plot(np.arange(len(train_accuracy3)), train_accuracy3, label="train accuracy full")
        plt.plot(validate_accuracy3, label="validate accuracy full")
        max_index = np.argmax(validate_accuracy3)
        plt.plot(max_index, validate_accuracy3[max_index], '^',
                 label="max validate_acc full %f" % validate_accuracy3[max_index])

        plt.xlabel("epoch")
        plt.title("accuracy")
        plt.legend()
        plt.show()

    def show_lamb_diff(self):
        loss0, train_accuracy0, validate_accuracy0 = self.train(1e-6, self.alpha, len(self.X_train))
        loss1, train_accuracy1, validate_accuracy1 = self.train(1e-4, self.alpha, len(self.X_train))
        loss2, train_accuracy2, validate_accuracy2 = self.train(1e-2, self.alpha, len(self.X_train))
        loss3, train_accuracy3, validate_accuracy3 = self.train(1, self.alpha, len(self.X_train))

        plt.subplot(121)
        plt.plot(np.arange(len(loss0)), loss0, label="lambda 1e-6")
        plt.plot(np.arange(len(loss1)), loss1, label="lambda 1e-4")
        plt.plot(np.arange(len(loss2)), loss2, label="lambda 1e-2")
        plt.plot(np.arange(len(loss3)), loss3, label="lambda 1")
        plt.xlabel("epoch")
        plt.title("loss")
        plt.legend()

        plt.subplot(122)

        plt.plot(np.arange(len(train_accuracy0)), train_accuracy0, label="train accuracy 1e-6")
        plt.plot(validate_accuracy0, label="validate accuracy 1e-6")
        max_index = np.argmax(validate_accuracy0)
        plt.plot(max_index, validate_accuracy0[max_index], '^', label="max validate_acc 1e-6 %f" % validate_accuracy0[max_index])

        plt.plot(np.arange(len(train_accuracy2)), train_accuracy2, label="train accuracy 1e-2")
        plt.plot(validate_accuracy2, label="validate accuracy 1e-2")
        max_index = np.argmax(validate_accuracy2)
        plt.plot(max_index, validate_accuracy2[max_index], '^',
                 label="max validate_acc 1e-2 %f" % validate_accuracy2[max_index])

        plt.plot(np.arange(len(train_accuracy3)), train_accuracy3, label="train accuracy 1")
        plt.plot(validate_accuracy3, label="validate accuracy 1")
        max_index = np.argmax(validate_accuracy3)
        plt.plot(max_index, validate_accuracy3[max_index], '^',
                 label="max validate_acc 1 %f" % validate_accuracy3[max_index])

        plt.xlabel("epoch")
        plt.title("accuracy")
        plt.legend()
        plt.show()

    def show_alpha_diff(self):
        loss0, train_accuracy0, validate_accuracy0 = self.train(self.lamb, 0.001, len(self.X_train))
        loss1, train_accuracy1, validate_accuracy1 = self.train(self.lamb, 0.01, len(self.X_train))
        loss2, train_accuracy2, validate_accuracy2 = self.train(self.lamb, 0.1, len(self.X_train))
        loss3, train_accuracy3, validate_accuracy3 = self.train(self.lamb, 1, len(self.X_train))

        plt.subplot(121)
        plt.plot(np.arange(len(loss0)), loss0, label="alpha 0.001")
        plt.plot(np.arange(len(loss1)), loss1, label="alpha 0.01")
        plt.plot(np.arange(len(loss2)), loss2, label="alpha 0.1")
        plt.plot(np.arange(len(loss3)), loss3, label="alpha 1")
        plt.xlabel("epoch")
        plt.title("loss")
        plt.legend()

        plt.subplot(122)

        plt.plot(np.arange(len(train_accuracy0)), train_accuracy0, label="train accuracy 0.001")
        plt.plot(validate_accuracy0, label="validate accuracy 0.001")
        max_index = np.argmax(validate_accuracy0)
        plt.plot(max_index, validate_accuracy0[max_index], '^', label="max validate_acc 0.001 %f" % validate_accuracy0[max_index])

        plt.plot(np.arange(len(train_accuracy2)), train_accuracy2, label="train accuracy 0.1")
        plt.plot(validate_accuracy2, label="validate accuracy 0.1")
        max_index = np.argmax(validate_accuracy2)
        plt.plot(max_index, validate_accuracy2[max_index], '^',
                 label="max validate_acc 0.1 %f" % validate_accuracy2[max_index])

        plt.plot(np.arange(len(train_accuracy3)), train_accuracy3, label="train accuracy 1")
        plt.plot(validate_accuracy3, label="validate accuracy 1")
        max_index = np.argmax(validate_accuracy3)
        plt.plot(max_index, validate_accuracy3[max_index], '^',
                 label="max validate_acc 1 %f" % validate_accuracy3[max_index])

        plt.xlabel("epoch")
        plt.title("accuracy")
        plt.legend()
        plt.show()

    def accuracy(self, X, y, w, b):
        y_pred_index = np.argmax(X @ w + b, axis=1)
        y_index = np.argmax(y, axis=1)
        return sum(y_pred_index == y_index) / len(y_pred_index)

    def run(self):
        print("Logistic regression:")
        # print(self.X_train.shape, self.y_train.shape)
        # print(self.X_validate.shape, self.y_validate.shape)
        # print(self.X.shape, self.y.shape)

        # self.check_gradient()
        # self.show()
        # self.show_batch_diff()
        # self.show_lamb_diff()
        self.show_alpha_diff()


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

np.random.seed(2333)
dataset_train, dataset_test = get_text_classification_datasets()
logistic = Logistic(dataset_train, dataset_test)
logistic.run()
