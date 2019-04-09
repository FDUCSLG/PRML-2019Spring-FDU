import os
os.sys.path.append('..')

from matplotlib import pyplot as plt
from handout import get_text_classification_datasets
import numpy as np
import string
import re


class Logistic:
    alpha = 0.25
    lamb = 1e-3
    train_validate_ratio = 4 / 5
    epoch = 200
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

    def check_gradient(self, w, b, w_gradient, b_gradient, delta=1e-6, epsilon=1e-6):
        print("Checking gradient...")

        error = []
        for i in range(w_gradient.shape[0]):
            for j in range(w_gradient.shape[1]):
                w_delta = w.copy()
                w_delta[i, j] = w_delta[i, j] + delta
                error.append(np.abs(w_gradient[i, j] - (self.loss(w_delta, b) - self.loss(w, b)) / delta))

        for i in range(len(b_gradient)):
            b_delta = b.copy()
            b_delta[i] = b_delta[i] + delta
            error.append(np.abs(b_gradient[i] - (self.loss(w, b_delta) - self.loss(w, b)) / delta))

        print("Max error %f" % np.max(error))
        if np.max(error) < epsilon:
            return True
        return False

    def shuffle_dataset(self, X, y):
        dataset = np.hstack((y, X))
        np.random.shuffle(dataset)
        return dataset[:, self.categories:], dataset[:, 0:self.categories]

    def train(self):
        N, M, K = self.X_train.shape[0], self.X_train.shape[1], self.categories
        w = np.zeros([M, K])
        b = np.zeros(K)

        # print("Start training...")
        # loss = self.loss(w, b)
        # loss_list = [loss]
        # print("First loss: %f" % loss, end=" ")
        # accuracy = self.accuracy(w, b)
        # accuracy_list = [accuracy]
        # print("First accuracy: %f" % accuracy)
        loss = []
        train_accuracy = []
        validate_accuracy = []

        for epoch in range(self.epoch):
            print("Epoch %d" % epoch, end=": ")
            X_train, y_train = self.shuffle_dataset(self.X_train, self.y_train)
            # X_train, y_train = self.X_train, self.y_train
            for i in range(0, N, self.batch):
                end = i + self.batch
                if end > N:
                    end = N
                w_gradient, b_gradient = self.gradient(X_train[i:end, :], y_train[i:end, :], w, b)
                # if i < 10 and not self.check_gradient(w, b, w_gradient, b_gradient):
                #     print("Wrong gradient!", end=" ")
                w = w - self.alpha * w_gradient
                b = b - self.alpha * b_gradient
                l = self.loss(X_train[i:end, :], y_train[i:end, :], w, b)
                loss.append(l)

            # accuracy?
            accuracy = self.accuracy(self.X_train, self.y_train, w, b)
            print("train_acc %f" % accuracy, end=" ")
            train_accuracy.append(accuracy)
            accuracy = self.accuracy(self.X_validate, self.y_validate, w, b)
            print("validate_acc %f" % accuracy)
            validate_accuracy.append(accuracy)

        plt.subplot(121)
        plt.title("loss")
        plt.plot(np.arange(len(loss)), loss)
        plt.xlabel("batch")
        plt.ylabel("loss")

        plt.subplot(122)
        plt.plot(np.arange(len(train_accuracy)), train_accuracy, label="train accuracy")
        plt.plot(validate_accuracy, label="validate accuracy")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        plt.show()

    def accuracy(self, X, y, w, b):
        y_pred_index = np.argmax(X @ w + b, axis=1)
        y_index = np.argmax(y, axis=1)
        return sum(y_pred_index == y_index) / len(y_pred_index)

    def run(self):
        print("Logistic regression:")
        # print(self.categories)
        # M, K = self.X_train.shape[1], self.categories
        # w = np.ones([M, K])
        # b = np.ones(K)
        # print(self.X_train.shape, w.shape, b.shape)
        # print(self.X.shape, self.y.shape)
        # w_gradient, b_gradient = self.gradient(self.X_train, self.y_train, w, b)
        # print(w_gradient.shape, b_gradient.shape)

        self.train()


import ssl
ssl._create_default_https_context = ssl._create_unverified_context


dataset_train, dataset_test = get_text_classification_datasets()
logistic = Logistic(dataset_train, dataset_test)
logistic.run()
# print(len(logistic.y_train))
# print(logistic.y_train[0])
# print(logistic.X_train.shape)
# for i in range(0, 10, 3):
#     print(i)
# print(i)
