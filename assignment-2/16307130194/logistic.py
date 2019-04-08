import os
os.sys.path.append('..')

from matplotlib import pyplot as plt
from handout import get_text_classification_datasets
import numpy as np
import string
import re


class Logistic:
    alpha = 0.25
    lamb = 0.5
    batch = 1
    epoch = 100
    min_count = 10

    def __init__(self, dataset_train, dataset_test):
        self.categories = len(dataset_train.target_names)

        self.vocabulary = self.get_vocabulary(dataset_train.data)
        self.X_train = self.preprocess_X(dataset_train.data)
        self.y_train = self.preprocess_y(dataset_train.target)
        self.X = self.preprocess_X(dataset_test.data)
        self.y = self.preprocess_y(dataset_test.target)

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

    def loss(self, w, b):
        return - np.sum(self.y_train * np.log(self.softmax(self.X_train @ w + b))) / len(self.y_train) + self.lamb * np.sum(w ** 2)

    def gradient(self, X_train, y_train, w, b):
        y_pred = self.softmax(X_train @ w + b)
        w_gradient = 2 * self.lamb * w - X_train.T @ (y_train - y_pred) / X_train.shape[0]
        b_gradient = - np.sum(y_train - y_pred, axis=0) / X_train.shape[0]
        return w_gradient, b_gradient

    def train(self):
        N, M, K = self.X_train.shape[0], self.X_train.shape[1], self.categories
        w = np.zeros([M, K])
        b = np.zeros(K)

        loss_list = [self.loss(w, b)]

        for epoch in range(self.epoch):
            dataset = np.hstack((self.y_train, self.X_train))
            np.random.shuffle(dataset)
            y_train, X_train = dataset[:, 0:self.categories], dataset[:, self.categories:]
            for i in range(0, N, self.batch):
                end = i + self.batch
                if end > N:
                    end = N
                w_gradient, b_gradient = self.gradient(X_train[i:end, :], self.y_train[i:end, :], w, b)
                w = w - self.alpha * w_gradient
                b = b - self.alpha * b_gradient
                loss = self.loss(w, b)
                print(loss)
                loss_list.append(loss)

            # accuracy?

        # print(loss_list)

    def accuracy(self):
        pass

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
