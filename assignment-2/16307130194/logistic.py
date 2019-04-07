import os
os.sys.path.append('..')

from matplotlib import pyplot as plt
from handout import get_text_classification_datasets
import numpy as np
import string
import re


class Logistic:
    def __init__(self, dataset_train, dataset_test):
        self.categories = len(dataset_train.target_names)
        self.alpha = 0.25
        self.lamb = 0.5
        self.batch = 1

        self.vocabulary = {}
        self.X_train = self.preprocess_X(dataset_train.data)
        self.y_train = self.preprocess_y(dataset_train.target)
        self.X = self.preprocess_X(dataset_test.data, save_vocabulary=False)
        self.y = self.preprocess_y(dataset_test.target)

    def preprocess_X(self, X_data, save_vocabulary=True):
        min_count = 10
        vocabulary = {}
        documents = []

        for text in X_data:
            text = text.translate(str.maketrans("", "", string.punctuation))
            text = re.sub('[' + string.whitespace + ']', ' ', text)
            words = text.lower().split(' ')
            documents.append(words)
            for word in words:
                if word in vocabulary.keys():
                    vocabulary[word] = vocabulary[word] + 1
                else:
                    vocabulary[word] = 1

        count = 0
        new_vocabulary = {}
        for key in vocabulary:
            if vocabulary[key] >= min_count:
                new_vocabulary[key] = count
                count = count + 1

        if save_vocabulary:
            self.vocabulary = new_vocabulary

        vocabulary_list = new_vocabulary.keys()
        dataset = np.zeros([len(documents), len(vocabulary_list)])

        for i, words in enumerate(documents):
            for word in words:
                if word in vocabulary_list:
                    dataset[i, new_vocabulary[word]] = 1

        return dataset

    def preprocess_y(self, y_data):
        dataset = np.zeros([len(y_data), self.categories])
        dataset[np.arange(len(y_data)), y_data] = 1
        return dataset

    def softmax(self, z):
        z = np.exp(z - np.max(z, axis=1))
        z = z / np.sum(z, axis=1)
        return z

    def loss(self, w, b):
        return - np.sum(self.y_train * np.log(self.softmax(w @ self.X_train + b))) / len(self.y_train) + self.lamb * np.sum(w ** 2)

    def batched_dataset(self):
        pass

    def gradient(self, w, b):
        y_pred = self.softmax(w @ self.X_train + b)
        w_gradient = 2 * self.lamb * w - self.X_train * (self.y_train - y_pred) / len(self.X_train)
        b_gradient = - np.sum(self.y_train - y_pred, axis=0) / len(self.X_train)
        return w_gradient, b_gradient

    def train(self):
        N, M, K = self.X_train.shape[0], self.X_train.shape[1], self.categories
        w = np.ones([M, K])
        b = np.ones(K)

        loss_list = [self.loss(w, b)]

        while True:
            for i in range(0, N, self.batch):
                w_gradient, b_gradient = self.gradient(w, b)
                w -= self.alpha * w_gradient
                b -= self.alpha * b_gradient
            loss_list.append(self.loss(w, b))

    def run(self):
        print("Logistic regression:")
        print(self.categories)


dataset_train, dataset_test = get_text_classification_datasets()
logistic = Logistic(dataset_train, dataset_test)
logistic.run()
# print(len(logistic.y_train))
# print(logistic.y_train[0])
# print(logistic.X_train.shape)
