import os
os.sys.path.append('..')

from matplotlib import pyplot as plt
from handout import get_text_classification_datasets
import numpy as np
import string
import re


class Logistic:
    def __init__(self, dataset_train, dataset_test):
        self.vocabulary = {}
        self.X_train = self.preprocess_X(dataset_train.data)
        self.y_train = self.preprocess_y(dataset_train.target)
        self.X = self.preprocess_X(dataset_test.data, save_vocabulary=False)
        self.y = self.preprocess_y(dataset_test.target, categories=4)

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

    def preprocess_y(self, y_data, categories=4):
        dataset = np.zeros([len(y_data), categories])
        dataset[np.arange(len(y_data)), y_data] = 1
        return dataset


dataset_train, dataset_test = get_text_classification_datasets()
logistic = Logistic(dataset_train, dataset_test)

print(len(logistic.y_train))
