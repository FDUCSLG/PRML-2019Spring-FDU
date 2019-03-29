import os
os.sys.path.append('..')

from handout import Dataset, get_linear_seperatable_2d_2c_dataset, get_text_classification_datasets
from matplotlib import pyplot as plt
import numpy as np
import string
import re

#Part 1
def Display(d, w, str):
    plt.title(str)
    d.plot(plt)
    x = [-2.0, 2.0]
    y = [0, 0]
    y[0] = (- w[2] - w[0] * x[0]) / w[1]
    y[1] = (- w[2] - w[0] * x[1]) / w[1]
    plt.plot(x, y)
    plt.show()

def Accuracy(d, w, str):
    N = d.X.shape[0]
    cnt = 0
    for pointX, y in zip(d.X, d.y):
        if y == True and np.dot(np.append(pointX, 1), w) > 0: cnt += 1
        elif y == False and np.dot(np.append(pointX, 1), w) <= 0: cnt += 1
    print(str + ":", (cnt/N))

def least_square_model(data):
    N = data.X.shape[0]
    phi = np.mat(np.c_[data.X, np.ones(N)])
    w = np.resize(((phi.T * phi).I * phi.T * np.mat(data.y).T), 3)
    print(w)
    return w

def Perceptron(data, eta = 0.1):
    w = np.array([.0, .0, .0])
    for pointX, y in zip(data.X, data.y):
        if y == True:
            if np.dot(np.append(pointX, 1), w) <= 0:
                w += eta * np.append(pointX, 1)
        else:
            if np.dot(np.append(pointX, 1), w) > 0: w -= eta * np.append(pointX, 1)
    print(w)
    return w

def Preprocess(data):
    words = []
    for text in data:
        text = re.sub('[' + string.punctuation + ']', "", text)
        text = re.sub('[' + string.whitespace + '\u200b]+', ' ', text)
        text = text.strip().lower()
        list = text.split(' ')
        words.append(list)
    return words

'''
d = get_linear_seperatable_2d_2c_dataset()
w = least_square_model(d)
#print(w)
Display(d, w, 'Least Square Classification')
Accuracy(d, w, 'Accuracy of Least Square Classification')
w = Perceptron(d)
#print(w)
Display(d, w, 'Perceptron Algorithm')
Accuracy(d, w, 'Accuracy of Perceptron Algorithm')
'''
dataset_train, dataset_test = get_text_classification_datasets()
print(type(dataset_train.data))
Preprocess(dataset_train.data)
#print(dataset_train)