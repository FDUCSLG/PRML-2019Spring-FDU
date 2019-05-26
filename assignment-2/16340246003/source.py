import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_linear_seperatable_2d_2c_dataset
from handout import get_text_classification_datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import string
from collections import Counter

data = get_linear_seperatable_2d_2c_dataset()

def least_square_model():
    cls1x = []
    cls1y = []
    cls2x = []
    cls2y = []
    for i in range(len(data.X)):
        if(data.y[i]):
            cls1x.append(data.X[i,0])
            cls1y.append(data.X[i,1])
        else:
            cls2x.append(data.X[i,0])
            cls2y.append(data.X[i,1])
    cls1x_mean = np.mean(cls1x)
    cls1y_mean = np.mean(cls1y)
    cls2x_mean = np.mean(cls2x)
    cls2y_mean = np.mean(cls2y)
    num1 = 0
    den1 = 0
    num2 = 0
    den2 = 0
    for i in range(len(cls1x)):
        num1 += (cls1x[i] - cls1x_mean)*(cls1y[i] - cls1y_mean)
        den1 += (cls1x[i] - cls1x_mean)**2
    for i in range(len(cls2x)):
        num2 += (cls2x[i] - cls2x_mean)*(cls2y[i] - cls2y_mean)
        den2 += (cls2x[i] - cls2x_mean)**2
    m = ((num1 / den1)+(num2 / den2))/2
    c = ((cls1y_mean - m*cls1x_mean)+(cls2y_mean - m*cls2x_mean))/2
    x = np.linspace(-1.5,1.5)
    y = m*x+c
    print(m)
    print(c)
    plt.plot(x, y, '-r')
    data.plot(plt).show()



class Perceptron(object):
    
    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.001):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
    
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
def perceptron():
    X = data.X
    y = np.zeros(len(data.y))
    for i in range(len(data.y)):
        if data.y[i]:
            y[i]=1
        else:
            y[i]=0
    p = Perceptron(2)
    p.train(X, y)
    m = -(p.weights[1]/p.weights[2])
    c = -(p.weights[0]/p.weights[2])
    x = np.linspace(-1.5,1.5)
    y = m*x+c
    print(m)
    print(c)
    plt.plot(x, y, '-r')
    data.plot(plt).show()

def logistic_regression():
    text_train, text_test = get_text_classification_datasets()
    labels_train = creat_labels(text_train.target)
    my_dict = creat_dict(text_train.data)
    vectors_train = get_vectors(text_train.data, my_dict)
    w = np.zeros([vectors_train.shape[1],4])
    lam = 1
    iterations = 1000
    learningRate = 0.005
    losses = []
    for i in range(0,iterations):
        loss,grad = getLoss(w,vectors_train,labels_train,lam)
        losses.append(loss)
        w = w - (learningRate * grad)
    plt.plot(losses)
    plt.show()

def creat_labels(targets):
    one_hot_labels = []
    for t in targets:
        one_hot_label = np.zeros(4)
        one_hot_label[t] = 1
        one_hot_labels.append(one_hot_label)
    return np.array(one_hot_labels)

def get_vectors(data,dict):
    muiti_hots = []
    for subtext in data:
        subwords = text_to_words(subtext)
        muiti_hot = np.zeros(len(dict))
        for word in subwords:
            if(word in dict):
                muiti_hot[dict[word]] = 1
        muiti_hots.append(muiti_hot)
    return np.array(muiti_hots)


def creat_dict(data):
    words_train = text_to_words(data)
    my_dict = {}
    for i in range(len(words_train)):
        my_dict[words_train[i]] = i
    return my_dict



def text_to_words(text):
    table = str.maketrans('', '', string.punctuation)
    text = str(text).translate(table).lower()
    #print(text)
    words = text.split()
    words = Counter(words)
    words10 = []
    for word in words:
        if words[word] >= 10:
            words10.append(word)
    words = sorted(words10)
    return words

def getLoss(w, x, y, lam):
    m = x.shape[0]
    scores = np.dot(x,w)
    prob = softmax(scores) #Next we perform a softmax on these scores to get their probabilities
    loss = (-1 / m) * np.sum(y * np.log(prob)) + (lam/2)*np.sum(w*w) #We then find the loss of the probabilities
    grad = (-1 / m) * np.dot(x.T,(y - prob)) + lam*w #And compute the gradient for that loss
    return loss,grad

def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

#least_square_model()
#perceptron()
logistic_regression()
