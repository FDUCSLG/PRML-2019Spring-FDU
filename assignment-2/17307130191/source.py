import os

os.sys.path.append('..')
from handout import get_linear_seperatable_2d_2c_dataset
from handout import get_text_classification_datasets
from matplotlib import pyplot as plt
import numpy as np
import re
import string
import math
import copy

sampled_data = get_linear_seperatable_2d_2c_dataset()
sampled_data.plot(plt)
X = np.column_stack((np.ones(200), np.array(sampled_data.X)))
alpha = 0.02
beta = 10
gama = 0.1
penal = 0.01
map = dict()
acc = 0.0001

def Lq():
    T = np.zeros((200, 2))
    for i in range(len(sampled_data.y)):
         if(sampled_data.y[i] == True):
             T[i][0] = 1
         else: T[i][1] = 1

    W = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X), T))
    Y = np.zeros((200, 2))
    Y = np.dot(X, W)
    Yn = np.zeros(200)
    for i in range(len(sampled_data.y)):
        if(Y[i][0] > Y[i][1]):
            Yn[i] = True
        else: Yn[i] = False
    sum = 0
    for i in range(len(sampled_data.y)):
        if(Yn[i] == sampled_data.y[i]):
            sum += 1
    print ('accuracy: ', '%.3f' %(sum / 200))
    xmin = min(sampled_data.X[:,0])
    xmax = max(sampled_data.X[:,0])
    x = np.linspace(xmin, xmax, 200)
    y = np.zeros(200)
    z = np.zeros(200)
    for i in range(len(x)):
        y[i] = (0.5 -(W[0][0] + W[1][0] * x[i]) )/ W[2][0]
        z[i] = (0.5 -(W[0][1] + W[1][1] * x[i]) )/ W[2][1]
    plt.plot(x, y, color = 'yellow')
    plt.plot(x, z, color = 'blue')

def Perceptron():
    print('hello')
    w = np.array([[0.1],[1],[1]])
    loss2 = loss1 = 0
    Tn = np.zeros(200)
    for i in range(len(sampled_data.y)):
        if(sampled_data.y[i] == 1):
            Tn[i] = 1
        else: Tn[i] = -1

    selected = []
    for i in range(len(X)):
        print(i, X[i], Tn[i], np.dot(X[i], w) * Tn[i])
        if(np.dot(X[i], w) * Tn[i] < 0):
            loss1 += -np.dot(X[i], w) * Tn[i]
            selected.append(i)

    for i in selected:
        for j in range(len(w)):
            w[j][0] += alpha * X[i][j] * Tn[i]


    for i in range(len(X)):
        if(np.dot(X[i], w) * Tn[i] < 0):
            loss2 += -np.dot(X[i], w) * Tn[i]

    print('loss1: ', loss1, 'loss2: ', loss2)

    while loss2 < loss1:
        print('loss1: ', loss1, ' loss2: ', loss2, ' w: ', w)
        selected = [i for i in range(len(X)) if np.dot(X[i], w) * Tn[i]  < 0]
        for i in selected:
            for j in range(len(w)):
                w[j][0] += alpha * X[i][j] * Tn[i]

        loss1 = loss2
        loss2 = 0
        for i in range(len(X)):
            if(np.dot(X[i], w) * Tn[i] < 0):
                loss2 += -np.dot(X[i], w) * Tn[i]

    xmin = min(sampled_data.X[:,0])
    xmax = max(sampled_data.X[:,0])
    x = np.linspace(xmin, xmax, 200)
    y = np.zeros(200)
    for i in range(len(x)):
        y[i] = -(w[0] + w[1] * x[i]) / w[2]
    plt.plot(x, y, color = 'blue')

    sum = 200
    for i in range(len(X)):
        if(np.dot(X[i], w) * Tn[i] < 0):
            sum -= 1

    print('accuracy: ', sum / 200)

def preprocess(train):
    train_set = []
    for i in range(len(train.data)): #remove #/?...
        train_set.append(re.sub(r'[^a-zA-Z0-9\s]','',train.data[i]))

    for i in range(len(train.data)): #replace whitespace
        train_set[i] = re.sub(r'['+string.whitespace+']+',' ',train_set[i])

    newone = []
    for i in range(len(train_set)): #union
        newone.append(re.split(r' +', train_set[i].strip().lower()))
    #print(newone)
    return newone

def softmax(z):
    sum = 0
    zmax = max(z)
    for i in range(len(z)):
        sum += math.exp(z[i] - zmax)

    soft = np.zeros(len(z))
    for i in range(len(z)):
        soft[i] = math.exp(z[i] - zmax) / sum
    return soft

def diction(newone1):
    myTrain = []
    for i in range(len(newone1)):
        myTrain += newone1[i]
    train_dic = {}
    for i in myTrain:
        if i not in train_dic: #dic
            train_dic[i] = 1
        else: train_dic[i] += 1

    for key in list(train_dic.keys()): #del < 10
        if(train_dic[key] < beta):
            del train_dic[key]

    num = 0
    for key in train_dic:
        map[key] = num
        num += 1
    return num

def Logistic():
    train, test = get_text_classification_datasets()
    newone1 = preprocess(train)
    newone2 = preprocess(test)

    num = diction(newone1)

    Train = []

    for i in range(len(newone1)):
        tmp = np.zeros(num)
        for j in range(len(newone1[i])):
            if(newone1[i][j] in map):
                tmp[map[newone1[i][j]]] = 1
        Train.append(tmp)

    Test = []

    for i in range(len(newone2)):
        tmp = np.zeros(num)
        for j in range(len(newone2[i])):
            if(newone2[i][j] in map):
                tmp[map[newone2[i][j]]] = 1
        Test.append(tmp)

    w = np.ones((4, num))
    b = np.ones(4)
    #print(Train, len(Train[0]))

    pred = np.zeros((len(Train), 4))
    z = np.zeros(4)

    Loss2 = Loss1 = 0
    for i in range(len(Train)):
        z = np.dot(w, Train[i]) + b
        pred[i] = softmax(z)

    #print(pred)
    for i in range(len(Train)):
        Loss1 -= np.log(pred[i][train.target[i]])
    Loss1 = Loss1 / len(Train) + 0.5 * penal * np.sum(w * w)

    cycle = 0
    while abs(Loss1 - Loss2) > acc:
        cycle += 1
        #print(Loss1, Loss2)
        for i in range(len(Train)):
            z = np.dot(w, Train[i]) + b
            #print(z)
            pred[i] = softmax(z)
            soft = copy.deepcopy(pred[i])
            soft[train.target[i]] -= 1
            w -= gama * (penal * w + np.dot(np.mat(soft).T, np.mat(Train[i])))
            b -= gama * soft

        Loss2 = Loss1
        Loss1 = 0
        for i in range(len(Train)):
            Loss1 -= np.log(pred[i][train.target[i]])
        Loss1 = Loss1 / len(Train) + 0.5 * penal * np.sum(w * w)

    print(cycle)
    sum = 0
    pred_t = np.zeros(4)
    for i in range(len(Train)):
        z = np.dot(w, Train[i]) + b
        pred_t = softmax(z)
        #print(pred_t, train.target[i])
        if(pred_t.tolist().index(max(pred_t)) == train.target[i]):
            sum += 1
            #print(i)
    print(sum, len(train.data), sum / len(train.data))

    sum = 0
    for i in range(len(Test)):
        z = np.dot(w, Test[i]) + b
        pred_t = softmax(z)
        #print(pred_t, test.target[i])
        if(pred_t.tolist().index(max(pred_t)) == test.target[i]):
            sum += 1
            #print(i)
    print(sum, len(test.data), sum / len(test.data))
Lq()
#Perceptron()
#Logistic()

plt.show()