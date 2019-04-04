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
import random

sampled_data = get_linear_seperatable_2d_2c_dataset()
train, test = get_text_classification_datasets()
#sampled_data.plot(plt)
X = np.column_stack((np.ones(200), np.array(sampled_data.X)))
alpha = 0.02 #learning rate
beta = 10 # threshold
gama = 0.03 # learning rate
penal = 0.01 #lambda
map = dict()
acc = 1e-3 #accuracy |loss1 - loss2|
check_time = 20 #gradient checking time

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
    w = np.array([[0.1],[1],[1]])
    loss2 = loss1 = 0
    Tn = np.zeros(200)
    for i in range(len(sampled_data.y)):
        if(sampled_data.y[i] == 1):
            Tn[i] = 1
        else: Tn[i] = -1

    selected = []
    for i in range(len(X)):
        if(np.dot(X[i], w) * Tn[i] < 0):
            loss1 += -np.dot(X[i], w) * Tn[i]
            selected.append(i)

    for i in selected:
        for j in range(len(w)):
            w[j][0] += alpha * X[i][j] * Tn[i]


    for i in range(len(X)):
        if(np.dot(X[i], w) * Tn[i] < 0):
            loss2 += -np.dot(X[i], w) * Tn[i]


    while loss2 < loss1:
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

def gradient_check(w, r, Train):
    Loss2 = Loss1 = 0
    pred1 = np.zeros((check_time, 4))
    pred2 = np.zeros((check_time, 4))
    lenth = len(Train[0])
    w1 = copy.deepcopy(w)
    b = copy.deepcopy(r)

    epsi = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    error = np.zeros(len(epsi))
    for epsilon in epsi:
        for i in range(check_time):
            w2 = copy.deepcopy(w1)

            xi, yi = random.randint(0, 3), random.randint(0, lenth - 1)
            w2[xi][yi] += epsilon
            z2 = np.dot(w2, Train[i]) + b
            z1 = np.dot(w1, Train[i]) + b

            pred1[i] = softmax(z1)
            pred2[i] = softmax(z2)

            soft1 = copy.deepcopy(pred1[i])
            soft1[train.target[i]] -= 1

            soft2 = copy.deepcopy(pred2[i])
            soft2[train.target[i]] -= 1

            Loss1 = -np.log(pred1[i][train.target[i]]) + 0.5 * penal * np.sum(w1 * w1)
            Loss2 = -np.log(pred2[i][train.target[i]]) + 0.5 * penal * np.sum(w2 * w2)

            w1 -= gama * (penal * w1 + np.dot(np.mat(soft1).T, np.mat(Train[i])))
            b -= gama * soft1

            error[epsi.index(epsilon)] += abs(((soft1[xi] * Train[i][yi] + penal * w1[xi][yi] - (Loss2 - Loss1) / epsilon)) / ((Loss2 - Loss1) / epsilon))

        error[epsi.index(epsilon)] /= check_time
    ax = plt.axes(xscale = 'log', yscale = 'log')
    ax.plot(epsi, error)
    ax.set_xlabel('epsilon')
    ax.set_ylabel('error')
    print(epsi, error)

def rate(Train, pred1, w1, b1, loss1, loss2):
    rate = [1e-3, 1e-2, 3e-2, 5e-2, 1e-1]
    for gama in rate:
        epoch = 0
        cycle = []
        Loss = []
        pred = copy.deepcopy(pred1)
        w = copy.deepcopy(w1)
        b = copy.deepcopy(b1)
        Loss1 = loss1
        Loss2 = loss2
        cycle.append(epoch)
        Loss.append(Loss1)
        while abs(Loss1 - Loss2) > acc:
            epoch += 1
            cycle.append(epoch)
            for i in range(len(Train)):
                z = np.dot(w, Train[i]) + b
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
            Loss.append(Loss1)
        plt.plot(cycle, Loss)
        plt.ylabel('Loss')
        plt.ylim([0, 2])
        plt.show()
        print(epoch)

def stochastic(Train, Test, pred1, w1, b1, Loss1, Loss2):
    pred = copy.deepcopy(pred1)
    w = copy.deepcopy(w1)
    b = copy.deepcopy(b1)
    epoch = 0
    while abs(Loss1 - Loss2) > acc:
        epoch += 1
        Loss2 = Loss1
        Loss1 = 0
        for i in range(len(Train)):
            z = np.dot(w, Train[i]) + b
            pred[i] = softmax(z)
            soft = copy.deepcopy(pred[i])
            soft[train.target[i]] -= 1
            w -= gama * (penal * w + np.dot(np.mat(soft).T, np.mat(Train[i])))
            b -= gama * soft
            Loss1 -= np.log(pred[i][train.target[i]])
        Loss1 = Loss1 / len(Train) + 0.5 * penal * np.sum(w * w)

    print("stochastic:", epoch)
    Verify(Train, w, b, 0)
    Verify(Test, w, b, 1)

def full_batched(Train, Test, pred1, w1, b1, Loss1, Loss2):
    pred = copy.deepcopy(pred1)
    w = copy.deepcopy(w1)
    b = copy.deepcopy(b1)
    epoch = 0
    soft = np.zeros((len(Train), 4))
    z = np.zeros(4)
    for i in range(len(Train)):
        z = np.dot(w, Train[i]) + b
        pred[i] = softmax(z)
        soft[i] = pred[i]
        soft[i][train.target[i]] -= 1

    while abs(Loss1 - Loss2) > acc:
        epoch += 1
        print(Loss2, Loss1)
        Loss2 = Loss1

        Loss1 = 0
        w -= gama * (np.dot(np.mat(soft).T, np.mat(Train)) + penal * w)
        b -= gama * np.sum(soft)
        for i in range(len(Train)):
            z = np.dot(w, Train[i]) + b
            pred[i] = softmax(z)
            soft[i] = pred[i]
            soft[i][train.target[i]] -= 1
            if pred[i][train.target[i]] == 0:
                pred[i][train.target[i]] = 3e-253
            Loss1 -= np.log(pred[i][train.target[i]])
        Loss1 = Loss1 / len(Train) + 0.5 * penal * np.sum(w * w)
    print("full_batched:", epoch)
    Verify(Train, w, b, 0)
    Verify(Test, w, b, 1)

def batched(Train, Test, pred1, w1, b1, Loss1, Loss2, k):
    pred = copy.deepcopy(pred1)
    w = copy.deepcopy(w1)
    b = copy.deepcopy(b1)
    epoch = 0
    soft = np.zeros((k, 4))
    while abs(Loss1 - Loss2) > acc:
        epoch += 1
        Loss2 = Loss1
        print(Loss1, Loss2)
        Loss1 = 0
        for i in range(len(Train)):
            z = np.dot(w, Train[i]) + b
            pred[i] = softmax(z)
            soft[i % k] = pred[i]
            soft[i % k][train.target[i]] -= 1
            if i % k == 0 and i != 0:
                for j in range(i - k, i):
                    w -= gama * (penal * w + np.dot(np.mat(soft[j % k]).T, np.mat(Train[i])))
                b -= np.sum(soft)
            Loss1 -= np.log(pred[i][train.target[i]])
        Loss1 = Loss1 / len(Train) + 0.5 * penal * np.sum(w * w)

    print("batched:", epoch)
    Verify(Train, w, b, 0)
    Verify(Test, w, b, 1)

def Verify(Train, w, b, num):
    sum = 0
    pred_t = np.zeros(4)
    z = np.zeros(4)
    lenth = 0
    if num == 0:
        Valid = train
        lenth = len(train.data)
    else :
        Valid = test
        lenth = len(test.data)
    for i in range(len(Train)):
        z = np.dot(w, Train[i]) + b
        pred_t = softmax(z)
        if(pred_t.tolist().index(max(pred_t)) == Valid.target[i]):
            sum += 1

    print(sum, lenth, sum / lenth)

def Logistic():
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

    w = np.zeros((4, num))
    b = np.zeros(4)
    pred = np.zeros((len(Train), 4))
    z = np.zeros(4)

    Loss2 = Loss1 = 0
    for i in range(len(Train)):
        z = np.dot(w, Train[i]) + b
        pred[i] = softmax(z)

    for i in range(len(Train)):
        Loss1 -= np.log(pred[i][train.target[i]])
    Loss1 = Loss1 / len(Train) + 0.5 * penal * np.sum(w * w)

    #gradient_check(w, b, Train)
    #rate(Train, pred, w, b, Loss1, Loss2)
    #stochastic(Train, Test, pred, w, b, Loss1, Loss2)
    #full_batched(Train, Test, pred, w, b, Loss1, Loss2)
    batched(Train, Test, pred, w, b, Loss1, Loss2, 2)

#Lq()
#Perceptron()
Logistic()
plt.show()