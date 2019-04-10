import os
os.sys.path.append('..')
from handout import get_linear_seperatable_2d_2c_dataset
from handout import get_text_classification_datasets
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from sklearn.datasets import fetch_20newsgroups
import string
import re
import collections


data = get_linear_seperatable_2d_2c_dataset()

def predict(a,b,c):
    y = data.y
    pre_y = np.zeros(len(y))
    X = data.X
    for i in range(len(pre_y)):
        x1 = X[i][0]
        x2 = X[i][1]
        res = a*x1 + b*x2 + c
        if res>0:
            pre_y[i] = True
        else:
            pre_y[i] = False
    print(data.acc(pre_y)) 

def LS():
    X = np.mat(data.X)
    one = np.mat(np.ones(len(X))).T
    y = data.y
    X_ = np.concatenate((X,one),axis = 1)
    X_T = X_.T

    #calculate (XT * X)^-1
    temp = np.dot(X_T,X_).I
    pre_y = np.zeros(len(y))
    for i in range(len(y)):
        if y[i]:
            pre_y[i] = 1
        else:
            pre_y[i] = -1

    M = temp * X_T 
    M = M * np.mat(pre_y).T
    M = np.array(M)
    (a,b,c) = (M[0],M[1],M[2])

    predict(a,b,c)
    plt.plot([-2,2],[(2*a-c)/b,(-2*a-c)/b])
    data.plot(plt).show()
    return (a,b,c)

def perceptron(a1=1,b1=1,c1=0,step=0.1):
    (a,b,c) = (a1,b1,c1)
    X = data.X
    y = data.y
    j = 0
    rank = 0
    while(j < 20000):
        if j > 1000*(1+rank) and j < 10000:
            rank += 1
            step /= 2
        j += 1
        L = []
        loss = 0
        for i in range(len(X)):
            if y[i]:
                t = 1
            else:
                t = -1
            result = a*X[i][0]+b*X[i][1]+c
            if result * t < 0:
                L.append((i,t))
                loss -= result*t
        if(loss == 0): 
            #print(a,b,c,L,result)
            break
        rand = random.randrange(0,len(L))
        temp = L[rand]
        a += X[temp[0]][0]*step*temp[1]
        b += X[temp[0]][1]*step*temp[1]
        c += step*temp[1]
    
    predict(a,b,c)
    plt.plot([-2,2],[(2*a-c)/b,(-2*a-c)/b])
    data.plot(plt).show()
    return (a,b,c)

def handle_data(data):
    min_count = 10
    dic_t = {}
    for i in range(len(data)):
        data[i] = re.sub("\d+|\s+|/", " ", data[i])
        data[i] = [word.strip(string.punctuation).lower() for word in data[i].split() if word.strip(string.punctuation) != '']
        for j in data[i]:
            if j in dic_t:
                dic_t[j] += 1
            else: 
                dic_t[j] = 1 
    dic = {}
    dic = collections.OrderedDict()
    index = 0
    for i,v in dic_t.items():
        if v >= min_count:
            dic[i] = index
            index += 1
    return dic

def softmax(x):
    if x.shape[1] == 1:
        temp = max(x)
        x -= temp
        return (np.exp(x)/np.sum(np.exp(x)))
    temp = np.max(x)
    x -= np.ones(x.shape)*temp
    vec = np.dot(np.exp(x.T),np.ones((x.shape[0],1)))
    vec = 1/(vec*np.ones((1,x.shape[0])))
    result = np.multiply(np.exp(np.mat(x)),vec.T)
    return result

class model:
    def __init__(self,dic,data,target,data_t,target_t):
        self.multi_hot = []
        for i in range(len(data)):
            hot = np.zeros(len(dic))
            for j in range(len(data[i])):
                if data[i][j] in dic:
                    hot[dic[data[i][j]]] = 1
            self.multi_hot.append(hot)
        one = np.mat(np.ones(len(self.multi_hot))).T
        self.multi_hot = np.mat(self.multi_hot)
        self.multi_hot = np.concatenate((self.multi_hot,one),axis = 1)

        self.multi_hot_t = []
        for i in range(len(data_t)):
            hot = np.zeros(len(dic))
            for j in range(len(data_t[i])):
                if data_t[i][j] in dic:
                    hot[dic[data_t[i][j]]] = 1
            self.multi_hot_t.append(hot)
        one = np.mat(np.ones(len(self.multi_hot_t))).T
        self.multi_hot_t = np.mat(self.multi_hot_t)
        self.multi_hot_t = np.concatenate((self.multi_hot_t,one),axis = 1)

        self.N = len(dic)
        self.sample = len(data)
        self.label = len(dataset_train.target_names)
        self.W = np.zeros((self.N+1,self.label))
        self.tar = np.zeros((self.sample,self.label))
        for i in range(len(self.tar)):
            self.tar[i][target[i]] = 1
    
    def init_W(self):
        self.W = np.zeros((self.N+1,self.label))

    def train_1(self,step,lamd):
        proc = []
        j = 0
        rank = 0
        while(j < 500):
            j+= 1
            add = np.mat(self.W)*2*lamd
            add[self.N,:] = [0,0,0,0]
            temp = softmax(self.W.T*self.multi_hot.T)
            #print(temp)
            #print(temp.shape,multi_hot[i].shape,W.shape)
            temp -= self.tar.T
            self.W -= (temp*self.multi_hot).T*step/self.sample
            #print(W)
            self.W -= add*step
            #print(W)
            loss = 0         
            y = np.log(softmax(self.W.T*self.multi_hot.T))
            y = np.multiply(y,self.tar.T)
            loss -= np.sum(y)/self.sample
            loss += sum(sum(np.square(self.W[:self.N,:]))) * lamd   
            proc.append(loss)
            if j > 100*(rank):
                rank += 1
                step /= 2
                print(loss)
        x = [i for i in range(len(proc))]
        plt.plot(x,proc)
        plt.show()

    def train_2(self,step,lamd):
        proc = []
        j = 0
        rank = 0
        while(j < 20000):
            j+= 1
            add = np.mat(self.W)*2*lamd
            add[self.N,:] = [0,0,0,0]
            i = random.randint(0,self.sample-1)
            temp = softmax(self.W.T*self.multi_hot[i].T)
            #print(temp)
            #print(temp.shape,multi_hot[i].shape,W.shape)
            temp[target[i]] -= 1
            self.W -= (temp*self.multi_hot[i]).T*step
                #print(W)
            self.W -= add*step
            #print(W)
            if j > 20*(rank):
                loss = 0           
                y = np.log(softmax(self.W.T*self.multi_hot.T))
                y = np.multiply(y,self.tar.T)
                loss -= np.sum(y)/self.sample
                loss += sum(sum(np.square(self.W[:self.N,:]))) * lamd  
                proc.append(loss)
                rank += 1
                if (rank % 200 == 0):
                    step /= 2
                    print(loss)
        x = [i for i in range(len(proc))]
        plt.plot(x,proc)
        plt.show()

    def train_3(self,step,lamd,batch):
        proc = []
        epoch = 0
        rank = 0
        while(epoch < 6):
            epoch += 1
            j = 0
            if epoch > rank:
                rank += 1
                step /= 2  
            while(j+batch < self.sample):
                loss = 0           
                y = np.log(softmax(self.W.T*self.multi_hot.T))
                y = np.multiply(y,self.tar.T)
                loss -= np.sum(y)/self.sample
                loss += sum(sum(np.square(self.W[:self.N,:]))) * lamd 
                proc.append(loss)
                add = np.mat(self.W)*2*lamd
                add[self.N,:] = [0,0,0,0]
                temp = softmax(self.W.T*self.multi_hot[j:j+batch].T)
            #print(temp)
            #print(temp.shape,multi_hot[i].shape,W.shape)
                temp -= self.tar[j:j+batch].T
                self.W -= (temp*self.multi_hot[j:j+batch]).T*step/batch
                    #print(W)
                self.W -= add*step
                j += batch
            print(loss)
        x = [i for i in range(len(proc))]
        plt.plot(x,proc)
        plt.show()
    
    def test(self,data_t):
        acc = 0
        sum_t = len(data_t)
        for i in range(len(self.multi_hot_t)):
            temp = softmax(self.W.T*self.multi_hot_t[i].T)
            ans = np.argmax(temp)
            if ans == target_t[i]:
                acc += 1
        print(acc/sum_t)
        

categories = ['comp.os.ms-windows.misc', 'rec.motorcycles', 'sci.space', 'talk.politics.misc', ]
dataset_train = fetch_20newsgroups(subset='train', categories=categories, data_home='../../..')
data = dataset_train.data
target = dataset_train.target
dataset_test = fetch_20newsgroups(subset='test', categories=categories, data_home='../../..')
data_t = dataset_test.data
target_t = dataset_test.target
for i in range(len(data_t)):
    data_t[i] = re.sub("\d+|\s+|/", " ", data_t[i])
    data_t[i] = [word.strip(string.punctuation).lower() for word in data_t[i].split() if word.strip(string.punctuation) != '']

step = 0.2
lamd = 0.001
dic = handle_data(data)
mod = model(dic,data,target,data_t,target_t)
mod.train_3(step,lamd,10)
mod.test(data_t)