#!/usr/bin/env python
# coding: utf-8

# In[2]:

#PART1

import os
os.sys.path.append('..')
from handout import get_linear_seperatable_2d_2c_dataset
data_sample = get_linear_seperatable_2d_2c_dataset()
import numpy as np
import matplotlib.pyplot as plt
import time

#list square model

def accurate(x,y,w):
    count = 0
    tr = 0
    for i in range(len(y)):
        count += 1
        if ((w[0]*x[i][0]+w[1]*x[i][1]+w[2] >= 0.5) and y[i] == True):
            tr += 1
        elif ((w[0]*x[i][0]+w[1]*x[i][1]+w[2] < 0.5) and y[i] == False):
            tr += 1
    return str(tr/count)
            
def ListSquareModel(X,y):
    dataLength = len(X)
    Xexpand = np.insert(X,2,1,axis = 1)
    # XW = Y
    w = np.linalg.solve(np.dot(Xexpand.T,Xexpand),np.dot(Xexpand.T,y))
    # print(w)
    xlist = np.linspace(-1.5,1.5,100)
    ylist = np.array(((0.5-w[2])-w[0]*xlist)/w[1])
    plt.xlabel("x")
    plt.ylabel("y")
    LSModelStr = "LSM: "+str(round(w[0],2))+"*x+"+str(round(w[1],2))+"*y+"+str(round(w[2],2))+" = 0.5"
    LSModel = plt.plot(xlist,ylist)
    plt.legend(LSModel,[LSModelStr])
    return w
    
w = ListSquareModel(data_sample.X,data_sample.y)
print("accuracy:"+accurate(data_sample.X,data_sample.y,w))
data_sample.plot(plt).show()


# In[4]:

# perceptron algorithm

def perceptron(X,y,times,rate):
    #initial w
    #w = np.random.rand(3)
    w = np.zeros(3)
    #expand X
    #print("w init:"+str(w))
    Xexpand = np.insert(X,2,1,axis = 1)
    k = 0
    for i in range(times):
        for j in range(len(X)):
            result = np.dot(Xexpand[j].T,w)
            if result > 0.5:
                yPre = 1
            else:
                yPre = 0
            w += rate*(y[j] - yPre)*Xexpand[j]
        accuracy = accurate(data_sample.X,data_sample.y,w)
        k += 1
        if float(accuracy) == 1:
            break
    #print("w final:"+str(w))
    #accuracy = accurate(data_sample.X,data_sample.y,w)
    xlist = np.linspace(-1.5,1.5,100)
    ylist = np.array(((0.5-w[2])-w[0]*xlist)/w[1])
    plt.xlabel("x")
    plt.ylabel("y")
    perceptronModelStr = "perceptron: "+str(round(w[0],2))+"*x+"+str(round(w[1],2))+"*y+"+str(round(w[2],2))+" = 0.5"
    perceptronModel = plt.plot(xlist,ylist)
#     plt.legend(perceptronModel,[perceptronModelStr])
    #plt.title("times:"+ str(times)+" accuracy:"+str(accuracy))
    plt.title("rate:"+ str(rate)+" times:"+str(k))
    data_sample.plot(plt).show()
    return w

ax = plt.subplot(321)
perceptron(data_sample.X,data_sample.y,30,0.001)

ax = plt.subplot(322)
perceptron(data_sample.X,data_sample.y,30,0.002)

ax = plt.subplot(323)
perceptron(data_sample.X,data_sample.y,30,0.003)

ax = plt.subplot(324)
perceptron(data_sample.X,data_sample.y,30,0.004)

ax = plt.subplot(325)
perceptron(data_sample.X,data_sample.y,30,0.005)

ax = plt.subplot(326)
perceptron(data_sample.X,data_sample.y,30,0.006)
# print("accuracy:"+accurate(data_sample.X,data_sample.y,w))
# data_sample.plot(plt).show()


# In[5]:


#part2
from handout import get_text_classification_datasets
trainData,testData = get_text_classification_datasets()


# In[6]:


import string
trainDataset = trainData['data']
def getListX(dataset):
    dic = {}
    for data in trainDataset:
        data = data.lower()
        for i in data:
            if i in string.punctuation:
                data = data.replace(i," ")
        data = data.split()
        #print(data)
        for word in data:
            if word in dic.keys():
                dic[word] += 1
            else:
                dic[word] = 1
    for i in list(dic.keys()):
        if dic[i] < 10:
            del(dic[i])
    for data in dataset:
        data = data.lower()
        for i in data:
            if i in string.punctuation:
                data = data.replace(i," ")
        data = data.split()
    multiHotList = []
    for data in dataset:
        j = 0
        multiHotData = []
        for word in list(dic.keys()):
            if word in data:
                multiHotData.append(1)
            else:
                multiHotData.append(0)
        multiHotList.append(multiHotData)
    return multiHotList
multiHotList = getListX(trainDataset)
# multiHotList


# In[7]:


def getListX_no_num(dataset):
    dic = {}
    for data in trainDataset:
        data = data.lower()
        for i in data:
            if i in string.punctuation:
                data = data.replace(i," ")
            elif i.isdigit():
                data = data.replace(i," ")
        data = data.split()
        #print(data)
        for word in data:
            if word in dic.keys():
                dic[word] += 1
            else:
                dic[word] = 1
    for i in list(dic.keys()):
        if dic[i] < 10:
            del(dic[i])
    for data in dataset:
        data = data.lower()
        for i in data:
            if i in string.punctuation:
                data = data.replace(i," ")
        data = data.split()
    multiHotList = []
    for data in dataset:
        j = 0
        multiHotData = []
        for word in list(dic.keys()):
            if word in data:
                multiHotData.append(1)
            else:
                multiHotData.append(0)
        multiHotList.append(multiHotData)
    return multiHotList


# In[8]:



y = trainData['target']
def getListY(y):
    target = []
    for i in range(len(y)):
        t = [0,0,0,0]
        t[y[i]] = 1
        target.append(t)
    return target
target = getListY(y)


# In[10]:


#vectorized style softmax
#every vector is a column
def softmaxTransfer(z):
    z = z - np.ones(z.shape)*np.max(z)
    sumVec = np.dot(np.exp(z.T),np.ones((z.shape[0],1)))
    #expand to a matrix
    sumRecip = 1/np.dot(sumVec,np.ones((1,z.shape[0])))
    return (np.multiply(np.exp(z),sumRecip.T))


# In[20]:


# the vertion having ternimal contition
# plot: change of loss function
def gradientDescentTermination(X,y,rate,lmd):
    Xexpand = np.insert(X,X.shape[1],1,axis = 1)
    Xexpand = Xexpand.T
    y = y.T
    w = np.zeros((y.shape[0],Xexpand.shape[0]))
    lossSet = []
    changeSet = []
    for i in range(500):
        penalWeight = (2*lmd)*w
        penalWeight[:,penalWeight.shape[1]-1] = 0
        softmax = softmaxTransfer(np.dot(w,Xexpand))
        softmax -= y
        # get derivative
        dataLength = Xexpand.shape[1]
        derivativeW = np.dot(softmax,Xexpand.T)*(1/dataLength) + penalWeight
        w -= rate * derivativeW
        #calculate loss
        tmp = softmaxTransfer(np.dot(w,Xexpand))
        loss = lmd*sum(sum(np.square(w[:,:w.shape[1]-1])))-np.sum(np.multiply(np.log(tmp),y))/dataLength
        lossSet.append(loss)
        if len(lossSet) >=2:
            change = lossSet[len(lossSet)-1]-lossSet[len(lossSet)-2]
            changeSet.append(change)
            if change < 0.0005 and change > -0.0005:
                break
    x = [i for i in range(len(changeSet))]
    plt.plot(x,changeSet)
    plt.show()
    return w

gradientDescentTermination(np.array(multiHotList),np.array(target),0.05,0.001)


# In[21]:


# normal vertion
# 300 round with rate changed
def gradientDescent(X,y,rate,lmd):
    Xexpand = np.insert(X,X.shape[1],1,axis = 1)
    Xexpand = Xexpand.T
    y = y.T
    w = np.zeros((y.shape[0],Xexpand.shape[0]))
    lossSet = []
    time1 = 0
    timePerRound = 0
    for i in range(300):
        penalWeight = (2*lmd)*w
        penalWeight[:,penalWeight.shape[1]-1] = 0
        softmax = softmaxTransfer(np.dot(w,Xexpand))
        softmax -= y
        if i == 1:
            time1 = time.time()
        if i == 2:
            timePerRound = time.time() - time1
            print("time per round for full batch gradient descent:" + str(timePerRound))
        # get derivative
        dataLength = Xexpand.shape[1]
        derivativeW = np.dot(softmax,Xexpand.T)*(1/dataLength) + penalWeight
        w -= rate * derivativeW
        #calculate loss
        tmp = softmaxTransfer(np.dot(w,Xexpand))
        loss = lmd*sum(sum(np.square(w[:,:w.shape[1]-1])))-np.sum(np.multiply(np.log(tmp),y))/dataLength
        lossSet.append(loss)
    x = [i for i in range(len(lossSet))]
    plt.plot(x,lossSet)
    plt.show()
    
ax = plt.subplot(321)
gradientDescent(np.array(multiHotList),np.array(target),0.01,0.001)

ax = plt.subplot(322)
gradientDescent(np.array(multiHotList),np.array(target),0.03,0.001)

ax = plt.subplot(323)
gradientDescent(np.array(multiHotList),np.array(target),0.05,0.001)

ax = plt.subplot(324)
gradientDescent(np.array(multiHotList),np.array(target),0.1,0.001)

ax = plt.subplot(325)
gradientDescent(np.array(multiHotList),np.array(target),0.2,0.001)

ax = plt.subplot(326)
gradientDescent(np.array(multiHotList),np.array(target),0.4,0.001)


# In[22]:


# check whether the gradient is correct
def checkGradient(X,y,lmd):
    Xexpand = np.insert(X,X.shape[1],1,axis = 1)
    Xexpand = Xexpand.T
    y = y.T
    w = np.zeros((y.shape[0],Xexpand.shape[0]))
    lossSet = []
    for i in range(20):
        dataLength = Xexpand.shape[1]
        tmp = softmaxTransfer(np.dot(w,Xexpand))
        lossBefore = lmd*sum(sum(np.square(w[:,:w.shape[1]-1])))-np.sum(np.multiply(np.log(tmp),y))/dataLength
        penalWeight = (2*lmd)*w
        penalWeight[:,penalWeight.shape[1]-1] = 0
        softmax = softmaxTransfer(np.dot(w,Xexpand))
        softmax -= y
        derivativeW = np.dot(softmax,Xexpand.T)*(1/dataLength) + penalWeight
        m,n = np.random.randint(y.shape[0]),np.random.randint(Xexpand.shape[0])
        w[m][n] += 0.001
        tmp = softmaxTransfer(np.dot(w,Xexpand))
        lossAfter = lmd*sum(sum(np.square(w[:,:w.shape[1]-1])))-np.sum(np.multiply(np.log(tmp),y))/dataLength
        print("calculation: %f   approximation：%f"%(derivativeW[m][n],((lossAfter-lossBefore)*1000)))
checkGradient(np.array(multiHotList),np.array(target),0.001)
    


# In[27]:


#SGD
def gradientDescentStochastic(X,y,rate,lmd):
    Xexpand = np.insert(X,X.shape[1],1,axis = 1)
    Xexpand = Xexpand.T
    y = y.T
    w = np.zeros((y.shape[0],Xexpand.shape[0]))
    lossSet = []
    time1 = 0
    timePerRound = 0
    for i in range(40):
        if i == 1:
            time1 = time.time()
        if i == 2:
            timePerRound = time.time() - time1
            print("time per round for stochastic gradient descent:" + str(timePerRound))
        dataLength = Xexpand.shape[1]
        for j in range(dataLength):
            indexBegin = np.random.randint(0,dataLength-1)
            penalWeight = (2*lmd)*w
            penalWeight[:,penalWeight.shape[1]-1] = 0
            trainData = Xexpand[:,indexBegin:indexBegin+1]
            trainY = y[:,indexBegin:indexBegin+1]
            softmax = softmaxTransfer(np.dot(w,trainData))
            softmax -= trainY
            # get derivative
            derivativeW = np.dot(softmax,trainData.T) + penalWeight
            w -= rate * derivativeW
            #calculate loss
        tmp = softmaxTransfer(np.dot(w,Xexpand))
        loss = lmd*sum(sum(np.square(w[:,:w.shape[1]-1])))-np.sum(np.multiply(np.log(tmp),y))/dataLength
        lossSet.append(loss)
    x = [i for i in range(len(lossSet))]
    plt.plot(x,lossSet)
    plt.show()
    return w
gradientDescentStochastic(np.array(multiHotList),np.array(target),0.01,0.001)


# In[11]:


# miniBGD
def gradientDescentBatched(X,y,rate,lmd,batchSize):
    Xexpand = np.insert(X,X.shape[1],1,axis = 1)
    Xexpand = Xexpand.T
    y = y.T
    w = np.zeros((y.shape[0],Xexpand.shape[0]))
    lossSet = []
    k = 0
    time1 = 0
    timePerRound = 0
    for i in range(40):
        if i == 1:
            time1 = time.time()
        if i == 2:
            timePerRound = time.time() - time1
            print("time per round for batched gradient descent:" + str(timePerRound))
        dataLength = Xexpand.shape[1]
        for j in range(int(dataLength/batchSize)):
            indexBegin = np.random.randint(0,dataLength-batchSize)
            penalWeight = (2*lmd)*w
            penalWeight[:,penalWeight.shape[1]-1] = 0
            trainData = Xexpand[:,indexBegin:indexBegin+batchSize]
            trainY = y[:,indexBegin:indexBegin+batchSize]
            softmax = softmaxTransfer(np.dot(w,trainData))
            softmax -= trainY
            # get derivative
            derivativeW = np.dot(softmax,trainData.T)*(1/batchSize) + penalWeight
            w -= rate * derivativeW
            #calculate loss
        tmp = softmaxTransfer(np.dot(w,Xexpand))
        loss = lmd*sum(sum(np.square(w[:,:w.shape[1]-1])))-np.sum(np.multiply(np.log(tmp),y))/dataLength
        lossSet.append(loss)
            
    x = [i for i in range(len(lossSet))]
    plt.plot(x,lossSet)
    plt.show()
    return w

ax = plt.subplot(321)
gradientDescentBatched(np.array(multiHotList),np.array(target),0.01,0.001,10)

ax = plt.subplot(322)
gradientDescentBatched(np.array(multiHotList),np.array(target),0.01,0.001,20)

ax = plt.subplot(323)
gradientDescentBatched(np.array(multiHotList),np.array(target),0.01,0.001,40)

ax = plt.subplot(324)
gradientDescentBatched(np.array(multiHotList),np.array(target),0.01,0.001,80)

ax = plt.subplot(325)
gradientDescentBatched(np.array(multiHotList),np.array(target),0.01,0.001,120)

ax = plt.subplot(326)
gradientDescentBatched(np.array(multiHotList),np.array(target),0.01,0.001,200)


# In[36]:


def testAccuracy(w,X,y):
    accNum = 0
    X = np.insert(X,X.shape[1],1,axis = 1)
    sumNum = len(X)
    for i in range(len(X)):
        temp = softmaxTransfer(np.dot(w,X[i].T))
        result = np.argmax(temp)
        if result == y[i]:
            accNum += 1
    return accNum/sumNum


# In[26]:


wFBGD = gradientDescentTermination(np.array(multiHotList),np.array(target),0.05,0.001)
wSGD = gradientDescentStochastic(np.array(multiHotList),np.array(target),0.01,0.001)
wMiniBGD = gradientDescentBatched(np.array(multiHotList),np.array(target),0.01,0.001,20)
testX = getListX(testData['data'])
testY = testData['target']
accFBGD = testAccuracy(wFBGD,np.array(testX),np.array(testY))
accSGD = testAccuracy(wSGD,np.array(testX),np.array(testY))
accMiniBGD = testAccuracy(wMiniBGD,np.array(testX),np.array(testY))
print("accuracy for FBGD: %f" %(accFBGD))
print("accuracy for SGD: %f" %(accSGD))
print("accuracy for miniBGD: %f" %(accMiniBGD))


# In[40]:


multiHotList2 = getListX_no_num(trainDataset)
wFBGD = gradientDescentTermination(np.array(multiHotList2),np.array(target),0.05,0.001)
wSGD = gradientDescentStochastic(np.array(multiHotList2),np.array(target),0.01,0.001)
wMiniBGD = gradientDescentBatched(np.array(multiHotList2),np.array(target),0.01,0.001,20)
testX = getListX_no_num(testData['data'])
testY = testData['target']
accFBGD = testAccuracy(wFBGD,np.array(testX),np.array(testY))
accSGD = testAccuracy(wSGD,np.array(testX),np.array(testY))
accMiniBGD = testAccuracy(wMiniBGD,np.array(testX),np.array(testY))
print("accuracy for FBGD: %f" %(accFBGD))
print("accuracy for SGD: %f" %(accSGD))
print("accuracy for miniBGD: %f" %(accMiniBGD))


