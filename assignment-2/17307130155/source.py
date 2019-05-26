import os
import math
import sys,getopt
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from handout import get_linear_seperatable_2d_2c_dataset
from handout import get_text_classification_datasets
import re
import string
import collections

BATCH_SIZE = 2247
alpha = 0.1
lamda = 0.01 #正则化项参数需要修改，正则化项需要推导

def perception(w,data,t):
    alpha = 1
    for i in range(0,data.shape[0]):
        y = np.dot(data[i],w)
        if (y*t[i]<0):
            w = w + alpha*t[i]*data[i]
    return w

def leastsquare(w,data,t):
    #print(t.shape)
    w = np.linalg.solve(np.dot(data.T,data),np.dot(data.T,t))
    return w

def cal_precision(w,x,t):
    result = np.dot(x,w)*t
    #print(x.shape)
    #print(w.shape)
    #print(t.shape)
    result[result<0] = 0
    print('The precision rate is :',np.count_nonzero(result)/x.shape[0])

def part1(choose):
    dataset = get_linear_seperatable_2d_2c_dataset()
    data = dataset.X

    label = dataset.y
    bias = np.ones(data.shape[0]).reshape(data.shape[0],1)
    data = np.hstack([data,bias])
    t = np.repeat(-1,label.shape[0]).reshape(label.shape[0],1)
    t[label] = 1

    w1 = np.zeros(3)
    w2 = np.ones(3)
    w1 = leastsquare(w1,data[0:data.shape[0]*4//5],t[0:data.shape[0]*4//5])
    w2 = perception(w2,data[0:data.shape[0]*4//5],t[0:data.shape[0]*4//5])

    cal_precision(w1.reshape(3,1),data[data.shape[0]*1//5:data.shape[0]],t[data.shape[0]*1//5:data.shape[0]])
    cal_precision(w2.reshape(3,1),data[data.shape[0]*1//5:data.shape[0]],t[data.shape[0]*1//5:data.shape[0]])

    plotx = np.linspace(-1.0,1.0,10)
        
    if (choose==0):
        plt.title('Least Square')
        ploty1 = -(w1[0]*plotx+w1[2])/w1[1]
        plt.xlabel("x")
        plt.ylabel("y")
        plt.scatter(dataset.X[label,0],dataset.X[label,1],c='orange')
        plt.scatter(dataset.X[~label,0],dataset.X[~label,1],c='purple')
        plt.plot(plotx,ploty1,c='black')
        print(w1)
    
    elif (choose==1):
        plt.title('Perception')
        ploty2 = -(w2[0]*plotx+w2[2])/w2[1]
        plt.xlabel("x")
        plt.ylabel("y")
        plt.scatter(dataset.X[label,0],dataset.X[label,1],c='orange')
        plt.scatter(dataset.X[~label,0],dataset.X[~label,1],c='purple')
        plt.plot(plotx,ploty2,c='black')
        print(w2)
        
    plt.show()

def word_count(train):   #对文本中所有的词进行统计
    word_freq = collections.defaultdict(int)
    for trainstr in train.data:
        for word in trainstr.lower().translate(str.maketrans('', '', string.punctuation)).split():
            word_freq[word] += 1
    return word_freq

def build_dict(train,min_word_freq=10):   #根据统计的词表生成字典
    word_freq = word_count(train)
    word_freq = filter(lambda x:x[1] > min_word_freq, word_freq.items())
    word_freq_sorted = sorted(word_freq, key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*word_freq_sorted))
    word_idx = dict(zip(words, range(len(words))))
    word_idx['<unk>'] = len(words)
    print ('Dictionary has been built.')
    return word_idx

def data_preprocess(train,dictionary):
    trainset_onehot = np.zeros((len(train.data),len(dictionary)))
    for i in range(len(train.data)):
        for word in train.data[i].lower().translate(str.maketrans('', '', string.punctuation)).split():
            if word in dictionary:
                trainset_onehot[i][dictionary[word]] = 1
            else:
                trainset_onehot[i][dictionary['<unk>']] = 1
    t = np.zeros((len(train.target),4))
    for i in range(0,len(train.target)):
        t[i][train.target[i]] = 1
    bias = np.ones((len(train.data),1))
    trainset_onehot = np.hstack([trainset_onehot,bias])
    return trainset_onehot,t

def traincycle(x,t,w): #训练一次的过程,少了一个loss
    h = np.e**(np.dot(x,w))/(np.sum(np.e**np.dot(x,w),axis=1).reshape(x.shape[0],1))
    w_cat = w.copy()
    w_cat[-1,:] = 0
    #print(x.shape)
    w = w + alpha*(1/x.shape[0]*np.dot(x.T,(t-h))-lamda*w_cat)
    #w = w + alpha*1/2247*(np.dot(x.T,(t-h))-x.shape[0]*lamda*w_cat)
    return w

def onecycle(x,t,w,batchsize): 
    i = np.random.randint(0,x.shape[0]//batchsize)
    # for i in range(0,x.shape[0]//batchsize):
    #     w = traincycle(x[i*batchsize:(i+1)*batchsize],t[i*batchsize:(i+1)*batchsize],w)
    # if x.shape[0] % batchsize != 0 :
    #     w = traincycle(x[(x.shape[0]//batchsize)*batchsize:x.shape[0]],t[(x.shape[0]//batchsize)*batchsize:x.shape[0]],w)  
    w = traincycle(x[i*batchsize:(i+1)*batchsize],t[i*batchsize:(i+1)*batchsize],w)
    h = np.e**(np.dot(x,w))/(np.sum(np.e**np.dot(x,w),axis=1).reshape(x.shape[0],1))
    w_cat = w.copy()
    w_cat[-1,:] = 0
    j = -1/x.shape[0]*(np.sum(t*np.log(h)))+0.5*lamda*(np.sum(w_cat**2))
    return j,w

def precision(w,x,t):
    result = np.e**(np.dot(x,w))/(np.sum(np.e**np.dot(x,w),axis=1)).reshape(x.shape[0],1)
    #判断分类
    count = 0
    y_cat = np.argmax(result,axis=1)
    for i in range(0,(result.shape)[0]):
        if y_cat[i] == t[i]:
            count += 1
    return (count/result.shape[0])

def part2(choose):
    train,test = get_text_classification_datasets()
    print(type(train))
    dictionary = build_dict(train,10)
    trainset_onehot,t = data_preprocess(train,dictionary)
    testset_onehot,t_ = data_preprocess(test,dictionary)

    w = np.zeros((len(dictionary)+1,4))
    w_ = w
    j=100
    j_=99
    count = 0
    c = []
    p = []

    if choose==1:
        while (count<5000):
            count += 1 
            c.append(count)
            j = j_
            w = w_
            j_, w_ = onecycle(trainset_onehot,t,w,1)
            p.append(j_)
            print("After ",count, "times training, the loss is:",j_)
    
    elif choose==2:
        while (count<2000):
            count += 1 
            c.append(count)
            j = j_
            w = w_
            j_, w_ = onecycle(trainset_onehot,t,w,10)
            p.append(j_)
            print("After ",count, "times training, the loss is:",j_)
    
    elif choose==3:
        while (count<1000):
            count += 1 
            c.append(count)
            j = j_
            w = w_
            j_, w_ = onecycle(trainset_onehot,t,w,100)
            p.append(j_)
            print("After ",count, "times training, the loss is:",j_)

    elif choose==4:
        while (np.abs(j-j_)>1e-4):
            count += 1 
            c.append(count)
            j = j_
            w = w_
            j_, w_ = onecycle(trainset_onehot,t,w,BATCH_SIZE)
            p.append(j_)
            print("After ",count, "times training, the loss is:",j_)
    
    #print(w)
    print('The train accuracy is',precision(w,trainset_onehot,train.target))
    print('The test accuracy is',precision(w,testset_onehot,test.target))
    plt.plot(c,p)
    plt.show()

def usage():
    print('<Usage>')
    print('python3 source.py [options]')
    print()
    print('Options')
    print('-h or --help     :  Show help')
    print('--leastsquare    :  Show leastsquare')
    print('--perception     :  Show perception algorithm')
    print('--SGD            :  Show SGD')
    print('--MBGD10         :  Show MBGD with batch_size = 10')
    print('--MBGD100        :  Show MBGD with batch_size = 100')
    print('--FBGD           :  Show FBGD')
#part1()
#part2()

def main(argv):
    if argv[1] == '-h' or argv[1] == '--help':
        usage()
        sys.exit()
    elif argv[1] == '--perception':
        part1(1);
    elif argv[1] == '--leastsquare':
        part1(0);
    elif argv[1] == '--SGD':
        part2(1);
    elif argv[1] == '--MBGD10':
        part2(2);
    elif argv[1] == '--MBGD100':
        part2(3)
    elif argv[1] == '--FBGD':
        part2(4);
    else:
        print ("Error: invalid parameters.You can type 'python3 source.py -h' or 'python3 source.py --help' for further help")
        sys.exit()

if __name__ == "__main__":
    main(sys.argv)
