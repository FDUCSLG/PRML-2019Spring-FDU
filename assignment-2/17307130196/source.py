# In this assignment, I will basically finish an assignment devided into two part.
# The first part focuses on the linear classification model
# while the second one is a small task on text classification

# import area
import matplotlib.pyplot as plt
import numpy as np
import math as mt
import string
import os
from nltk.corpus import stopwords
import nltk
import collections
# np.set_printoptions(threshold=np.inf) # prevent ...
# nltk.download('stopwords')
from __init__ import *
# os.sys.path.append("..")
# use the above line of code to surpass the top module barrier
# from handout import *

#=================================================#
#                      TASK1                      #
#=================================================#
# Part one
# two Linear Classification Model based on a given dataset
# And discuss the accuracy and ways to improve them
# Methods in order: least square model,  perceptron algorithm

def plot_0(w):
    x_=[-1,1]
    y_=[(0.5-w[2]-w[0]*x_[0])/w[1],(0.5-w[2]-w[0]*x_[1])/w[1]]
    plt.plot(x_,y_)
    plt.show()

def augment(X):
    X=X.T
    newX=np.append(X,[np.ones(len(X.T))],axis=0)
    return newX.T

# Least square model
def LS(X,y): #train dataset X, y
    X=X.T
    X_=np.linalg.inv(np.dot(X,X.T))
    # print(X_)
    X_X=np.dot(X_,X)
    # print(X_X)
    return np.dot(X_X,y+0)


def main_1():
    dataset=get_linear_seperatable_2d_2c_dataset()
    dataset.plot(plt)

    newX=augment(dataset.X)
    y=dataset.y

    #compute
    w=LS(newX,y)
    print(w)

    #show
    plot_0(w)
    return w

# main_1()


#=================================================#
#                      TASK2                      #
#=================================================#
# Perceptron Algorithm
def perceptron(X,y):
    sz=len(X)
    newY=[]
    for i in range(0,sz):
        if y[i]==True:
            newY.append(1)
        else:
            newY.append(-1)
    
    w=np.zeros(3)
    # print(newY)
    for i in range(0,500):
        # random here
        order=np.random.permutation(sz)
        for j in order:
            x_=X[j]
            y_=newY[j]
            # print(x_)
            # print(w)
            # print(y_)
            w_x=np.dot(x_,w)
            # print(w_x)
            if w_x*y_<=0:
                # print(1)
                w=w+y_*x_
    return w            

def plot_1(w):
    x_=[-1,1]
    y_=[(-w[2]-w[0]*x_[0])/w[1],(-w[2]-w[0]*x_[1])/w[1]]
    plt.plot(x_,y_)
    plt.show()            

def main_2():
    dataset=get_linear_seperatable_2d_2c_dataset()
    dataset.plot(plt)

    newX=augment(dataset.X)
    y=dataset.y

    #compute
    w=perceptron(newX,y)
    # print(w)

    #show
    print(w)
    plot_1(w)
    return w

# main_2()



#=================================================#
#                      TASK3                      #
#=================================================#
# Part two
# Text classification based on logistic regression.

lamda=1
alpha=0.1

# softmax to onehot
def get_onehot(W_c,X): #W_c=d*c X=d*N
    ori=np.exp(np.dot(W_c.T,X)) #ori=c*N
    ori=ori.T
    argmax = np.argmax(ori, axis=1)
    one_hot = np.zeros((len(argmax), ori.shape[1]))
    one_hot [np.arange(len(argmax)), argmax] = 1
    # print(one_hot)
    return one_hot.T #one_hot=c*N

def cross_entropy_gradient(W_c,X,Y): #W_c=(d+1)*c X=(d+1)*N Y=c*N when usual train
    #W_c=(d+1)*c x=(d+1)*1 y=c*1 when stochastic gradient descent train
    Y_tilta=get_onehot(W_c,X) #Y_tilta=c*N
    Y_sub=(Y-Y_tilta).T
    return np.dot(X,Y_sub) #X_Y=d*c

def normal_gradient_descent_train(X,Y,T):
    W_c=np.zeros((X.shape[0],Y.shape[0]))
    step_error=[]
    N=X.shape[1]
    for i in range(0,T):
        W_c=W_c*(1-alpha*lamda/N)+alpha*cross_entropy_gradient(W_c,X,Y)/N
        step_error.append(test(W_c,X,Y))
    return W_c,step_error

def stochastic_gradient_descent_train(X,Y,T): #W_c=(d+1)*c X=(d+1)*N Y=c*N when usual train
    W_c=np.zeros((X.shape[0],Y.shape[0]))
    N=X.shape[1]
    step_error=[]
    for i in range(0,T):
        order=np.random.permutation(N)
        for j in order:
            x=(np.array([X[:,j]])).T
            y=(np.array([Y[:,j]])).T
            W_c=W_c*(1-alpha*lamda/N)+alpha*cross_entropy_gradient(W_c,x,y)
            step_error.append(test(W_c,X,Y))
    return W_c,step_error

def batched_gradient_descent_train(X,Y,k): #k marks the num of train data
    W_c=np.zeros((X.shape[0],Y.shape[0]))
    T=10000
    N=X.shape[1]
    for i in range(0,T):
        order=np.random.randint(0,N)
        X_cmp=X[:,order:min(order+k,N-1)] #X_cmp=(d+1)*k
        Y_cmp=Y[:,order:min(order+k,N-1)]
        W_c=W_c*(1-alpha*lamda/N)+alpha*cross_entropy_gradient(W_c,X_cmp,Y_cmp)/N
    return W_c

# text preprocess
def remove_punctuation(str):
    str=str.lower()
    for c in string.punctuation:
        str=str.replace(c,' ')
    return str

def remove_stopwords(list):
    en_stops = set(stopwords.words('english'))
    # print(en_stops)
    new_list=[]
    for word in list: 
        if word not in en_stops:
            new_list.append(word)
    return new_list

# input: dictionary, a list of words
def build_one_hot(dictionary,words):
    # pre
    remove_dup = sorted(set(words))
    # onehot array
    one_hot= np.zeros(len(dictionary))
    for word in remove_dup:
        if word in dictionary:
            one_hot[dictionary[word]]=1
    return one_hot

# return dictionary and a list of built one-hot vectors
def build_dict(list_of_papers):
    words_all=[] # collect of all words
    papers=[] # collect a list of lists of words
    for paper in list_of_papers:
        # pre-
        words_exist=remove_punctuation(paper)
        words_list=words_exist.split()
        words_list=remove_stopwords(words_list)
        # add into the all
        papers.append(words_list) 
        words_all+=words_list
    # build dic
    remove_dup = sorted(set(words_all))
    dictionary = collections.Counter(words_all)
    i=0
    for word in remove_dup:
        if dictionary[word]<10:
            del dictionary[word]
        else:
            dictionary[word]=i
            i+=1
    # build one hot
    one_hot_collections=np.array([np.zeros(len(dictionary))])
    for words_list in papers:
        one_hot_collections=np.row_stack((one_hot_collections,build_one_hot(dictionary,words_list)))
    return np.delete(one_hot_collections,0,0), dictionary

# turn target into one-hot
def build_one_hot_target(list_of_target):
    sz = len(set(list_of_target))
    target=np.zeros((len(list_of_target),sz))
    for i in range(len(list_of_target)):
        target[i][list_of_target[i]]=1
    return target

def test(W_c,X,Y):
    result=get_onehot(W_c,X)
    ct=0
    for i in range(0,X.shape[1]):
        if (result[:,i]==Y[:,i]).all():
            ct=ct+1
            # print('True')
        # else:
        #     print("False!")
    return (float(ct)/float(X.shape[1]))

def build_env():
    dataset_train,dataset_test=get_text_classification_datasets()
    target=build_one_hot_target(dataset_train.target)
    # # C=len(dataset_train.categories)
    one_hot_collections,dictionary=build_dict(dataset_train.data)
    X=augment(one_hot_collections).T
    Y=target.T
    np.savetxt('X.txt',X)
    np.savetxt('Y.txt',Y)
    # return X,Y

def main_3(model,T):
    X=np.loadtxt('X.txt')
    Y=np.loadtxt('Y.txt')

    if model==1:
        W_c,step_error=normal_gradient_descent_train(X,Y,T)
    elif model==2:
        W_c,step_error=stochastic_gradient_descent_train(X,Y,T)  
    else:
        k=3
        W_c=batched_gradient_descent_train(X,Y,k)
    # print(W_c)

    plt.plot(range(0,len(step_error)),step_error)
    plt.show()
    # # test here, and with nice answer
    # docs_toy = ["Hi!How are you?","Do you have a dog?"]
    # one_hot_collections,dictionary=build_dict(docs_toy)
    # print(one_hot_collections)


# main_3(2)


# TODO list 
# write down how to comput the gradient
# L2 regularization, should regularize the bias term?
# how to check the correctness
# plot loss curve
# how to decide the learning rate
# when to determinate the train
# compare three descent methods, about pros and cons
# report the three different model