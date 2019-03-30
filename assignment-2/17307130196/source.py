# In this assignment, I will basically finish an assignment devided into two part.
# The first part focuses on the linear classification model
# while the second one is a small task on text classification

# import area
import matplotlib.pyplot as plt
import numpy as np
import math as mt
import string
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
import collections
nltk.download('stopwords')
from __init__ import *
# os.sys.path.append("..")
# use the above line of code to surpass the top module barrier
# from handout import *
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

# Part two
# Text classification based on logistic regression.

#main code here
def softmax(w,x,bottom): #count the p(y=c|C=i)
    divide=mt.exp(w.T*x)/bottom
    return divide

def logistic(W_c,x):
    result=[]
    bottom=0

    for w in W_c:
        bottom=bottom+mt.exp(w.T*x)
    
    for w in W_c:
        result.append(softmax(w,x,bottom))
    return np.argmax(result)

#degree
# W_t, traned parameters
# alpha, learned speed
# N, num of data
# X, the i-th sample
# Y, output of the model now
def integrate(W_t,alpha,N,X,Y):
    sum=0
    for i in range(0,N):
        sum=sum+X[i]*(Y[i]-logistic(W_t,X[i])).T
    W_t=W_t+alpha/N*sum
    return W_t


#train progress
def train(train_data, target):
    sz=len(train_data)
    W_0=np.zeros(sz)
    turns=400
    for i in range(0,turns):
        W_0=integrate(W_0,0.01,sz,train_data,target)
    return W_0


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
        # words_list=remove_stopwords(words_list)
        # add into the all
        papers.append(words_list) 
        words_all+=words_list
    # build dic
    remove_dup = sorted(set(words_all))
    dictionary = collections.Counter(words_all)
    i=0
    for word in remove_dup:
        if dictionary[word]<0:
            del dictionary[word]
        else:
            dictionary[word]=i
            i+=1
    # build one hot
    one_hot_collections=np.array([np.zeros(len(dictionary))])
    for words_list in papers:
        one_hot_collections=np.row_stack((one_hot_collections,build_one_hot(dictionary,words_list)))
    return np.delete(one_hot_collections,0,0), dictionary


def main_3():
    dataset_train,dataset_test=get_text_classification_datasets()
    target=dataset_train.target
    
    one_hot_collections,dictionary=build_dict(dataset_train.data)
    
    # # test here, and with nice answer
    # docs_toy = ["Hi!How are you?","Do you have a dog?"]
    # one_hot_collections,dictionary=build_dict(docs_toy)
    # print(one_hot_collections)
    
    


main_3()