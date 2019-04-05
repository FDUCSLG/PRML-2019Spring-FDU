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
# from __init__ import *
os.sys.path.append("..")
# use the above line of code to surpass the top module barrier
from handout import *

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
def test_perceptron(W,X,y):
    predict=np.dot(W.T,X.T)
    ct=0
    for i in range(0,len(y)):
        if (predict[i]>=0 and y[i]==-1) or (predict[i]<=0 and y[i]==1):
            ct+=1
    return ct/len(y)

def perceptron(X,y):
    sz=len(X)
    newY=[]
    for i in range(0,sz):
        if y[i]==True:
            newY.append(1)
        else:
            newY.append(-1)
    acc=[]
    w=np.zeros(3)
    # print(newY)
    for i in range(0,2):
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
                w=w+1*y_*x_
            acc.append(test_perceptron(w,X,y))
    # return w
    return acc            

def plot_1(w):
    x_=[-1,1]
    y_=[(-w[2]-w[0]*x_[0])/w[1],(-w[2]-w[0]*x_[1])/w[1]]
    plt.plot(x_,y_)
    plt.show()            

def main_2():
    dataset=get_linear_seperatable_2d_2c_dataset()
    # dataset.plot(plt)

    newX=augment(dataset.X)
    y=dataset.y

    #compute
    w=perceptron(newX,y)

    #show
    print(w)
    # plot_1(w)
    plt.plot(range(0,len(w)),w)
    plt.show()
    return w

# main_2()



#=================================================#
#                      TASK3                      #
#=================================================#
# Part two
# Text classification based on logistic regression.

# softmax to onehot
def get_onehot(W_c,X): #W_c=d*c X=d*N
    ori=np.exp(np.dot(W_c.T,X)) #ori=c*N
    ori=ori.T
    argmax = np.argmax(ori, axis=1)
    one_hot = np.zeros((len(argmax), ori.shape[1]))
    one_hot [np.arange(len(argmax)), argmax] = 1
    # print(one_hot)
    return one_hot.T #one_hot=c*N

def softmax(W_c,X): #X=(d+1)*N
    z=np.dot(W_c.T,X) # z = c*N
    z=z-np.max(z,axis=0)
    up=np.exp(z) # up = c*N
    down=np.sum(up,axis=0) # down = 1*N
    softmax_error=(np.array([up[:,i]/down[i] for i in range(0,X.shape[1])])).T
    return softmax_error

def cross_entropy_gradient(W_c,X,Y): #W_c=(d+1)*c X=(d+1)*N Y=c*N when usual train
    Y_tilta=softmax(W_c,X) #Y_tilta=c*N
    Y_sub=(Y-Y_tilta).T
    return np.dot(X,Y_sub) #X_Y=d*c

def loss_funct(W,X,Y):
    soft_error=softmax(W,X)
    # soft_error=np.maximum(0.0000001,soft_error)
    error=0
    for i in range(0,X.shape[1]):
        error+=np.dot(Y[:,i],np.log(soft_error[:,i]))
        # print(error)
    error=-error/X.shape[1]
    return error

def gradient_descent(model,X,Y,flag,lamda,epsilon,alpha,k):
    W_c=np.zeros((X.shape[0],Y.shape[0]))
    T=0
    step_error=[]
    loss_error=[]
    N=X.shape[1]
    last_W_c=np.zeros((X.shape[0],Y.shape[0]))
    while 1:
        T+=1
        print("You are now at epoch %d" %T)
        if model==1: #normal
            W_c=W_c*(1-alpha*lamda/N)+alpha*cross_entropy_gradient(W_c,X,Y)/N
        elif model==2: #stochastic
            order=np.random.permutation(N)
            for j in order:
                x=(np.array([X[:,j]])).T
                y=(np.array([Y[:,j]])).T
                W_c=W_c*(1-alpha*lamda)+alpha*cross_entropy_gradient(W_c,x,y)
            # print(W_c)            
        else: #batched
            x=0
            X_=X.T
            Y_=Y.T
            while x+k<N:
                # print(x+k)
                X_cmp=X_[x:x+k] #X_cmp=(d+1)*k
                Y_cmp=Y_[x:x+k]
                X_cmp=X_cmp.T
                Y_cmp=Y_cmp.T
                W_c=W_c*(1-alpha*lamda/k)+alpha*cross_entropy_gradient(W_c,X_cmp,Y_cmp)/k
                x+=k    
            # print(W_c)
        gradient=loss_funct(W_c,X,Y)
        if flag==1:
            print("The loss is %f" %gradient)
            loss_error.append(gradient)
        elif flag==2:
            acc=test(W_c,X,Y)
            print("The acc is %f" %acc)
            step_error.append(acc)
        if  T>100 or np.sum((W_c-last_W_c)**2)<epsilon:
            break
        else:
            last_W_c=W_c
    return W_c,step_error,loss_error,T

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

def build_testset(list_of_papers,dictionary):
    papers=[] # collect a list of lists of words
    for paper in list_of_papers:
        # pre-
        words_exist=remove_punctuation(paper)
        words_list=words_exist.split()
        words_list=remove_stopwords(words_list)
        # add into the all
        papers.append(words_list) 
    one_hot_collections=np.array([np.zeros(len(dictionary))])
    for words_list in papers:
        one_hot_collections=np.row_stack((one_hot_collections,build_one_hot(dictionary,words_list)))
    return np.delete(one_hot_collections,0,0)

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
    return (float(ct)/float(X.shape[1]))

def build_env():
    dataset_train,dataset_test=get_text_classification_datasets()
    target_train=build_one_hot_target(dataset_train.target)
    traget_test=build_one_hot_target(dataset_test.target)
    # # C=len(dataset_train.categories)
    one_hot_collections,dictionary=build_dict(dataset_train.data)
    testset_collections=build_testset(dataset_test.data,dictionary)
    X=augment(one_hot_collections).T
    X_=augment(testset_collections).T
    Y=target_train.T
    np.savetxt('X.txt',X)
    np.savetxt('Y.txt',Y)
    np.savetxt('testY.txt',traget_test.T)
    np.savetxt('testX.txt',X_)
    # return X,Y

def main_3(model=1,flag=0,lamda=0.1,epsilon=0,alpha=0.01,k=30):
    X=np.loadtxt('X.txt')
    Y=np.loadtxt('Y.txt')
    W_c,step_error,loss_error,T=gradient_descent(model,X,Y,flag,lamda,epsilon,alpha,k)
    print("You have run %d epochs" %T)
    # print(W_c)
    print(test(W_c,X,Y))
    if flag==1:
        plt.plot(range(0,len(loss_error)),loss_error)
    elif flag==2:
        plt.plot(range(0,len(step_error)),step_error)
    else:
        pass
    # plt.plot(range(0,len(loss_error)),loss_error)
    plt.show()
    Xtest=np.loadtxt('testX.txt')
    Ytest=np.loadtxt('testY.txt')
    print(test(W_c,Xtest,Ytest))
    # # test here, and with nice answer
    # docs_toy = ["Hi!How are you?","Do you have a dog?"]
    # one_hot_collections,dictionary=build_dict(docs_toy)
    # print(one_hot_collections)

def plot_on_alpha_of_loss(model=1,flag=1,lamda=0.001,epsilon=0,k=30):
    X=np.loadtxt('X.txt')
    Y=np.loadtxt('Y.txt')
    # print(W_c)
    # print(test(W_c,X,Y))
    W_c,step_error,loss_error,T=gradient_descent(model,X,Y,flag,lamda,epsilon,0.7,3)
    plt.plot(range(0,len(loss_error)),loss_error,color='red',label='3')
    print('You reach convergence at epoch %d' %T)
    W_c,step_error,loss_error,T=gradient_descent(model,X,Y,flag,lamda,epsilon,0.7,10)
    plt.plot(range(0,len(loss_error)),loss_error,color='orange',label='10')
    print('You reach convergence at epoch %d' %T)
    W_c,step_error,loss_error,T=gradient_descent(model,X,Y,flag,lamda,epsilon,0.7,30)
    plt.plot(range(0,len(loss_error)),loss_error,color='blue',label='30')
    print('You reach convergence at epoch %d' %T)
    W_c,step_error,loss_error,T=gradient_descent(model,X,Y,flag,lamda,epsilon,0.7,50)
    plt.plot(range(0,len(loss_error)),loss_error,color='green',label='50')
    print('You reach convergence at epoch %d' %T)
    W_c,step_error,loss_error,T=gradient_descent(model,X,Y,flag,lamda,epsilon,0.7,70)
    plt.plot(range(0,len(loss_error)),loss_error,color='red',label='70',linestyle='--')
    print('You reach convergence at epoch %d' %T)
    W_c,step_error,loss_error,T=gradient_descent(model,X,Y,flag,lamda,epsilon,0.7,100)
    plt.plot(range(0,len(loss_error)),loss_error,color='orange',label='100',linestyle='--')
    print('You reach convergence at epoch %d' %T)
    plt.legend(loc='best')
    # plt.plot(range(0,len()),loss_error)
    plt.show()
# main_3(2,2)

def plot_on_alpha_of_acc(model=1,flag=2,lamda=300,epsilon=1,k=30):
    X=np.loadtxt('X.txt')
    Y=np.loadtxt('Y.txt')
    # print(W_c)
    # print(test(W_c,X,Y))
    W_c,step_error,loss_error,T=gradient_descent(model,X,Y,flag,lamda,epsilon,0.001,k)
    plt.plot(range(0,len(step_error)),step_error,color='red',label='0.001')
    W_c,step_error,loss_error,T=gradient_descent(model,X,Y,flag,lamda,epsilon,0.01,k)
    plt.plot(range(0,len(step_error)),step_error,color='orange',label='0.01')
    W_c,step_error,loss_error,T=gradient_descent(model,X,Y,flag,lamda,epsilon,0.1,k)
    plt.plot(range(0,len(step_error)),step_error,color='blue',label='0.1')
    W_c,step_error,loss_error,T=gradient_descent(model,X,Y,flag,lamda,epsilon,0.2,k)
    plt.plot(range(0,len(step_error)),step_error,color='green',label='0.2')
    W_c,step_error,loss_error,T=gradient_descent(model,X,Y,flag,lamda,epsilon,0.5,k)
    plt.plot(range(0,len(step_error)),step_error,color='red',label='0.5',linestyle='--')
    W_c,step_error,loss_error,T=gradient_descent(model,X,Y,flag,lamda,epsilon,0.7,k)
    plt.plot(range(0,len(step_error)),step_error,color='orange',label='0.7',linestyle='--')
    W_c,step_error,loss_error,T=gradient_descent(model,X,Y,flag,lamda,epsilon,1,k)
    plt.plot(range(0,len(step_error)),step_error,color='blue',label='1', linestyle='--')
    W_c,step_error,loss_error,T=gradient_descent(model,X,Y,flag,lamda,epsilon,1.5,k)
    plt.plot(range(0,len(step_error)),step_error,color='green',label='1.5',linestyle='--')
    plt.legend(loc='best')
    # plt.plot(range(0,len()),loss_error)
    plt.show()


def three_models(flag=1,lamda=0.01,epsilon=0.0001,alpha=0.7,k=30):
    X=np.loadtxt('X.txt')
    Y=np.loadtxt('Y.txt')
    W_c,step_error,loss_error,T=gradient_descent(1,X,Y,flag,lamda,epsilon,alpha,k)
    plt.plot(range(0,len(loss_error)),loss_error,color='red',label='normal')
    W_c,step_error,loss_error,T=gradient_descent(2,X,Y,flag,lamda,epsilon,alpha,k)
    plt.plot(range(0,len(loss_error)),loss_error,color='blue',label='stochastic')
    W_c,step_error,loss_error,T=gradient_descent(3,X,Y,flag,lamda,epsilon,alpha,k)
    plt.plot(range(0,len(loss_error)),loss_error,color='green',label='batched=30')
    plt.legend(loc='best')
    plt.show()
