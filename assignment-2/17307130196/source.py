# In this assignment, I will basically finish an assignment devided into two part.
# The first part focuses on the linear classification model
# while the second one is a small task on text classification

# import area
import matplotlib.pyplot as plt
import numpy as np
import math as mt

import os

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

main_2()

# Part two
# Text classification based on logistic regression.