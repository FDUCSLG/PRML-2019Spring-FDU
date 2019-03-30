# In this assignment, I will basically finish an assignment devided into two part.
# The first part focuses on the linear classification model
# while the second one is a small task on text classification

# import area
import matplotlib.pyplot as plt
import numpy as np
import math as mt;

import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import *
# Part one
# two Linear Classification Model based on a given dataset
# And discuss the accuracy and ways to improve them
# Methods in order: least square model,  perceptron algorithm

# Least square model
def LS(X,y): #train dataset X, y
    X=X.T
    X_=np.linalg.inv(np.dot(X,X.T))
    print(X_)
    X_X=np.dot(X_,X)
    print(X_X)
    return np.dot(X_X,y+0)

dataset=get_linear_seperatable_2d_2c_dataset()

dataset.plot(plt)
# plt.show()
print(dataset)

def get_w(dataset):
    w=LS(dataset.X,dataset.y)
    return w

w=get_w(dataset)

print(w)

# Perceptron Algorithm



# Part two
# Text classification based on logistic regression.