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
from handout import get_linear_seperatable_2d_2c_dataset
# Part one
# two Linear Classification Model based on a given dataset
# And discuss the accuracy and ways to improve them
# Methods in order: least square model,  perceptron algorithm

# Least square model
def LS(X,y): #train dataset X, y
    X_=np.linalg.inv(X*X.T)
    return X_*X*y



# Perceptron Algorithm



# Part two
# Text classification based on logistic regression.