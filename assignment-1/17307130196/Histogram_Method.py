#First the Histogram Method
#Simply partition x into bins of width Delta_i
#and count the number in the bin as n_i. And 
# P_i=n_i/(N*Delta_i)
# And we shoose the same width at the beginning.
# TODO: Consider a situation where the width is not the same.
import math as mt
#Set the Delta as 0.08 at first
import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt
#--------------
#parameters
#N=10000
Delta=0.08
#--------------
# sampled_data = get_data(N)
# plt.hist(sampled_data, normed=True, bins=50)
# plt.show()

def Hist(N,Delta):
    sampled_data = get_data(N)
    plt.hist(sampled_data, normed=True, bins=50)
    plt.show()

Hist(10000,0.08)