import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt
import math as mt
import scipy
import Nearest_Neighbor_Method as NNM
N=500
def M_0(K):
    sample_data=get_data(N)
    sample_data.sort()
    # print(sample_data)
    # print(len(sample_data))
    def NNM_new(x):
        return (NNM.KNN_Pro(x,sample_data,N-1,K))**2
    
    the_first=scipy.integrate.quad(NNM_new,20,40)
    the_second=0
    flag=0

    for x in sample_data:
        tp_sample_data=sample_data[:]
        del tp_sample_data[flag]
        flag=flag+1
        the_second=the_second+NNM.KNN_Pro(x,tp_sample_data,N-1,K)
        # print(x)
    the_second=the_second*2/N

    M0=the_first[0]-the_second
    return M0