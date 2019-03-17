import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt
import math as mt
import scipy
import Kernel_Method as KDE
def M_0(h):
    sample_data=get_data()
    N=100
    h_2=h**2
    para=1/(float(N)*mt.sqrt(2*mt.pi*h_2))
    para_i=1/(float(N-1)*mt.sqrt(2*mt.pi*h_2))
    def KDE_new(x):
        return (KDE.KernelGaussian(x,sample_data,h_2,para))**2
    the_first=scipy.integrate.quad(KDE_new,20,40)
    the_second=0
    flag=0

    for x in sample_data:
        tp_sample_data=sample_data[:]
        del tp_sample_data[flag]
        flag=flag+1
        the_second=the_second+KDE.KernelGaussian(x,tp_sample_data,h_2,para_i)
    the_second=the_second*2/N

    M0=the_first[0]-the_second
    return M0