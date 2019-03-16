#FINAL GOAL: Use Nonparametric Methods to get probability distribution

#This is the first task of Assignment 1
#In this file, I will try to figure out what will happen with different numbers of samples.

#Histogram Method, Kernel Method and the Nearest Neighbor Method will be showed here.
#-------------
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

def Hist(N):
    sampled_data = get_data(N)
    plt.hist(sampled_data, normed=True, bins=50)
    plt.show()

#Hist(10000)

#----------------------------------
#Kernel Method
#Give a window of every point of x, and we can get the neighbor area
#count the number of points lay in such area we get the probability densuty
#However, the choice of counts of the points varies
#Cosider a simple situation. We simply count the number in the region.
#For the problem 1, we will fix the h, which will be more carefully dicussed in the later work.
h=1
N=10000
h_2=h**2
para=1/float(N)*mt.sqrt(2*mt.pi*h_2)

def KernelGaussian(target,dataset):
    sum=0
    for x in dataset:
        #print(x)
        sum=sum+mt.exp(-(target-x)**2/2*h_2)    
    sum=sum*para
    ##print(sum)
    return sum

def KernelPlot(N):
    output_data=[]
    sampled_data = get_data(N)
    # for x in np.linspace(0,1-h,h):
    #     print(1)
    #     output_data.append(KernelGaussian(x,sampled_data))
    for i in range(20,40-h,h):
        output_data.append(KernelGaussian(i*h,sampled_data))
    plt.plot(np.linspace(20,40-h,(20-h)/h),output_data)
    plt.show()

#KernelPlot(N) 

#--------------------------
#Nearest Neighbor Method
#Different from Kernel Method, we choose to fix K and varies V, and simply uses the function:
#p(x)=K/(V*N)
K=60
num=200 #the length of the target
def NNM_Pro(target,dataset_ordered,N):
    flag_data=0
    while flag_data<N and dataset_ordered[flag_data]<=target:
        # print(flag_data)
        flag_data=flag_data+1
        #In this way, we get the first point larger than x
        # and the flag_data marks the first number larger than target
    ct=0 #ct marks the number of numbers met
    flag_l=max(flag_data-1,0)
    flag_r=min(flag_data,N-1)
    while ct<K:
        if flag_l>0 and flag_r<N-1:
            if target-dataset_ordered[flag_l]<dataset_ordered[flag_r]-target:
                flag_l=flag_l-1
            else:
                flag_r=flag_r+1
        else:
            if flag_l==0 and flag_r<N-1:
                flag_r=flag_r+1
            elif flag_r==N-1 and flag_l>0:
                flag_l=flag_l-1
        ct=ct+1
    # print(flag_l)
    # print(flag_r)
    V=dataset_ordered[flag_r]-dataset_ordered[flag_l]
    # return  V#get the volume of the box
    return float(K)/(float(V)*N)

def NNM(N):
    sampled_data = get_data(N)
    sampled_data.sort()
    output=[]
    for x in np.linspace(20,40,num):
        output.append(NNM_Pro(x,sampled_data,N))
    plt.plot(np.linspace(20,40,num),output)
    plt.show()

NNM(N)


