import math as mt
#Set the Delta as 0.08 at first
import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt
#----------------------------------
#Kernel Method
#Give a window of every point of x, and we can get the neighbor area
#count the number of points lay in such area we get the probability densuty
#However, the choice of counts of the points varies
#Cosider a simple situation. We simply count the number in the region.
#For the problem 1, we will fix the h, which will be more carefully dicussed in the later work.

#parameter and constant list

#functions:
#compute the probability density at target with sample_data as dataset
def KernelGaussian(target,dataset,h_2,para):
    sum=0
    for x in dataset:
        #print(x)
        sum=sum+mt.exp(-(target-x)**2/(2*h_2))    
    sum=sum*para
    ##print(sum)
    return sum

# Main function
# the only input here is the number of dataset
def Kernel(num,N,h):
    output_data=[]
    h_2=h**2
    para=1/(float(N)*mt.sqrt(2*mt.pi*h_2))
    sampled_data = get_data(N)
    # for x in np.linspace(0,1-h,h):
    #     print(1)
    #     output_data.append(KernelGaussian(x,sampled_data))
    for x in np.linspace(20,40,num):
        output_data.append(KernelGaussian(x,sampled_data,h_2,para))
    plt.plot(np.linspace(20,40,num),output_data)
    plt.show()

Kernel(5000,10000,0.08)
