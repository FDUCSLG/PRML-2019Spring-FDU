#Set the Delta as 0.08 at first
import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt
from handout import GaussianMixture1D
import math as mt

### Histogram Method
#First the Histogram Method
#Simply partition x into bins of width Delta_i
#and count the number in the bin as n_i. And 
# P_i=n_i/(N*Delta_i)
# And we shoose the same width at the beginning.
np.random.seed(0)
#--------------
#parameters
#N=10000
Delta=0.08
#--------------
# sampled_data = get_data(N)
# plt.hist(sampled_data, normed=True, bins=50)
# plt.show()
def Hist(N,bins_in):
    sampled_data = get_data(N)
    plt.hist(sampled_data, normed=True, bins=bins_in)
    gm1d = GaussianMixture1D(mode_range=(0, 50))
    gm1d.plot(200)
    plt.show()

#Hist(10000,0.08)


###KDE
np.random.seed(0)

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
np.random.seed(0)
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
    for x in np.linspace(0,50,num):
        output_data.append(KernelGaussian(x,sampled_data,h_2,para))
    plt.plot(np.linspace(0,50,num),output_data)
    gm1d = GaussianMixture1D(mode_range=(0, 50))
    gm1d.plot(200)
    plt.show()
    return output_data

# Kernel(5000,10000,0.08)


##k-NN
np.random.seed(0)
#--------------------------
#Nearest Neighbor Method
#Different from Kernel Method, we choose to fix K and varies V, and simply uses the function:
#p(x)=K/(V*N)
def KNN_Pro(target,dataset_ordered,N,K):
    flag_data=0
    # print(N)
    # print(len(dataset_ordered))
    while flag_data<N and dataset_ordered[flag_data]<=target:
        flag_data=flag_data+1
        # print(flag_data)
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
    # print(V)
    return float(K)/(float(V)*N)

def KNN(num,N,K):
    sampled_data = get_data(N)
    sampled_data.sort()
    output=[]
    for x in np.linspace(20,40,num):
        output.append(KNN_Pro(x,sampled_data,N,K))
    plt.plot(np.linspace(20,40,num),output)
    gm1d = GaussianMixture1D(mode_range=(0, 50))
    gm1d.plot(200)
    plt.show()
    

#KNN(10000,10000,50)

###main
def main(c,num,bins,K,H,N):  
    if(c=='H'): Histogram_Method.Hist(N,bins)
    elif(c=='K'): Kernel_Method.Kernel(num,N,H)
    else: Nearest_Neighbor_Method.KNN(num,N,K)
    

# bins is the number of bins in Histogram Method
# H is the parameter in Kernel Method of the volume
# K is the key parameter in Nearest Neighbor Method
# num is the number of test in Kernel and Nearest Neighbor Method

###Requirement-1
import main
print('H:')
#num,K,bins,K,h,N
# main.main('H',1000,80,5,0.8,200)
# main.main('H',1000,50,20,0.8,500)
# main.main('H',1000,50,20,0.8,1000)
# main.main('H',1000,50,20,0.8,10000)

# print('K:')
# main.main('K',1000,50,20,0.275,200)
# main.main('K',1000,50,20,0.6,500)
# main.main('K',1000,50,20,0.8,1000)
# main.main('K',1000,50,20,0.08,100)
# main.main('K',1000,50,20,0.08,10000)

# print('N:')
main.main('N',1000,50,20,0.8,100)
# main.main('N',1000,50,20,0.8,500)
# main.main('N',1000,50,20,0.8,1000)
# main.main('N',1000,50,20,0.8,10000)

