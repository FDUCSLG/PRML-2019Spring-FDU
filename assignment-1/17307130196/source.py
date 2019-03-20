#Set the Delta as 0.08 at first
import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt
from handout import GaussianMixture1D
import math as mt
import scipy
import random
### Histogram Method
#First the Histogram Method
#Simply partition x into bins of width Delta_i
#and count the number in the bin as n_i. And 
# P_i=n_i/(N*Delta_i)
# And we shoose the same width at the beginning.
#--------------
#parameters
#--------------
# sampled_data = get_data(N)
# plt.hist(sampled_data, normed=True, bins=50)
# plt.show()
def Hist(N,bins_in):
    np.random.seed(0)
    sampled_data = get_data(N)
    plt.hist(sampled_data, normed=True, bins=bins_in)
    gm1d = GaussianMixture1D(mode_range=(0, 50))
    gm1d.plot(200)
    plt.show()

#Hist(10000,0.08)


###KDE
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
    np.random.seed(0)
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
    np.random.seed(0)
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
    if(c=='H'): Hist(N,bins)
    elif(c=='K'): Kernel(num,N,H)
    else: KNN(num,N,K)
    

# bins is the number of bins in Histogram Method
# H is the parameter in Kernel Method of the volume
# K is the key parameter in Nearest Neighbor Method
# num is the number of test in Kernel and Nearest Neighbor Method


# ass functions
# Simple CV for Histogram
def sum_of_squire(l1,l2,N):
    sum=0
    for i in range(0,N):
        sum=sum+(l1[i]-l2[i])**2
    return sum

def Hist_new(N,dataset,bins): #get the prediction estimation
    size=20/bins
    ans=[]
    i=0
    j=0 #marks the bin_id
    while i<N-1 and j<bins:
        tp=0
        while i<N-1 and dataset[i]<=j*size+20:
            tp=tp+1
            i=i+1
        ans.append(tp/N)
        # print(tp/N)
        j=j+1
    return ans

def simple_CV(bins,N): #return the result of CV test
    test_num=int(N/2)
    sampled_data = get_data(N)
    sampled_data_exchange=sampled_data[test_num:N]
    sampled_data_exchange.sort()
    size=20/float(bins)
    
    pred_distribution=Hist_new(N-test_num,sampled_data_exchange,bins)
    # print(pred_distribution)
    # plt.plot(range(0,len(pred_distribution)),pred_distribution)
    # plt.show()
    pred=[]
    for i in range(0,test_num):
        s=random.random()
        j=0
        while s>0 and j<len(pred_distribution):
            s=s-pred_distribution[j]
            j=j+1
        pred.append(j*size+20)
        # print(pred)
    return sum_of_squire(pred,sampled_data[0:len(pred)],test_num)

def simple_CV_test(bins,N,i):
    random.seed(i)
    test_num=int(N/2)
    sampled_data = get_data(N)
    sampled_data_exchange_1=sampled_data[0:int(N/2)]
    sampled_data_exchange_1.sort()
    sampled_data_exchange_2=sampled_data[int(N/2):N]
    sampled_data_exchange_2.sort()
    size=20/float(bins)
    pred_distribution_1=Hist_new(int(N/2),sampled_data_exchange_1,bins)
    pred_distribution_2=Hist_new(int(N/2),sampled_data_exchange_2,bins)
    return sum_of_squire(pred_distribution_1,pred_distribution_2,len(pred_distribution_1))

def M_HIS(bins):
    N=200
    sample_data=get_data(N)
    def SCV_new(x):
        return (simple_CV(x,sample_data,h_2,para))**2
#compute M-0 for KDE

def M_KDE(h):
    sample_data=get_data()
    N=100
    h_2=h**2
    para=1/(float(N)*mt.sqrt(2*mt.pi*h_2))
    para_i=1/(float(N-1)*mt.sqrt(2*mt.pi*h_2))
    def KDE_new(x):
        return (KernelGaussian(x,sample_data,h_2,para))**2
    the_first=scipy.integrate.quad(KDE_new,20,40)
    the_second=0
    flag=0

    for x in sample_data:
        tp_sample_data=sample_data[:]
        del tp_sample_data[flag]
        flag=flag+1
        the_second=the_second+KernelGaussian(x,tp_sample_data,h_2,para_i)
    the_second=the_second*2/N

    M0=the_first[0]-the_second
    return M0

#compute M_0 for k-NN
def M_k_NN(K):
    N=500
    sample_data=get_data(N)
    sample_data.sort()
    def NNM_new(x):
        return (KNN_Pro(x,sample_data,N-1,K))**2
    
    the_first=scipy.integrate.quad(NNM_new,20,40)
    the_second=0
    flag=0

    for x in sample_data:
        tp_sample_data=sample_data[:]
        del tp_sample_data[flag]
        flag=flag+1
        the_second=the_second+KNN_Pro(x,tp_sample_data,N-1,K)
        # print(x)
    the_second=the_second*2/N

    M0=the_first[0]-the_second
    return M0

#compute the best bins
def best_bins_simple_cv_error():
    ans=[]
    for i in range(1,200):
        sum=0
        for j in range(1,2):
            sum=sum+simple_CV(i,200)
        # sum=sum/30
        ans.append(sum)
    plt.plot(range(1,200),ans)
    plt.show()

def best_bins_simple_cv():
    ans=[]
    for i in range(1,200):
        sum=0
        for j in range(1,30):
            sum=sum+simple_CV_test(i,200,j)
        sum=sum/30
        ans.append(sum)
    plt.plot(range(1,200),ans)
    plt.show()

#compute the best h
def best_h():
    ans=[]
    for i in range(1,20):
        ans.append(M_KDE(i*0.05))

    plt.plot([i*0.5 for i in range(1,20)],ans)
    plt.show()

#compute best K
def best_K():
    ans=[]
    for i in range(2,50):
        ans.append(M_k_NN(i))

    plt.plot(range(2,50),ans)
    plt.show()



#----------------------------
#----------------------------
# Requirements starts now
##  REQUIREMENT 1
def requirement_1():
    c=input("Please input a char: H for Hist, K for KDE and N for k-NN:")
    if c== 'H':
        print('Histogram:')
        # num,K,bins,K,h,N
        print("num=200")
        main('H',1000,50,20,0.8,200)
        print("num=500")
        main('H',1000,50,20,0.8,500)
        print("num=1000")
        main('H',1000,50,20,0.8,1000)
        print("num=10000")
        main('H',1000,50,20,0.8,10000)
    elif c=='K':
        print('KDE:')
        print("num=200")
        main('K',1000,50,20,0.08,200)
        print("num=500")
        main('K',1000,50,20,0.08,500)
        print("num=1000")
        main('K',1000,50,20,0.08,1000)
        print("num=10000")
        main('K',1000,50,20,0.08,10000)
    else:
        print('k-NN')
        print("num=200")
        main('N',1000,50,20,0.8,100)
        print("num=500")
        main('N',1000,50,20,0.8,500)
        print("num=1000")
        main('N',1000,50,20,0.8,1000)
        print("num=10000")
        main('N',1000,50,20,0.8,10000)

##  REQUIREMENT 2
def requirement_2():
    c=input("Please input a bool. 0 for viewing plot under different bins, while 1 for the plot for CV in bins:")
    if c=='0':
        x=eval(input("Please input a num for N:"))
        print("N=%d now, and bins varies in order: 50, 100, 150, 200"%(x)) 
        print("bins=50")
        main('H',1000,50,20,0.8,x)
        print("bins=100")
        main('H',1000,100,20,0.8,x)
        print("bins=150")
        main('H',1000,150,20,0.8,x)
        print("bins=200")
        main('H',1000,200,20,0.8,x)
    else:
        best_bins_simple_cv()
        best_bins_simple_cv_error()

##  REQUIREMENT 3
def requirement_3():
    c=input("Please input a bool. 0 for viewing plot under different h, while 1 for the plot for CV in h:")
    if c=='0':
        x=eval(input("Please input a num for N:"))
        print("N=%d now, and h varies in order: 0.01, 0.08, 0.2, 0.4, 0.8"%(x)) 
        print("h=0.01")
        main('K',1000,50,20,0.01,x)
        print("h=0.08")
        main('K',1000,50,20,0.08,x)
        print("h=0.2")
        main('K',1000,50,20,0.2,x)
        print("h=0.4")
        main('K',1000,50,20,0.4,x)
        print("h=0.8")
        main('K',1000,50,20,0.8,x)
    else:
        print("N=100 here")
        best_h()
        main('K',1000,50,20,0.4,100)

        
##  REQUIREMENT 4
def requirement_4():
    c=input("Please input a bool. 0 for viewing plot under different K, while 1 for the plot for CV in K:")
    if c=='0':
        x=eval(input("Please input a N:"))
        print("K are in order: 3, 6, 9, 12, 20")
        main('N',1000,3,20,0.8,x)
        main('N',1000,6,20,0.8,x)
        main('N',1000,9,20,0.8,x)
        main('N',1000,12,20,0.8,x)
        main('N',1000,20,20,0.8,x)
    else:
        print("N=500 here")
        best_K()
        main('N',1000,50,10,0.8,500)


print("Bonjor!")
while 1:
    c=input("Please input 1,2,3,4 according to the require_id you are interested,others to exit:")
    if c=='1':
        requirement_1()
    elif c=='2':
        requirement_2()
    elif c=='3':
        requirement_3()
    elif c=='4':
        requirement_4()
    else:
        print("So, that's all about this work. See you!")
        break
