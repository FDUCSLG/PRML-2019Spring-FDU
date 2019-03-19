import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt
import math

#sampled_data = get_data(500)
#plt.hist(sampled_data, normed=True, bins=50)
#plt.show()

def histogram_plot(data,bins=50):
    plt.hist(data, normed=True,bins=bins)
    plt.title("Histogram\nbin="+str(bins)+"\ndata size="+str(len(data)),fontsize=20)
    plt.show()
## guassian slef
## para:h namely the standard deviation
def gaussian_kde(data,h):
    assert h!=0,"h cannot be 0!\n"
    N=len(data)
    term=[]
    mini=min(data)
    maxi=max(data)
    maxi_y=0;
    coefficient=1/(N*(2*math.pi*h**2)**0.5)
    x=np.linspace(mini,maxi,num=200)
    for e in x:
        s=0
        for d in data:
            s=s+math.e**(-(e-d)**2/(2*h**2))
        term.append(s*coefficient)
        if s*coefficient>maxi_y:
            maxi_y=s*coefficient
    maxi_y=maxi_y*1.1
    ##fig, ax = plt.subplots()
    ##ax.set_ylim(0,2)
    ##ax.set_xlim(mini,maxi)
    plt.plot(x,term)
    plt.title("Gaussian\nh="+str(h)+"\ndata size="+str(len(data)),fontsize=20)
    plt.show()
    
## nearest neighbour
## para: sample data:     data
##       constant K:      k
##       estimated point: x
## returnee:              int or assert error infomation
def minimum_volume(data, k, x):
    assert len(data)>=k,"number of data less than k!\n"
    distance=[];
    for e in data:
        if e<=x:
            distance.append(x-e)
        else:
            distance.append(e-x)
    distance.sort()
    return distance[k-1]

def knn_plot(data,k=1):
    ## equation p=K/(N*V)
    mini=min(data)
    maxi=max(data)
    y=[]
    N=len(data)
    X=np.linspace(mini, maxi, num=200)
    for d in X:
        V=minimum_volume(data,k,d)
        if V==0:
            y.append(float('inf'))
            ##print("inf")
        else:
            y.append(k*1.0/(V*N))    
    fig, ax = plt.subplots()
    ax.set_ylim(0,2)
    ax.set_xlim(mini,maxi)
    plt.title("Nearest Neighbor\nK="+str(k)+"\ndata size="+str(len(data)),fontsize=20)
    plt.plot(X,y)
    plt.show()

## empirical assertion
def get_empirical_assertion():
    sampled_data=get_data(100)
    histogram_plot(sampled_data)
    gaussian_kde(sampled_data,len(sampled_data)**-0.2)
    knn_plot(sampled_data,100)
    
    sampled_data=get_data(500)
    histogram_plot(sampled_data)
    gaussian_kde(sampled_data,len(sampled_data)**-0.2)
    knn_plot(sampled_data,100)
    
    sampled_data=get_data(1000)
    histogram_plot(sampled_data)
    gaussian_kde(sampled_data,len(sampled_data)**-0.2)
    knn_plot(sampled_data,100)
    
    sampled_data=get_data(10000)
    histogram_plot(sampled_data)
    gaussian_kde(sampled_data,len(sampled_data)**-0.2)
    knn_plot(sampled_data,100)

##tune histogram bins
def tune_histogram_bins():
    sampled_data=get_data(200)
    histogram_plot(sampled_data,5)
    histogram_plot(sampled_data,9)
    histogram_plot(sampled_data,15)
    histogram_plot(sampled_data,75)

##1,2,3
def tune_gaussian_h():
    sampled_data=get_data(200)
    gaussian_kde(sampled_data,1)
    gaussian_kde(sampled_data,3)
    gaussian_kde(sampled_data,0.1)
    gaussian_kde(sampled_data,0.34657)
def tune_gaussian_100_h():
    sampled_data=get_data(100)
    gaussian_kde(sampled_data,1)
    gaussian_kde(sampled_data,0.3981)

def tune_nn_K():
    sampled_data=get_data(200)
    knn_plot(sampled_data,1)
    knn_plot(sampled_data,5)
    knn_plot(sampled_data,30)
    knn_plot(sampled_data,7)

while True:
    print("To get empirical_assertion press 1!\n\
To tunne histograms bins press 2!\n\
To tunne gaussian h press 3!\n\
To get suitable h for 100 data of gaussian kde press 4!\n\
To tunne nearest neighbour K press 5!\n\
To show real distribution press 6!\n\
To end press 0\n\
Good day!\n")
    s=input("Input your choice please##: ")
    if s=="1":
        get_empirical_assertion()
    if s=="2":
        tune_histogram_bins()
    if s=="3":
        tune_gaussian_h()
    if s=="4":
        tune_gaussian_100_h()
    if s=="5":
        tune_nn_K()
## show real distribution
    if s=="6":
        from handout import gm1d
        gm1d.plot()
    if s=='0':
        print("Good bye!\n")
        break;
