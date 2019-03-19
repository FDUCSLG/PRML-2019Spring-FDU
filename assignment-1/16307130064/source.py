import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt
import math
import random

def guassian_kde(data,x,h):
    N=len(data)    
    s=np.zeros_like(x)
    for i in range(N):
        s+=np.exp(-np.square(x-data[i])/2/np.square(h))
    
    return s/N/np.sqrt(2*np.pi)/h

def knn(data,x,k):
    N=len(data)
    dis=np.zeros((len(x),N))
    for i in range(N):
        dis[:,i]=np.abs(x-data[i])
    dis.sort(1)
    #print(dis[0,:k])
    return 1.0*k/N/2/dis[:,k]

def cross_validation(data,k,h):
    s=0
    N=len(data)
    m=N//k
    batch=[data[i:i+m] for i in range(0,N,m)]
    for i in range(k):
        train=[]
        for j in range(k):
            if i!=j: train.extend(batch[j])
        s+=sum(np.log(guassian_kde(train,np.array(batch[i]),h)))
    return s/k

def find_h(data):
    x=np.linspace(0.1,1,1000)
    score=np.zeros_like(x)
    best=0
    s_max=-10000
    for i,h in enumerate(x):
        s=cross_validation(data,5,h)
        if s>s_max:
            s_max=s
            best=h
        score[i]=s
    plt.figure("3-0")
    plt.title("log likelihood")
    plt.plot(x,score)
    return best
        
sampled_data = get_data(10000)
x=np.linspace(20,40,2000)

ty=input()
ty=int(ty)
if ty==1:
    plt.figure("hist")
    plt.subplot(1,3,1)
    plt.title("num_data=100")
    plt.hist(sampled_data[:100], normed=True, bins=50)
    plt.subplot(1,3,2)
    plt.title("num_data=1000")
    plt.hist(sampled_data[:1000], normed=True, bins=50)
    plt.subplot(1,3,3)
    plt.title("num_data=10000")
    plt.hist(sampled_data, normed=True, bins=50)

    plt.figure("density")
    h=0.2
    plt.subplot(1,3,1)
    plt.title("num_data=100")
    plt.plot(x,guassian_kde(sampled_data[:100],x,h))
    plt.subplot(1,3,2)
    plt.title("num_data=1000")
    plt.plot(x,guassian_kde(sampled_data[:1000],x,h))
    plt.subplot(1,3,3)
    plt.title("num_data=10000")
    plt.plot(x,guassian_kde(sampled_data,x,h))

    plt.figure("knn")
    k=20
    plt.subplot(1,3,1)
    plt.title("num_data=100")
    plt.plot(x,knn(sampled_data[:100],x,k))
    plt.subplot(1,3,2)
    plt.title("num_data=1000")
    plt.plot(x,knn(sampled_data[:1000],x,k))
    plt.subplot(1,3,3)
    plt.title("num_data=10000")
    plt.plot(x,knn(sampled_data,x,k))

elif ty==2:
    plt.figure(1)
    plt.subplot(1,3,1)
    plt.title("bins=5")
    plt.hist(sampled_data[:200], normed=True, bins=5)
    plt.subplot(1,3,2)
    plt.title("bins=20")
    plt.hist(sampled_data[:200], normed=True, bins=20)
    plt.subplot(1,3,3)
    plt.title("bins=50")
    plt.hist(sampled_data[:200], normed=True, bins=50)
    
    plt.figure(2)
    plt.subplot(1,3,1)
    plt.title("Square-root choice bins=15")
    plt.hist(sampled_data[:200], normed=True, bins=15)
    plt.subplot(1,3,2)
    plt.title("Sturges' formula bins=9")
    plt.hist(sampled_data[:200], normed=True, bins=9)
    plt.subplot(1,3,3)
    plt.title("Rice Rule bins=12")
    plt.hist(sampled_data[:200], normed=True, bins=12)

elif ty==3:
    h=find_h(sampled_data[:100])
    print(h)
    plt.figure(3)
    plt.title("num_data=100 h={}".format(h))
    plt.plot(x,guassian_kde(sampled_data[:100],x,h))

elif ty==4:
    plt.figure(4)
    plt.plot(x,knn(sampled_data[:200],x,k=5))
    plt.plot(x,knn(sampled_data[:200],x,k=10))
    plt.plot(x,knn(sampled_data[:200],x,k=50))
    plt.legend(["k=5","k=10","k=50"])

plt.show()

