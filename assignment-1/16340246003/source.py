import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt
import math


sampled_data = get_data(1000)


def hist_est(bins_num):
    plt.hist(sampled_data, normed=True, bins=bins_num)
    plt.show()

def kde(h):
    N=len(sampled_data)
    term=[]
    mini=min(sampled_data)
    maxi=max(sampled_data)
    maxi_y=0;
    c=1/(N*(2*math.pi*h**2)**0.5)
    x=np.linspace(mini,maxi,num=200)
    for e in x:
        s=0
        for d in sampled_data:
            s=s+math.e**(-(e-d)**2/(2*h**2))
        term.append(s*c)
        if s*c>maxi_y:
            maxi_y=s*c
    maxi_y=maxi_y*1.1
    plt.plot(x,term)
    plt.show()

def knn(K):
    mini=min(sampled_data)
    maxi=max(sampled_data)
    x = np.linspace(mini, maxi, 200)
    y = np.zeros_like(x)
    ld=len(sampled_data)
    yi=0;
    for i in x:
        dis=np.zeros_like(sampled_data)
        index=0
        for q in sampled_data:
            dis[index]=abs(q-i);
            index=index+1;
        dis.sort()
        V=dis[K-1]
        y[yi]=K/(ld*V)
        yi=yi+1
    plt.plot(x, y)
    plt.show()

#hist_est(5)
silverman_h = 1.06*np.std(sampled_data)*math.pow(len(sampled_data),-1/5)
#print(silverman_h)
#kde(silverman_h)
#knn(1)
