import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt
import math
from handout import *
def gaosi(h,x,xn):
    expx=math.exp(-(x-xn)*(x-xn)/(2*h*h))
    gen=math.sqrt(2*math.pi)
    return expx/(h*gen)

def gaosikd(h,data):
    mind=min(data)
    maxd=max(data)
    xs = np.linspace(mind, maxd, 2000)
    ys = np.zeros_like(xs)
    index=0
    ld=len(data)
    for i in xs:
        xxx=0
        for j in data:
            xxx=xxx+gaosi(h,i,j)
        ys[index]=xxx/ld
        index=index+1
    plt.plot(xs, ys)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    #plt.show()
def knn(K,data):
    mind=min(data)
    maxd=max(data)
    xs = np.linspace(mind, maxd, 2000)
    ys = np.zeros_like(xs)
    ld=len(data)
    yi=0;
    for i in xs:
        dis=np.zeros_like(data)
        index=0
        for q in data:
            dis[index]=abs(q-i);
            index=index+1;
        dis.sort()
        V=dis[K-1]
        ys[yi]=K/(ld*V)
        yi=yi+1
    plt.plot(xs, ys)
    plt.xlabel("x")
    plt.ylabel("f(x)")
sampled_data = get_data(200)

plt.hist(sampled_data, normed=True, bins=100)
#plt.show()
gm1d.plot()

gaosikd(0.45,sampled_data)
gm1d.plot()

knn(1,sampled_data)
gm1d.plot()
knn(5,sampled_data)
gm1d.plot()
knn(30,sampled_data)
gm1d.plot()
