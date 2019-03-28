import numpy as np
import math
import matplotlib.pyplot as plt
from handout import get_data

interval_size = 0.1
n = 2000
data = get_data(n)

sigma = np.std(data, ddof=1)
besth = 1.06 * sigma * np.power(n, -0.2)

print(sigma, besth)

k = 400

def sqr(x):
    return x * x

def zhifangtu(data, b):
    plt.hist(data, bins = b, label="bins="+str(b))

def kernel(data, h, eps):

    mn = min(data) - eps
    mx = max(data) + eps
    x = []
    while mn <= mx:
        x.append(mn)
        mn += eps
    
    llen = len(x)
    p = []
    
    for i in range(llen):
        px = 0
        cc = 1 / math.sqrt(2 * math.pi * sqr(h))
        for j in range(n):
            px += cc * math.exp(-sqr(x[i] - data[j]) / (2 * sqr(h)))
        px /= n
        p.append(px)

    stlabel = "h="+str(h)
    plt.plot(x, p, label=stlabel)


def kneighbors(data, k, eps):
    mn = min(data) - eps
    mx = max(data) + eps
    x = []
    while mn <= mx:
        x.append(mn)
        mn += eps
    
    llen = len(x)
    p = []
    
    for i in range(llen):
        v = []
        for j in range(n):
            v.append(abs(x[i] - data[j]))
        v = sorted(v)

        p.append(k / (n * v[k - 1]))
        
    stlabel = "k="+str(k)
    plt.plot(x, p, label=stlabel)

#for i in range(1, 11):
#    zhifangtu(data, i * 15)

kernel(data, 0.1, interval_size)

for i in range(1, 6):
    h = i * 0.2
    kernel(data, h, interval_size)


plt.grid(True)

plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
plt.show()


