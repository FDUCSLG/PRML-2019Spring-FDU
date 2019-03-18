import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_data
from handout import GaussianMixture1D
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as ss

N = 10000


sampled_data_temp = get_data(N)
sampled_data = sampled_data_temp[:200]
min_range = min(sampled_data_temp)
max_range = max(sampled_data_temp)

#------histogram estimation------

def histogram_est(bin):
    plt.hist(sampled_data, normed=True, bins=bin)


#------kernel density estimation------
def func(h,x,xn):
    return (1.0/(2*math.pi*h*h)**(0.5)) * math.e**(-(x-xn)**2/(2*h*h))

def Gaussian(N,x,h):
    result = 0
    for i in range(N):
        xn = sampled_data[i]
        result += func(h,x,xn)        
    result *= 1.0/N
    return result

def kernel_dens_est(N,h,min_range,max_range):
    x = np.linspace(min_range, max_range, 2000)
    y=[Gaussian(N,i,h) for i in x]
    plt.plot(x,y)


#------nearest neighbor method------
def nearest(N,k,x):
    dis = [0 for i in range(N)]
    for i in range(N):
        dis[i] = abs(sampled_data[i]-x)
    dist = sorted(dis)
    v = 2*dist[k-1]+0.001
    return (k/N*1.0)/v

def nearest_ng_method(N,k,min_range,max_range):
    x = np.linspace(min_range, max_range, 2000)
    y=[nearest(N,k,i) for i in x]
    plt.plot(x,y)


def nearest_ng_method_cal(N,k,min_range,max_range):
    x = np.linspace(min_range, max_range, 2000)
    y=[nearest(N,k,i) for i in x]
    sum = 0
    for i in y:
        sum += i * (max_range-min_range) / 2000
    return sum


#---------------------
bin = 50
h = 0.5
k = 5
#------task_1---------
'''
ax = plt.subplot(321)
ax.set_title("N = 100")
sampled_data = sampled_data_temp[:100]
histogram_est(bin = bin)
ax = plt.subplot(322)
ax.set_title("N = 1000")
sampled_data = sampled_data_temp[:1000]
histogram_est(bin = bin)

plt.subplot(323)
sampled_data = sampled_data_temp[:100]
kernel_dens_est(N = 100, h = h, min_range = min_range, max_range = max_range)
plt.subplot(324)
sampled_data = sampled_data_temp[:1000]
kernel_dens_est(N = 1000, h = h, min_range = min_range, max_range = max_range)

plt.subplot(325)
sampled_data = sampled_data_temp[:100]
nearest_ng_method(N = 100, k = k, min_range = min_range, max_range = max_range)
plt.subplot(326)
sampled_data = sampled_data_temp[:1000]
nearest_ng_method(N = 1000, k = k, min_range = min_range, max_range = max_range)
plt.show()
'''




#-------task_2_1------
'''
ax = plt.subplot(221)
ax.set_title("bin = 6")
histogram_est(bin = 6)
ax = plt.subplot(222)
ax.set_title("bin = 20")
histogram_est(bin = 20)
ax = plt.subplot(223)
ax.set_title("bin = 50")
histogram_est(bin = 50)
ax = plt.subplot(224)
ax.set_title("bin = 100")
histogram_est(bin = 100)
plt.show()
'''
#-------task_2_2--------------
'''
ax = plt.subplot(221)
ax.set_title("bin = 6")
histogram_est(bin = 6)
ax = plt.subplot(223)
ax.set_title("bin = 100")
histogram_est(bin = 100)
sampled_data = sampled_data_temp[200:400]
ax = plt.subplot(222)
ax.set_title("bin = 6")
histogram_est(bin = 6)
ax = plt.subplot(224)
ax.set_title("bin = 100")
histogram_est(bin = 100)
plt.show()
'''

#------task3_1---------
'''
ax = plt.subplot(141)
ax.set_title("h = 0.1")
kernel_dens_est(N =200, h = 0.1, min_range = min_range, max_range = max_range)
ax = plt.subplot(142)
ax.set_title("h = 0.5")
kernel_dens_est(N =200, h = 0.5, min_range = min_range, max_range = max_range)
ax = plt.subplot(143)
ax.set_title("h = 1.0")
kernel_dens_est(N =200, h = 1.0, min_range = min_range, max_range = max_range)
ax = plt.subplot(144)
ax.set_title("h = 2.0")
kernel_dens_est(N =200, h = 2.0, min_range = min_range, max_range = max_range)
plt.show()
'''
#------task3_2------
'''
ax = plt.subplot(111)
ax.set_title("h = 0.5")
sampled_data = sampled_data_temp
histogram_est(bin = 100)
sampled_data = sampled_data_temp[:100]
kernel_dens_est(N =100, h = 0.6, min_range = min_range, max_range = max_range)
sampled_data = sampled_data_temp[100:200]
kernel_dens_est(N =100, h = 0.6, min_range = min_range, max_range = max_range)
sampled_data = sampled_data_temp[200:300]
kernel_dens_est(N =100, h = 0.6, min_range = min_range, max_range = max_range)
plt.show()
'''

#-------task4_1-------------
'''
np.random.seed(0)
gm1d = GaussianMixture1D(mode_range=(0, 50))
def cur():
    data = gm1d.sample([1000])
    min_range = min(gm1d.modes) - 3 * gm1d.std_range[1]
    max_range = max(gm1d.modes) + 3 * gm1d.std_range[1]
    xs = np.linspace(min_range, max_range, 2000)
    ys = np.zeros_like(xs)

    for l, s, w in zip(gm1d.modes, gm1d.stds, gm1d.weights):
        ys += ss.norm.pdf(xs, loc=l, scale=s) * w

    plt.plot(xs, ys)
    return

ax = plt.subplot(221)
ax.set_title("K = 1")
nearest_ng_method(N = 200, k = 1, min_range = min_range, max_range = max_range)
cur()

ax = plt.subplot(222)
ax.set_title("K = 5")
nearest_ng_method(N = 200, k = 5, min_range = min_range, max_range = max_range)
cur()

ax = plt.subplot(223)
ax.set_title("K = 10")
nearest_ng_method(N = 200, k = 10, min_range = min_range, max_range = max_range)
cur()

ax = plt.subplot(224)
ax.set_title("K = 30")
nearest_ng_method(N = 200, k = 30, min_range = min_range, max_range = max_range)
cur()
plt.show()
'''

#-------task 4_2------------
for i in range(1,30):
    sum = nearest_ng_method_cal(N = 200, k = i, min_range = min_range, max_range = max_range)
    print(sum)