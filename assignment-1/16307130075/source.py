import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_data, GaussianMixture1D
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import scipy.stats as ss

np.random.seed(0)
gm1d = GaussianMixture1D(mode_range=(0, 50))
def truth():
    data = gm1d.sample([1000])
    min_range = min(gm1d.modes) - 3 * gm1d.std_range[1]
    max_range = max(gm1d.modes) + 3 * gm1d.std_range[1]
    xs = np.linspace(min_range, max_range, 2000)
    ys = np.zeros_like(xs)
    for l, s, w in zip(gm1d.modes, gm1d.stds, gm1d.weights):
        ys += ss.norm.pdf(xs, loc=l, scale=s) * w
    plt.plot(xs, ys)
    

def hist(num_data = 200, num_bins = 50, ptype = "varbin"):
    sampled_data = get_data(num_data)
    if ptype == "varbin":
        plt.title("histogram bin={}".format(num_bins))
    else:
        plt.title("histogram n={}".format(num_data))
    
    plt.hist(sampled_data, normed = True, bins = num_bins)
    

def calc_density(x, data, h):
    p = 0
    n = len(data)
    const = 1 / math.sqrt(math.pi * 2 * h * h)
    for i in data:
        num_exp = - ((i - x) * (i - x) / (2 * h * h))
        p += math.exp(num_exp)
    p = p * const / n
    return p

def calc_gauss_likelihood(h, trainset, testset):
    l = 0
    for x in testset:
        l += math.log(calc_density(x, trainset, h))
    return l

def find_maxli(dataset, k_fold = 5):
    n = len(dataset)
    random.shuffle(dataset)
    h_list = np.linspace(0.1, 1, 100)
    maxi, best_h = - 1e7, 0
    for h in h_list:
        li = 0
        for i in range(k_fold):
            st, ed = i * n // k_fold, (i + 1) * n // k_fold
            testset = dataset[st:ed]
            trainset = dataset[:st] + dataset[ed:]
            li += calc_gauss_likelihood(h, trainset, testset)

        if li > maxi:
            maxi = li
            best_h = h
    return best_h

def gauss_kernel(num_data = 100, h = None, ptype = "varh", num_inter = 2000):
    sampled_data = get_data(num_data)
    mini, maxi = min(sampled_data), max(sampled_data)
    interval = maxi - mini
    x_list = np.linspace(mini - interval * 0.05, maxi + interval * 0.05, num_inter)
    if h is None:
        h = find_maxli(sampled_data)
    
    print(h)
        
    p_list = []
    for x in x_list:
        p = calc_density(x, sampled_data, h)
        p_list.append(p)

    if ptype == "varh":
        plt.title("gauss h={:.2f}".format(h))
    else:
        plt.title("gauss n={}".format(num_data))
    plt.plot(x_list, p_list)
    
    

def knn_estimation(num_data, k, ptype = "vark", num_inter = 2000):
    if k > num_data:
        print("Error!k is larger than the size of dataset")
        return
    sampled_data = get_data(num_data)
    mini, maxi = min(sampled_data), max(sampled_data)
    interval = maxi - mini
    x_list = np.linspace(mini - interval * 0.05, maxi + interval * 0.05, num_inter)
    p_list = []
    for x in x_list:
        dist = []
        for y in sampled_data:
            dist.append(abs(x - y))
        dist = sorted(dist)

        v = 2 * dist[k - 1]
        p = k / (num_data * v)
        p_list.append(p)

    if ptype == "vark":
        plt.title("knn k={}".format(k))
    else:
        plt.title("knn n={}".format(num_data))
    plt.plot(x_list, p_list)
    truth()
    
    
def find(data, n, x):
    if x >= data[n - 1]:
        return n - 1
    if x < data[0]:
        return -1
    ans, l, r = 0, 0, n - 1
    while l <= r:
        mid = (l + r) // 2
        if data[mid] <= x:
            ans = mid
            l = mid + 1
        else:
            r = mid - 1
    return ans

def fast_knn_estimation(num_data, k, ptype = "vark", num_inter = 2000):
    if k > num_data:
        print("Error!k is larger than the size of dataset")
        return
    sampled_data = sorted(get_data(num_data))
    mini, maxi = min(sampled_data), max(sampled_data)
    interval = maxi - mini
    x_list = np.linspace(mini - interval * 0.05, maxi + interval * 0.05, num_inter)
    p_list = []
    for x in x_list:
        pos = find(sampled_data, num_data, x)
        l, r = pos + 1, pos
        while r - l + 1 < k:
            if l == 0:
                r += 1
            elif r == num_data - 1 or x - sampled_data[l - 1] < sampled_data[r + 1] - x:
                l -= 1
            else:
                r += 1

        v = 2 * max(x - sampled_data[l], sampled_data[r] - x)
        p = k / (num_data * v)
        p_list.append(p)
    
    if ptype == "vark":
        plt.title("knn k={}".format(k))
    else:
        plt.title("knn n={}".format(num_data))
    plt.plot(x_list, p_list)
    
    

def plot_hist_varbin():
    plt.figure(figsize = (8, 8))
    plt.subplot(2, 2, 1)
    hist(200, 5, "varbin")
    plt.subplot(2, 2, 2)
    hist(200, 10, "varbin")
    plt.subplot(2, 2, 3)
    hist(200, 15, "varbin")
    plt.subplot(2, 2, 4)
    hist(200, 20, "varbin")
    plt.show()    
    
    

def plot_hist_varn():
    plt.figure(figsize = (8, 8))
    plt.subplot(2, 2, 1)
    hist(100, 20, "varn")
    plt.subplot(2, 2, 2)
    hist(500, 20, "varn")
    plt.subplot(2, 2, 3)
    hist(1000, 20, "varn")
    plt.subplot(2, 2, 4)
    hist(10000, 20, "varn")
    plt.show()
    
def plot_gauss_varn():
    plt.figure(figsize = (8, 8))
    plt.subplot(2, 2, 1)
    gauss_kernel(100, 0.5, "varn")
    plt.subplot(2, 2, 2)
    gauss_kernel(500, 0.5, "varn")
    plt.subplot(2, 2, 3)
    gauss_kernel(1000, 0.5, "varn")
    plt.subplot(2, 2, 4)
    gauss_kernel(10000, 0.5, "varn")
    plt.show()
    
def plot_knn_varn():
    plt.figure(figsize = (8, 8))
    plt.subplot(2, 2, 1)
    fast_knn_estimation(100, 20, "varn")
    plt.subplot(2, 2, 2)
    fast_knn_estimation(500, 20, "varn")
    plt.subplot(2, 2, 3)
    fast_knn_estimation(1000, 20, "varn")
    plt.subplot(2, 2, 4)
    fast_knn_estimation(10000, 20, "varn")
    plt.show()


    
def plot_knn_vark():
    plt.figure(figsize = (8, 8))
    plt.subplot(2, 2, 1)
    knn_estimation(200, 5, "vark")
    plt.subplot(2, 2, 2)
    knn_estimation(200, 20, "vark")
    plt.subplot(2, 2, 3)
    knn_estimation(200, 50, "vark")
    plt.subplot(2, 2, 4)
    knn_estimation(200, 100, "vark")
    plt.show()