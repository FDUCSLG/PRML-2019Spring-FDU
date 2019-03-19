import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import time

def histogram(sampled_data, num_bins = 75):
    plt.hist(sampled_data, normed=True, bins=num_bins, facecolor='slateblue')

def kernel(sampled_data, test_data, h = 0.2):
    sampled_num = len(sampled_data)
    ys = np.zeros_like(test_data)
    for x in sampled_data:
        ys += ss.norm.pdf(test_data, loc = x, scale = h)
    ys /= sampled_num
    return ys

def IFGT(Sampled_data, test_data, h = 0.2, p = 10, K = 100):
    Sampled_data.sort()
    N = len(Sampled_data)
    M = len(test_data)
    ys = np.zeros_like(test_data)
    max_range, min_range = max(Sampled_data), min(Sampled_data)
    size = (max_range - min_range) / K
    h *= np.sqrt(2)
    L, R = 0, 0
    l, r = 0, 0
    for k in range(0, K):
        L = R
        while (R < N and Sampled_data[R] <= min_range + (k + 1) * size): R += 1
        if (L >= R): continue
        else: sampled_data = Sampled_data[L: R]
        x_star = np.mean(sampled_data)
        C = [[0 for i in range(p)] for i in range(p)]
        for a in range(0, p):
            for b in range(0, p - a):
                for x in sampled_data:
                    C[a][b] += np.exp(-((x - x_star)**2)/(h**2))*(((x - x_star)/h)**a)
                C[a][b] *= (2**(a+b)) / (np.math.factorial(a) * np.math.factorial(b) * N * 2 * np.sqrt(np.math.pi) * h)
        sumC = [0 for i in range(p)]
        for b in range(0, p):
            for a in range(0, p - b):
                sumC[b] += C[a][b]
        while (r < M and test_data[r] - x_star <= h * 1.2): r += 1
        while (l < r and x_star - test_data[l] > h * 1.2): l += 1
        for i in range(l, r):
            for b in range(0, p):
                ys[i] += sumC[b] * np.exp(-((test_data[i] - x_star)**2)/(h**2))*(((test_data[i] - x_star)/h)**b)
    return ys

def kNN(sampled_data, test_data, K = 1):
    sampled_num = len(sampled_data)
    N = len(test_data)
    ys = np.zeros_like(test_data)
    for i in range(0, N):
        dis = abs(sampled_data - test_data[i])
        dis.sort()
        ys[i] = K / (sampled_num * dis[K - 1] * 2)
    return ys

def cross_validation(sampled_data, h = 0.2, k = 5):
    sampled_num = len(sampled_data)
    total_likelihood = 0
    size = sampled_num // k
    for i in range(0, k):
        learning_data = np.append(sampled_data[0: (i * size)], sampled_data[((i + 1) * size + 1): sampled_num])
        test_data = sampled_data[(i * size): ((i + 1) * size)]
        test_likelihood = kernel(learning_data, test_data, h)
        total_likelihood += sum(np.log(test_likelihood))
    total_likelihood /= sampled_num
    return total_likelihood

def bandwidth_selection(min_range = 0.1, max_range = 0.6, N = 50):
    xH = np.linspace(min_range, max_range, N)
    yH = np.zeros_like(xH)
    maxH, max_likelihood = 100, -100
    for i in range(0, N):
        yH[i] = cross_validation(sampled_data, h = xH[i], k = 5)
        if yH[i] > max_likelihood: maxH, max_likelihood = xH[i], yH[i]
    print("Best Bandwidth: ", maxH)
    plt.plot(xH, yH)
    plt.annotate('global max', xy=(maxH, max_likelihood), xytext=(maxH + 0.05, max_likelihood - 0.07),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 )
    plt.title('Log Maximum Likelihood Estimation')
    plt.xlabel("h")
    plt.ylabel("Likelihood")
    plt.show()
    return maxH

def compare(min_range = 10000, max_range = 50001):
    xT = range(min_range, max_range, 2000)
    yT_kernel = []
    yT_IFGT = []
    xs = np.linspace(min_range, max_range, 10000)
    for x in xT:
        sampled_data = get_data(x)
        T = time.time()
        kernel(sampled_data, xs, h=0.1389)
        yT_kernel.append(time.time() - T)
        T = time.time()
        IFGT(sampled_data, xs, h=0.1389, K=100)
        yT_IFGT.append(time.time() - T)

    plt.title('Time Comparision')
    plt.plot(xT, yT_kernel, color='blue')
    plt.plot(xT, yT_IFGT, color='red')
    plt.xlabel("x")
    plt.ylabel("Time")
    plt.show()


sampled_num = 10000

sampled_data = get_data(sampled_num)
min_range = min(sampled_data) - 3 #* np.std(sampled_data, ddof = 1)
max_range = max(sampled_data) + 3 #* np.std(sampled_data, ddof = 1)
xs = np.linspace(min_range, max_range, 2000)
ys = np.zeros_like(xs)

#histogram(sampled_data, num_bins = 25)

#h = bandwidth_selection(min_range = 0.1, max_range = 0.6, N = 50)
#ys = kernel(sampled_data, xs, h = 0.1389)
#ys = IFGT(sampled_data, xs, h = 0.1389, K = 100)

#compare()

#ys = kNN(sampled_data, xs, K = 50)

#plt.ylim([0, 0.5])
plt.plot(xs, ys, color = 'blue')
#plt.plot(xs, ys1, color = 'red')
plt.xlabel("x")
plt.ylabel("p(x)")
plt.show()