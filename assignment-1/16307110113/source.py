import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import *
import math
import numpy as np
import matplotlib.pyplot as plt
import sys


def histmd(data_num = 200, bin_num = 50):
    sample_data = get_data(data_num)
    plt.hist(sample_data, density=True, bins=bin_num)
    plt.title('Histogram Method\ndata_num='+str(data_num)+'  bins='+str(bin_num))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    gm1d.plot(num_sample=1000)
    plt.show()


# histmd(bin_num=5)
# histmd(bin_num=20)
# histmd(bin_num=50)
# histmd(bin_num=100)
# histmd(bin_num=500)

def kde(sample_data, test_data, h):

    p = np.zeros_like(test_data)
    for i, x in enumerate(test_data):
        for xn in sample_data:
            p[i] += math.exp(-(x-xn)**2 / (2 * h**2)) / math.sqrt(2 * math.pi * (h**2))

    p /= len(sample_data)
    return p


def kde_plt(data_num, spot_num, h):
    sample_data = get_data(data_num)
    max_x = max(sample_data)
    min_x = min(sample_data)
    spot_data = np.linspace(min_x, max_x, spot_num)

    p = kde(sample_data, spot_data, h)
    plt.plot(spot_data, p, label = 'h=' + str(h))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Kernel Density Estimation\ndata_num=' + str(data_num) + '  h=' + str(h))
    plt.legend()
    gm1d.plot(num_sample=1000)
    plt.show()


# kde_plt(100, 100, 0.05)
# kde_plt(100, 100, 0.2)
# kde_plt(100, 100, 0.6)
# kde_plt(100, 100, 0.5)
# kde_plt(100, 100, 1)
# kde_plt(100, 100, 2)

def cross_validation(data_num, h_min, h_max):
    sample_data = get_data(data_num)
    steps = (h_max - h_min) / 10
    lh_fun = np.zeros(10)
    h_set = np.arange(h_min, h_max, steps)

    def kde2(N, sample, t_i, h):
        p = 0
        for n, xn in enumerate(sample):
            if(n == t_i):
                continue
            p += math.exp(-(sample[t_i] - xn) ** 2 / (2 * h ** 2)) / math.sqrt(2 * math.pi * (h ** 2))

        p /= N-1
        return p

    for n in range(0, 10):
        h = h_min + n * steps
        for i in range(0, data_num):
            p = kde2(data_num, sample_data, i, h)
            lh_fun[n] += math.log10(p)

    lh_fun /= data_num
    plt.plot(h_set, lh_fun)
    plt.title("Maximum likelihood cross-validation")
    plt.xlabel('h')
    plt.ylabel('likelihood function')
    plt.show()
    lh_arg = lh_fun.argsort()
    return h_set[lh_arg[-1]]


h_fit = cross_validation(100, 0.3, 0.4)
kde_plt(100, 100, round(h_fit, 2))

def knn(sample_data, test_data, data_num, k):
    p = np.empty_like(test_data)

    for i, x in enumerate(test_data):
        x_dist = abs(sample_data - x)
        x_dist.sort()
        if x_dist[k-1] == 0:
            p[i] = float('Inf')
        else:
            p[i] = k / (data_num * 2 * x_dist[k-1])

    return p


def knn_plt(data_num, spot_num, k):
    sample_data = get_data(data_num)
    max_x = max(sample_data)
    min_x = min(sample_data)
    spot_data = np.linspace(min_x, max_x, spot_num)

    p = knn(sample_data, spot_data, data_num, k)
    plt.plot(spot_data, p, label = 'k=' + str(k))
    plt.title('K-nearest neighbor\ndata_num=' + str(data_num) + '  K=' + str(k))
    plt.legend()
    plt.axis([15, 40, -0.02, 0.45])
    gm1d.plot(num_sample=1000)
    plt.show()


# knn_plt(200, 200, 1)
# knn_plt(200, 200, 5)
# knn_plt(200, 200, 10)
# knn_plt(200, 200, 20)
# knn_plt(200, 200, 50)
# knn_plt(200, 200, 25)

def main(argv=sys.argv):
    if(len(argv) < 4):
        print("You should select the method and parameters")
        exit(1)

    data_num = int(argv[2])

    if(argv[1] == '--hist'):
        histmd(data_num, int(argv[3]))
    elif(argv[1] == '--kde'):
        kde_plt(data_num, data_num, float(argv[3]))
    elif(argv[1] == '--knn'):
        knn_plt(data_num, data_num, int(argv[3]))
    else:
        print("You should select the method and parameters")
        exit(1)

if(__name__ == "__main__"):
    main()
