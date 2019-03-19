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
    plt.show()


# histmd(bin_num=10)
# histmd(bin_num=20)
# histmd(bin_num=50)
# histmd(bin_num=100)
# histmd(bin_num=500)

def kde(sample_data, test_data, h):

    p = np.zeros_like(test_data)
    for i, x in enumerate(test_data):
        for xn in sample_data:
            p[i] += math.exp(-(x-xn)**2 / (2 * h**2)) / math.sqrt(2 * math.pi * (h**2))

    p /= len(test_data)
    return p


def kde_plt(data_num, spot_num, h):
    sample_data = get_data(data_num)
    max_x = max(sample_data)
    min_x = min(sample_data)
    spot_data = np.linspace(min_x, max_x, spot_num)

    p = kde(sample_data, spot_data, h)
    plt.plot(spot_data, p)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Kernel Density Estimation\ndata_num=' + str(data_num) + '  h=' + str(h))
    plt.show()


# kde_plt(100, 100, 0.002)
# kde_plt(100, 100, 0.1)
# kde_plt(100, 100, 0.5)
# kde_plt(100, 100, 1)
# kde_plt(100, 100, 2)
# kde_plt(100, 100, 5)


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
    plt.plot(spot_data, p)
    plt.title('K-nearest neighbor\ndata_num=' + str(data_num) + '  K=' + str(k))
    gm1d.plot(num_sample=1000)
    plt.show()


# knn_plt(200, 200, 1)
# knn_plt(200, 200, 5)
# knn_plt(200, 200, 20)
# knn_plt(200, 200, 35)
# knn_plt(200, 200, 40)
# knn_plt(200, 200, 100)

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
