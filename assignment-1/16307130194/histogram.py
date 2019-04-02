# path to top module
import os
os.sys.path.append('..')

# lib
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt
from plot import plot_gm1d


def histogram_method(N=200, num_bin=25, show=True):
    assert num_bin <= N

    data = get_data(N)

    plt.hist(data, density=True, bins=num_bin, label="histogram method")

    plt.legend()
    plt.title("N = %d, num_bin = %d" % (N, num_bin))
    plot_gm1d()
    if show:
        plt.show()


def histogram_result():
    plt.subplot(131)
    plt.xlabel("x")
    plt.ylabel("p(x)")
    histogram_method(N=200, num_bin=10, show=False)

    plt.subplot(132)
    plt.xlabel("x")
    plt.ylabel("p(x)")
    histogram_method(N=200, num_bin=25, show=False)

    plt.subplot(133)
    plt.xlabel("x")
    plt.ylabel("p(x)")
    histogram_method(N=200, num_bin=75, show=False)

    plt.show()


def optimal_bin(N=200):
    data = get_data(N)

    data_max = max(data)
    data_min = min(data)
    num_bin = np.arange(4, 50, 1)
    size_bin = (data_max - data_min) / num_bin
    cost = []

    for i in range(np.size(num_bin)):
        edges = np.linspace(data_min, data_max, num_bin[i] + 1)
        ki = plt.hist(data, edges)[0]
        k = np.mean(ki)
        v = sum((ki - k) ** 2) / num_bin[i]
        cost.append((2 * k - v) / (size_bin[i] ** 2))

    cost_min = min(cost)
    index = np.where(cost == cost_min)[0]
    optimal_size = size_bin[index]

    optimal_cnt = (data_max - data_min) / optimal_size

    plt.clf()

    plt.subplot(121)
    plt.title("N = %d, optimal_cnt = %d / bin_width = %f" % (N, optimal_cnt, optimal_size))
    plt.plot(size_bin, cost, '.b', optimal_size, cost_min, '*r')
    plt.xlabel("bin width")
    plt.ylabel("cost")

    plt.subplot(122)
    plt.xlabel("x")
    plt.ylabel("p(x)")
    histogram_method(N=N, num_bin=int(optimal_cnt), show=False)
    plt.legend()
    plot_gm1d()
    plt.show()
