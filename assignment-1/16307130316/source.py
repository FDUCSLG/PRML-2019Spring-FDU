import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import *
import numpy as np
import matplotlib.pyplot as plt
import math


np.random.seed(0)
global data_size


# histogram
def histogram_estimation(num_bins):
    sampled_data = get_data(data_size)
    plt.hist(sampled_data, normed=True, bins=num_bins)
    plt.title("data_size=%d,%dbins" % (data_size, num_bins))
    a = GaussianMixture1D(mode_range=(0, 50))
    a.plot(num_sample=1000)
    plt.show()


def gaussian_kernel_implement(h):
    sampled_data = get_data(data_size)
    X = np.array(sampled_data)[:, np.newaxis]  # convert to 2-D array

    def prob(sampled_data, x):
        N = len(sampled_data)
        return 1/N/math.sqrt(2*math.pi*h*h) * sum([math.exp(-(x-xn)**2/2/h/h) for xn in sampled_data])

    x_min = min(sampled_data) - 1
    x_max = max(sampled_data) + 1
    X_plot = np.linspace(x_min, x_max, 10000)[:, np.newaxis]  # sampling for plot
    fig = plt.figure()
    p = [prob(sampled_data, x) for x in X_plot[:, 0]]
    plt.plot(X_plot[:, 0], p)
    plt.title("data_size=%d, h=%f" % (data_size, h))
    a = GaussianMixture1D(mode_range=(0, 50))
    a.plot(num_sample=1000)
    plt.show()


def KNN_implement(K):
    sampled_data = get_data(data_size)
    X = np.array(sampled_data)[:, np.newaxis]
    x_min = min(sampled_data) - 1
    x_max = max(sampled_data) + 1
    X_plot = np.linspace(x_min, x_max, 1000)[:, np.newaxis]

    def KNN_radius(sampled_data, x):
        distance = [abs(x-xn) for xn in sampled_data]
        return sorted(distance)[K-1]

    y = [K / (len(sampled_data) * 2 * KNN_radius(sampled_data, x)) for x in X_plot[:, 0]]
    fig = plt.figure()
    plt.plot(X_plot[:, 0], y)
    plt.title("data_size=%d, K=%d" % (data_size, K))
    a = GaussianMixture1D(mode_range=(0, 50))
    a.plot(num_sample=1000)
    plt.show()


def main(argv):
    global data_size
    data_size = int(argv[3])
    if argv[1] == "histogram":
        histogram_estimation(int(argv[2]))
    elif argv[1] == "gaussian":
        gaussian_kernel_implement(float(argv[2]))
    elif argv[1] == "KNN":
        KNN_implement(int(argv[2]))

    else:
        exit(1)

if __name__ == "__main__":
    main(os.sys.argv)