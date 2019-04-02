import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import gm1d
import numpy as np
import matplotlib.pyplot as plt


def histogram_estimation(num_bins=25, sample_data=[], status=False, title=""):
    num_data = len(sample_data)
    plt.hist(sample_data, normed=True, bins=num_bins)
    if len(title) != 0:
        plt.title(title)
    if status:
        gm1d.plot(num_sample=1000)
    else:
        plt.show()

def kernel_density_estimation(h=0.1, sample_data=[], num_test_base=10000, status=False, title=""):
    # process the sample data
    num_data = len(sample_data)
    sd_min = min(sample_data)
    sd_max = max(sample_data)
    N = num_data

    # organize the test base
    num_test_base = num_test_base
    estimation_pdf = np.zeros(num_test_base)
    xs = np.linspace(sd_min, sd_max, num_test_base).reshape(num_test_base, 1)
    extended_xs = np.repeat(xs, N, axis=1)

    # get the pdf according to gaussian kernel
    mid_gaussian = np.exp(- (extended_xs - sample_data)**2 / (2 * h**2)) / (
        2 * np.pi * h**2)**0.5
    estimation_pdf = np.sum(mid_gaussian, axis=1) / N

    # show the result
    plt.plot(xs, estimation_pdf)
    # plt.show()
    if len(title) != 0:
        plt.title(title)
    if status:
        gm1d.plot(num_sample=1000)
    else:
        plt.show()


def knn_V(K, sample_data, x):
    N = len(sample_data)
    assert K <= N, "too large K!"
    len_x = len(x)
    # organize test base
    extended_xs = x.reshape(len_x, 1).repeat(N, axis = 1)

    # get distance with every sample data point
    distance = np.absolute(extended_xs - sample_data)

    # sort for knn distance
    k_dis = np.argsort(distance,axis = 1)
    result = 2 * distance[np.arange(len_x), k_dis[:, K-1]]

    return result

def knn_estimation(K=1, sample_data=[], num_test_base=10000, status=False, title=""):
    # process sample data
    num_data = len(sample_data)
    sd_min = min(sample_data)
    sd_max = max(sample_data)
    N = num_data

    # test base
    num_test_base = num_test_base
    xs = np.linspace(sd_min, sd_max, num_test_base)

    # get V of knn
    V = knn_V(K, sample_data, xs)

    estimation_pdf = K / (N * V)

    # show the result
    plt.plot(xs, estimation_pdf)
    if len(title) != 0:
        plt.title(title)
    if status:
        gm1d.plot(num_sample=1000)
    else:
        plt.show()



# https://stats.stackexchange.com/questions/218514/how-to-decide-whether-a-kernel-density-estimate-is-good
# https://amstat.tandfonline.com/doi/abs/10.1080/07350015.2018.1424633?af=R&journalCode=ubes20#.XI51oBMzZbU
# https://stats.stackexchange.com/questions/262658/why-does-maximum-likelihood-estimation-maximizes-probability-density-instead-of

# optimize
# https://blog.csdn.net/suzyu12345/article/details/70046826