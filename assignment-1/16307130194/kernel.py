# path to top module
import os
os.sys.path.append('..')

# lib
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from plot import plot_gm1d


def kernel_density_estimate(N=100, h=0.35, show=True):
    assert h > 0

    data = get_data(N)

    x = np.linspace(min(data), max(data), 1000)
    px = np.sum(np.exp(np.square(x - np.reshape(data, (N, 1))) / (-2 * h ** 2)), axis=0) / (np.sqrt(2 * np.pi) * h) / N

    plt.plot(x, px, label="kernel density estimate")
    plt.legend()
    plt.title("N = %d, h = %f" % (N, h))
    plot_gm1d()
    if show:
        plt.show()


def kde_result():
    plt.subplot(131)
    plt.xlabel("x")
    plt.ylabel("p(x)")
    kernel_density_estimate(N=100, h=0.15, show=False)

    plt.subplot(132)
    plt.xlabel("x")
    plt.ylabel("p(x)")
    kernel_density_estimate(N=100, h=0.35, show=False)

    plt.subplot(133)
    plt.xlabel("x")
    plt.ylabel("p(x)")
    kernel_density_estimate(N=100, h=1, show=False)

    plt.show()


def optimal_h(N=100, show=True):
    # likelihood function for minimize
    def likelihood_func(h):
        h = h[0]
        data = get_data(N)
        px = (np.sum(np.exp(np.square(data - np.reshape(data, (N, 1))) / (-2 * h ** 2)), axis=0) - np.ones(N)) / (
                    np.sqrt(2 * np.pi) * h) / N
        L = np.sum(np.log(px + 1e-9))
        return -L

    # find the optimal h
    h = np.array([10])
    opt_h = minimize(likelihood_func, h, method='SLSQP')
    kernel_density_estimate(N=N, h=opt_h.x[0], show=show)
