# path to top module
import os
os.sys.path.append('..')

# lib
from handout import get_data
from handout import gm1d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from plot import plot_gm1d


def kNN(N=200, K=10):
    assert K > 0
    assert K <= N

    data = sorted(get_data(N))

    x = np.linspace(min(data), max(data), 100)
    px = []
    left = 0
    right = K - 1
    for xi in x:
        while right < N - 1 and data[right + 1] + data[left] < 2 * xi:
            right = right + 1
            left = left + 1
        px.append(0.5 / max(data[right] - xi, xi - data[left]))
    px = np.array(px) * K / N

    plt.plot(x, px, label="kNN")
    plt.legend()
    plt.title("N = %d, K = %d" % (N, K))
    gm1d.plot()


def kNN_matrix(N=200, K=10):
    assert K > 0
    assert K <= N

    data = get_data(N)

    x = np.linspace(min(data), max(data), 100)
    distance = np.abs(x - np.reshape(data, (N, 1)))
    px = K / N * 0.5 / np.sort(distance, axis=0)[K - 1, :]

    plt.plot(x, px, label="kNN_matrix")
    plt.legend()
    plt.title("N = %d, K = %d" % (N, K))
    gm1d.plot()


def kNN_kdtree(N=200, K=10, show=True):
    assert K > 0
    assert K <= N

    data = get_data(N)
    tree = KDTree(np.reshape(data, (N, 1)))

    x = np.linspace(min(data), max(data), 100).reshape((100, 1))
    matrix = tree.query(x, k=K, p=1)
    px = K / N * 0.5 / np.abs(tree.data[matrix[1][:, K - 1]] - x)

    plt.plot(x, px, label="kNN_kdtree")
    plt.legend()
    plt.title("N = %d, K = %d" % (N, K))
    plot_gm1d()
    if show:
        plt.show()


def kNN_result():
    plt.subplot(131)
    plt.xlabel("x")
    plt.ylabel("p(x)")
    kNN_kdtree(N=200, K=5, show=False)

    plt.subplot(132)
    plt.xlabel("x")
    plt.ylabel("p(x)")
    kNN_kdtree(N=200, K=14, show=False)

    plt.subplot(133)
    plt.xlabel("x")
    plt.ylabel("p(x)")
    kNN_kdtree(N=200, K=50, show=False)

    plt.show()


def optimal_K(N=200, show=True):
    opt_K = np.sqrt(N)
    kNN_kdtree(N=N, K=int(opt_K), show=show)
