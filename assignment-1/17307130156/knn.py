import os
os.sys.path.append('..')
import math
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

from handout import get_data
from handout import GaussianMixture1D

def plot_true_distribution(self, num_sample = 100) :
    sampled_data = self.sample([num_sample])
    min_range = min(self.modes) - 3 * self.std_range[1]
    max_range = max(self.modes) + 3 * self.std_range[1]
    xs = np.linspace(min_range, max_range, 2000)
    ys = np.zeros_like(xs)
    for l, s, w in zip(self.modes, self.stds, self.weights):
        ys += ss.norm.pdf(xs, loc=l, scale=s) * w
    plt.plot(xs, ys)

class knn(object):
    def __init__(self, dataset, k = 1) : 
        dataset.sort()
        self.d = dataset
        self.k = k
        self.datasetLen = len(dataset)
    
    def evaluate(self, xs) :
        y = []
        n = len(self.d)
        k = self.k
        d = self.d
        for x in xs :
            i = 0
            for i in range(self.datasetLen - k) :
                if abs(d[i] - x) < abs(d[i+k] - x): break
            v = max(d[i + k - 1] - d[i], abs(x - d[i]))
            y.append(k / (n * v))
        return y
    __call__ = evaluate

if __name__ == '__main__' : 

    N = 200

    data = get_data(N)
    Ks = [5, 30, 50]

    legends = []

    l = min(data)
    r = max(data)
    d = (r - l) / 10
    l -= d
    r += d

    # True distribution
    np.random.seed(0)
    gm1d = GaussianMixture1D(mode_range=(0, 50))
    plot_true_distribution(gm1d, num_sample=1000)
    legends.append('True distribution')

    # KNN
    for k in Ks :
        density = knn(data, k)
        xs = np.linspace(l, r, 1000)
        plt.plot(xs, density(xs))
        legends.append('K = ' + str(k))
    
    plt.axis([l, r, 0, 0.4])
    plt.legend(legends)
    # plt.savefig('knn.png', dpi = 300)
    plt.show()
