import os
os.sys.path.append('..')
import math

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

from handout import get_data
from handout import GaussianMixture1D

from knn import knn
from kde import kde

global min_range
global max_range

if __name__ == '__main__' :
    np.random.seed(0)
    gm1d = GaussianMixture1D(mode_range=(0, 50))

    global min_range
    global max_range
    min_range = min(gm1d.modes) - 3 * gm1d.std_range[1]
    max_range = max(gm1d.modes) + 3 * gm1d.std_range[1]


def plot_true_distribution(num_sample = 100) :

    np.random.seed(0)
    gm1d = GaussianMixture1D(mode_range=(0, 50))
    self = gm1d
    sampled_data = self.sample([num_sample])
    min_range = min(self.modes) - 3 * self.std_range[1]
    max_range = max(self.modes) + 3 * self.std_range[1]
    xs = np.linspace(min_range, max_range, 2000)
    ys = np.zeros_like(xs)
    for l, s, w in zip(self.modes, self.stds, self.weights):
        ys += ss.norm.pdf(xs, loc=l, scale=s) * w
    plt.plot(xs, ys)

def requirement1() :

    global min_range
    global max_range

    ds = [100, 500, 1000, 10000]
    b = 100
    h = 0.1
    k = 10

    xs = np.linspace(min_range, max_range, 200)

    # Histogram as example
    legends = []
    data = get_data(200)
    plot_true_distribution(1000)
    legends.append('True distribution')
    for d in ds :
        data = get_data(d)
        plt.hist(data, density=True, bins=b, alpha=0.4)
        legends.append('#bin = ' + str(b) + ', #data = ' + str(d))
    plt.legend(legends)
    plt.title('Requirement 1-1')
    plt.savefig('req1-1', dpi=300)
    plt.show()

    # KDE as example
    plt.figure()
    legends = []
    data = get_data(200)
    plot_true_distribution(1000)
    legends.append('True distribution')
    density = kde(data)
    for d in ds :
        data = get_data(d)
        density = kde(data)
        density.set_bandwidth(h)
        plt.plot(xs, density(xs))
        legends.append('h = ' + str(h) + ', #data = ' + str(d))
    plt.legend(legends)
    plt.title('Requirement 1-2')
    plt.savefig('req1-2', dpi=300)
    plt.show()

    # KNN as example
    plt.figure()
    legends = []
    data = get_data(200)
    plot_true_distribution(1000)
    legends.append('True distribution')
    for d in ds :
        data = get_data(d)
        density = knn(data, k)
        plt.plot(xs, density(xs))
        legends.append('k = ' + str(k) + ', #data = ' + str(d))
    plt.legend(legends)
    plt.ylim([0, 0.4])
    plt.title('Requirement 1-3')
    plt.savefig('req1-3', dpi=300)
    plt.show()

def requirement2() :
    global min_range
    global max_range

    data = get_data(200)
    bs = [2, 10, 30]
    xs = np.linspace(min_range, max_range, 200)
    
    legends = []

    plot_true_distribution()
    legends.append('True Distribution')

    # Plotting histogram with different bins
    for b in bs :
        plt.hist(data, density=True, bins=b, alpha=0.4)
        legends.append('bins = ' + str(b))

    plt.title('Requirement 2')
    plt.legend(legends)
    plt.show()


def requirement3() :

    global min_range
    global max_range

    data = get_data(200)
    hs = [0.1, 1, 2]
    legends = []

    xs = np.linspace(min_range, max_range, 200)

    density = kde(data)

    plot_true_distribution(1000)
    legends.append('True distribution')

    # KDE with different h
    for h in hs :
        density.set_bandwidth(h)
        plt.plot(xs, density(xs))
        legends.append('h = ' + str(h))
    plt.legend(legends)
    plt.title('Requirement 3')
    plt.show()

    # Comment out to get Cross-validation KDE and Variable KDE
    '''
    plt.figure()
    legends = []
    density = kde(data)

    plot_true_distribution(1000)
    legends.append('True distribution')
    
    # KDE with Cross-validation
    cv = density.cross_validation()
    density.set_bandwidth(cv)
    plt.plot(xs, density(xs))
    legends.append('Cross-validation h = ' + str(round(cv, 2)    ))
    
    # Variable KDE with Cross-validation and KNN
    density.variable_cv()
    plt.plot(xs, density(xs, method='variable'))
    legends.append('Variable KDE')
    
    plt.legend(legends)
    plt.title('Requirement 3')
    plt.show()
    '''

def requirement4() :
    global min_range
    global max_range

    data = get_data(10000)
    ks = [1, 20, 50]
    legends = []

    xs = np.linspace(min_range, max_range, 200)
    
    plot_true_distribution()
    legends.append('True Distribution')

    # KNN with different K
    for k in ks :
        density = knn(data, k)
        plt.plot(xs, density(xs))
        legends.append('K = ' + str(k))
    plt.legend(legends)
    plt.title('Requirement 4')
    plt.ylim([0, 0.4])
    plt.show()
    
requirement1()
requirement2()
requirement3()
requirement4()
