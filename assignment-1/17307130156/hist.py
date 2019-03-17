import os
os.sys.path.append('..')
import math
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

from handout import get_data
from handout import GaussianMixture1D

def plot_true_distribution(num_sample = 1000) :

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

if __name__ == '__main__' :

    N = 200
    Bs = [2, 10, 30] 

    # Percent of empty bins
    Ks = [0.01, 0.1, 0.2]

    data = get_data(N)

    # Histogram with different #bins
    legends = ['True Distribution']
    plot_true_distribution()
    for b in Bs :
        plt.hist(data, density=True, bins=b, alpha=0.4)
        legends.append('bins = ' + str(b))
   
    plt.title('Histogram')
    plt.legend(legends)
    # plt.savefig('hist', dpi=300)

    # Histogram with fixed percent of empty bins
    for k in Ks :
        t = int(N * k)
        l = 0
        r = N

        # Binary Search
        while (l < r) :
            m = (l + r) // 2 
            plt.figure()
            (n, bins, patches) = plt.hist(data, bins=m, histtype='step') # histtype='step' is faster
            # empty bins
            cnt = sum(1 if i == 0 else 0 for i in n)
            if cnt > t : r = m
            elif cnt < t : l = m + 1 
            else :# to choose the largest one 
                l = m
                if m == l : r = l # break
            plt.close()

        plt.figure()
        legends = ['True Distribution']
        plot_true_distribution()
        plt.hist(data, bins=l, density=True, alpha=0.6)
        legends.append('k = '+ str(k) + ', bins = ' + str(l))
        plt.legend(legends)    
        plt.title('Histogram with fixed percent of empty bins')
        # plt.savefig('histp'+ str(l), dpi=300)

    plt.show()

