import os
os.sys.path.append('..')

import math
import random

import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from scipy.optimize import minimize

from handout import get_data
from handout import GaussianMixture1D

class kde(object) :

    def __init__(self, dataset, bandwidth = None) : 
        dataset.sort()
        self.d = dataset
        self.n = len(dataset)
        
        self.h = bandwidth 
        # if bandwidth == None : 
            # self.h = self.cross_validation()
            # print ("Using cross-validation on h: " + str(self.h))
        '''
        # Rule of thumb of h
        self.mean = sum(dataset) / self.n
        self.var = sum( (d - self.mean) ** 2 for d in self.d) / self.n
        self.sd = self.var ** 0.5
        h = (4 * (self.sd ** 5) / ( 3 * self.n)) ** (0.2)
        print ('Rule of thumb h: ' + str(h))
        '''


    # Kernel functions
    # It seems that result is almost same
    def gaussian(self, u) :
        return math.exp(- (u ** 2) / 2) / math.sqrt(2 * math.pi)
    def logistic(self, u) :
        return 1 / (math.exp(u) + 2 + math.exp(-u))
    def sigmoid(self, u) :
        return 2 / (math.pi * (math.exp(u) + math.exp(-u)))
    def silverman(self, u) :
        return 0.5 * math.exp(- abs(u) / math.sqrt(2)) * math.sin(abs(u) / math.sqrt(2) + math.pi / 4)

    def set_kernel(self, k) :
        if k == 'gaussian' :
            self.kernel = self.gaussian

    def set_bandwidth(self, h) :
        self.h = h
    
    def evaluate(self, xs, kernel = 'gaussian', method = None) :
        y = []
        self.set_kernel(kernel)
        kernel = self.kernel

        n = self.n
        d = self.d
        # KDE
        if method == None :
            h = self.h
            for x in xs:
                y.append((sum(kernel((x-xd)/h)/h for xd in d))/n)

        # Variable KDE
        else :
            hs = self.hs
            for x in xs :
                y.append(1 / n * sum(kernel((x-xd)/h)/h for xd, h in zip(d, hs)))
        return y

    __call__ = evaluate

    '''
    This function computes bandwidths(h) for each data point and works as follows :

    1. Use KNN to compute the max distance (width) between the Kth Nearest Neighbour of each point in the trainning data.
    2. Using sigmoid function to smooth bandwidth with parameters upper/lower, which govern the range of the output.
    
    '''
    def compute_variable_bandwidth(self, k = 10, upper = 1.0, lower = 0.4) :
        h = []
        d = self.d
        n = self.n

        for x in d :
            i = 0
            for i in range (n - k) :
                if abs(d[i] - x) < abs(d[i + k] - x): break
            width = d[i + k - 1] - d[i]
            h.append(width)

        m = sum(h) / len(h) # Mean

        # Using Sigmoid function to smooth h
        return [(upper * math.exp(x - m) / (1 + math.exp(x - m)) + lower) for x in h]
            
    def pdf(self, x) :
        x = [x]
        return self.evaluate(x)[0]

    '''
    Cross-validation to generate a bandwidth
    s: Split the whole randomly into s sets
    h: Among which the best h is chosen
    '''
    def cross_validation(self, s = 4, h = np.linspace(0.1, 1, 50)) :

        d = self.d
        n = self.n
        t = len(d) // s * (s - 1)
        
        # Split
        split = [[] for i in range(s)]
        for i in range(0, self.n, s) :
            j = s
            if j + i > n : j = n - i
            r = [x for x in range(j)]
            random.shuffle(r)
            for ii in range(j) : split[ii].append(d[i + r[ii] - 1])
        # Leave-one-out
        score = []
        for hh in h :
            self.set_bandwidth(hh)
            sc = 0
            for i in range(s) :
                test = []
                for j in range(s) :
                    if j != i : test += split[j]
                self.d = test
                sc += np.prod([self.pdf(x) for x in split[i]])
            score.append(sc / s)

        best = h[np.argmax(score)]

        self.d = d

        return best

    def variable_cv(self) :
        cv = self.cross_validation()
        self.hs = self.compute_variable_bandwidth(lower = cv/2, upper = cv * 2)
        
def plot_true_distribution(self, num_sample = 100) :
    sampled_data = self.sample([num_sample])
    min_range = min(self.modes) - 3 * self.std_range[1]
    max_range = max(self.modes) + 3 * self.std_range[1]
    xs = np.linspace(min_range, max_range, 2000)
    ys = np.zeros_like(xs)
    for l, s, w in zip(self.modes, self.stds, self.weights):
        ys += ss.norm.pdf(xs, loc=l, scale=s) * w
    plt.plot(xs, ys)

# Following will work only if the .py is run as a script, not when imported
if __name__ == '__main__' :

    Hs = [0.5, 2]
    num_data = 100
    legends = []

    data = get_data(num_data)

    density = kde(data)

    l = min(data)
    r = max(data)

    d = (r - l) / 10
    l -= d
    r += d
    xs = np.linspace(l, r, 100)

    # True Distribution
    np.random.seed(0)
    gm1d = GaussianMixture1D(mode_range=(0, 50))
    plot_true_distribution(gm1d, num_sample=1000)
    legends.append('True distribution')

    # Cross-validation
    cv = density.cross_validation()
    density.set_bandwidth(cv)
    plt.plot(xs, density(xs))
    legends.append('Cross-validation h = ' + str(round(cv, 2)))

    # Variable Cross-validation
    vcv = density.variable_cv()
    plt.plot(xs, density(xs, method = 'variable'))
    legends.append('Variable Cross-validation')

    # Normal KDE
    for h in Hs :
        density.set_bandwidth(h)
        plt.plot(xs, density(xs))
        legends.append('h = ' + str(h))

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Kernel Density Estimation')
    plt.legend(legends)
    # plt.savefig('kde.png', dpi = 300)
    plt.show()
