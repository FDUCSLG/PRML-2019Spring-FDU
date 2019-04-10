import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse

parser = argparse.ArgumentParser(description='Nonparametric \
                                density estimation. ')
parser.add_argument('-method', type=str, default='NN', \
    choices=['Histogram', 'Gaussian', 'NearestNeighbour', \
             'H', 'GK', 'NN'], \
    help='Choose a method in Histogram(or H), Gaussian(or GK), \
            NearestNeighbour(or NN). ')
parser.add_argument('-N', type=int, default=200, \
    help='The number of data points. Default is 200. ')
parser.add_argument('-B', type=int, default=50, \
    help='Bins in Histogram method. Default is 50. ')
parser.add_argument('-H', type=float, default=1.0, \
    help='H in Gaussian Kernel method. Default is 1. ')
parser.add_argument('-K', type=int, default=5, \
    help='K in Nearest Neighbour method. Default is 5. ')

args = parser.parse_args()
print(args)

def gaussian_kernel(x, xn:list, h, N):
    '''
    Gaussian Density Estimation
    
    input:
        -x: single data point
        -xn: list of sampled data
        -h: smooth parameter
        -N: number of data points
    
    output: probability density of x
    '''
    res = 0
    for xi in xn:
        delta = abs(x - xi)
        res += 1 / math.sqrt(2 * math.pi * (h ** 2)) \
                * math.exp(delta ** 2 / -2 / (h ** 2))
    return res / N

def nearest_neighbour(x, xn:list, k, N):
    '''
    K-Nearest Neighbour Density Estimation
    
    input:
        -x: single data point
        -xn: list of sampled data
        -k: smooth parameter
        -N: number of data points
    
    output: probability density of x
    '''
    dis = []
    for xi in xn:
        if not xi == x:
            dis.append(abs(xi - x))
    dis.sort()
    if dis[k-1] == 0:
        return float('Inf')
    return k / N / dis[k-1]

def plot_H(sampled_data, bins):
    plt.title('Histogram Density Estimation \n' \
            + str(N) + ' data points, ' + str(bins) + ' bins')
    plt.hist(sampled_data, density=True, bins=bins)
    plt.show()

def plot_GK(sampled_data, H):
    plt.title('Gaussian Kernel Density Estimation \n' \
                + str(N) + ' data points, ' + 'H is ' + str(H))
    
    mx = max(sampled_data)
    mi = min(sampled_data)

    X = np.linspace(mi, mx, num=N)
    y = [gaussian_kernel(xi, sampled_data, H, N) for xi in X]

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.plot(X, y)
    plt.show()

def plot_NN(sampled_data, K):
    plt.title('K Nearest Neighbour Desntity Estimation \n' \
                + str(N) + ' data points, ' + 'K is ' + str(K))

    mx = max(sampled_data)
    mi = min(sampled_data)

    X = np.linspace(mi, mx, num=N)
    y = [nearest_neighbour(xi, sampled_data, K, N) for xi in X]

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.plot(X, y)
    plt.show()

'''
def plot_mul():
    # for report
    N = [200, 200, 200, 200]
    K = [1, 5, 25, 50]
    H = []
    data = [get_data(i) for i in N]
    mx = [max(dat) for dat in data]
    mi = [min(dat) for dat in data]
    X = [np.linspace(mi[i], mx[i], num=N[i]) for i in range(4)]
    y = [0, 0, 0, 0]
    for j in range(4):
        y[j] = [nearest_neighbour(xi, data[j], K[j], N[j]) for xi in X[j]]

    for i in range(4):
        plt.subplot(221+i)
        plt.title('KNN, '+str(N[i])+'data points, K is '+str(K[i]))
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.plot(X[i], y[i])
    plt.show()
'''

# plot_mul()

if __name__ == '__main__':  
    N = args.N
    sampled_data = get_data(N)

    if args.method in ['Histogram', 'H']:
        plot_H(sampled_data, args.B)
    elif args.method in ['Gaussian', 'GK']:
        plot_GK(sampled_data, args.H)
    elif args.method in ['NearestNeighbour', 'NN']:
        plot_NN(sampled_data, args.K)
