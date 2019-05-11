import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse as ap
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import sys
from scipy.optimize import minimize



# Histogram method
def Histogram(bins_size=50,num_data=200):
    sampled_data = get_data(num_data)
    plt.hist(sampled_data, normed=True,bins=bins_size, facecolor='slateblue')
    plt.title("num_data = %d & bins = %d" % (num_data, bins_size))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.savefig("img/histogram_sample_"+str(num_data)+"_bins_"+str(bins_size)+".png") 



def kNN(num_data=200, K=10):
    data = sorted(get_data(num_data))

    x = np.linspace(min(data), max(data), 100)
    px = []
    left = 0
    right = K - 1
    for xi in x:
        while right <num_data- 1 and data[right + 1] + data[left] < 2 * xi:
            right = right + 1
            left = left + 1
        px.append(0.5 / max(data[right] - xi, xi - data[left]))
    px = np.array(px) * K / num_data

    plt.plot(x, px, label="K = %d" % (K))
    plt.legend()
    plt.title("N = %d, K = %d" % (num_data, K))
    plt.savefig("img/knn_sample_"+str(num_data)+"_k_"+str(K)+".png") 

    

# Kernel Density Estimation
def KernelDensity(num_data, h):
    sampled_data = get_data(num_data)
    min_range = min(sampled_data) - 3
    max_range = max(sampled_data) + 3
    xs = np.linspace(min_range, max_range, 2000)
    ys = np.zeros_like(xs)
    print(h)
    index = 0
    for x in xs:
        tmp = 0
        for xn in sampled_data:
            tmp += np.exp(-(np.power(x-xn, 2))/(2*np.power(h, 2)))/(np.sqrt(2*np.pi*np.power(h, 2)))
        ys[index] = tmp / num_data
        index += 1

    plt.title("num_data = %d & h = %f" % (num_data, h))
    plt.plot(xs, ys)
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.savefig("img/kernel_sample_"+str(num_data)+"_h_"+str(h)+".png") 
    plt.close()


# Finding optimal h
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

    KernelDensity(num_data=N, h=opt_h.x[0])




if __name__ == '__main__':  
    parser = ap.ArgumentParser(description="Non-parametric density estimation algorithms")
    parser.add_argument('--num_samp', help='The number of samples', type=int)

    parser.add_argument('--knn', help='knn plot', action='store_true')
    parser.add_argument('--K', help='Parameter K for knn', type=int)

    parser.add_argument('--histogram', help='Histogram plot', action='store_true')
    parser.add_argument('--num_bins', help='Parameter bins used in Histogram', type=int)

    parser.add_argument('--kde', help='Kernel_Density_Estimation', action='store_true')
    parser.add_argument('--h', help='Parameter h used in Kernel_Density_Estimation', type=float)
    parser.add_argument('--opt_h', help='Find the optimal value for h', action='store_true')



    args = parser.parse_args()
    if args.histogram:
        Histogram(args.num_bins,args.num_samp)
    elif args.kde:
        KernelDensity(args.num_samp, args.h)
    elif args.opt_h:
        optimal_h()
    elif args.knn:
        kNN(args.num_samp, args.K)
    elif args.opt_h:
        optimal_h()
     
