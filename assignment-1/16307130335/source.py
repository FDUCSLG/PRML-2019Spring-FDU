import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import *
import numpy as np
import matplotlib.pyplot as plt
import argparse

from math import *


def gaussian(x):
    return (1/(sqrt(2*pi))) * exp(-0.5*x*x)


def best_h(sampled_data):
    std = np.std(sampled_data, ddof=1)
    h = 1.06*std*pow(len(sampled_data), -1/5)
    return h

def handler_sample(num=500):
    sampled_data = get_data(num)
    min_data = min(sampled_data)
    max_data = max(sampled_data)
    x_plot = np.linspace(min_data, max_data, 10000)
    return sampled_data, min_data, max_data, x_plot


def display(x, fx, string):
    plt.plot(x, fx)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(string)
    gm1d.plot()


def histogram(num=500, bin=100):
    sampled_data = get_data(num)
    plt.hist(sampled_data, normed=True, bins=bin, edgecolor="black",alpha=0.7)
    plt.title('histogram')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()


def kernel_density(k=gaussian, h=0.75, num=500):
    sampled_data, min_data, max_data, x_plot = handler_sample(num)
    fx = []
    for i in x_plot:
        px = 0
        for j in sampled_data:
            px += k((j-i)/h)
        px = px / (h*num)
        fx.append(px)
    display(x_plot, fx, "kernel density of gaussian")


def k_nearest_neighbor(k=5, num=500):
    sampled_data, min_data, max_data, x_plot = handler_sample(num)
    fx = []
    sum = 0
    for i in x_plot:
        dist = []
        for j in sampled_data:
            dist.append(abs(j-i))
        dist.sort()
        fx.append(k/(4*num*dist[k-1]))
    display(x_plot, fx, "k nearest neighbor")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", "-m", default="hist", choices=["hist", "kde", "knn"])
    parser.add_argument("--number", "-n", default=100, type=int)
    parser.add_argument("--bins", "-b", default=50, type=int)
    parser.add_argument("--k_near", "-k", default=10, type=int)
    parser.add_argument("--bandwidth", '-d', default=0.75, type=float)
    args = parser.parse_args()
    if(args.methods == "hist"):
        histogram(args.number, args.bins)
    if(args.methods == "kde"):
        kernel_density(gaussian, args.bandwidth, args.number)
    if(args.methods == "knn"):
        k_nearest_neighbor(args.k_near, args.number)





if __name__ == "__main__":
    main()