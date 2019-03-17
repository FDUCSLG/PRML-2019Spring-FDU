import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt
from math import exp,pi,pow,sqrt
import argparse

def he(num_data, num_bins):
    sampled_data = get_data(num_data)
    plt.hist(sampled_data, normed=True, bins=num_bins)
    plt.show()


def kde(num_data,h):
    sampled_data = get_data(num_data)
    xs = np.linspace(min(sampled_data)-3*np.std(sampled_data),
                     max(sampled_data)+3*np.std(sampled_data), 2000)
    ys = np.zeros_like(xs)
    for i,x in enumerate(xs):
        for xi in sampled_data:
            ys[i] += exp(-pow(x-xi,2)/(2*h*h))/(sqrt(2*pi*h*h)*num_data)
    plt.plot(xs, ys)
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.show()

def nnde(num_data,k):
    sampled_data = get_data(num_data)
    xs = np.linspace(min(sampled_data)-3*np.std(sampled_data),
                     max(sampled_data)+3*np.std(sampled_data), 2000)
    ys = np.zeros_like(xs)
    for i,x in enumerate(xs):
        dist = []
        for xi in sampled_data:
            dist.append(abs(x-xi))
        dist.sort()
        ys[i]=k/(num_data*2*(dist[k]+1e-9))
    plt.plot(xs, ys)
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--method",type=str,choices=['hist','kde','nearest'],help="method used to estimate density: hist,kde or nearest")
    parser.add_argument("-n","--numdata",type=int,help="num of data used")
    parser.add_argument("-pb","--bins",type=int,help="number of bins used in hist")
    parser.add_argument("-ph","--parah",type=float,help="h used in kde")
    parser.add_argument("-pk","--parak",type=int,help="k used in nearest neighbor")
    args = parser.parse_args()
    if args.method == 'hist':
        he(args.numdata,args.bins)
    elif args.method == 'kde':
        kde(args.numdata,args.parah) 
    elif args.method == 'nearest':
        nnde(args.numdata,args.parak)
