import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
import numpy as np
import matplotlib.pyplot as plt
import argparse as ap
import math as m
from handout import get_data

# Histogram Estimation
def Histogram_Estimation(num_data, bins_num):
    sampled_data = get_data(num_data)
    plt.title("num_data = %d & bins = %d" % (num_data, bins_num))
    plt.hist(sampled_data, density=True, bins=bins_num)
    plt.show()


# Kernel Density Estimation
def Kernel_Density_Estimation(num_data, h):
    sampled_data = get_data(num_data)
    min_range = min(sampled_data) - 3
    max_range = max(sampled_data) + 3
    xs = np.linspace(min_range, max_range, 2000)
    ys = np.zeros_like(xs)

    index = 0
    for x in xs:
        tmp = 0
        for xn in sampled_data:
            tmp += m.exp(-(m.pow(x-xn, 2))/(2*m.pow(h, 2)))/(m.sqrt(2*m.pi*m.pow(h, 2)))
        ys[index] = tmp / num_data
        index += 1

    plt.title("num_data = %d & h = %f" % (num_data, h))
    plt.plot(xs, ys)
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.show()


# Nearest Neighbor Estimation
def Nearest_Neighbor_Estimation(num_data, k):
    sampled_data = get_data(num_data)
    min_range = min(sampled_data) - 3
    max_range = max(sampled_data) + 3
    xs = np.linspace(min_range, max_range, 2000)
    ys = np.zeros_like(xs)

    index = 0
    for x in xs:
        data_list = []
        for xn in sampled_data:
            data_list.append(abs(x-xn))
        data_list.sort()
        ys[index] = k / (num_data*2*(data_list[k]+1e-10))
        index += 1

    plt.title("num_data = %d & k = %d" % (num_data, k))
    plt.plot(xs, ys)
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.show()


if __name__ == '__main__':
    parser = ap.ArgumentParser(description="Assignment-1 non-parametric density estimation algorithms")
    parser.add_argument('--num', help='the number of data', type=int)

    parser.add_argument('--he', help='Histogram_Estimation', action='store_true')
    parser.add_argument('--bins', help='Parameter bins used in Histogram_Estimation', type=int)

    parser.add_argument('--kde', help='Kernel_Density_Estimation', action='store_true')
    parser.add_argument('--h', help='Parameter h used in Kernel_Density_Estimation', type=float)

    parser.add_argument('--nne', help='Nearest_Neighbor_Estimation', action='store_true')
    parser.add_argument('--k', help='Parameter k used in Nearest_Neighbor_Estimation', type=int)

    args = parser.parse_args()
    if args.he:
        Histogram_Estimation(args.num, args.bins)
    elif args.kde:
        Kernel_Density_Estimation(args.num, args.h)
    elif args.nne:
        Nearest_Neighbor_Estimation(args.num, args.k)
        # gm1d.plot(args.num)
