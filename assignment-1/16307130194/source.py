# path to top module
import os
os.sys.path.append('..')

# histogram
from histogram import histogram_method
from histogram import histogram_result
from histogram import optimal_bin

# kernel density estimate
from kernel import kernel_density_estimate
from kernel import kde_result
from kernel import optimal_h

# kNN
from knn import kNN_kdtree
from knn import kNN_result
from knn import optimal_K

# other lib
import argparse


def parser():
    parser = argparse.ArgumentParser(description='Non-parameter Density Estimation')

    parser.add_argument('--algorithm',
                        choices=["histogram_method", "histogram_result", "optimal_bin",
                                 "kernel_density_estimate", "kde_result", "optimal_h",
                                 "kNN_kdtree", "kNN_result", "optimal_K"],
                        help='The algorithms.')

    parser.add_argument('--n', default=200, type=int,
                        help='The amount of sample data.')
    parser.add_argument('--b', default=25, type=int,
                        help='The amount of bins for histogram.')
    parser.add_argument('--h', default=0.35, type=float,
                        help='The value of h in gaussian kernel.')
    parser.add_argument('--k', default=14, type=int,
                        help='The amount of nearest neighbors.')

    args = parser.parse_args()

    if args.n > 10000 or args.n <= 0:
        parser.error('Wrong n!')

    if args.b > args.n or args.b < 1:
        parser.error('Wrong b!')

    if args.h < 0:
        parser.error('Wrong h!')

    if args.k < 0 or args.k > args.n:
        parser.error('Wrong k!')

    n = args.n
    b = args.b
    h = args.h
    k = args.k

    algos = {
        "histogram_method": histogram_method,
        "histogram_result": histogram_result,
        "optimal_bin": optimal_bin,

        "kernel_density_estimate": kernel_density_estimate,
        "kde_result": kde_result,
        "optimal_h": optimal_h,

        "kNN_kdtree": kNN_kdtree,
        "kNN_result": kNN_result,
        "optimal_K": optimal_K
    }

    if args.algorithm == "histogram_result" or args.algorithm == "kde_result" or args.algorithm == "kNN_result":
        algos[args.algorithm]()
    elif args.algorithm == "optimal_bin" or args.algorithm == "optimal_h" or args.algorithm == "optimal_K":
        algos[args.algorithm](N=n)
    elif args.algorithm == "histogram_method":
        algos[args.algorithm](N=n, num_bin=b)
    elif args.algorithm == "kernel_density_estimate":
        algos[args.algorithm](N=n, h=h)
    elif args.algorithm == "kNN_kdtree":
        algos[args.algorithm](N=n, K=k)
    else:
        parser.print_help()


if __name__ == "__main__":
    parser()
