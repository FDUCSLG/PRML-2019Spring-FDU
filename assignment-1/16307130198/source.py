import os
os.sys.path.append('..')
os.sys.path.append('../..')
# use the above line of code to surpass the top module barrier
import handout
#from handout import gm1d 
import numpy as np
import matplotlib.pyplot as plt 
import argparse
import os
os.sys.path.append("..")
from histogram import estimation_histogram
from different_rule_histogram import estimation_dif_rule_histogram
from kernel import estimation_kernel
from auto_kernel import estimation_auto_kernel  
from knn import estimation_knn



parser = argparse.ArgumentParser()
parser.add_argument("--num_data", type=int, default=500, help="number of the data points")
parser.add_argument("--estimation_method", type=str, choices=['histogram','different_rule_histogram', 'kernel', 'auto_kernel', 'knn'],
                    default='histogram', help="choose a den-estimation method")
parser.add_argument("--bin_num", type=int, default=50, help="bin number of histogram method")
parser.add_argument("--kernel_h", type=float, default=0.5, help="parameter h of gaussian kernel")
parser.add_argument("--knn_k", type=int, default=50, help="k of knn algorithm")
args = parser.parse_args()


# gain parameters from args
num_data = args.num_data
estimation_method = args.estimation_method
bin_num = args.bin_num
kernel_h = args.kernel_h
knn_k = args.knn_k 
# create class gm1d and get the training data
gm1d = handout.gm1d
sampled_data = handout.get_data(num_data)

if estimation_method == 'histogram':
    estimation_histogram(sampled_data, gm1d, bin_num)
elif estimation_method == 'different_rule_histogram':
    estimation_dif_rule_histogram(sampled_data, gm1d)
elif estimation_method == 'kernel':
    estimation_kernel(sampled_data, gm1d, kernel_h)
elif estimation_method == 'auto_kernel':
    estimation_auto_kernel(sampled_data, gm1d)
elif estimation_method == 'knn':
    estimation_knn(sampled_data, gm1d, knn_k)


