import os
os.sys.path.append('..')
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt


def histogram_estimation_graph(data_num, bins_num):
    sampled_data = get_data(data_num)
    plt.hist(sampled_data, density=True, bins=bins_num)
    plt.title('Histogram - bins = ' + str(bins_num) + ', num_data = ' + str(data_num))
    plt.show()


def kernel_estimation_implement(x_list, data_list, h):
    n = len(data_list)
    res = []
    for i in range(len(x_list)):
        temp = x_list[i]
        rec = 0
        for j in range(n):
            rec += np.exp((-0.5) * (np.power(temp - data_list[j], 2)) / np.power(h, 2))
        res.append((1/n) * (1/(np.power(2 * np.pi * np.power(h, 2), 0.5))) * rec)
    return res


def kernel_estimation_graph(data_num, h):
    sample_data = get_data(data_num)
    x_list = np.linspace(min(sample_data) - 1, max(sample_data) + 1, 2000)
    p_x_list = kernel_estimation_implement(x_list, sample_data, h)

    plt.plot(x_list, p_x_list)
    plt.title('Kernel - h = ' + str(h) + ', num_data = ' + str(data_num))
    plt.show()


def nearest_neighbor_method_implement(x_list, data_list, k):
    n = len(data_list)
    data_list = sorted(data_list)
    res = []
    for i in range(len(x_list)):
        rec = []
        for j in range(n):
            rec.append(abs(x_list[i] - data_list[j]))
        rec.sort()
        res.append(k/(n*rec[k - 1]))
    return res


def nearest_neighbor_method_graph(data_num, k):
    sample_data = get_data(data_num)
    x_list = np.linspace(min(sample_data) - 1, max(sample_data) + 1, 2000)
    p_x_list = nearest_neighbor_method_implement(x_list, sample_data, k)

    plt.plot(x_list, p_x_list)
    plt.title('Nearest_neighbor - K = ' + str(k) + ', num_data = ' + str(data_num))
    plt.show()

if __name__ == '__main__':
    '''
    Simply change the parameters of a certain funtion to plot the estimation.
    
    1. For histogram estimation:
        histogram_estimation_graph(num_data, bins)
        
    2. For kernel estimation:
        kernel_estimation_graph(num_data, h)
        
    3. For nearest ntighbor estimation:
        nearest_neighbor_method_graph(num_data, K)
    
    '''
    histogram_estimation_graph(200, 45)  # num_data = 200, bins = 45
    kernel_estimation_graph(100, 0.834)  # num_data = 100, h = 0.834
    nearest_neighbor_method_graph(200, 5)  # num_data = 200, K = 5
