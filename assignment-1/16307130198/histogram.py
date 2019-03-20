import numpy as np
import matplotlib.pyplot as plt
from gm1d_plot import gm1d_plot 

def estimation_histogram(sampled_data, gm1d, bin_num):
    # generate sub-graphs
    gm1d_plot(gm1d)
    a = plt.hist(sampled_data, normed=True, bins=bin_num, label='bin_num: '+str(bin_num))
    plt.title("Histogram Method\nData Number: " + str(len(sampled_data)))
    plt.legend(loc='best', prop={'size':12})
    plt.show()


