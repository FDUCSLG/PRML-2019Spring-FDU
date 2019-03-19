import numpy as np
import matplotlib.pyplot as plt 
import math
from gm1d_plot import gm1d_plot 

def graussian_kernel_cal(prediction_x, sequence_x, h):
    N = len(sequence_x)
    es_density = []
    
    prediction_num = len(prediction_x)
    sequence_num = len(sequence_x)

    prediction_matrix = np.array(prediction_x).repeat(sequence_num).reshape((prediction_num, sequence_num))
    temp = np.exp((prediction_matrix - sequence_x)**2 / ((-1)*(2*h*h)))
    es_density = (np.sum(temp, axis=1)/((N)*(math.pow((2*math.pi*h*h),1/2))))

    test_es_density = []
    #for i in prediction_x:
    #    sum = 0
    #    for j in sequence_x:
    #        sum += math.exp((-1) * math.pow(i-j, 2) / ( 2 * h * h))
        
    #    test_es_density.append(sum * weight1 / N)
    
    #print(test_es_density-es_density)
    #import sys
    #sys.exit()
    return es_density 
        
def estimation_kernel(sampled_data, gm1d, h):
    gm1d_plot(gm1d)
    data_min = min(sampled_data)-2
    data_max = max(sampled_data)+2
    prediction_x = np.linspace(data_min, data_max,2000)
    #new_h = 1.06 * np.std(sampled_data) * math.pow(len(sampled_data), -1/5)
    es_density = np.array(graussian_kernel_cal(prediction_x, sampled_data, h))
    plt.plot(prediction_x, es_density, label="kernel estimation\nh: "+str(h))
    plt.title('Kernel Estimation Method\nData Number: '+str(len(sampled_data))) 
    plt.legend(loc='best', prop={'size':12})
    plt.show()
