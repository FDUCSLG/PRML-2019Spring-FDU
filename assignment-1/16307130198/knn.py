import numpy as np
import matplotlib.pyplot as plt

import math
from gm1d_plot import gm1d_plot 

def distance_cal(x1,x2):
    #return math.pow((x1-x2)**2, 1/2)
    return np.abs(x1-x2)

def knn_estimate_density(prediction_x, sequence_x, k):
    max_length = len(sequence_x)
    es_density = []
        
    mid_position = 0
    for i in prediction_x:
        left = 0
        right = 0
        
        while sequence_x[mid_position]<i:
            if mid_position == max_length-1:
                break
            else:
                mid_position += 1
        
        left = mid_position-1
        right = mid_position 
        count_k = 0
        
        while True:
            if left == -1:
                rest_num = k - count_k
                right = right + rest_num 
                break 
            elif right == (max_length-1):
                rest_num = k - count_k 
                left = left - rest_num
                break 
            else:
                d1 = distance_cal(i, sequence_x[left])
                d2 = distance_cal(i, sequence_x[right])
                if d1 < d2:
                    count_k += 1
                    left -= 1
                    if count_k == k:
                        break
                else:
                    count_k += 1
                    right += 1
                    if count_k == k:
                        break
        left += 1
        right -= 1
        
        if np.abs(sequence_x[left]-i) < np.abs(sequence_x[right] - i):
            V = 2 * (1e-50 + np.abs(sequence_x[right] - i))
        else:
            V = 2 * (1e-50 + np.abs(sequence_x[left] - i))
        
        es_density.append(k/(V * max_length))
    
    return es_density 
                    

def estimation_knn(sampled_data, gm1d, k):
    #gm1d.plot()
    gm1d_plot(gm1d)
    prediction_x = np.linspace(0,50,100)
    sampled_data = np.array(sampled_data)
    sampled_data.sort()
    es_density = knn_estimate_density(prediction_x, sampled_data, k) 
    plt.plot(prediction_x, es_density, label="K: "+str(k))
    plt.legend(loc='best', prop={'size':12})
    plt.title("KNN Estimation Method\nData Number: "+str(len(sampled_data)))
    plt.show()
