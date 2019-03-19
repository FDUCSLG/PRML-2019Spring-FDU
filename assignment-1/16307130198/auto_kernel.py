import numpy as np
import matplotlib.pyplot as plt 
import math
import scipy
from gm1d_plot import gm1d_plot

def middle_process_cal(a):
    param1 = math.exp((-1) * math.pow(a, 2) / 2)
    return param1 / math.pow(2*math.pi, 1/2)

def MLCV(sequence_x):
    N = len(sequence_x)
    prediction_num = len(sequence_x)
    prediction_x = sequence_x
    select_h = [0.01*(i+1) for i in range(100)]
    best_h = 0
    maximum_MLCV=-1
    for h in select_h:
        prediction_matrix = np.array(prediction_x).repeat(N).reshape((prediction_num, N))
        
        temp = (np.exp((-1)*(((prediction_matrix - sequence_x)/h)**2)/2))
        temp = (np.sum(temp, axis=1) - 1 )/(math.pow(1/(2*math.pi), 1/2))
        
        temp += 1e-50
        temp = np.log10(temp)

        MLCV = np.sum(temp)/N - np.log10((N-1)*h)
        
        print(h)
        print(MLCV)
        if maximum_MLCV == -1 or MLCV > maximum_MLCV:
            maximum_MLCV = MLCV 
            best_h = h

    print(best_h)
    print(maximum_MLCV)
    return best_h

def MLCV_cal(h, sequence_x): 
    h = h[0]
    print(h)
    N = len(sequence_x)
    prediction_num = len(sequence_x)
    prediction_x = sequence_x
     
    prediction_matrix = np.array(prediction_x).repeat(N).reshape((prediction_num, N))
        
    temp = (np.exp((-1)*(((prediction_matrix - sequence_x)/h)**2)/2))
    temp = (np.sum(temp, axis=1) - 1 )/(math.pow(1/(2*math.pi), 1/2))
        
    temp += 1e-50
    temp = np.log10(temp)

    MLCV = np.sum(temp)/N - np.log10((N-1)*h)
    
    return -1 * MLCV

def graussian_kernel_density_estimation(prediction_x, sequence_x):
    N = len(sequence_x)
    es_density = []
    #h = MLCV(sequence_x)
    res = scipy.optimize.minimize(MLCV_cal, 0.2, args=(sequence_x,),method='SLSQP')
    print(res)
    h = res.x[0]
    weight1 = (math.pow(1/(2*math.pi*h*h), 1/2))
    for i in prediction_x:
        sum = 0
        for j in sequence_x:
            sum += math.exp((-1) * math.pow(i-j, 2) / ( 2 * h * h))
                
        es_density.append(sum * weight1 / N)
    
    return es_density,h
        
def estimation_auto_kernel(sampled_data, gm1d):
    gm1d_plot(gm1d)
    data_min = min(sampled_data)-2
    data_max = max(sampled_data)+2
    prediction_x = np.linspace(data_min, data_max,500)
    #prediction_x = np.sort(sampled_data)
    es_density,h = np.array(graussian_kernel_density_estimation(prediction_x, sampled_data))
    plt.title('Kernel Estimation Method\nData Number: '+str(len(sampled_data)))
    plt.plot(prediction_x, es_density, label='cross_validation-maximum_likelihood\nh: '+ str(h)[0:8])
    plt.legend(loc='best', prop={'size':12})
    plt.show()
