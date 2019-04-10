import numpy as np
import matplotlib.pyplot as plt


# assessment method for histogram
h2k = lambda h, x : np.ceil((max(x) - min(x)) / h)

square_root_choice = lambda x : np.ceil(np.sqrt(len(x)))

sturges_formula = lambda x : np.ceil(np.log2(len(x))) + 1

rice_rule = lambda x : np.ceil(2 * len(x)**(1/3))

scotts_normal_reference_rule = lambda x : h2k(3.5 * np.std(x) / len(x)**(1/3), x)

def shimazaki_and_shinomoto(x):
    x_max = max(x)
    x_min = min(x)
    N_MIN = 4

    N_MAX = 50
    N = range(N_MIN,N_MAX)
    N = np.array(N)
    D = (x_max-x_min)/N 
    C = np.zeros(shape=(np.size(D),1))

    for i in range(np.size(N)):
        edges = np.linspace(x_min,x_max,N[i]+1)
        ki = plt.hist(x,edges)
        ki = ki[0]    
        k = np.mean(ki)
        v = np.sum((ki-k)**2)/N[i]
        C[i] = (2*k-v)/((D[i])**2)

    cmin = min(C)
    idx  = np.where(C==cmin)
    idx = int(idx[0])
    return N[idx]

# assessment method for KDE
hMISE = lambda x : 1.06 * np.std(x) / len(x)**(1 / 5)

def MLCV_KDE(h, sample_data):
    gaussian_kernel = lambda x : np.exp(- x**2 / 2) / (2 * np.pi)**0.5

    x = sample_data
    N = len(x)

    xi = np.array(x).reshape(N, 1)
    xj = np.array(x).reshape(1, N)
    extended_xi = xi.repeat(N, axis=1)
    extended_xj = xj.repeat(N, axis=0)

    input_x = (extended_xj - extended_xi) / h

    gaussian_value = gaussian_kernel(input_x)
    inner_sum = gaussian_value.sum(axis=1) - 1 / (2 * np.pi)**0.5
    log_inner_sum = np.log(inner_sum+1e-16)

    result = np.sum(log_inner_sum) / N - np.log((N - 1) * h)
    return - result