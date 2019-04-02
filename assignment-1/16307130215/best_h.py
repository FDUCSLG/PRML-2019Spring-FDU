import os,argparse
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# failuer
# def Histogram_bins(bins,sampled_data):

#     h = plt.hist(sampled_data, normed=True, bins=int(bins[0]))
#     y,x=h[0],h[1]
#     print(bins[0])
#     likelihood = 0
#     bias = 1e-50
#     first_heigher = 0
#     for data in sampled_data:
#         while first_heigher <len(y) and x[first_heigher]<=data:
#             first_heigher+=1
#         likelihood -= np.log(y[first_heigher-1]+bias)

#     return likelihood

def Gaussian_kernel_h(h,sampled_data):
    h=h[0]
    print(h)
    x = np.sort(sampled_data)
    num_data = len(x)
    cons = 1/( (np.sqrt(2*np.pi*h*h))*num_data )
    y = np.zeros(num_data)
    for i in range (num_data):
        y[i] = np.sum( np.exp( -((x-x[i])**2) / (2*h*h) ) )
        y[i] -= 1
    y = y*cons
    return np.sum(-np.log(y+1e-50))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_data',type=int,required=True,help="数据量")
    
    args = parser.parse_args()
    sampled_data = get_data(args.num_data)

    h=[0.2]
    x=minimize(Gaussian_kernel_h,h,args=(sampled_data,),method='SLSQP')
    print(x)