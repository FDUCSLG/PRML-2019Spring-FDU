import os,argparse
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_data
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

def Sqrt(sampled_data):
    return np.ceil(np.sqrt(len(sampled_data)))

def Log(sampled_data):
    return np.ceil(np.log(len(sampled_data))) + 1

def Rice_Rule(sampled_data):
    return np.ceil(2 * len(sampled_data)**(1/3))

def Scott_Rule(sampled_data):
    max_range = max(sampled_data)
    min_range = min(sampled_data)
    sigma = np.std(sampled_data)
    h = 3.5*sigma/(len(sampled_data)**(1/3))
    return np.ceil((max_range-min_range)/h)

def Minimum_cross_h(h,sampled_data):
    h=h[0]
    print(h)
    n = len(sampled_data)
    bins = max(int ( n/h),1)
    print(bins)
    his = plt.hist(sampled_data, bins=bins)
    y = his[0]
    return 2/((n-1)*h) - (np.sum(np.square(y))*(n+1))/(n*n*(n-1)*h)

def Minimum_cross(h,sampled_data):
    max_range = max(sampled_data)
    min_range = min(sampled_data)
    return np.ceil((max_range-min_range)/h)

def Shimazaki_Shinomoto(x):
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
        v = sum((ki-k)**2)/N[i] 
        C[i] = (2*k-v)/((D[i])**2) 

    cmin = min(C)
    idx  = np.where(C==cmin)
    idx = int(idx[0])
    return N[idx]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_data',type=int,required=True,help="数据量")
    
    args = parser.parse_args()
    sampled_data = get_data(args.num_data)

    # h = [1.5]
    # x=minimize(Minimum_cross_h,h,args=(sampled_data,),method='SLSQP')
    # print(x)
    # h=x.x[0]

    print("Square-root choice:",Sqrt(sampled_data))
    print("Sturges' formula:",Log(sampled_data))
    print("Rice Rule:",Rice_Rule(sampled_data))
    print("Scott's normal reference rule:",Scott_Rule(sampled_data))
    # print("Minimizing cross-validation estimated squared error:",Minimum_cross(h,sampled_data))
    print("Shimazaki and Shinomoto's choice:",Shimazaki_Shinomoto(sampled_data))
