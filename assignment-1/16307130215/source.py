import os,argparse
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_data,gm1d
import numpy as np
import matplotlib.pyplot as plt


def Histogram(data,bins=50):
    assert bins>0
    plt.hist(data, normed=True, bins=bins)
    plt.title("Histogram\nbin="+str(bins)+"\nnum_data="+str(len(data)),fontsize=10)

def Gaussian_kernel(data,h=0.4):
    assert h>0
    min_range = min(data) - 3
    max_range = max(data) + 3
    x = np.linspace(min_range, max_range, 2000)
    y = np.zeros_like(x)

    for i in range (len(x)):
        y[i] = np.sum( np.exp( -((data-x[i])**2) / (2*h*h) ) )

    cons = 1/( (np.sqrt(2*np.pi*h*h))*len(data) )
    y = y*cons
    plt.plot(x, y,color='red', label='estimate')
    plt.title("Gaussian_kernel\nh="+str(h)+"\nnum_data="+str(len(data)),fontsize=10)
    plt.legend()

def Knn(data,k=10):
    assert k>0
    data = np.sort(data)
    num_data = len(data)
    assert k <= num_data

    min_range = min(data) - 3
    max_range = max(data) + 3
    x = np.linspace(min_range, max_range, 2000)
    y = np.zeros_like(x)

    first_higher = 0
    for i in range (len(x)):
        while first_higher<num_data and data[first_higher]<x[i] :
            first_higher+=1

        l,r=first_higher-1,first_higher
        cnt = 0
        while cnt<k:
            if l>=0 and r<=num_data-1:
                if x[i]-data[l]<data[r]-x[i]:
                    l-=1
                else:
                    r+=1
            elif l<0:
                r+=1
            else:
                l-=1
            cnt+=1

        y[i]=2*max(data[r-1]-x[i],x[i]-data[l+1])+1e-30
        
    y= k/(num_data*y)
    plt.plot(x, y,color='red', label='estimate')
    plt.title("Knn\nK="+str(k)+"\nnum_data="+str(len(data)),fontsize=10)
    plt.legend()
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type=str, choices=['hist','kernel','knn'],required=True, help='密度估计方法')
    parser.add_argument('--num_data',type=int,default=200,help="数据量")
    parser.add_argument('--bins',type=int,default=50)
    parser.add_argument('--h',type=float,default=0.4)
    parser.add_argument('--k',type=int,default=10)

    args = parser.parse_args()

    sampled_data = get_data(args.num_data)

    if args.func=="hist":
        Histogram(sampled_data,args.bins)
    elif args.func=="kernel":
        Gaussian_kernel(sampled_data,args.h)
    elif args.func=="knn":
        Knn(sampled_data,args.k)

    gm1d.plot(num_sample=1000)