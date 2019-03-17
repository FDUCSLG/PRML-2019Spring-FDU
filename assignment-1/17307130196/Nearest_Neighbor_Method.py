import math as mt
#Set the Delta as 0.08 at first
import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt
#--------------------------
#Nearest Neighbor Method
#Different from Kernel Method, we choose to fix K and varies V, and simply uses the function:
#p(x)=K/(V*N)
def KNN_Pro(target,dataset_ordered,N,K):
    flag_data=0
    while flag_data<N and dataset_ordered[flag_data]<=target:
        # print(flag_data)
        flag_data=flag_data+1
        #In this way, we get the first point larger than x
        # and the flag_data marks the first number larger than target
    ct=0 #ct marks the number of numbers met
    flag_l=max(flag_data-1,0)
    flag_r=min(flag_data,N-1)
    while ct<K:
        if flag_l>0 and flag_r<N-1:
            if target-dataset_ordered[flag_l]<dataset_ordered[flag_r]-target:
                flag_l=flag_l-1
            else:
                flag_r=flag_r+1
        else:
            if flag_l==0 and flag_r<N-1:
                flag_r=flag_r+1
            elif flag_r==N-1 and flag_l>0:
                flag_l=flag_l-1
        ct=ct+1
    # print(flag_l)
    # print(flag_r)
    V=dataset_ordered[flag_r]-dataset_ordered[flag_l]
    # return  V#get the volume of the box
    return float(K)/(float(V)*N)

def KNN(num,N,K):
    sampled_data = get_data(N)
    sampled_data.sort()
    output=[]
    for x in np.linspace(20,40,num):
        output.append(KNN_Pro(x,sampled_data,N,K))
    # plt.plot(np.linspace(20,40,num),output)
    # return output
    # plt.show()

#KNN(10000,10000,50)