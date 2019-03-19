import os
os.sys.path.append('..')
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt

def KNN(NUM,k,c,l):
    sampled_data = get_data(NUM)
    minvalue = min(sampled_data)-3
    maxvalue = max(sampled_data)+3
    x = np.linspace(minvalue,maxvalue,1000)
    y = np.zeros(x.shape)
    for i in range(0,1000):
        t = abs(x[i] - sampled_data)
        t.sort(axis=0)
        y[i] = k/(2*NUM*t[k])
    plt.plot(x,y,color=c,label=l,linewidth=1.0)
def KNN1():
    KNN(100,1,'red','NUM=100, k=1')
    KNN(100,3,'grey','NUM=100, k=3')
    KNN(100,10,'grey','NUM=100, k=10')
    KNN(100,20,'orange','NUM=100, k=20')
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.title('Figure 5-2')
    plt.axis([17,41,-0.01,0.5])
    plt.legend()
    plt.show()