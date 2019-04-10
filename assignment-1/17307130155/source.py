import os
import math
os.sys.path.append('..')
import sys, getopt
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt

def usage():
    print('<Usage>')
    print('python3 source.py [options]')
    print()
    print('Options')
    print('-h or --help:  Show help.')
    print('--hist      :  Show histogram.')
    print('--kde1      :  Show kernel density estimators with different num_data.')
    print('--kde2      :  Show kernel density estimators with different h.')
    print('--kde3      :  Show cross-validation to find the best h for kernel density estimators.')
    print('--knn       :  Show k nearest neigbor methods with different k.')

def HIST1():
    sampled_data = get_data(1000)
    plt.hist(sampled_data, normed=True, bins=2,color='blue',label='bins=2',alpha=0.4)
    plt.hist(sampled_data, normed=True, bins=10,color='orange', label='bins=10',alpha=0.4)
    plt.hist(sampled_data, normed=True, bins=100,color='red',label='bins=100',alpha=0.4)
    plt.hist(sampled_data, normed=True, bins=500,color='grey',label='bins=100',alpha=0.4)

    plt.legend()
    plt.show()

def KDE(NUM,h,c,l):
    sampled_data = get_data(NUM)
    minvalue = min(sampled_data)-3
    maxvalue = max(sampled_data)+3
    bins = 500 #(int)((maxvalue-minvalue)/h)
    x = np.linspace(minvalue,maxvalue,bins)
    y = np.zeros(x.shape, dtype = np.float)
    for i in sampled_data:
        y = y+(1/NUM)*(1/(2*math.pi*h*h)**0.5)*math.e**(-(x-i)**2/(2*h**2))
    plt.plot(x,y,color=c,label=l)

def KDE_CV(NUM,h):
    sampled_data = get_data(NUM)
    train_data = sampled_data[0:NUM*8//10]
    valid_data = sampled_data[NUM*8//10:NUM]
    train_data_size = NUM*8//10
    valid_data_size = NUM//5
    minvalue = min(sampled_data)-3
    maxvalue = max(sampled_data)+3
    bins = 500 #(int)((maxvalue-minvalue)/h)
    deltax = (maxvalue-minvalue)/500
    x = np.linspace(minvalue,maxvalue,bins)
    y = np.zeros(valid_data_size, dtype = np.float)

    for i in train_data:
        y = y+(1/train_data_size)*(1/(2*math.pi*h*h)**0.5)*math.e**(-(valid_data-i)**2/(2*h**2))
    loss = 0
    for i in range(0,valid_data_size):
        loss = loss - math.log(y[i])
    return(loss);

def KDE1():
    KDE(100,0.2,'black','NUM=100')
    KDE(500,0.2,'red','NUM=500')
    KDE(1000,0.2,'grey','NUM=1000')
    KDE(10000,0.2,'orange','NUM=10000')
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.legend()
    plt.show()

def KDE2():
    KDE(100,0.05,'red','h=0.05')
    #KDE(100,0.1,'purple','h=0.1')
    KDE(100,0.25,'black','h=0.25')

    KDE(100,0.5,'grey','h=0.5')
    KDE(100,1,'orange','h=1')
    #KDE(100,2,'red','h=2')
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.legend()
    plt.show()

def KDE3():
    x = np.linspace(0.05,1,96)
    y = np.zeros(x.shape, dtype=np.float)
    for i in range(0,96):
        y[i]=KDE_CV(100,x[i])
    plt.cla()
    plt.plot(x,y,color='blue')
    plt.xlabel('h')
    plt.ylabel('Loss')
    plt.show()

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

def main(argv):
    for arg in argv[1:]:
        if arg == '-h' or arg == '--help':
            usage()
            sys.exit()
        elif arg == '--hist':
            HIST1();
        elif arg == '--kde1':
            KDE1();
        elif arg == '--kde2':
            KDE2();
        elif arg == '--kde3':
            KDE3();
        elif arg == '--knn':
            KNN1();
        else:
            print ("Error: invalid parameters")
            sys.exit()

if __name__ == "__main__":
    main(sys.argv)
