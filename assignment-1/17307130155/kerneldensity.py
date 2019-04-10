import os 
os.sys.path.append('..')
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt
import math

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
#KDE(100,0.23,'red','h=0.23')
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

#plt.title('Figure4-2')
def KDE3():
    x = np.linspace(0.05,1,96)
    y = np.zeros(x.shape, dtype=np.float)
    for i in range(0,96):
        y[i]=KDE_CV(100,x[i])
        print(x[i])
    plt.cla()
    plt.plot(x,y,color='blue')
    plt.xlabel('h')
    plt.ylabel('Loss')
    plt.show()

plt.title('Figure4-1')
plt.show()
