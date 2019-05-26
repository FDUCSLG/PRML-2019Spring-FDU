# -*- coding: utf-8 -*-
import os
os.sys.path.append('../')
from handout import get_linear_seperatable_2d_2c_dataset as get2d2c
from handout import get_text_classification_datasets as get_text
import numpy as np
import matplotlib.pyplot as plt
import string

class Part1:
    def __init__(self):
        self.dataset = get2d2c()
    
    def least_square_model(self):
        #self.dataset.plot(plt).show()
        num = len(self.dataset.X)
        npX = np.insert(self.dataset.X, 0, values = np.ones(num),axis=1)
        W = np.linalg.solve(npX.T.dot(npX), npX.T.dot(self.dataset.y))
        print("W is:",W)
        dbx = np.linspace(-1.5,1.5,50)
        dby = np.array(((0.5-W[0])-W[1]*dbx)/W[2])
        plt.title("The Least Square Model:\nW={}".format(W))
        lqm = plt.plot(dbx,dby,'r')
        plt.xlabel("x1")
        plt.ylabel("x2")
        strl = "{:.2f}+{:.2f}*x1-{:.2f}*x2=0.5".format(W[0],W[1],-W[2])
        plt.legend(lqm,[strl])
        self.dataset.plot(plt).show()
        
    def perceptron_model(self):
        #pre processing
        num = len(self.dataset.X)
        npy = self.dataset.y
        #npy = dataset.y + dataset.y - np.ones(len(dataset.y))
        #npy = np.where(npy<0,npy,1)
        
        npX = np.insert(self.dataset.X, 0, values = np.ones(num),axis=1)
        # print(npy)
        
        #initialize parameters
        W = np.array([0,-1,1])
        eta = 0.01
        print(W)
        terminate = False
        while(not terminate):
            terminate = True
            #print(W)
            for n in range(num):
                xn = np.array(npX[n])
                if npy[n]:
                    tn = 1
                else:
                    tn = -1
                if W.T.dot(xn) < 0.5:
                    test = -1
                else:
                    test = 1
                if test*tn < 0:
                    terminate = False
                    W = W + tn*eta*xn
                    break
        print(W)
        # draw plot
        dbx = np.linspace(-1.5,1.5,50)
        dby = np.array(((0.5-W[0])-W[1]*dbx)/W[2])
        plt.title("The Perceptron Model:\nW={}".format(W))
        lqm = plt.plot(dbx,dby,'r')
        plt.xlabel("x1")
        plt.ylabel("x2")
        strl = "{:.2f}+{:.2f}*x1-{:.2f}*x2=0.5".format(W[0],W[1],-W[2])
        plt.legend(lqm,[strl])
        self.dataset.plot(plt).show()
    
    def perceptron_model2(self):
        #pre processing
        num = len(self.dataset.X)
        npy = self.dataset.y
        #npy = dataset.y + dataset.y - np.ones(len(dataset.y))
        #npy = np.where(npy<0,npy,1)
        
        npX = np.insert(self.dataset.X, 0, values = np.ones(num),axis=1)
        # print(npy)
        
        #initialize parameters
        W = np.array([0,1,-1])
        eta = 0.01
        print(W)
        terminate = False
        while(not terminate):
            terminate = True
            # print(W)
            for n in range(num):
                xn = np.array(npX[n])
                if npy[n]:
                    tn=1
                else:
                    tn=-1
                test = W.T.dot(xn)
                if (test<=0.5) ^ (tn<0):
                    terminate = False
                    W = W + tn*eta*xn
                    break
        print(W)
        # draw plot
        dbx = np.linspace(-1.5,1.5,50)
        dby = np.array(((0.5-W[0])-W[1]*dbx)/W[2])
        plt.title("The Perceptron Model:\nW={}".format(W))
        lqm = plt.plot(dbx,dby,'r')
        plt.xlabel("x1")
        plt.ylabel("x2")
        strl = "{:.2f}+{:.2f}*x1-{:.2f}*x2=0.5".format(W[0],W[1],-W[2])
        plt.legend(lqm,[strl])
        self.dataset.plot(plt).show()