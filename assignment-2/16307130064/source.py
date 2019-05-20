import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import *
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import argparse
import part2

class LSA:
    def __init__(self,dim=2):
        self.w=np.ones(dim+1)

    def fit(self,X,y):
        self.w=np.matmul(np.matmul(np.linalg.inv(np.matmul(X.transpose((1,0)),X)),X.transpose((1,0))),y)
          
    def plot(self,x):
        return (-self.w[0]*x-self.w[2])/self.w[1]
    
class Percetron:
    def __init__(self,dim=2):
        self.w=np.ones(dim)
        self.b=0
        
    def fit(self,X,y,lr=0.1):
        it=0
        n=len(X)
        while True:
            i=random.randint(0,n-1)
            if y[i]*(np.dot(X[i],self.w)+self.b)<=0:
                self.w+=lr*y[i]*X[i]
                self.b+=lr*y[i]
                it+=1
            predict=np.dot(X,self.w)+self.b
            predict[predict>0]=1
            predict[predict<0]=-1
            predict[predict==0]=0
            if np.all(predict==y):
                print("percepton iteration:",it)
                break
    def plot(self,x):
        return (-self.w[0]*x-self.b)/self.w[1]

parser = argparse.ArgumentParser()
parser.add_argument("--part", required=True, type=float,dest="part", help="part")
parser.add_argument("--lr", default=0.01, dest="lr", type=float,help="learning rate")
parser.add_argument("--batch", default=20, dest="batch", type=int,help="batch_size")
parser.add_argument("--epoch", default=100, dest="epoch", type=int,help="epoch")
parser.add_argument("--lamda", default=0, dest="lamda", type=float,help="l2 regularization")
parser.add_argument("--skip-dev", dest="skip_dev", action="store_true", help="Skip dev set")
parser.add_argument("--full", dest="full", action="store_true", help="full batch")
options = parser.parse_args()

if options.part==1:
    d = get_linear_seperatable_2d_2c_dataset()
    x=np.linspace(-1.5,1.5,1000)
    p1=LSA(2)
    trainX=np.ones((len(d.X),3))
    trainX[:,:-1]=d.X
    p1.fit(trainX,d.y.astype(int)*2-1)
    plt.figure("least square method")
    plt.scatter(d.X[:, 0], d.X[:, 1], c=d.y)
    plt.plot(x,p1.plot(x))
    
    p2=Percetron(2)
    p2.fit(d.X,d.y.astype(int)*2-1,lr=options.lr)
    plt.figure("percepton")
    plt.plot(x,p2.plot(x))
    plt.scatter(d.X[:, 0], d.X[:, 1], c=d.y)
    
elif options.part==2:
    loss,vacc,acc=part2.start(epoch=options.epoch,lr=options.lr,lamda=options.lamda,batch=options.batch,skip_dev=options.skip_dev,full=options.full)
    plt.figure("loss")
    plt.plot(loss)
    if vacc is not None:
        plt.figure("accuracy on Validation set")
        plt.plot(vacc)
    print(acc)
    
plt.show()
