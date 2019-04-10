import os,argparse
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_linear_seperatable_2d_2c_dataset
import numpy as np
import matplotlib.pyplot as plt

def Plot(W):
    x = np.array([-1.5,1.5])
    y = (-W[0]-W[1]*x)/W[2]
    d.plot(plt)
    plt.plot(x, y,color='red')
    plt.show()

def Least_Square(d):
    ones = np.expand_dims(np.ones(len(d.X)), axis=0)
    X = np.concatenate((ones.T, d.X), axis=1)
    y = 2*d.y-1
    W = np.linalg.pinv(X).dot(y)
    
    acc = (np.sign(X.dot(W)) == y).mean()
    print("The accuracy of least square is {}%.".format(str(acc*100)))
    Plot(W)


def Perceptron(d):
    eta = 1e-2
    ones = np.expand_dims(np.ones(len(d.X)), axis=0)
    X = np.concatenate((ones.T, d.X), axis=1)
    y = 2*d.y-1
    W = np.ones(3)
    for _ in range(2000):
        flag = True
        for i in range(len(X)):
            temp = W.dot(X[i].T)*(y[i])
            if temp<0:
                W = W + eta*X[i]*y[i]
                flag = False
        if flag:
            break

    acc = (np.sign(X.dot(W)) == y).mean()
    print("The accuracy of perceptron is {}%.".format(str(acc*100)))
    Plot(W)

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', dest='Lesat_Square',action='store_true', default=False, help='least square')
    parser.add_argument('-p', dest='Perceptron',action='store_true', default=False, help='perceptron')
    args = parser.parse_args()

    d = get_linear_seperatable_2d_2c_dataset()
    
    if args.Lesat_Square:
        Least_Square(d)
    elif args.Perceptron:
        Perceptron(d)
    else:
        print("usage: linear.py [-h] [-l] [-p]")