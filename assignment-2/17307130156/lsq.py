import os 
os.sys.path.append('..')
import math

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from sklearn.datasets import fetch_20newsgroups

from handout import get_linear_seperatable_2d_2c_dataset 

# Multi-class Linear Regression with Least Square
class lsq_classifier(object):
    '''
    K classes
    xs = [x1, ..., xn], where xi is a D-dimensional vector
    ts = [t1, ..., tn], where ti is a one-hot K-dimensional vector
    '''
    def __init__(self, xs, ts): 
        self.K = len(ts[0]) 
        self.D = len(xs[0]) + 1
        T = np.array(ts)
        # T_t = np.matrix.transpose(T)
        
        X = []
        for x in xs:
            t = [1] # bias w0 = 1
            for xx in x: t.append(xx)
            X.append(t)
        
        X = np.array(X)

        X_pinv = np.linalg.pinv(X)
        W = np.matmul(X_pinv, T)

        # y = W'x = ((X'X)^-1 * X' * T)' x = (X_pinv * T)'x
        #   = T' * X_pinv' * x = c * x
        self.W_t = np.matrix.transpose(W)

    # x = [x1, ..., xD]
    def evaluate(self, x):
        X = [1]
        for xx in x: X.append(xx)
        y = np.matmul(self.W_t, X)
        return y.argmax()
        


if __name__ == '__main__':
    d = get_linear_seperatable_2d_2c_dataset()
    ts = [[1, 0] if t == True else [0, 1] for t in d.y]
    
    lsq = lsq_classifier(d.X, ts)

    # Calculate accuracy
    hit_cnt = 0
    miss_cnt = 0
    for x, t in zip(d.X, d.y):
        p = lsq.evaluate(x)
        r = 0 if t == True else 1
        if p == r: hit_cnt = hit_cnt + 1
        else: miss_cnt = miss_cnt + 1
    print ('The accuracy is ' + str(hit_cnt / (hit_cnt + miss_cnt)))

    # Plot
    W_t = lsq.W_t
    w = W_t[0] - W_t[1]

    xs = np.linspace(-1.5, 1.5, 100)
    ys = [-(w[1] * x + w[0]) / w[2] for x in xs]
 
    d.plot(plt)
    plt.plot(xs, ys)
    plt.title('Classification with Least Square')
    # plt.savefig('lsq', dpi=300)
    plt.show()




