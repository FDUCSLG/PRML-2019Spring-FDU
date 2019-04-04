import os
os.sys.path.append('..')

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from sklearn.datasets import fetch_20newsgroups
from handout import get_linear_seperatable_2d_2c_dataset

class perceptron_classifier(object):
    '''
    K classes
    xs = [x1, ..., xn], where xi is a D-dimensional vector
    ts = [t1, ..., tn], where ti should be 1 or -1
    '''
    def __init__(self, xs, ts):
        
        self.D = len(xs[0]) + 1

        xs = np.array(xs)
        ts = np.array(ts)

        temp = [[t, x] for t, x in zip(ts, xs)]
        random.shuffle(temp)

        self.w = np.array(self.D * [0])

        for t in temp:
            ti = t[0]
            xi = np.array([1] + [i for i in t[1]])
            tt = np.matmul(np.matrix.transpose(self.w), xi)
            if tt * ti <= 0:
                # Lazy update ?
                self.w = self.w + xi * ti
         
            
    def evaluate(self, x):
        xi = np.array([1] + [i for i in x])
        tt = np.matmul(np.matrix.transpose(self.w), xi)
        return 1 if tt >= 0 else -1 

       

if __name__ == '__main__':
    d = get_linear_seperatable_2d_2c_dataset()
    ts = [1 if y else -1 for y in d.y]

    pc = perceptron_classifier(d.X, ts)

    # Calculate accuracy
    hit_cnt = 0
    miss_cnt = 0
    for x, t in zip(d.X, d.y):
        p = pc.evaluate(x)
        r = 1 if t == True else -1
        if p == r: hit_cnt = hit_cnt + 1
        else: miss_cnt = miss_cnt + 1
    print ('The accuracy is ' + str(hit_cnt / (hit_cnt + miss_cnt)))

    w = pc.w

    xs = np.linspace(-1.5, 1.5, 100)
    ys = [-(w[1] * x + w[0]) / w[2] for x in xs]

    d.plot(plt)
    plt.plot(xs, ys)
    plt.title('Perceptron Classifier')
    # plt.savefig('perceptron', dpi=300)
    plt.show()





