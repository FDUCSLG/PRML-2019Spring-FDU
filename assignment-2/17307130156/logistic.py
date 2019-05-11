import os
os.sys.path.append('..')

import random
import math
import string 

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from sklearn.datasets import fetch_20newsgroups

from handout import get_text_classification_datasets

# Multi-class Logistic Regression
class logistic_classifier(object):
    '''
    xs = [x1, ..., xn], where xi is a D-dimensional vector
    ts = [t1, ..., tn], where ti is a K-dimensional one-hot vector
    '''
    def __init__(self, xs, ts):

        self.K = len(ts[0])
        self.D = len(xs[0]) + 1
        self.N = len(ts)

        temp = []

        # add x0 = 1
        
        for x in xs: 
            tt = [1]
            for xi in x: tt.append(xi)
            temp.append(tt)

        xs = np.array(temp)
        
        self.W = np.array([[0] * self.D] * self.K)
        self.T = np.array(ts).transpose()
        self.Phi = np.array(xs)


        # Plot SGD variances
        '''
        legends = []
        plt.figure()
        plt.xlabel('Times')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        times = 2000

        self.W = np.array([[0] * self.D] * self.K)
        accs = self.SGD(xs, ts, batches=3000, num_time=times) 
        plt.plot(range(times), accs)
        legends.append('Normal SGD')

        self.W = np.array([[0] * self.D] * self.K)
        accs = self.SGD(xs, ts, batches=3000, num_time=times, method='momentum') 
        plt.plot(range(times), accs)
        legends.append('SGD with Momentum')

        self.W = np.array([[0] * self.D] * self.K)
        accs = self.SGD(xs, ts, batches=3000, num_time=times, method='N momentum') 
        plt.plot(range(times), accs)
        legends.append('SGD with Nesterov Momentum')

        plt.legend(legends)
        plt.title('SGD Variances')
        plt.savefig('sgd', dpi=300)
        plt.show()
        '''

        '''
        # Plot mini-batch SGD
        batches = 100
        times = batches
        legends = []
        plt.figure()
        plt.xlabel('Times')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)

        self.W = np.array([[0] * self.D] * self.K)
        accs = self.SGD(xs, ts, batches=batches, num_time=times) 
        plt.plot(range(times), accs)
        legends.append('Normal')

        '''

        '''
        self.W = np.array([[0] * self.D] * self.K)
        accs = self.SGD(xs, ts, batches=batches, num_time=times, method='momentum') 
        plt.plot(range(times), accs)
        legends.append('Momentum')

        self.W = np.array([[0] * self.D] * self.K)
        accs = self.SGD(xs, ts, batches=batches, num_time=times, method='N momentum') 
        plt.plot(range(times), accs)
        legends.append('Nesterov Momentum')

        plt.legend(legends)
        plt.title('Mini-SGD with ' + str(batches) + ' batches')
        plt.savefig('minisgd'+str(batches)+'b', dpi=300)
        plt.show()
        '''

        batches = 100
        times = 2000
        l = 0.01
        self.W = np.array([[0] * self.D] * self.K)
        accs = self.SGD(xs, ts, batches=batches, num_time=times, method='N momentum', l=l) 


        '''
        legends = []
        rate = 0.1
        l = 0.01
        epochs = 100
        self.W = np.array([[0] * self.D] * self.K)
        accs = self.BGD(xs, ts, num_epoch=epochs, learning_rate=rate, l=l) 
        
        '''

        # self.SGD(xs, ts, 10)
        # self.SGD(xs, ts, 20000)

    def SGD(self, xs, ts, batches=1, num_time=10, method='normal', learning_rate=0.1, l=0):
        # Shuffle
        xts = list(zip(xs, ts))
        random.shuffle(xts)
        xs = [xsr for xsr, tsr in xts]
        ts = [tsr for xsr, tsr in xts]

        Phi = np.array(xs) 
        W = self.W
        T = self.T
        N = self.N

        a = learning_rate
        l = l
        b = 0.5
        min_d = 0.1

        V = np.array([[0] * self.D] * self.K)
        pre_V = np.array([[0] * self.D] * self.K)

        s = 0
       
        cnt = 0

        INF = 1e20

        accs = []

        size = N // batches + 1

        # while(True):
        for e in range(num_time):
            
            # Shuffle
            xts = list(zip(xs, ts))
            random.shuffle(xts)
            xs = [xsr for xsr, tsr in xts]
            ts = [tsr for xsr, tsr in xts]

            # mini-batch
            for i in range(0, N, size): 

                j = i + size if i + size <= N else N

                d = self.compute_gradient(xs[i:j], ts[i:j], W, l=l)

                if method == 'normal':
                    dW = - a * d
                    W = W + dW
                elif method == 'momentum':
                    V = b * V - a * d
                    dW = V
                    W = W + V 
                else: 
                    # Nesterov Momentum
                    pre_V = V
                    V = b * V - a * d
                    dW = (- b * pre_V + (1 + b) * V)
                    W = W + dW

                self.W = W
                # s = self.self_evaluate(xs, ts)
                # print (str(cnt) + ': '+ str(s))
                # accs.append(s)
                cnt += 1
                print (cnt)
                if cnt == num_time: break

            s = self.compute_loss(xs, ts, W)
            print (s)
            self.W = W
            if cnt >= num_time: break
            # if s < min_d: break
        return accs
        # plt.ylim(Es[-1], 10000)
        # plt.plot(times, Es)
        # plt.show()


    def BGD(self, xs, ts, num_epoch=1, learning_rate=0.1, l=0.1):

        W = self.W
        T = self.T
        Phi = self.Phi
        N = self.N

        # Learning rate
        a = learning_rate
        b = 0.5
        min_d = 0.01

        V = np.array([[0] * self.D] * self.K)

        accs = []

        # Batch Gradient Descent
        for e in range(num_epoch):
        # while (True) :

            d = self.compute_gradient(xs, ts, W, l)

            dW = a * d
            W = W - dW
            self.W = W

            acc = self.compute_loss(xs, ts, W)
            # acc = self.self_evaluate(xs, ts)
            accs.append(acc) 
            print (acc)

            print ('epoch')

            # s = self.matrix_norm(dW)
            # print (s)
            # if s < min_d: break
            # a = self.get_learning_rate(epoch)
        return accs
   

    def compute_loss(self, xs, ts, W):
        INF = 1e20
        N = self.N
        Phi = np.array(xs)
        Y = np.array([self.softmax(W @ pn) for pn in Phi])
        E = - 1/N * sum(sum( tk * (math.log(yk) if yk > 0 else -INF) for tk, yk in zip(t, y)) for t, y in zip(ts, Y))
        
        return E

    def compute_gradient(self, xs, ts, W, l=0):
        Phi = np.array(xs)
        N = len(xs)
        Y = np.array([self.softmax(W @ phi) for phi in Phi]).transpose()
        T = np.array(ts).transpose()
        R = np.array([[l * r for r in row] for row in W]).transpose()

        R[0] = [0] * len(R[0])
        R = R.transpose()
        d = (Y - T) @ Phi/N + R
        
        return d
        
        
        
    # z is a K-dimensional vector
    def softmax(self, z):
        m = max(z)
        z = np.array(z) - np.array([m] * len(z))
        s = sum(math.exp(zi) for zi in z)
        return np.array([math.exp(zi) / s for zi in z])

    def get_learning_rate(self, epoch_num, v0=0.2, decay_rate=1):
        return v0 / (1 + decay_rate * epoch_num)

    def evaluate(self, x):
        t = [1]
        for xi in x: t.append(xi) 
        rtn = np.argmax(self.softmax(self.W @ np.array(t)))
        return rtn
    def self_evaluate(self, xs, ts):
        total = len(ts)
        hit_cnt = 0
        for x, t in zip(xs, ts):
            a = np.argmax(self.softmax(self.W @ x))
            if t[a] == 1: hit_cnt += 1
        return hit_cnt / total

    # Frobenius Matrix Norm
    def matrix_norm(self, a):
        return math.sqrt(sum(sum( aij**2  for aij in row) for row in a))


def preprocess_data(dataset, test_dataset):
    # Threshold
    min_count = 10
    max_count = 100

    cnt = {}
    texts = []
    for text in dataset:
        t = text.lower()
        for s in string.punctuation:
            t = t.replace(s, ' ')
        for s in string.whitespace:
            t = t.replace(s, ' ')
        t = t.split()
        texts.append(t)
        for w in t:
            cnt[w] = cnt[w] + 1 if w in cnt else 1

    dic = {}
    i = 0
    # Remove low frequent letter
    for word, freq in cnt.items():
        if freq >= min_count and freq < max_count:
            # dic[word] = freq
            dic[word] = i
            i += 1

    l = len(dic)
    print ('dic length ' + str(l))

    ts = []
    for text in texts:
        t = [0] * l
        cnt = {}

        for word in text:
            cnt[word] = cnt[word] + 1 if word in cnt else 1

        for w, c in cnt.items():
            if w in dic: t[dic[w]] = c

        ts.append(t)
        # print (t)
        # input()
     

    # Test dataset
    tts = []
    for d in test_dataset:
        text = d
        for s in string.punctuation:
            text = text.replace(s, ' ')
        for s in string.whitespace:
            text = text.replace(s, ' ')
        text = text.split()

        t = [0] * l
        cnt = {}

        for word in text:
            cnt[word] = cnt[word] + 1 if word in cnt else 1

        for w, c in cnt.items():
            if w in dic: t[dic[w]] = c

        tts.append(t)

    return ts, tts



def preprocess_target(K, ts, tts):
    ys = []
    tys = []
    for t in ts:
        temp = [0] * K
        temp[t] = 1
        ys.append(temp)

    for t in tts:
        temp = [0] * K
        temp[t] = 1
        tys.append(temp)

    return ys, tys
    

if __name__ == '__main__':

    x = get_text_classification_datasets()
    
    # x[0].data[i]
    # x[0].target[i]

    xs, txs = preprocess_data(x[0].data, x[1].data)
    ts, tts = preprocess_target(len(x[0].target_names), x[0].target, x[1].target)
    # print (len(x[0].target_names))
    # print (ts)

    print ('preprocessing finish')


    lc = logistic_classifier(xs, ts)
    
    # Test
    rc = 0
    wc = 0
    for x, t in zip(txs, x[1].target):
    # for x, t in zip(xs, x[0].target):
        y = lc.evaluate(x)
        if t == y: rc = rc + 1
        else: wc = wc + 1
    print ('right: ' + str(rc))
    print ('wrong: ' + str(wc))
    print ('accuracy: ' + str(rc/(rc + wc)))
    





