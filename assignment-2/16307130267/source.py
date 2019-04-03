import os
os.sys.path.append('..')
from handout import *

import numpy as np
import random
import string, re
from matplotlib import pyplot as plt

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

mean = lambda x: sum(x) / len(x)

class Perceptron:
    def __init__(self, n_iter, lr, train_X, train_y, test_X, test_y):
        '''
        paras:
            n_iter: number of iterations
            lr: learning rate
        '''
        self.n_iter = n_iter
        self.lr = lr
        self.w = np.random.rand(train_X.shape[-1])
        self.b = 0
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.acti = lambda x: 1 if x > 0 else 0
        self.errors = []

    def train(self):
        train_X, train_y = self.train_X, self.train_y
        for _ in range(self.n_iter):
            error = 0
            state = np.random.get_state()
            np.random.shuffle(train_X)
            np.random.set_state(state)
            np.random.shuffle(train_y)
            for (X, y) in zip(train_X, train_y):
                pred = np.dot(X, self.w) + self.b
                err = y - self.acti(pred)
                self.w += self.lr * err * X
                self.b += self.lr * err
                error += err != 0
            self.errors.append(error)
        #print(self.errors)

    def test(self):
        error = 0
        for (X, y) in zip(self.test_X, self.test_y):
            pred = np.dot(X, self.w) + self.b
            err = y - self.acti(pred)
            error += err != 0
        print(error)

    def plot(self, plt):
        mi, mx = np.min(self.train_X), np.max(self.train_X) 
        mi_ = -1 * (self.b + mi * self.w[0]) / self.w[1]
        mx_ = -1 * (self.b + mx * self.w[0]) / self.w[1]
        plt.title('Perceptron Algorithm')
        plt.plot([mi, mx], [mi_, mx_])
        plt.show()

class LeastSquare:
    def __init__(self, train_X, train_y, test_X, test_y):

        self.w = np.random.rand(train_X.shape[-1] + 1, 2)

        self.train_X = np.concatenate(
            (train_X, np.array([np.ones(train_X.shape[0])]).transpose()), axis=-1
            )
        self.train_y = np.array([[0, 1] if y else [1, 0] for y in train_y])
        
        self.test_X = np.concatenate(
            (test_X, np.array([np.ones(test_X.shape[0])]).transpose()), axis=-1
            )
        self.test_y = np.array([[0, 1] if y else [1, 0] for y in test_y])
        
        self.acti = lambda x: [1, 0] if x[0] > x[1] else [0, 1]

    def train(self):
        self.w = np.dot(
            np.dot(
                np.linalg.inv(np.dot(self.train_X.transpose(), self.train_X)),
                self.train_X.transpose()
                ),
            self.train_y
            )
        error = 0
        for (X, y) in zip(self.train_X, self.train_y):
            pred = self.acti(np.dot(X, self.w))
            err = y - pred
            error += (sum(err) != 0)
        #print(error)

    def test(self):
        error = 0
        for (X, y) in zip(self.test_X, self.test_y):
            pred = self.acti(np.dot(X, self.w))
            print(pred, y)
            err = y - pred
            error += (sum(err) != 0)
        print(error)

    def plot(self, plt):
        mi, mx = np.min(self.train_X), np.max(self.train_X)
        mi_ = -1 * ((self.w[2][0]-self.w[2][1]) + mi * (self.w[0][0]-self.w[0][1])) / (self.w[1][0]-self.w[1][1])
        mx_ = -1 * ((self.w[2][0]-self.w[2][1]) + mx * (self.w[0][0]-self.w[0][1])) / (self.w[1][0]-self.w[1][1])
        plt.title('Least Square Model')
        plt.plot([mi, mx], [mi_, mx_])
        plt.show()

class DataProcessor:
    def __init__(self, train, test, min_count=10):
        trans = str.maketrans({key: None for key in string.punctuation})
        train_X = [dat.lower().translate(trans) for dat in train.data]
        test_X = [dat.lower().translate(trans) for dat in test.data]
        train_X = [re.sub(pattern=r'\s', repl=' ', string=dat) for dat in train_X]
        test_X = [re.sub(pattern=r'\s', repl=' ', string=dat) for dat in test_X]
        train_X = [dat.split(' ') for dat in train_X]
        test_X = [dat.split(' ') for dat in test_X]
        for i in range(len(train_X)):
            while '' in train_X[i]:
                train_X[i].remove('')
        for i in range(len(test_X)):
            while '' in test_X[i]:
                test_X[i].remove('')
        self.train_X = train_X
        self.test_X = test_X
        self.train_y = np.array(train.target)
        self.test_y = np.array(test.target)
        self.build_vocab(min_count)
        self.map2vec()

    def build_vocab(self, min_count):
        self.vocab = set()
        word_count = dict()
        for i in range(len(self.train_X)):
            for j in range(len(self.train_X[i])):
                word_count[self.train_X[i][j]] = word_count.get(self.train_X[i][j], 0) + 1

        for i in range(len(self.train_X)):
            for j in range(len(self.train_X[i])):
                if word_count[self.train_X[i][j]] >= min_count:
                    self.vocab.add(self.train_X[i][j])
        
        self.vocab = sorted(list(self.vocab))
        self.vocab_dict = dict()
        
        for i in range(len(self.vocab)):
            self.vocab_dict[self.vocab[i]] = i

    def map2vec(self):
        num_w = len(self.vocab)
        train_X = [[0 for i in range(num_w)] for j in range(len(self.train_X))]
        test_X = [[0 for i in range(num_w)] for j in range(len(self.test_X))]
        for i in range(len(self.train_X)):
            for j in range(len(self.train_X[i])):
                if self.vocab_dict.get(self.train_X[i][j]):
                    train_X[i][self.vocab_dict[self.train_X[i][j]]] = 1
        self.train_X = np.array(train_X)
        for i in range(len(self.test_X)):
            for j in range(len(self.test_X[i])):
                if self.vocab_dict.get(self.test_X[i][j]):
                    test_X[i][self.vocab_dict[self.test_X[i][j]]] = 1
        self.test_X = np.array(test_X)


class Classfier:
    def __init__(self, train_X, train_y, test_X, test_y, lr, lambd, n_iter, n_cat=4):
        '''
        paras:
            n_iter: number of iterations
            lr: learning rate
            lambd: lambda which is the regularization coefficient
            n_cat: number of categories
        '''
        self.train_X = train_X
        self.train_y = np.zeros([len(train_y), n_cat])
        for i in range(len(train_y)):
            self.train_y[i][train_y[i]] = 1
        self.test_X = test_X
        self.test_y = np.zeros([len(test_y), n_cat])
        for i in range(len(test_y)):
            self.test_y[i][test_y[i]] = 1
        self.lr = lr
        self.n_iter = n_iter
        self.w = np.random.rand(train_X.shape[-1], n_cat)
        self.b = np.random.rand(n_cat)
        self.lambd = lambd
        self.n_cat = n_cat
        self.criterion = lambda y, pred: -np.mean(y * np.log(pred)) + self.lambd * np.sum(self.w ** 2)
        self.get_acc = lambda y, pred: float((np.argmax(pred, axis=-1) == np.argmax(y, axis=-1)).mean())
        self.loss_array = []
        self.acc_array = []

    def forward(self, X, y):
        z = np.dot(X, self.w) + self.b
        z -= np.max(z, axis=-1).reshape(-1, 1)
        total = np.sum(np.exp(z), axis=-1).reshape(-1, 1)
        self.pred = np.exp(z) / total
        #self.pred += (self.pred == 0) * 1e-3

    def backward(self, X, y):
        db = np.mean(y - self.pred, axis=0)
        dw = (-np.dot(X.transpose(), (y - self.pred)) + 2 * self.lambd * self.w) / len(self.pred)
        assert self.w.shape == dw.shape, "shapes of w and dw don't match! "
        assert self.b.shape == db.shape, "shapes of b and db don't match! "
        self.w -= self.lr * dw
        self.b -= self.lr * db

    def train_BGD(self):
        last_loss = 0
        for i in range(self.n_iter):
            self.forward(self.train_X, self.train_y)
            loss = self.criterion(self.train_y, self.pred)
            acc = self.get_acc(self.train_y, self.pred)
            self.loss_array.append(loss)
            self.acc_array.append(acc)
            '''
            The procedure can be terminated automatically if you add the following codes
            '''
            #if abs(loss - last_loss) < 1e-3:
            #    break
            #last_loss = loss
            self.backward(self.train_X, self.train_y)
            print('BGD')
            print('iteration', i + 1)
            print('train loss is', loss)
            print('train acc is', acc)

    def train_MBGD(self, bs=100):
        '''
        paras:
            bs: batch size
        '''
        mx = len(self.train_X)
        nums = mx // bs
        last_loss = 0
        for i in range(self.n_iter):
            train_X, train_y = self.randomize()
            loss_, acc_ = [], []
            for j in range(nums):
                X, y = train_X[j*bs:min((j+1)*bs, mx)], train_y[j*bs:min((j+1)*bs, mx)]
                self.forward(X, y)
                loss = self.criterion(y, self.pred)
                acc = self.get_acc(y, self.pred)
                loss_.append(loss)
                acc_.append(acc)
                '''
                The procedure can be terminated automatically if you add the following codes
                '''
                #if abs(loss - last_loss) < 1e-3:
                #    i = self.n_iter
                #    break
                #last_loss = loss
                self.backward(X, y)
                '''
                The following codes are only used for ploting the figures in report.
                An iteration's loss and accuracy should be recorded out of this loop. 
                '''
                #self.loss_array.append(loss)#mean(loss_))
                #self.acc_array.append(acc)#mean(acc_))
            self.loss_array.append(mean(loss_))
            self.acc_array.append(mean(acc_))
            print('MBGD')
            print('iteration', i + 1)
            print('train loss is', mean(loss_))
            print('train acc is', mean(acc_))

    def train_SGD(self):
        j = 0
        for i in range(self.n_iter):
            train_X, train_y = self.randomize()
            loss_, acc_ = [], []
            for (X, y) in zip(train_X, train_y):
                self.forward(X.reshape(1, -1), y.reshape(1, -1))
                loss = self.criterion(y, self.pred)
                acc = self.get_acc(y, self.pred)
                loss_.append(loss)
                acc_.append(acc)
                '''
                The procedure can be terminated automatically if you add the following codes
                '''
                #if abs(loss - last_loss) < 1e-3:
                #    i = self.n_iter
                #    break
                #last_loss = loss
                self.backward(X.reshape(1, -1), y.reshape(1, -1))
                '''
                The following codes are only used for ploting the figures in report.
                An iteration's loss and accuracy should be recorded out of this loop. 
                '''
                #j += 1
                #if j % 50 == 0:
                #    self.loss_array.append(loss)
                #    self.acc_array.append(acc)
            self.loss_array.append(mean(loss_))
            self.acc_array.append(mean(acc_))
            print('SGD')
            print('iteration', i + 1)
            print('train loss is', mean(loss_))
            print('train acc is', mean(acc_))

    def test(self):
        self.forward(self.test_X, self.test_y)
        loss = self.criterion(self.test_y, self.pred)
        acc = self.get_acc(self.test_y, self.pred)
        print('test loss is', loss)
        print('test acc is', acc)

    def randomize(self):
        train_X, train_y = self.train_X, self.train_y
        state = np.random.get_state()
        np.random.shuffle(train_X)
        np.random.set_state(state)
        np.random.shuffle(train_y)
        return train_X, train_y

    def clear(self):
        self.w = np.random.rand(self.train_X.shape[-1], self.n_cat)
        self.b = np.random.rand(self.n_cat)
        self.loss_array = []
        self.acc_array = []

    def plot(self, name):
        nums = len(self.loss_array)
        x = range(nums)
        plt.suptitle('loss and accuracy curves in training procedure', fontsize=16)
        plt.title('optimizer: %s, iterations: %d, learning rate: %f, lambda: %f' % (name, self.n_iter, self.lr, self.lambd), fontsize=10)
        plt.plot(x, self.loss_array, label='loss', color='#FFA500')
        plt.plot(x, self.acc_array, label='accuracy', color='cyan')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    #---------- Part I ----------
    d = get_linear_seperatable_2d_2c_dataset()

    l = LeastSquare(train_X=d.X, train_y=d.y, test_X=np.zeros_like(d.X), test_y=np.zeros_like(d.y))
    '''
    The parameters of Least Square here are just for report. 
    You can split the data into training and test data.
    '''
    l.train()
    l.plot(d.plot(plt))

    p = Perceptron(n_iter=100, lr=0.1, train_X=d.X, train_y=d.y, test_X=np.zeros_like(d.X), test_y=np.zeros_like(d.y))
    '''
    The parameters of Perceptron here are just for report. 
    You can split the data into training and test data.
    '''
    p.train()
    p.plot(d.plot(plt))
    
    
    #---------- Part II ----------
    train, test = get_text_classification_datasets()
    data = DataProcessor(train, test)

    clf = Classfier(data.train_X, data.train_y, data.test_X, data.test_y, lr=0.1, lambd=1e-4, n_iter=100, n_cat=4)
    
    clf.train_MBGD()
    clf.test()
    clf.plot('MBGD')
    
    clf.clear()
    clf.train_BGD()
    clf.test()
    clf.plot('BGD')
    
    clf.clear()
    clf.train_SGD()
    clf.test()
    clf.plot('SGD')