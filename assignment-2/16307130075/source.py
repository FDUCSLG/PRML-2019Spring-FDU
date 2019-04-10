import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import *
import numpy as np
import matplotlib.pyplot as plt
import math, time
import random, string, re

def solve1_lsq():
    dataset = get_linear_seperatable_2d_2c_dataset()
    n = dataset.y.shape[0]
    t = (dataset.y * 2 - 1).reshape(-1, 1)
    X = np.ones([n, 3])
    X[:, 1:] = dataset.X
    X = np.matrix(X)
    X_inv = np.linalg.pinv(X)
    W = np.matmul(X_inv, t)
    pred = np.matmul(X, W) > 0
    pred = pred.reshape(-1)
    print(np.mean(pred == dataset.y))
    xmin, xmax = float(np.min(dataset.X[:, 0])), float(np.max(dataset.X[:, 0]))
    x_dot = [xmin, xmax]
    ymin = xmin * - W[1, 0] / W[2, 0] - W[0, 0] / W[2, 0]
    ymax = xmax * - W[1, 0] / W[2, 0] - W[0, 0] / W[2, 0]
    y_dot = [ymin, ymax]
    
    plt.title('least square')
    plt.scatter(dataset.X[:, 0], dataset.X[:, 1], c = dataset.y)
    plt.plot(x_dot, y_dot)
    
    
def solve1_perceptron():
    dataset = get_linear_seperatable_2d_2c_dataset()
    n = dataset.y.shape[0]
    t = (dataset.y * 2 - 1).reshape(-1, 1)
    X = np.ones([n, 3])
    X[:, 1:] = dataset.X
    W = np.zeros([3, 1])
    for iter_num in range(10000):
        flag = 1
        for j in range(X.shape[0]):
            x = X[j, :].reshape(-1, 1)
            pred = np.matmul(W.T, x) >= 0
            if pred != dataset.y[j]:
                W = W + t[j] * x
                flag = 0
        if flag == 1:
            break
    
    pred = np.matmul(X, W) > 0
    pred = pred.reshape(-1)
    print(np.mean(pred == dataset.y))
    xmin, xmax = float(np.min(dataset.X[:, 0])), float(np.max(dataset.X[:, 0]))
    x_dot = [xmin, xmax]
    ymin = xmin * - W[1, 0] / W[2, 0] - W[0, 0] / W[2, 0]
    ymax = xmax * - W[1, 0] / W[2, 0] - W[0, 0] / W[2, 0]
    y_dot = [ymin, ymax]
    
    plt.title('perceptron')
    plt.scatter(dataset.X[:, 0], dataset.X[:, 1], c = dataset.y)
    plt.plot(x_dot, y_dot)
    
    
def part1_plot():
    plt.figure(figsize = (8, 4))
    plt.subplot(1, 2, 1)
    solve1_lsq()
    plt.subplot(1, 2, 2)
    solve1_perceptron()
    plt.show()

def sub_dict(dic):
    del_key = []
    for k in dic.keys():
        if dic[k] < 10:
            del_key.append(k)
    for k in del_key:
        dic.pop(k)
    return dic

def cross_entropy(W, P, C, lamda):
    offset = 1e-8
    loss = - np.sum(np.log(P + offset) * C)
    return loss + 2 * lamda * np.sum(W)

def softmax(W, b, X):
    P = np.matmul(W.T, X) + b
    P = np.exp(P)
    s = np.sum(P, axis = 0).reshape(1, -1)
    P = P / s
    return P

def test(W, b, X, target, C, lamda):
    P = softmax(W, b, X)
    pred = np.argmax(P, axis = 0)
    ACC = np.mean(pred == target)
    loss = cross_entropy(W, P, C, lamda) / X.shape[1]
    return loss, ACC
   
def check_gd(W, b, gradw, gradb, X, C, lamda):
    eps = 1e-5
    err = 1e-2
    l, r = W.shape
    P1 = softmax(W, b, X) 
    l1 = cross_entropy(W, P1, C, lamda) 
    chk_time = 500
    flag = 1
    for i in range(chk_time):
        x, y = random.randint(0, l-1), random.randint(0, r-1)
        W[x, y] += eps
        P2 = softmax(W, b, X)
        l2 = cross_entropy(W, P2, C, lamda) 
        delta_l = l2 - l1
        error = abs(gradw[x, y] * eps - delta_l)
        if error > err:
            flag = 0
            print("gradient of w is wrong")
            break
        W[x, y] -= eps
    
    if flag:
        print("gradient of w is right")
    flag = 1
    l = b.shape[0]
    for i in range(chk_time):
        x = random.randint(0, l-1)
        b[x, 0] += eps
        P2 = softmax(W, b, X)
        l2 = cross_entropy(W, P2, C, lamda) 
        delta_l = l2 - l1
        error = abs(gradb[x, 0] * eps - delta_l)
        if error > err:
            flag = 0
            print("gradient of w is wrong")
            break
        b[x, 0] -= eps
    if flag:
        print("gradient of b is right")
   
    

def engine(X, C, target, bs, lr, lamda):
    num_word, n = X.shape
    num_train = int(n * 0.9)
    num_val = n - num_train
    X_train, X_test = X[:, :num_train], X[:, num_train:]
    tar_train, tar_test = target[:num_train], target[num_train:]
    C_train, C_test = C[:, :num_train], C[:, num_train:]
    W = np.zeros([num_word, 4])
    b = np.zeros([4, 1])
    vw = np.zeros([num_word, 4])
    vb = np.zeros([4, 1])
    loss_sum = 0
    val_loss_list, train_loss_list, epoch_list = [], [], []
    i = 0
    while True:
        i += 1
        tf = time.time()
        W, b, vw, vb, train_loss, train_ACC = train(X_train, C_train, tar_train, bs, lr, lamda, W, b, vw, vb)
        epoch_list.append(i)
        train_loss_list.append(train_loss)
        val_loss, val_ACC = test(W, b, X_test, tar_test, C_test, lamda)
        ed = time.time()
        interval = ed - tf
        tf = ed
        print("Time:{:.2f}\tEpoch:{:2d}\tTrain_loss:{:.3f}\tTrain_Acc:{:.3f}\tVal_loss:{:.3f}\tVal_Acc:{:.3f}".format(interval, i, train_loss, train_ACC, val_loss, val_ACC ))
        
        val_loss_list.append(val_loss)
        loss_sum += val_loss
        
        
        if len(val_loss_list) == 11:
            loss_sum -= val_loss_list[0]
            val_loss_list.pop(0)
            if loss_sum / 10 - val_loss < 5e-3:
                #plt.plot(epoch_list, train_loss_list)
                #plt.title("lr = {:.1f}".format(lr))
                #plt.xlabel("epoch")
                #plt.ylabel("train loss")
                return W, b
        
    return W, b

    

def train(X, C, target, bs, lr, lamda, W, b, vw, vb, momentum = 0):
    n = X.shape[1]
    loss, ACC, itertime = 0, 0, 0
    loss_list, iter_list = [], []
    
    
    for i in range(0, n, bs):
        itertime += 1
        st, ed = i, min(i + bs, n)
        batch_size = ed - st
        X_ = X[:, st:ed]
        P = softmax(W, b, X_)
        pred = np.argmax(P, axis = 0)
        train_loss = cross_entropy(W, P, C[:, st:ed], lamda)
        loss += train_loss
        
        '''
        if itertime <= 30:
            loss_list.append(train_loss / bs)
            iter_list.append(itertime)
        '''
        
        ACC += np.sum(pred == target[st:ed])
        P = P - C[:, st:ed]
        if batch_size == 1:
            X_t = X_.reshape(1, -1)
        else:
            X_t = X_.T
        gradw = np.matmul(P, X_t).T / batch_size + W * lamda * 2
        gradb = np.sum(P, axis = 1).reshape(-1, 1) / batch_size 
        #check_gd(W, b, gradw, gradb, X, C, lamda)
        
        vw = momentum * vw + (1 - momentum) * gradw
        vb = momentum * vb + (1 - momentum) * gradb
        W = W - lr * vw
        b = b - lr * vb 
    
    '''
    if bs == 1:
        plt.title('SGD')
    else:
        plt.title('BGD')
    plt.plot(iter_list, loss_list)
    plt.xlabel('iterations')
    plt.ylabel('train loss')
    '''
    
    loss /= n
    ACC /= n
    return W, b, vw, vb, loss, ACC
        
    
def clean_data(dataset):
    data = []
    for ss in dataset.data:
        s = ''
        for ch in ss:
            if ch in string.punctuation:
                continue
            if ch in string.whitespace:
                ch = ' '
            ch = ch.lower()
            s += ch
        rs = re.sub(' +', ' ', s)
        word_list = rs.split()
        data.append(word_list)
    return data

def find(word, word_list, n):
    l, r = 0, n - 1
    while l <= r:
        mid = (l + r) // 2
        if word_list[mid] == word:
            return mid
        if word_list[mid] < word:
            l = mid + 1
        else:
            r = mid - 1
    return -1

def trans(n, data, dataset, num_word, pos):
    X = np.zeros([n, num_word])
    C = np.zeros([4, n])
    for i, wl in enumerate(data):
        for word in wl:
            p = pos.get(word, -1)
            if p != -1:
                X[i, p] = 1
    for i, label in enumerate(dataset.target):
        C[label, i] = 1
    X = X.T
    return X, C

def cross_val_lr(X, C, target, bs, lamda):
    #lr_list = [5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    lr_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    #lr_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    #lr_list = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
    num_word, n = X.shape
    k_fold = 5
    acc_list = []
    
    for lr in lr_list:
        ACC = 0
        for i in range(5):
            st, ed = i * n // k_fold, (i + 1) * n // k_fold
            X_test, C_test, tar_test = X[:, st:ed], C[:, st:ed], target[st:ed]
            num_train = st + n - ed
            X_train, C_train, tar_train = np.zeros([num_word, num_train]), np.zeros([4, num_train]), np.zeros([num_train])
            j = 0
            for i in range(0, st):
                X_train[:, j], C_train[:, j], tar_train[j] = X[:, i], C[:, i], target[i]
                j += 1
            
            for i in range(ed, n):
                X_train[:, j], C_train[:, j], tar_train[j] = X[:, i], C[:, i], target[i]
                j += 1
            
            W, b = engine(X_train, C_train, tar_train, bs, lr, lamda)
            loss, ACC1 = test(W, b, X_test, tar_test, C_test, lamda)
            ACC += ACC1
        
        ACC /= 5
        acc_list.append(ACC)
    
    plt.plot(lr_list, acc_list)
    plt.show()
    

def solve2(mode = "all"):
    trainset, testset = get_text_classification_datasets()
    len_train, len_test = len(trainset.data), len(testset.data)
    train_data = clean_data(trainset)
    
    dic, pos = {}, {}
    for word_list in train_data:
        for word in word_list:
            if word in dic.keys():
                dic[word] = dic[word] + 1
            else:
                dic[word] = 1
    
    dic = sub_dict(dic)
    for i, key, in enumerate(dic.keys()):
        pos[key] = i 
    
    num_word = len(dic)
    X, C = trans(len_train, train_data, trainset, num_word, pos)
    #cross_val_lr(X, C, trainset.target, len_train, 0.001)
    if mode == "all":
        bs, lr, lamda = len_train, 0.5, 0.001
    elif mode == "one":
        bs, lr, lamda = 1, 0.006, 0.001
    elif mode == "batch":
        bs, lr, lamda = 64, 0.06, 0.001
    
    '''
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    bs = 1
    engine(X, C, trainset.target, bs, lr, lamda)
    plt.subplot(1,2,2)
    bs = 64
    engine(X, C, trainset.target, bs , lr, lamda)
    plt.show()
    '''
    W, b = engine(X, C, trainset.target, bs, lr, lamda)
    test_data = clean_data(testset)
    X_, C_ = trans(len_test, test_data, testset, num_word, pos)
    loss, ACC = test(W, b, X_, testset.target, C_, lamda)
    print("Test ACC:{:.3f}".format(ACC))
            
    
    
if __name__ == "__main__":
    
    #part1_plot()
    #all means FBGD, one means SGD, batch means BGD
    solve2('all')

    