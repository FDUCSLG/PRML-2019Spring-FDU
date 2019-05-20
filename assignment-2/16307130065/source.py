import os
os.sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
import string
import re
import random
from sklearn.datasets import fetch_20newsgroups
from handout import get_linear_seperatable_2d_2c_dataset


def pre_acc(a, b, c, dataset):
    pre = []
    for tup in dataset.X:
        x = tup[0]
        y = tup[1]
        res = a*x+b*y+c
        if res > 0:
            pre.append(True)
        else:
            pre.append(False)
    return dataset.acc(pre)


def least_square_model(dataset):
    xn = np.mat(dataset.X)
    bias = np.mat(np.ones(len(xn))).T
    X = np.hstack((bias, xn))
    X_Pseudo_Inverse = np.linalg.pinv(X)
    tmp = []
    for ele in dataset.y:
        if ele:
            tmp.append(1)
        else:
            tmp.append(-1)
    T = np.mat(np.array(tmp)).T
    W = np.matmul(X_Pseudo_Inverse, T)
    Wa = np.array(W)
    (c, a, b) = (Wa[0][0], Wa[1][0], Wa[2][0])
    print(a, b, c)
    print("acc = %f" % pre_acc(a, b, c, dataset))
    xp = np.linspace(min(dataset.X[:, 0]), max(dataset.X[:, 0]), 1000)
    yp = -(a*xp+c)/b
    plt.plot(xp, yp, color="red")
    dataset.plot(plt).show()


def perceptron(dataset):
    a = b = c = 1
    rate = 0.1
    Y = []
    for ele in dataset.y:
        if ele:
            Y.append(1)
        else:
            Y.append(-1)
    step = 0
    while True:
        step += 1
        wrong_pre = []
        loss = 0
        for i in range(len(Y)):
            judge = Y[i]*(a*dataset.X[i][0]+b*dataset.X[i][1]+c)
            if judge < 0:
                wrong_pre.append(i)
                loss += -judge
        if loss == 0:
            print(step)
            break
        index = random.randrange(0, len(wrong_pre))
        a += rate*dataset.X[wrong_pre[index]][0]*Y[wrong_pre[index]]
        b += rate*dataset.X[wrong_pre[index]][1]*Y[wrong_pre[index]]
        c += rate*Y[wrong_pre[index]]
    print(a, b, c)
    print("acc = %f" % pre_acc(a, b, c, dataset))
    xp = np.linspace(min(dataset.X[:, 0]), max(dataset.X[:, 0]), 1000)
    yp = -(a*xp+c)/b
    plt.plot(xp, yp, color="red")
    dataset.plot(plt).show()


def preprocess(data, mincount):
    word_dict = {}
    data_list = []
    for ele in data:
        ele = re.sub('\d+', ' ', ele)
        for c in string.punctuation:
            ele = ele.replace(c, '')
        for c in string.whitespace:
            ele = ele.replace(c, ' ')
        ele = ele.lower().split()
        data_list.append(ele)
        for word in ele:
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1
    word_dict = sorted(word_dict.items(), key=lambda w: w[0], reverse=False)
    new_dict = {}
    pos = 0
    for ele in word_dict:
        if int(ele[1]) >= mincount:
            new_dict[ele[0]] = pos
            pos += 1
    return data_list, new_dict


def multi_hot(data_list, word_dict):
    mul_hot = []
    for data in data_list:
        code = np.zeros(len(word_dict))
        for word in data:
            if word in word_dict:
                code[word_dict[word]] = 1
        mul_hot.append(code)
    bias = np.mat(np.ones(len(mul_hot))).T
    hot = np.hstack((np.mat(mul_hot), bias))
    return hot


def one_hot_category(dataset):
    one_hot = np.zeros((len(dataset.data), len(dataset.target_names)))
    for index in range(len(dataset.target)):
        one_hot[index][dataset.target[index]] = 1
    return np.mat(one_hot)


def softmax(mat):
    if mat.shape[1] == 1:
        mat -= np.max(mat)
        res = np.exp(mat)/np.sum(np.exp(mat))
    else:
        mat -= np.matmul(np.ones((mat.shape[0], 1)), np.max(mat, axis=0))
        col_sum = 1/(np.matmul(np.exp(mat.T), np.ones((mat.shape[0], 1))))
        res = np.multiply(np.exp(mat), col_sum.T)
    return res


def LR_Model_FBGD(x, y, data_num, category_num, dic_num, rate=0.1, param=0.001, iteration=1000):
    W = np.zeros((dic_num+1, category_num))
    loss = []
    step = 0
    while step < iteration:
        step += 1
        modify = ((softmax(W.T * x.T) - y.T) * x).T / data_num
        bias = 2 * param * W
        bias[dic_num] = [0]*category_num
        W -= (modify + bias) * rate
        cur_loss = -np.sum(np.multiply(y.T, np.log(softmax(W.T*x.T))))/data_num
        cur_loss += param * np.linalg.norm(np.square(W[:dic_num]), 1)
        loss.append(cur_loss)
        print("Iteration: %d , current_loss: %.20f" % (step, cur_loss))
    plt.plot(np.linspace(1, step, step), loss, color="red")
    plt.show()
    return W


def LR_Model_SGD(x, y, data_num, category_num, dic_num, rate=0.1, param=0.001, iteration=10000):
    W = np.zeros((dic_num+1, category_num))
    loss = []
    step = 0
    while step < iteration:
        step += 1
        i = random.randint(0, data_num - 1)
        modify = ((softmax(W.T * x[i].T) - y[i].T) * x[i]).T
        bias = 2 * param * W
        bias[dic_num] = [0] * category_num
        W -= (modify + bias) * rate
        cur_loss = -np.sum(np.multiply(y.T, np.log(softmax(W.T*x.T)))) / data_num
        cur_loss += param * np.linalg.norm(np.square(W[:dic_num]), 1)
        loss.append(cur_loss)
        print("Iteration: %d , current_loss: %.20f" % (step, cur_loss))
    plt.plot(np.linspace(1, step, step), loss, color="red")
    plt.show()
    return W


def LR_Model_BGD(x, y, data_num, category_num, dic_num, batch, rate=0.1, param=0.001, iteration=10000):
        W = np.zeros((dic_num + 1, category_num))
        loss = []
        step = 0
        while step < iteration:
            step += 1
            i = random.randint(0, data_num - batch)
            modify = ((softmax(W.T * x[i:i+batch].T) - y[i:i+batch].T) * x[i:i+batch]).T / batch
            bias = 2 * param * W
            bias[dic_num] = [0] * category_num
            W -= (modify + bias) * rate
            cur_loss = -np.sum(np.multiply(y.T, np.log(softmax(W.T * x.T)))) / data_num
            cur_loss += param * np.linalg.norm(np.square(W[:dic_num]), 1)
            loss.append(cur_loss)
            print("Iteration: %d , current_loss: %.20f" % (step, cur_loss))
        plt.plot(np.linspace(1, step, step), loss, color="red")
        plt.show()
        return W


def test_model(test_data, test_target, hot_code, matrix):
    correct_pre = 0
    for i in range(len(hot_code)):
        pre = np.argmax(softmax(matrix.T * hot_code[i].T))
        if pre == test_target[i]:
            correct_pre += 1
    acc = correct_pre/len(test_data)
    return acc


if __name__ == '__main__':
    # Part1
    # d = get_linear_seperatable_2d_2c_dataset()
    # least_square_model(d)
    # perceptron(d)

    # Part2
    categories = ['comp.os.ms-windows.misc', 'rec.motorcycles', 'sci.space', 'talk.politics.misc', ]
    dataset_train = fetch_20newsgroups(subset='train', categories=categories, data_home='../../..')
    data, word = preprocess(dataset_train.data, 10)
    multi_hot_code = multi_hot(data, word)
    one_hot_code = one_hot_category(dataset_train)
    data_n = len(dataset_train.data)
    category_n = len(dataset_train.target_names)
    dic_n = len(word)
    # mat_W = LR_Model_FBGD(multi_hot_code, one_hot_code, data_n, category_n, dic_n, 0.1, 0.001, 2000)
    # mat_W = LR_Model_SGD(multi_hot_code, one_hot_code, data_n, category_n, dic_n, 0.1, 0.001, 5000)
    mat_W = LR_Model_BGD(multi_hot_code, one_hot_code, data_n, category_n, dic_n, 4, 0.1, 0.001, 3500)

    acc_train = test_model(dataset_train.data, dataset_train.target, multi_hot_code, mat_W)
    print("acc_train = %.20f" % acc_train)
    dataset_test = fetch_20newsgroups(subset='test', categories=categories, data_home='../../..')
    data_test, word_test = preprocess(dataset_test.data, 10)
    multi_hot_test = multi_hot(data_test, word)
    acc_test = test_model(dataset_test.data, dataset_test.target, multi_hot_test, mat_W)
    print("acc_test = %.20f" % acc_test)