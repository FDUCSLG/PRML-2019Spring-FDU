import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import *
import numpy as np
import matplotlib.pyplot as plt
import string
import argparse


def least_square(d):
    T = []
    X = []
    for i in range(0, len(d.X)):
        T.append(([1,0] if(d.y[i]) else [0, 1]))
        X.append([1, d.X[i][0], d.X[i][1]])
    X_pinv = np.linalg.pinv(X)
    W = X_pinv.dot(T)
    #[W1-W2]^T * [1, x, y]
    k = (W[1][0] - W[1][1])/(W[2][1] - W[2][0])
    b = (W[0][0] - W[0][1])/(W[2][1] - W[2][0])
    return k, b


def perceptron(d):
    num = len(d.X)
    W = (np.random.randn(1, 3))[0]
    y = d.y
    X = normalization(d.X)
    while 1:
        loss_exist = 0
        for i in range(0, num):
            E = W[0] + W[1]* d.X[i][0] + W[2]* d.X[i][1]
            E = E * (1 if(y[i]) else -1)
            if E < 0:
                loss_exist = 1
                W += (X[i] if(y[i]) else -X[i])

        if loss_exist == 0:
            k = -W[1]/W[2]
            b = -W[0]/W[2]
            return k, b


def straight_fun(k, b, x):
    return k*x+b


def normalization(X):
    # mean
    u = np.mean(X, axis=0)
    # standard
    v = np.std(X, axis=0)
    X = (X - u) / v
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    return X


def split_doc(s):
    for i in s:
        if i in string.punctuation:
            s = s.replace(i, "")
        elif i in string.whitespace:
            s = s.replace(i, " ")
    ans = s.split()
    # to lower case
    for i in range(0, len(ans)):
        ans[i] = ans[i].lower()
    return ans


def handler_doc(data):
    data_set = []
    data_count = {}
    store_key = []
    vocabulary = {}
    for s in data:
        ans = split_doc(s)
        for x in ans:
            data_count[x] = data_count.get(x, 0) + 1
            # only choose the words which occur at least 10 times in the overall training set.
            if data_count[x] == 10:
                store_key.append(x)
        data_set.append(ans)
    store_key.sort()
    count = 0
    for key in store_key:
        vocabulary[key] = count
        count += 1
    data_vec = to_vector(data_set, vocabulary)
    return data_vec, vocabulary


def to_vector(data_set, vocabulary):
    data_vec = []
    for s in data_set:
        vec = [0]*len(vocabulary)
        for i in s:
            if i in vocabulary:
                vec[vocabulary[i]] = 1
        vec.insert(0, 1) # 将bias 加到向量头部
        data_vec.append(vec)
    data_vec = np.array(data_vec)
    print("Generate doc_vec, Shape: ", data_vec.shape)
    return data_vec


def get_target(data, num):
    target = []
    for i in range(0, len(data)):
        vec = [0]*num
        vec[data[i]] = 1
        target.append(vec)
    target = np.array(target)
    print("Generate Target(one hot), Shape: ", target.shape)
    return target


def softmax(Xn, W):
    data = np.exp(np.dot(Xn, W.T))
    sum_data = np.sum(data, axis=1).T
    data = data/sum_data[:,None]
    #print("Generate ynk, Shape: ", data.shape)
    return data


def cross_entropy(data, target, step, gMethod, num):
    W = np.random.randn(len(target[0]), len(data[0]))
    print("Generate W, Shape: " + str(W.shape))
    loss_vector = []
    for _ in range(0, step):
        res = 0
        if gMethod == 2:
            n_list = np.random.randint(len(data), size=min(1, num))
        elif gMethod == 1:
            n_list = range(len(data))
        ynk = softmax(data, W)
        for i in range(0, len(data)):
            if i in n_list:
                y_Xi = ynk[i]  # 1 X 4
                res -= np.dot(target[i], np.log(y_Xi).T)  # (1 X 4) * (4 X 1) : yi * ti
        print("step: ", _," loss: ", res/len(data))
        loss_vector.append(res/len(data))
        W = W - gradient_cross(ynk, target, data, n_list)
    return W, loss_vector


def gradient_cross(ynk, target, data, n_list, a=0.001):
    matrix = (ynk - target).T
    for i in range(0, len(matrix)):
        if i not in n_list:
            data[i] = [0]* len(data[0])
    gradient = np.dot(matrix, data)
    #print("Generate gradient, Shape: ", gradient.shape)
    return a * gradient


def forecast(data, W, real_target):
    correct = 0
    ynk = softmax(data, W)
    for i in range(0, len(data)):
        res = ynk[i].tolist()
        index = res.index(max(res))
        if index == real_target[i]:
            correct += 1
    print("correct: ", correct, "/", len(data), "=", correct/len(data))


def display(x, fx, string):
    plt.plot(x, fx)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title(string)
    plt.show()


def part1(method):
    d = get_linear_seperatable_2d_2c_dataset()
    if method == "lsq":
        k, b = least_square(d)
    elif method == "ptr":
        k, b = perceptron(d)
    else:
        return
    plt.plot([-1.5, 1.5], [straight_fun(k, b, -1.5), straight_fun(k, b, 1.5)])
    plt.title("y = " + str(k) + "x + " + str(b))
    d.plot(plt).show()


def part2(step, gMethod, num):
    dataset_train, dataset_test = get_text_classification_datasets()
    train_vec, vocabulary = handler_doc(dataset_train.data)
    target = get_target(dataset_train.target, len(dataset_train.target_names))
    W, loss = cross_entropy(train_vec, target, step, gMethod, num)
    x_plot = np.linspace(0, step, len(loss))
    test = []
    for s in dataset_test.data:
        test.append(split_doc(s))
    test_vec = to_vector(test, vocabulary)
    forecast(test_vec, W, dataset_test.target)
    display(x_plot, loss, "cross entropy loss function")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", "-m", default="lsq", choices=["lsq", "ptr", "logistic"])
    parser.add_argument("--step", "-s", default=500, type=int)
    parser.add_argument("--gMethods", "-g", default=1, choices=[1, 2], type=int)
    parser.add_argument("--number", "-n", default=50, type=int)
    args = parser.parse_args()
    if args.methods != "logistic":
        part1(args.methods)
    else:
        part2(args.step, args.gMethods, args.number)






if __name__=="__main__":
    main()

