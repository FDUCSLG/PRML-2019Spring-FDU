import sys
sys.path.append('..')
from handout import get_linear_seperatable_2d_2c_dataset
from handout import get_text_classification_datasets
from matplotlib import pyplot as plt
import numpy as np
import sklearn
import random
import string
import json

data_dot = get_linear_seperatable_2d_2c_dataset()

def least_square_model(ds):
    x_list = ds.X
    y_list = ds.y

    n = len(x_list)
    rec_list = []
    for i in range(n):
        rec_list.append((1, x_list[i][0], x_list[i][1]))
    rec_array = np.array(rec_list)
    X_mat = np.mat(rec_array)

    rec_list.clear()
    for i in range(n):
        if y_list[i]:
            rec_list.append(1)
        else:
            rec_list.append(-1)
    rec_array = np.array(rec_list)
    T_mat = np.mat(rec_array)

    W_mat = ((X_mat.T * X_mat).I * X_mat.T) * T_mat.T
    W_mat = W_mat.T
    x = np.arange(-2, 2)
    w0 = W_mat[0, 0]
    w1 = W_mat[0, 1]
    w2 = W_mat[0, 2]
    y = -(w1/w2)*x - w0/w2
    ds.plot(plt)
    plt.plot(x, y)
    plt.show()

    classify_res = []
    for i in range(n):
        classify_res.append(w0 + w1 * x_list[i][0] + w2 * x_list[i][1])
    print(w1, w2, w0, 1 - ds.acc(classify_res))


def perceptron_algorithm(ds):
    x_list = ds.X
    y_list = ds.y
    n = len(x_list)

    X_array = []
    T_array = []
    for i in range(n):
        X_array.append((1, x_list[i][0], x_list[i][1]))
        T_array.append(1 if y_list[i] else -1)

    W_array = [np.random.rand(), np.random.rand(), np.random.rand()]

    num = 0
    while num < 1000:
        for i in range(n):
            if T_array[i] * (W_array[0]*X_array[i][0] + W_array[1]*X_array[i][1] + W_array[2]*X_array[i][2]) < 0:
                W_array[0] += 0.1 * T_array[i] * X_array[i][0]
                W_array[1] += 0.1 * T_array[i] * X_array[i][1]
                W_array[2] += 0.1 * T_array[i] * X_array[i][2]

        for i in range(n):
            if T_array[i] * (W_array[0]*X_array[i][0] + W_array[1]*X_array[i][1] + W_array[2]*X_array[i][2]) < 0:
                num += 1
                continue
        break
    w0 = W_array[0]
    w1 = W_array[1]
    w2 = W_array[2]

    x = np.arange(-1, 2)
    y = -(w1/w2)*x - w0/w2

    ds.plot(plt)
    plt.plot(x, y)
    plt.show()

    classify_res = []
    for i in range(n):
        classify_res.append(w0 + w1 * x_list[i][0] + w2 * x_list[i][1])

    print(w0, w1, w2, 1 - ds.acc(classify_res))
    return ds.acc(classify_res)


def preprocess_get_dic_train(data_list, res_list):
    number_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    text_word = []
    dic = []
    sum = []

    for i in range(len(data_list)):
        data_list[i] = data_list[i].lower()
        for x in data_list[i]:
            if x in string.punctuation:
                data_list[i] = data_list[i].replace(x, ' ')
                continue
            elif x in string.whitespace:
                data_list[i] = data_list[i].replace(x, ' ')
            else:
                pass
        rec = []
        for row in data_list[i].split(" "):
            if row == ' ' or row == '':
                pass
            else:
                flag_t = 0
                for t in range(10):
                    if number_list[t] in row:
                        flag_t = 1
                        break
                if flag_t == 0:
                    rec.append(row)
        text_word.append(rec)

    for i in range(len(text_word)):
        temp = text_word[i]
        for j in range(len(temp)):
            if temp[j] in dic:
                sum[dic.index(temp[j])] += 1
            else:
                dic.append(temp[j])
                sum.append(1)
    word_dic = []
    for i in range(len(sum)):
        if not sum[i] < 10:
            word_dic.append(dic[i])
    k = len(word_dic)
    print('Dimension: ', k)
    word_vector = []
    for i in range(len(text_word)):
        v = []
        temp = text_word[i]
        for u in word_dic:
            if u in temp:
                v.append(1)
            else:
                v.append(0)
        word_vector.append(v)
    with open('train_text.json', 'w', encoding='utf-8') as fp:
        json.dump(word_vector, fp)

    train_res = []
    for i in range(len(res_list)):
        if res_list[i] == 0:
            train_res.append([0, 0, 0, 1])
        elif res_list[i] == 1:
            train_res.append([0, 0, 1, 0])
        elif res_list[i] == 2:
            train_res.append([0, 1, 0, 0])
        elif res_list[i] == 3:
            train_res.append([1, 0, 0, 0])
    with open('train_res.json', 'w', encoding='utf-8') as fp:
        json.dump(train_res, fp)

    return word_dic

def preprocess_get_test(data_list, res_list, word_dic):
    number_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    text_word = []

    for i in range(len(data_list)):
        data_list[i] = data_list[i].lower()
        for x in data_list[i]:
            if x in string.punctuation:
                data_list[i] = data_list[i].replace(x, ' ')
                continue
            elif x in string.whitespace:
                data_list[i] = data_list[i].replace(x, ' ')
            else:
                pass
        rec = []
        for row in data_list[i].split(" "):
            if row == ' ' or row == '':
                pass
            else:
                flag_t = 0
                for t in range(10):
                    if number_list[t] in row:
                        flag_t = 1
                        break
                if flag_t == 0:
                    rec.append(row)
        text_word.append(rec)

    word_vector = []
    for i in range(len(text_word)):
        v = []
        temp = text_word[i]
        for u in word_dic:
            if u in temp:
                v.append(1)
            else:
                v.append(0)
        word_vector.append(v)
    with open('test_text.json', 'w', encoding='utf-8') as fp:
        json.dump(word_vector, fp)

    train_res = []
    for i in range(len(res_list)):
        if res_list[i] == 0:
            train_res.append([0, 0, 0, 1])
        elif res_list[i] == 1:
            train_res.append([0, 0, 1, 0])
        elif res_list[i] == 2:
            train_res.append([0, 1, 0, 0])
        elif res_list[i] == 3:
            train_res.append([1, 0, 0, 0])
    with open('test_res.json', 'w', encoding='utf-8') as fp:
        json.dump(train_res, fp)


def softmax(x):
    exp_x = lambda t: np.exp(t - np.max(t))
    divide_x = lambda t: (1.0/np.sum(t)) * t
    x = np.apply_along_axis(exp_x, 1, x)
    x = np.apply_along_axis(divide_x, 1, x)
    return x


def get_update(x, y, w, b, lam):
    y_p = softmax(x * w + b.T)
    w_update = - x.T * (y - y_p)/x.shape[0] + 2 * lam * w
    b_update = (- np.sum((y - y_p), axis=0) / x.shape[0]).T
    return w_update, b_update


def get_loss(x, y, w, b, lam):
    y_p = softmax(x * w + b.T)
    loss = - np.trace((y * np.log(y_p).T)) / x.shape[0] + lam * np.sum(np.multiply(w, w))
    return loss


def get_accuracy(x, y, w, b):
    y_p = softmax(x * w + b.T)
    max_t = lambda t: 3 - np.argmax(t)
    return np.sum(np.apply_along_axis(max_t, 1, y_p) == y)/x.shape[0]


def logistic_algorithm_BGD(text_v, res_v, learn_rate, train_time, lam, sample_num, epsilon):
    text_a = np.array(text_v)
    res_a = np.array(res_v)
    n = text_v.shape[0]
    m = text_v.shape[1]
    c = res_v.shape[1]

    w = np.random.rand(m*c).reshape(m, c)
    b = np.random.rand(c).reshape(c, 1)
    loss_rec = [(0, 0)]

    cnt = 0
    while 1:
        sample_pick = []
        res_pick = []
        pick = random.sample(range(n), sample_num)
        for i in pick:
            sample_pick.append(text_a[i])
            res_pick.append(res_a[i])
        sample_pick = np.mat(sample_pick)
        res_pick = np.mat(res_pick)
        for i in range(sample_num):
            w_update, b_update = get_update(sample_pick, res_pick, w, b, lam)
            w -= learn_rate * w_update
            b -= learn_rate * b_update
        loss = get_loss(text_v, res_v, w, b, lam)
        cnt += 1
        print(cnt, '********loss = ', loss)
        if abs(loss - loss_rec[-1][1]) < epsilon or cnt >= train_time:
            loss_rec.append((cnt, loss))
            break
        loss_rec.append((cnt, loss))
    loss = [x[1] for x in loss_rec]
    plt.plot([i + 1 for i in range(cnt + 1)], loss)
    plt.show()
    return w, b


def logistic_algorithm_SGD(text_v, res_v, learn_rate, train_time, lam, epsilon):
    text_a = np.array(text_v)
    res_a = np.array(res_v)
    n = text_v.shape[0]
    m = text_v.shape[1]
    c = res_v.shape[1]

    w = np.random.rand(m * c).reshape(m, c)
    b = np.random.rand(c).reshape(c, 1)
    loss_rec = [(0, 0)]

    r_learn_rate = learn_rate
    cnt = 0
    while 1:
        sample_pick = []
        res_pick = []
        pick = random.randint(0, n-1)
        sample_pick.append(text_a[pick])
        res_pick.append(res_a[pick])
        sample_pick = np.mat(sample_pick)
        res_pick = np.mat(res_pick)
        w_update, b_update = get_update(sample_pick, res_pick, w, b, lam)
        w -= r_learn_rate * w_update
        b -= r_learn_rate * b_update
        loss = get_loss(text_v, res_v, w, b, lam)
        if loss > 8:
            r_learn_rate = 0.5
        elif loss > 3:
            r_learn_rate = 0.2
        else:
            r_learn_rate = learn_rate
        cnt += 1
        print(cnt, '********loss = ', loss)
        if abs(loss - loss_rec[-1][1]) < epsilon or cnt >= train_time:
            loss_rec.append((cnt, loss))
            break
        loss_rec.append((cnt, loss))
    loss = [x[1] for x in loss_rec]
    plt.plot([i + 1 for i in range(cnt + 1)], loss)
    plt.show()
    return w, b


def logistic_algorithm_FBGD(text_v, res_v, learn_rate, train_time, lam, epsilon):
    n = text_v.shape[0]
    m = text_v.shape[1]
    c = res_v.shape[1]

    w = np.random.rand(m * c).reshape(m, c)
    b = np.random.rand(c).reshape(c, 1)
    loss_rec = [(0, 0)]

    r_learn_rate = learn_rate
    cnt = 0
    while 1:
        sample_pick = text_v
        res_pick = res_v

        w_update, b_update = get_update(sample_pick, res_pick, w, b, lam)

        w -= r_learn_rate * w_update
        b -= r_learn_rate * b_update
        loss = get_loss(text_v, res_v, w, b, lam)
        if loss > 3:
            r_learn_rate = 1
        elif loss > 1:
            r_learn_rate = 0.5
        else:
            r_learn_rate = learn_rate
        cnt += 1
        print(cnt, '********loss = ', loss)
        if abs(loss - loss_rec[-1][1]) < epsilon or cnt >= train_time:
            loss_rec.append((cnt, loss))
            break
        loss_rec.append((cnt, loss))
    loss = [x[1] for x in loss_rec]
    plt.plot([i+1 for i in range(cnt+1)], loss)
    plt.show()
    return w, b


def part2():
    data_text_train, data_text_test = get_text_classification_datasets()
    try:
        with open('train_text.json', 'r', encoding='utf-8') as fp:
            train_text = json.load(fp)
        with open('train_res.json', 'r', encoding='utf-8') as fp:
            train_res = json.load(fp)
        with open('test_text.json', 'r', encoding='utf-8') as fp:
            test_text = json.load(fp)
        with open('test_res.json', 'r', encoding='utf-8') as fp:
            test_res = json.load(fp)
    except Exception as err:
        preprocess_get_test(data_text_test['data'], data_text_test['target'], preprocess_get_dic_train(data_text_train['data'], data_text_train['target']))
        with open('train_text.json', 'r', encoding='utf-8') as fp:
            train_text = json.load(fp)
        with open('train_res.json', 'r', encoding='utf-8') as fp:
            train_res = json.load(fp)
        with open('test_text.json', 'r', encoding='utf-8') as fp:
            test_text = json.load(fp)
        with open('test_res.json', 'r', encoding='utf-8') as fp:
            test_res = json.load(fp)

    train_text, train_res = np.mat(train_text), np.mat(train_res)
    test_text, test_res = np.mat(test_text), np.mat(test_res)


    # select the function for FBGD, SGD or BGD
    w, b = logistic_algorithm_FBGD(train_text, train_res, 0.1, 2000, 0.001, 1e-5)
    # w, b = logistic_algorithm_SGD(train_text, train_res, 0.1, 5000, 0.001, 1e-5)
    # w, b = logistic_algorithm_BGD(train_text, train_res, 0.1, 300, 0.001, 100, 1e-5)
    acc_train, acc_test = get_accuracy(train_text, data_text_train.target, w, b), get_accuracy(test_text, data_text_test.target, w, b)
    print('The training accuracy is: ', acc_train)
    print('The test accuracy is: ', acc_test)


if __name__ == '__main__':

    '''
    For Part1:
    
    least_square_model(data_dot)
    perceptron_algorithm(data_dot)
    
    '''



    '''
    For Part2:
    
    Simply select the functions for the three methods in function part2,  
    and change the arguments of learn_rate, learn_time and lambda.
    
    '''
    part2()
