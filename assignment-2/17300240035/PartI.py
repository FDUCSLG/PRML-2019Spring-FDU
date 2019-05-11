import numpy as np
from matplotlib import pyplot as plt

#Part 1
def Display(d, w, str):
    plt.title(str)
    d.plot(plt)
    x = [-2.0, 2.0]
    y = [0, 0]
    y[0] = (- w[2] - w[0] * x[0]) / w[1]
    y[1] = (- w[2] - w[0] * x[1]) / w[1]
    plt.plot(x, y)
    plt.show()

def Accuracy(d, w, str):
    N = d.X.shape[0]
    cnt = 0
    for pointX, y in zip(d.X, d.y):
        if y == True and np.dot(np.append(pointX, 1), w) > 0: cnt += 1
        elif y == False and np.dot(np.append(pointX, 1), w) <= 0: cnt += 1
    print(str + ":", (cnt/N))

def Least_Square_Model(data):
    N = data.X.shape[0]
    phi = np.mat(np.c_[data.X, np.ones(N)])
    t = [1 if i == True else -1 for i in data.y]
    w = np.resize(((phi.T * phi).I * phi.T * np.mat(t).T), 3)
    print(w)
    return w

def Perceptron(data, eta = 0.1):
    w = np.array([.0, .0, .0])
    while True:
        old_w = w.copy()
        for pointX, y in zip(data.X, data.y):
            if y == True:
                if np.dot(np.append(pointX, 1), w) <= 0: w += eta * np.append(pointX, 1)
            else:
                if np.dot(np.append(pointX, 1), w) > 0: w -= eta * np.append(pointX, 1)
        if (old_w == w).all(): break
    print(w)
    return w