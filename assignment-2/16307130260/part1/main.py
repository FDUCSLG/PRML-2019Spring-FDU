import os
os.sys.path.append('../..')

from handout import get_linear_seperatable_2d_2c_dataset

import numpy as np
import matplotlib.pyplot as plt

def least_square():
    # get data
    data = get_linear_seperatable_2d_2c_dataset()

    # preprocess data
    # extend x using bias trick
    x = data.X
    x_extended = np.insert(x, 0, 1, axis=1)
    # Change two type into 1 and -1
    y = data.y
    c = y * 2 - 1

    # get w using pseudo-inverse
    w = np.linalg.pinv(x_extended) @ c

    # show the hyperplane
    xl = np.linspace(-1.5, 1.5, 2)
    yl = - (w[0] + w[1] * xl) / w[2]
    line_info = '{:.4f} + {:.4f} * x + {:.4f} * y = 0'.format(w[0], w[1], w[2])
    plt.plot(xl, yl, label=line_info)
    plt.legend()

    # predict
    pred_y = x_extended @ w
    pred_y[pred_y<0] = 0
    pred_y[pred_y>0] = 1
    plt.title("least square, acc: " + str(data.acc(pred_y)))
    print(data.acc(pred_y))

    # show the result
    """need to add legend and title"""
    data.plot(plt).show()

def perceptron(eta=1e-2):
    # get data
    data = get_linear_seperatable_2d_2c_dataset()

    # preprocess data
    # extend x using bias trick
    x = data.X
    x_extended = np.insert(x, 0, 1, axis=1)
    # Change two type into 1 and -1
    y = data.y
    c = y * 2 - 1
    # shuffle data
    N = len(x)
    rng = np.random.RandomState(233)
    rand_ind = np.arange(N)
    rng.shuffle(rand_ind)
    rand_x = x_extended[rand_ind]
    rand_c = c[rand_ind]

    # get w using SGD
    # initialize weight
    w = np.zeros(3)
    # SGD for w
    epoch = 0
    while True:
        flag = True
        epoch += 1
        for i in range(N):
            p = rand_x[i] @ w
            p = (p > 0) * 2 - 1
            if p != rand_c[i]:
                flag = False
                w += eta * rand_x[i] * rand_c[i]
        if flag:
            break
    print(epoch)

    # show the hyperplane
    xl = np.linspace(-1.5, 1.5, 2)
    yl = - (w[0] + w[1] * xl) / w[2]
    line_info = '{:.4f} + {:.4f} * x + {:.4f} * y = 0'.format(w[0], w[1], w[2])
    plt.plot(xl, yl, label=line_info)
    plt.legend()

    # predict
    pred_y = x_extended @ w
    pred_y[pred_y<0] = 0
    pred_y[pred_y>0] = 1
    plt.title("perceptron, acc: " + str(data.acc(pred_y)) + ", epoch: " + str(epoch))
    print(data.acc(pred_y))

    # show the result
    """need to add legend and title"""
    data.plot(plt).show()
    

if __name__ == "__main__":
    least_square()
    perceptron()