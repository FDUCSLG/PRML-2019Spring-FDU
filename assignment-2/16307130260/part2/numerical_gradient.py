import numpy as np

from layer_utils import *

def numerical_gradient(f, x, df, h=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        old_value = x[ix]
        x[ix] = old_value + h
        pos = f(x).copy()
        x[ix] = old_value - h
        neg = f(x).copy()
        x[ix] = old_value

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad

def check_affine():
    np.random.seed(0)
    x = np.random.randn(10, 5)
    w = np.random.randn(5, 4)
    b = np.random.randn(4)
    dout = np.random.randn(10, 4)

    dw_num = numerical_gradient(lambda w:affine_forward(w, x, b)[0], w, dout)
    db_num = numerical_gradient(lambda b:affine_forward(w, x, b)[0], b, dout)

    _, cache = affine_forward(w, x, b)
    dw, db = affine_backward(dout, cache)

    print("dw - dw_num: ")
    print(dw - dw_num)
    print("db - db_num: ")
    print(db - db_num)

def check_softmax():
    np.random.seed(0)
    x = np.random.randn(10, 4)
    rand_ind = np.random.randint(4, size=(10)) - 1
    y = np.zeros((10,4))
    y[np.arange(10), rand_ind] = 1

    dx_num = numerical_gradient(lambda x:softmax_loss(x, y)[0], x, 1)

    _, dx = softmax_loss(x, y)

    print("dx - dx_num: ")
    print(dx - dx_num)

if __name__ == "__main__":
    check_affine()
    check_softmax()