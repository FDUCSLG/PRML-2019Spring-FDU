import os
os.sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/..')
from handout import get_linear_seperatable_2d_2c_dataset, get_text_classification_datasets
import numpy as np
import matplotlib.pyplot as plt
import string
import re
from random import randint

#------------
# Part 1
#------------
def least_square_model():
  data = get_linear_seperatable_2d_2c_dataset()
  X = []
  for x in data.X:
    X.append([1, x[0], x[1]])
  X = np.array(X)
  # calculate the W matrix
  X_ = np.linalg.inv(X.transpose() @ X) @ X.transpose()
  T = np.array([[int(x), 1-int(x)] for x in data.y])
  W = X_ @ T
  # draw the plot
  data.plot(plt)
  line_x = np.linspace(-1.5, 1.5, 10)
  line_y = -( W[0][0]-W[0][1]+( W[1][0]-W[1][1] )*line_x )/( W[2][0]-W[2][1] )
  plt.plot(line_x, line_y)
  plt.title('y={} x {}'.format(-( W[1][0]-W[1][1] )/( W[2][0]-W[2][1] ), -( W[0][0]-W[0][1] )/( W[2][0]-W[2][1] )))
  plt.show()

def perceptron_algorithm():
  data = get_linear_seperatable_2d_2c_dataset()
  w = np.array([-1.0, 1.0])
  offset = np.array([0.1, -0.30])
  # within 3 cycles the w can be figure out
  for cycle in range(3):
    for x, y in zip(data.X, data.y):
      x_ = x - offset
      if (x_ @ w) * (int(y) - 0.5) < 0:
        w += x_ * (int(y) - 0.5) * 2
        data.plot(plt)
        plt.scatter([x[0]], [x[1]], c='#ff0000')
        plt.scatter([offset[0]],[offset[1]], c='#0000ff')
        line_x = np.linspace(-1.5, 1.5, 10)
        line_y = -w[0]/w[1] * (line_x - offset[0]) + offset[1]
        plt.plot(line_x, line_y)
        plt.show()

#-------------
# Part 2
#-------------
def data_preprocess():
  train, test = get_text_classification_datasets()
  # build vacabulary
  train_item = []
  word_count = {}
  vacab = {}
  regular = re.compile(r'[\s]+')
  for item in train.data:
    words = regular.split( item.translate( str.maketrans('', '', string.punctuation) ).lower() )
    train_item.append(words)
    for word in words:
      if word in word_count:
        word_count[word] += 1
      else:
        word_count[word] = 1
  for word, count in word_count.items():
    if count >= 10:  # only record those >= 10 times
      vacab[word] = len(vacab)
  return train, test, vacab

def get_multi_hot(data, vacab):
  regular = re.compile(r'[\s]+')
  data_item = []
  for item in data.data:
    words = regular.split( item.translate( str.maketrans('', '', string.punctuation) ).lower() )
    data_item.append(words)
  # build multi-hot
  X = []
  for words in data_item:
    x = [1] + [0] * len(vacab)  # [1, 0, 0 ... 0]
    for word in words:
      if word in vacab:
        x[ vacab[word]+1 ] = 1  # enable word's bit
      else:
        print('"{}" not in vacabulary!'.format(word))
    X.append(x)  # append one record
  X = np.array(X)
  # T matrix
  T = []
  for target in data.target:
    t = [0] * 4
    t[target] = 1
    T.append(t)
  T = np.array(T)
  return X, T

def softmax(Y):
  for (i, y) in enumerate(Y):
    exp_sum = sum(np.exp(y))
    for (j, a) in enumerate(y):
      Y[i][j] = np.exp(a) / exp_sum
  return Y

def compute_y(X, W):
  Y = X @ W.transpose() # N * K
  Y = softmax(Y)
  return Y

def compute_dew(X, T, Y):
  # N = len(X)
  # K = len(W)
  # D = len( X[0] )-1
  # dEW = []
  # for j in range(K):
  #   dew = [0.0]*( D+1 )
  #   for n in range(N):
  #     dew += ( Y[n][j]-T[n][j] )*X[n]
  #   dEW.append(dew)
  # dEW = []
  # for j in range(K):
  #   dew = sum( ( Y[n][j]-T[n][j] )*X[0:N] )
  #   dEW.append(dew)
  dEW = (Y - T).transpose() @ X
  return dEW

def compute_dew_part(X, T, Y, i, j):
  dEW = ( (Y - T)[i:j] ).transpose() @ X[i:j]
  return dEW

def compute_loss(X, T, Y):
  N = len(X)
  loss = -( T*np.log(Y) ).sum()
  return loss/N

def det_derivative():
  pass

def full_logistic_regression(X, T, step, a):
  K = len(T[0])
  D = len(X[0]) - 1
  W = np.array([ [0.01]*(D+1) ]*K)
  # start to train
  # step = 100
  plot_y = []
  plot_x = [i for i in range(step)]
  # a = 0.0001
  for i in range(step):
    Y = compute_y(X, W)
    plot_y.append(compute_loss(X, T, Y))
    dEW = compute_dew(X, T, Y)
    W -= a * dEW
    print('-----------')
    print(Y)
  plt.xlabel('step')
  plt.ylabel('loss')
  plt.title('Full Batch Logistic Regression\nstep = {}   rate = {}'.format(step, a))
  plt.plot(plot_x, plot_y)
  plt.show()
  return W

def batched_logistic_regression(X, T, step, a, partition):
  N = len(X)
  K = len(T[0])
  D = len(X[0]) - 1
  W = np.array([ [0.01]*(D+1) ]*K)
  # start to train
  plot_y = []
  plot_x = [i for i in range(step)]
  for i in range(step):
    p_l = (i * partition) % N
    p_r = p_l + partition
    Y = compute_y(X, W)
    dEW = compute_dew_part(X, T, Y, p_l, p_r)
    W -= a * dEW
    plot_y.append(compute_loss(X, T, Y))
    print('-----------')
    print(Y)
  plt.xlabel('step')
  plt.ylabel('loss')
  plt.title('Batched Logistic Regression\nstep = {}   rate = {}   partition = {}'.format(step, a, partition))
  plt.plot(plot_x, plot_y)
  plt.show()
  return W

def one_logistic_regression(X, T, step, a):
  N = len(X)
  K = len(T[0])
  D = len(X[0]) - 1
  W = np.array([ [0.01]*(D+1) ]*K)
  # start to train
  plot_y = []
  plot_x = [i for i in range(step)]
  for i in range(step):
    j = randint(0, N-1)
    Y = compute_y(X, W)
    dEW = compute_dew_part(X, T, Y, j, j+1)
    W -= a * dEW
    plot_y.append(compute_loss(X, T, Y))
    print('-----------')
    print(Y)
  plt.xlabel('step')
  plt.ylabel('loss')
  plt.title('Stochastic Logistic Regression\nstep = {}   rate = {}'.format(step, a))
  plt.plot(plot_x, plot_y)
  plt.show()
  return W

def train_model(step, a, partition):
  train, test, vacab = data_preprocess()
  X, T = get_multi_hot(train, vacab)
  W = batched_logistic_regression(X, T, step, a, partition)  # 1000, 0.0001, 10
  return train, test, vacab, W

def test_model(Xtest, Ttest, W):
  Ytest = compute_y(Xtest, W)
  correct = 0
  wrong = 0
  for (y, t) in zip(Ytest, Ttest):
    if np.argmax(y) == np.argmax(t):  # if classification is right
      correct += 1
    else:
      wrong +=1
  return correct, wrong