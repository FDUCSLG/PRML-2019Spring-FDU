import os
os.sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/..')
from handout import get_linear_seperatable_2d_2c_dataset, get_text_classification_datasets
import numpy as np
import matplotlib.pyplot as plt
import string

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
  vacab = {}
  regular = re.compile(r'[\s]+')
  for item in train.data:
    words = regular.split( item.translate( str.maketrans('', '', string.punctuation) ).lower() )
    train_item.append(words)
    for word in words:
      if not word in vacab:
        vacab[word] = len(vacab)
  # build multi-hot
  X = []
  for words in train_item:
    x = [1] + [0] * len(vacab)  # [1, 0, 0 ... 0]
    for word in words:
      if word in vacab:
        x[ vacab[word]+1 ] = 1  # enable word's bit
      else:
        print('"{}" not in vacabulary!'.format(word))
    X.append(x)  # append one record
  X = np.array(X)

  T = []
  for target in train.target:
    t = [0] * 4
    t[target] = 1
    T.append(t)
  T = np.array(T)

  W = np.array([ [0.01]*(D+1) ]*K)
  return X, T, W

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

def compute_loss(X, T, Y):
  N = len(X)
  loss = -( T*np.log(Y) ).sum()
  return loss/N

def det_derivative():
  pass

def logistic_regression():
  X, T, W = data_preprocess()
  W = np.array([ [0.01]*(D+1) ]*K)
  step = 50
  plot_y = []
  plot_x = [i for i in range(step)]
  a = 0.0001
  for i in range(step):
    Y = compute_y(X, W)
    plot_y.append(compute_loss(X, T, Y))
    dEW = compute_dew(X, T, Y)
    W -= a * dEW
    print('-----------')
    print(compute_y(X, W))
  plt.plot(plot_x, plot_y)
  plt.show()