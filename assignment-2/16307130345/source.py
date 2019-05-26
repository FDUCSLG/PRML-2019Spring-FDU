import os
os.sys.path.append('..')
import numpy as np
from matplotlib import pyplot as plt
from handout import get_linear_seperatable_2d_2c_dataset
from handout import get_text_classification_datasets
import string
import math
import random

# =========================================== 
#
#  PART 1
#
# ===========================================

dataset_a, dataset_b = get_linear_seperatable_2d_2c_dataset().split_dataset()

def handleX(x):
  return [1, x[0], x[1]]

X = np.array([handleX(item) for item in dataset_a.X]) 

def transform_to_vector(t):
  return [1, 0] if t else [0, 1]

def get_func_value(A, B, C, x):
  return (A*x + B) / C

def draw_line(A, B, C):
  x1 = -1.5
  x2 = 1.5
  y1 = get_func_value(A, B, C, x1)
  y2 = get_func_value(A, B, C, x2)
  plt.plot([x1, x2], [y1, y2])

def get_accuracy(dataset, W_tsp, tag):
  correct_sum = 0
  input_vectors = np.array([handleX(item) for item in dataset.X]) 
  data_len = len(input_vectors)
  for i in range(data_len):
    t = np.dot(W_tsp, input_vectors[i].T)
    if tag == 'least_square':
      t = t[0] - t[1]
    if (dataset.y[i] == True and t >= 0) or (dataset.y[i] == False and t < 0):
      correct_sum += 1
  return correct_sum / data_len

# -----------  the least square algorithm  -------------

def least_square():
  T = np.array([transform_to_vector(item) for item in dataset_a.y])
  W_tsp = np.dot(T.T, np.linalg.pinv(X).T)
  return W_tsp

def draw_least_square_classification(dataset, tag):
  W_tsp = least_square()
  A = W_tsp[0][1]-W_tsp[1][1]
  B = W_tsp[0][0]-W_tsp[1][0]
  C = W_tsp[1][2]-W_tsp[0][2]
  accuracy = get_accuracy(dataset, W_tsp, 'least_square')
  dataset.plot(plt)
  draw_line(A, B, C)
  plt.title('least square model, ' + tag + ', accuracy = ' + str(accuracy))
  plt.show()

# -----------  the perceptron algorithm  -------------

def perceptron():
  W_tsp = np.random.random_sample((1, 3))
  for i in range(len(X)):
    a = np.dot(W_tsp, X[i].T)
    if a[0] >= 0 and dataset_a.y[i] == False:
      W_tsp = W_tsp - X[i]
    elif a[0] < 0 and dataset_a.y[i] == True:
      W_tsp = W_tsp + X[i]
  return W_tsp

def draw_perceptron_classification(dataset, tag):
  W_tsp = perceptron()
  accuracy = get_accuracy(dataset, W_tsp, 'perceptron')
  dataset.plot(plt)
  draw_line(W_tsp[0][1], W_tsp[0][0], -W_tsp[0][2])
  plt.title('perceptron model, ' + tag + ', accuracy = ' + str(accuracy))
  plt.show()

# ----------------------------------------------------

# draw_least_square_classification(dataset_a, 'training set')
# draw_least_square_classification(dataset_b, 'test set')
# draw_perceptron_classification(dataset_a, 'training set')
# draw_perceptron_classification(dataset_b, 'test set')



# =========================================== 
#
#  PART 2
#
# ===========================================

vocabulary_dict = {}          # word and index
min_count = 10
D = 0

# -----------  get vocabulary dictionary  -------------

def get_vocabulary_dict(data_set):
  record_dict = {}              # record all word and frequency
  for doc in data_set:
    for word in doc:
      if word in record_dict:
        record_dict[word] += 1
      else:
        record_dict[word] = 1
  temp_list = []
  for key, value in record_dict.items():
    if value >= min_count:
        temp_list.append(key)
  temp_list.sort()
  for i in range(len(temp_list)):
    vocabulary_dict[temp_list[i]] = i

# -----------  data preprocess  -------------

def split_string(doc):
  new_doc = ""
  for c in doc:
    if string.punctuation.find(c) == -1:
      if string.whitespace.find(c) != -1:
        c = ' '
      new_doc += c
  return new_doc.lower().split()

def preprocess_dataset(data_set):
  return [split_string(item) for item in data_set]

# -----------  get multi-hot-vector  -------------

def handle_string(doc):
  vector = [0]*(D+1)
  vector[0] = 1
  for item in doc:
    if item in vocabulary_dict:
      vector[vocabulary_dict[item]] = 1
  return vector

def handle_dataset(data_set):
  return np.array([handle_string(item) for item in data_set])

# -----------  function for logistic regression  -------------

def softmax(z):
  ez_sum = 0
  ez_list = []
  y_pred = []
  for item in z:
    ezi = math.exp(item)
    ez_sum += ezi
    ez_list.append(ezi)
  for item in ez_list:
    y_pred.append(item / ez_sum)
  return y_pred

def L_derivative_to_Wtsp(Xi, Yi, Ti, N):
  temp_list = []
  for i in range(4):
    temp_list.append((Yi[i]-Ti[i]) * Xi)
  return (1/N) * np.array(temp_list)  

def draw_curve(times, loss_list):
  plt.plot([i+1 for i in range(times)], loss_list)
  
# -----------  logistic regression by BGD -------------

def full_batch_gradient_descent(input_vectors, input_target, W_tsp, N, learning_rate):
  loss = 0
  for i in range(N):
    Zi = np.dot(W_tsp, input_vectors[i].T)
    Yi = softmax(Zi)
    loss += math.log(Yi[input_target[i]])
    # modify W_tsp
    Ti = [0]*4
    Ti[input_target[i]] = 1
    W_tsp = W_tsp - learning_rate*L_derivative_to_Wtsp(input_vectors[i], Yi, Ti, N)
  return (-1/N)*loss, W_tsp

def logistic_regression_byFBGD(input_vectors, input_target, times, learning_rate):
  loss_list = []
  initial_lrate = learning_rate
  N = len(input_vectors)
  W_tsp = np.random.random_sample((4, D+1))     # initialize W_tsp
  for i in range(times):
    if learning_rate > 0.01:
      learning_rate = initial_lrate * math.pow(0.95, math.floor((1+i)/10))
    loss, W_tsp = full_batch_gradient_descent(input_vectors, input_target, W_tsp, N, learning_rate)
    loss_list.append(loss)
  draw_curve(times, loss_list)
  return W_tsp

# -----------  logistic regression by SGD -------------

def stochastic_gradient_descent(input_vectors, input_target, W_tsp, N, learning_rate):
  loss = 0
  choosed_index = random.randint(0, N)
  for i in range(N):
    Zi = np.dot(W_tsp, input_vectors[i].T)
    Yi = softmax(Zi)
    loss += math.log(Yi[input_target[i]])
    # modify W_tsp
    if i == choosed_index:
      Ti = [0]*4
      Ti[input_target[i]] = 1
      W_tsp = W_tsp - learning_rate*L_derivative_to_Wtsp(input_vectors[i], Yi, Ti, N)
  return (-1/N)*loss, W_tsp

def logistic_regression_bySGD(input_vectors, input_target, times, learning_rate):
  loss_list = []
  initial_lrate = learning_rate
  N = len(input_vectors)
  W_tsp = np.random.random_sample((4, D+1))     # initialize W_tsp
  for i in range(times):
    if learning_rate > 0.01:
      learning_rate = initial_lrate * math.pow(0.95, math.floor((1+i)/1000))
    loss, W_tsp = stochastic_gradient_descent(input_vectors, input_target, W_tsp, N, learning_rate)
    loss_list.append(loss)
  draw_curve(times, loss_list)
  return W_tsp

# -----------  logistic regression by mini-BGD -------------

def mini_batch_gradient_descent(input_vectors, input_target, W_tsp, N, learning_rate, batch_size, begin_index):
  loss = 0
  for i in range(N):
    Zi = np.dot(W_tsp, input_vectors[i].T)
    Yi = softmax(Zi)
    loss += math.log(Yi[input_target[i]])
    # modify W_tsp
    if i >= begin_index and i < begin_index+batch_size:
      Ti = [0]*4
      Ti[input_target[i]] = 1
      W_tsp = W_tsp - learning_rate*L_derivative_to_Wtsp(input_vectors[i], Yi, Ti, N)
  return (-1/N)*loss, W_tsp

def logistic_regression_byBGD(input_vectors, input_target, times, learning_rate, batch_size):
  loss_list = []
  initial_lrate = learning_rate
  N = len(input_vectors)
  W_tsp = np.random.random_sample((4, D+1))     # initialize W_tsp
  for i in range(times):
    if learning_rate > 0.01:
      learning_rate = initial_lrate * math.pow(0.95, math.floor((1+i)/200))    
    loss, W_tsp = mini_batch_gradient_descent(input_vectors, input_target, W_tsp, N, learning_rate, batch_size, (i*batch_size)%N)
    loss_list.append(loss)
  draw_curve(times, loss_list)
  return W_tsp

# -----------  get the accuracy of test dataset  -------------

def get_dataset_test_accuracy(W_tsp, input_vector, input_target):
  correct_sum = 0
  input_sum = len(input_vector)
  for i in range(input_sum):
    Zi = np.dot(W_tsp, input_vector[i].T)
    Yi = softmax(Zi)
    yi = max(Yi)
    if Yi.index(yi) == input_target[i]:
      correct_sum += 1
  return correct_sum / input_sum

# ----------------------------------------------------

# dataset_train, dataset_test = get_text_classification_datasets()

# preprocessed_dataset_train = preprocess_dataset(dataset_train.data)
# get_vocabulary_dict(preprocessed_dataset_train)
# D = len(vocabulary_dict)

# dataset_train_vector = handle_dataset(preprocessed_dataset_train)
# W_tsp_byFBGD = logistic_regression_byFBGD(dataset_train_vector, dataset_train.target, 5000, 0.2)
# W_tsp_bySGD = logistic_regression_bySGD(dataset_train_vector, dataset_train.target, 18000, 15)
# W_tsp_byBGD = logistic_regression_byBGD(dataset_train_vector, dataset_train.target, 8000, 1.0, 64)

# preprocessed_dataset_test = preprocess_dataset(dataset_test.data)
# dataset_test_vector = handle_dataset(preprocessed_dataset_test)
# print(get_dataset_test_accuracy(W_tsp_byFBGD, dataset_test_vector, dataset_test.target))
# print(get_dataset_test_accuracy(W_tsp_bySGD, dataset_test_vector, dataset_test.target))
# print(get_dataset_test_accuracy(W_tsp_byBGD, dataset_test_vector, dataset_test.target))

# plt.show()
