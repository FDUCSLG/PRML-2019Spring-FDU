import os
os.sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/..')
# use the above line of code to surpass the top module barrier
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt

def draw_hist(data_num, bin_num):
  sampled_data = get_data(data_num)
  plt.hist(sampled_data, normed=True, bins=bin_num)
  plt.show()

def count_gaussian(sampled_data, x, h):
  px = 0
  for sample in sampled_data:
    px += np.exp( -np.power(x-sample, 2)/(2*h*h) )
  px /= len(sampled_data) * np.power(2*np.pi*h*h, 0.5)
  # print(px)
  return px

def draw_kernel(data_num, h):
  sampled_data = get_data(data_num)
  num_max = max(sampled_data)
  num_min = min(sampled_data)
  print(num_min)
  print(num_max)
  x_array = []
  y_array = []
  for i in range(0, int( (num_max-num_min)/h )):
    x = i*h + h/2 + num_min
    px = count_gaussian(sampled_data, x, h)
    x_array.append(x)
    y_array.append(px)
  plt.plot(x_array, y_array)
  plt.show()