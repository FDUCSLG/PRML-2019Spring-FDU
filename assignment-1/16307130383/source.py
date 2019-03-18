import os
os.sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/..')
# use the above line of code to surpass the top module barrier
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt

def draw_hist(data_num, bin_num):
  sampled_data = get_data(data_num)
  plt.hist(sampled_data, density=True, bins=bin_num)
  # plt.show()

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
  # print(num_min)
  # print(num_max)
  x_array = []
  y_array = []
  sampled_data = sorted(sampled_data)
  for sample in sampled_data:
    x = sample
    px = count_gaussian(sampled_data, x, h)
    x_array.append(x)
    y_array.append(px)
  # for i in range(0, int( (num_max-num_min)/h )):
  #   x = i*h + h/2 + num_min
  #   px = count_gaussian(sampled_data, x, h)
  #   x_array.append(x)
  #   y_array.append(px)
  plt.plot(x_array, y_array)
  # plt.show()

def get_volume(sampled_data, index, K):
  assert K <= len(sampled_data)
  # print('--------------')
  l = index
  r = index
  count = 1
  while(count < K):
    # print("{} {} {}".format( sampled_data[l], sampled_data[index], sampled_data[r] ))
    if l <= 0 or (r < len(sampled_data)-1 and sampled_data[index] - sampled_data[l-1] > sampled_data[r+1] - sampled_data[index]):
      count += 1
      r += 1
    else:
      count += 1
      l -= 1
  return sampled_data[r] - sampled_data[l]

def draw_nearest(data_num, K):
  sampled_data = get_data(data_num)
  x_array = []
  y_array = []
  N = len(sampled_data)
  sampled_data = sorted(sampled_data)
  for i, sample in enumerate(sampled_data):
    x = sample
    px = K /( N*get_volume(sampled_data, i, K) )
    x_array.append(x)
    y_array.append(px)
  plt.plot(x_array, y_array)
  # plt.show()

fig = plt.figure(figsize=(15, 6))
plt.subplot(4, 3, 1)
plt.ylabel('10 bins')
draw_hist(200, 10)
plt.subplot(4, 3, 4)
plt.ylabel('25 bins')
draw_hist(200, 25)
plt.subplot(4, 3, 7)
plt.ylabel('50 bins')
draw_hist(200, 50)
plt.subplot(4, 3, 10)
plt.ylabel('100 bins')
draw_hist(200, 100)
# plt.show()

plt.subplot(4, 3, 2)
plt.ylabel('h = 0.05')
draw_kernel(100, 0.05)
plt.subplot(4, 3, 5)
plt.ylabel('h = 0.1')
draw_kernel(100, 0.1)
plt.subplot(4, 3, 8)
plt.ylabel('h = 0.25')
draw_kernel(100, 0.25)
plt.subplot(4, 3, 11)
plt.ylabel('h = 1.0')
draw_kernel(100, 1.0)
# plt.show()

plt.subplot(4, 3, 3)
plt.ylabel('K = 2')
draw_nearest(200, 2)
plt.subplot(4, 3, 6)
plt.ylabel('K = 5')
draw_nearest(200, 5)
plt.subplot(4, 3, 9)
plt.ylabel('K = 10')
draw_nearest(200, 10)
plt.subplot(4, 3, 12)
plt.ylabel('K = 20')
draw_nearest(200, 20)
plt.show()