import os
os.sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/..')
# use the above line of code to surpass the top module barrier
from handout import get_data, gm1d
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss


#-------------------------
# draw histogram estimation
#-------------------------
def draw_hist(data_num, bin_num):
  sampled_data = get_data(data_num)
  plt.hist(sampled_data, density=True, bins=bin_num)
  # plt.show()


#-------------------------
# draw kernel density estimation
#-------------------------
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
  # for sample in sampled_data:
  #   x = sample
  #   px = count_gaussian(sampled_data, x, h)
  #   x_array.append(x)
  #   y_array.append(px)
  plot_num = 1000
  for i in range(0, plot_num):
    x = i*(num_max - num_min)/plot_num + num_min
    px = count_gaussian(sampled_data, x, h)
    x_array.append(x)
    y_array.append(px)
  plt.plot(x_array, y_array)
  # plt.show()


#-------------------------
# draw nearest neighbor estimation
#-------------------------
# def get_volume(sampled_data, index, K):
#   assert K <= len(sampled_data)
#   # print('--------------')
#   l = index
#   r = index
#   count = 1
#   while(count < K):
#     # print("{} {} {}".format( sampled_data[l], sampled_data[index], sampled_data[r] ))
#     if l <= 0 or (r < len(sampled_data)-1 and sampled_data[index] - sampled_data[l-1] > sampled_data[r+1] - sampled_data[index]):
#       count += 1
#       r += 1
#     else:
#       count += 1
#       l -= 1
#   return sampled_data[r] - sampled_data[l]

def get_next_index(sampled_data, x):
  l = 0
  r = len(sampled_data)
  while r - l > 1:
    mid = int( (l+r)/2 )
    if sampled_data[mid] > x:
      r = mid
    else:
      l = mid
  return l

def get_volume_new(sampled_data, index, K):
  # print('------------')
  i = max(index - K + 2, 0)
  volume = 10000
  while i <= index and i+K <= len(sampled_data):
    # print([i, i+K-1, sampled_data[i+K-1], sampled_data[i]])
    volume = min(volume, sampled_data[i+K-1] - sampled_data[i])
    i += 1
  return volume

def draw_nearest(data_num, K):
  sampled_data = get_data(data_num)
  num_max = max(sampled_data)
  num_min = min(sampled_data)
  x_array = []
  y_array = []
  N = len(sampled_data)
  sampled_data = sorted(sampled_data)
  plot_num = 200
  for i in range(1, plot_num):
    x = i*(num_max - num_min)/plot_num + num_min
    index = get_next_index(sampled_data, x)
    px = K /( N*get_volume_new(sampled_data, index, K) )
    x_array.append(x)
    y_array.append(px)
  # for i, sample in enumerate(sampled_data):
  #   x = sample
  #   px = K /( N*get_volume(sampled_data, i, K) )
  #   x_array.append(x)
  #   y_array.append(px)
  plt.plot(x_array, y_array)
  # plt.show()


#-------------------------
# explore answer for 4 questions
#-------------------------
def show_date_num_influence():
  nums = [100, 500, 1000, 2000]
  fig = plt.figure(figsize=(16, 6))
  for i, num in enumerate(nums):
    plt.subplot(4, 3, i*3+1)
    plt.ylabel('num = {}'.format(num))
    plt.xlabel('hist')
    draw_hist(num, 50)
    plt.subplot(4, 3, i*3+2)
    plt.ylabel('num = {}'.format(num))
    plt.xlabel('kernel')
    draw_kernel(num, 0.2)
    plt.subplot(4, 3, i*3+3)
    plt.ylabel('num = {}'.format(num))
    plt.xlabel('nearest')
    draw_nearest(num, 15)
  plt.show()

def show_bin_method():
  N = 200
  sampled_data = get_data(N)
  stdev = np.std(sampled_data)
  # Sturge’s Rule   k = 1+log2(N)
  # Scott’s Rule    h = 3.49σN^(−1/3)
  # Rice’s Rule     k = pow(N, 1/3)*2
  names = ['Sturge’s Rule', 'Scott’s Rule', 'Rice’s Rule', '', '', '', '']
  bin_num = [int( 1 + np.ceil(np.log2(N)) ),
            int(np.ceil( (max(sampled_data) - min(sampled_data)) / (3.49*stdev/np.power(N, 1.0/3.0)) )),
            int(np.ceil( np.power(N, 1.0/3.0)*2 )), 20, 25, 30, 50]
  print(bin_num)
  fig = plt.figure(figsize=(6, 10))
  for i, bins in enumerate(bin_num):
    plt.subplot(3, 3, i+1)
    plt.title(names[i])
    plt.ylabel(bins)
    draw_hist(N, bins)
  plt.show()

def show_h_influence():
  N = 100
  sampled_data = get_data(N)
  sampled_data = sorted(sampled_data)
  distance = 0
  for i, sample in enumerate(sampled_data[1:]):
    distance += sample - sampled_data[i-1]
  distance /= (N-1)
  print(distance)
  # choose sqrt(avg(distance)) * 2
  hs = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5, np.power(distance, 0.5)*2]
  fig = plt.figure(figsize=(12, 6))
  for i, h in enumerate(hs):
    plt.subplot(4, 2, i+1)
    plt.ylabel(h)
    if i == 7:
      plt.xlabel('sqrt(average_interval) * 2')
    draw_kernel(N, h)
  plt.show()

def gm_plot(self, num_sample):
    sampled_data = self.sample([num_sample])
    min_range = min(self.modes) - 3 * self.std_range[1]
    max_range = max(self.modes) + 3 * self.std_range[1]
    xs = np.linspace(min_range, max_range, 2000)
    ys = np.zeros_like(xs)
    for l, s, w in zip(self.modes, self.stds, self.weights):
        ys += ss.norm.pdf(xs, loc=l, scale=s) * w
    plt.plot(xs, ys, scaley=2)

def show_k_influence():
  N = 200
  sampled_data = get_data(N)
  # Ks = [2, 5, 10, 20, 30, 40, 50, 60, 80, 100]
  Ks = [2, 5, 20, 30]
  fig = plt.figure(figsize=(12, 6))
  for i, K in enumerate(Ks):
    plt.subplot(4, 1, i+1)
    plt.ylim(0, 0.35)
    plt.ylabel('K = {}'.format(K))
    draw_nearest(N, K)
    gm_plot(gm1d, N)
  plt.show()


#-------------------------
# draw final plots
#-------------------------
fig = plt.figure(figsize=(14, 6))
plt.subplot(3, 2, 1)
plt.ylabel('bin = {}'.format(20))
plt.title('histogram estimation')
draw_hist(200, 20)
plt.subplot(3, 2, 5)
plt.ylabel('h = {}'.format(0.5))
plt.title('kernel density estimation')
draw_kernel(100, 0.5)

plt.subplot(3, 2, 2)
plt.ylim(0, 0.35)
plt.title('nearest neighbor estimation')
plt.ylabel('K = {}'.format(5))
draw_nearest(200, 5)
gm_plot(gm1d, 200)
plt.subplot(3, 2, 4)
plt.ylim(0, 0.35)
plt.ylabel('K = {}'.format(20))
draw_nearest(200, 20)
gm_plot(gm1d, 200)
plt.subplot(3, 2, 6)
plt.ylim(0, 0.35)
plt.ylabel('K = {}'.format(30))
draw_nearest(200, 30)
gm_plot(gm1d, 200)
plt.show()