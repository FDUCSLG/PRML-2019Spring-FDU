import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import math
from handout import GaussianMixture1D
np.random.seed(0)
gm1d = GaussianMixture1D(mode_range=(0, 50))


# true distribution
def true_distribution(num_sample=1000):
    sampled_data = gm1d.sample([num_sample])
    min_range = min(gm1d.modes) - 3 * gm1d.std_range[1]
    max_range = max(gm1d.modes) + 3 * gm1d.std_range[1]
    xs = np.linspace(min_range, max_range, 2000)
    ys = np.zeros_like(xs)

    for l, s, w in zip(gm1d.modes, gm1d.stds, gm1d.weights):
        ys += ss.norm.pdf(xs, loc=l, scale=s) * w

    plt.plot(xs, ys)


# histogram estimation
def draw_histogram_estimation(num_data, bins):
    assert num_data > 0
    assert bins > 0
    sampled_data = get_data(num_data)
    plt.text(21, 0.22, r'num_data='+str(num_data))
    plt.text(21, 0.15, r'bin='+str(bins))
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.hist(sampled_data, bins, density=1, facecolor='g')
    true_distribution()
    # plt.show()


# kernel density estimation
def draw_kernel_density_estimation(num_data, h):
    assert num_data > 0
    sampled_data = get_data(num_data)
    sampled_data.sort()
    h_square2 = 2 * h * h
    abs_h = abs(h)
    prob = []
    for i in range(num_data):
        kernel_prob = 0
        for data in sampled_data:
            index = -1 * (sampled_data[i] - data)**2 / h_square2
            kernel_prob += math.exp(index)
        kernel_prob /= (math.sqrt(2 * math.pi) * num_data * abs_h)
        prob.append(kernel_prob)
    plt.text(21, 0.22, r'num_data='+str(num_data))
    plt.text(21, 0.15, r'h='+str(h))
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.axis([sampled_data[0] - 1, sampled_data[num_data - 1] + 1, 0, max(prob) + 0.05])
    plt.plot(sampled_data, prob)
    true_distribution()
    # plt.show()


# nearest neighbor
def draw_nearest_neighbor(num_data, k):
    assert num_data > 0
    assert k > 0 and k < num_data
    sampled_data = get_data(num_data)
    sampled_data.sort()
    max_dist = sampled_data[num_data - 1] - sampled_data[0] + 1
    prob = []
    for i in range(num_data):
        data = sampled_data[i]
        dist = 0
        l = i - 1
        r = i + 1
        cnt = 0
        flag = 1
        temp_dist = data - sampled_data[l] if l >= 0 else max_dist
        while cnt < k:
            if flag == 1:
                if r < num_data:
                    temp = sampled_data[r] - data
                    if temp > temp_dist:
                        dist = temp_dist
                        temp_dist = temp
                        flag = 0
                        l -= 1
                    else:
                        dist = temp
                        r += 1
                    cnt += 1
                else:
                    flag = 0
                    temp_dist = max_dist
            else:
                if l >= 0:
                    temp = data - sampled_data[l]
                    if temp > temp_dist:
                        dist = temp_dist
                        temp_dist = temp
                        flag = 1
                        r += 1
                    else:
                        dist = temp
                        l -= 1
                    cnt += 1
                else:
                    flag = 1
                    temp_dist = max_dist
        prob.append(k / (num_data * 2 * dist))
    plt.text(21, 0.22, r'num_data='+str(num_data))
    plt.text(21, 0.15, r'K='+str(k))
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.axis([sampled_data[0] - 1, sampled_data[num_data - 1] + 1, 0, max(prob) + 0.05])
    plt.plot(sampled_data, prob)
    true_distribution()
    # plt.show()


def question_1():
    plt.figure(num = 1, figsize=(80, 70))

    plt.subplot(331)
    draw_histogram_estimation(100, 50)
    plt.subplot(334)
    draw_histogram_estimation(500, 50)
    plt.subplot(337)
    draw_histogram_estimation(2000, 50)

    plt.subplot(332)
    draw_kernel_density_estimation(100, 0.4)
    plt.subplot(335)
    draw_kernel_density_estimation(500, 0.4)
    plt.subplot(338)
    draw_kernel_density_estimation(2000, 0.4)

    plt.subplot(333)
    draw_nearest_neighbor(100, 40)
    plt.subplot(336)
    draw_nearest_neighbor(500, 40)
    plt.subplot(339)
    draw_nearest_neighbor(2000, 40)

    plt.show()


def question_2():
    plt.figure(num = 1, figsize=(50, 40))

    plt.subplot(321)
    draw_histogram_estimation(200, 5)
    plt.subplot(322)
    draw_histogram_estimation(200, 20)
    plt.subplot(323)
    draw_histogram_estimation(200, 30)
    plt.subplot(324)
    draw_histogram_estimation(200, 50)
    plt.subplot(325)
    draw_histogram_estimation(200, 70)
    plt.subplot(326)
    draw_histogram_estimation(200, 100)

    plt.show()


def question_3():
    plt.figure(num = 1, figsize=(50, 40))

    plt.subplot(321)
    draw_kernel_density_estimation(100, 0.1)
    plt.subplot(322)
    draw_kernel_density_estimation(100, 0.3)
    plt.subplot(323)
    draw_kernel_density_estimation(100, 0.4)
    plt.subplot(324)
    draw_kernel_density_estimation(100, 0.5)
    plt.subplot(325)
    draw_kernel_density_estimation(100, 0.7)
    plt.subplot(326)
    draw_kernel_density_estimation(100, 1.0)

    plt.show()

def question_4():
    plt.figure(num = 1, figsize=(50, 40))

    plt.subplot(321)
    draw_nearest_neighbor(200, 1)
    plt.subplot(322)
    draw_nearest_neighbor(200, 10)
    plt.subplot(323)
    draw_nearest_neighbor(200, 30)
    plt.subplot(324)
    draw_nearest_neighbor(200, 50)
    plt.subplot(325)
    draw_nearest_neighbor(200, 70)
    plt.subplot(326)
    draw_nearest_neighbor(200, 100)

    plt.show()

# true_distribution()
# question_1()
# question_2()
# question_3()
# question_4()
# draw_nearest_neighbor(500, 40)
# draw_histogram_estimation(500, 50)
# draw_kernel_density_estimation(500, 0.4)
# plt.show()
