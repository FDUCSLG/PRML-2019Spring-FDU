import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import get_data
import numpy as np
import matplotlib.pyplot as plt
import math

N = 200
sampled_data = get_data(N)

# bins_num = 50
# plt.hist(sampled_data, normed=True, bins=bins_num)
# plt.show()


min_range = min(sampled_data)
max_range = max(sampled_data)
x = np.linspace(min_range, max_range, 2000)
px = np.empty(len(x))
# h = 2
#
# for i in range(len(x)):
#     px[i] = 1/N*np.sum( np.exp(-np.power((x[i]-np.array(sampled_data)), 2)/(2*h**2)) / pow((2*math.pi*h**2),0.5) )
#
# plt.plot(x, px)
# plt.show()


K = 1

for i in range(len(x)):
    for h in np.linspace(0, (max_range - min_range)/2, 100):
        cnt = 0
        for d in sampled_data:
            condition = (x[i] - h) <= d and (x[i] + h) >= d
            if condition:
                cnt += 1
        if cnt >= K:
            px[i] = 1/N*K/h
            break

plt.plot(x, px)
plt.show()

print(np.sum(px))